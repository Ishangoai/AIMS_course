from itertools import product
from typing import Optional

import dagster as dg
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from ..ml.resources import mlflow_resource
from .resources import EXPERIMENT_NAME, FraudDataAPI


def _pd_to_result(df: pd.DataFrame, key: Optional[dg.AssetKey] = None) -> dg.MaterializeResult:
    """
    Helper wrapper for turning pd.Dataframe to dg.MaterializeResult with predefined metadata
    :param:
        df: pd.Dataframe
    :return:
        result: dg.MaterializeResult
    """
    columns = [dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]

    return dg.MaterializeResult(
        asset_key=key,
        value=df,
        metadata={
            "preview": dg.MetadataValue.md(df.head().to_markdown() or ""),
            "dagster/row_count": len(df),
            "dagster/column_schema": dg.TableSchema(columns=columns)
        }
    )


@dg.asset(
    description="Download credit card fraud data",
    compute_kind="python",
    resource_defs={"data_api": FraudDataAPI()},
    group_name="ml_fraud_ingest",
)
def fraud_data(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:

    context.log.info("Downloading the Data")

    api_url = context.resources.data_api.url
    df = pd.read_csv(api_url)

    return _pd_to_result(df)


@dg.asset(
    description="Data transformation",
    compute_kind="python",
    group_name="ml_fraud_transform"
)
def transformed_data(
    context: dg.AssetExecutionContext,
    fraud_data: pd.DataFrame,
) -> dg.MaterializeResult:

    df_trans = fraud_data.drop(columns="Time")

    return _pd_to_result(df_trans)


@dg.multi_asset(
    description="Train test split",
    compute_kind="python",
    outs={"training_data": dg.AssetOut(), "test_data": dg.AssetOut()},
    group_name="ml_fraud_split",
)
def train_test_data(
    context: dg.AssetExecutionContext,
    transformed_data: pd.DataFrame
):
    df_train, df_test = train_test_split(
        transformed_data,
        test_size=0.2,
        stratify=transformed_data["Class"]
    )

    train_data = _pd_to_result(df_train, dg.AssetKey("training_data"))
    test_data = _pd_to_result(df_test, dg.AssetKey("test_data"))

    return train_data, test_data


@dg.asset(
    description="Model hyperparameter tuning",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_train"
)
def tuned_hyperparameters(
    context: dg.AssetExecutionContext,
    training_data: pd.DataFrame,
) -> dg.MaterializeResult:

    mlflow_client = context.resources.mlflow_tracking

    X_train = training_data.drop(columns="Class")
    y_train = training_data["Class"]

    try:
        experiment = mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            experiment = mlflow_client.create_experiment(EXPERIMENT_NAME)
            experiment_id = experiment.experiment_id
        else:
            experiment_id = experiment.experiment_id
    except Exception:
        experiment_id = mlflow_client.create_experiment(EXPERIMENT_NAME)

    p_grid = {
        "n_estimators": [10, 50, 100, 200],
        "random_state": [42],
    }

    cv = StratifiedKFold(
        n_splits=3, shuffle=True, random_state=42
    )

    params_key = list(p_grid.keys())
    params_val = [p_grid[k] for k in params_key]

    best_param, best_score = None, -1.0

    run_num = 0

    for params in product(*params_val):
        train_param = {params_key[i]: params[i] for i in range(len(params_key))}
        run_name = f"gridsearch_{run_num}_params_{"-".join(map(str, params))}"

        with mlflow_client.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
            # mlflow_client.log_params(params)

            model = RandomForestClassifier(**train_param)
            scores = cross_val_score(
                estimator=model,
                X=X_train,
                y=y_train,
                scoring="f1",
                cv=cv,
            )

            score = np.mean(scores)

            mlflow_client.log_metric("Average score", score)

        if best_score < score:
            best_param = train_param
            best_score = score

    return dg.MaterializeResult(
        value=best_param,
        metadata={
            "parameters": dg.MetadataValue.md(pd.DataFrame(best_param, index=["value"]).to_markdown() or ""),
            "f1 score": f"{best_score:.2f}"
        }
    )


@dg.asset(
    description="Model training",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_train"
)
def tuned_model(
    context: dg.AssetExecutionContext,
    training_data: pd.DataFrame,
    tuned_hyperparameters: dict,
) -> dg.MaterializeResult:

    mlflow_client = context.resources.mlflow_tracking

    X_train = training_data.drop(columns="Class")
    y_train = training_data["Class"]

    model = RandomForestClassifier(**tuned_hyperparameters)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_train)

    f1 = f1_score(y_train, y_pred)
    acc = accuracy_score(y_train, y_pred)
    cm = confusion_matrix(y_train, y_pred)

    report = classification_report(y_train, y_pred, output_dict=True)

    return dg.MaterializeResult(
        value=model,
        metadata={
            "f1 score": f"{f1:.2f}",
            "accuracy score": f"{acc:.2f}",
            "confusion matrix": dg.MetadataValue.md(pd.DataFrame(cm).to_markdown() or ""),
            "classification report": dg.MetadataValue.md(pd.DataFrame(report).iloc[:, :2].to_markdown() or ""),
        }
    )


@dg.asset(
    description="Model evaluation",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_eval"
)
def model_evaluation(
    context: dg.AssetExecutionContext,
    test_data: pd.DataFrame,
    tuned_model: RandomForestClassifier,
) -> dg.MaterializeResult:

    mlflow_client = context.resources.mlflow_tracking

    X_test = test_data.drop(columns="Class")
    y_test = test_data["Class"]

    y_pred = tuned_model.predict(X_test)

    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)

    return dg.MaterializeResult(
        value=f1,
        metadata={
            "f1 score": f"{f1:.2f}",
            "accuracy score": f"{acc:.2f}",
            "confusion matrix": dg.MetadataValue.md(pd.DataFrame(cm).to_markdown() or ""),
            "classification report": dg.MetadataValue.md(pd.DataFrame(report).iloc[:, :2].to_markdown() or ""),
        }
    )
