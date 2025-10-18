import io
import os
import time
from itertools import product
from typing import Optional

import dagster as dg
import dagster_slack
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn as ms
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .resources import FraudDataAPI, FraudPromotionConfig, mlflow_resource


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
            "dagster/column_schema": dg.TableSchema(columns=columns),
        },
    )


def _dump_figure(fig: Figure) -> np.ndarray:
    """
    Helper function for saving matplotlib figure to numpy array
    :param:
        fig: matplotlib.figure.Figure
    :return:
        image_numpy: np.array
    """

    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    image = np.array(Image.open(buf))
    buf.close()

    return image


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
    resource_defs={"mlflow_fraud": mlflow_resource},
    group_name="ml_fraud_transform",
)
def transformed_data(
    context: dg.AssetExecutionContext,
    fraud_data: pd.DataFrame,
) -> dg.MaterializeResult:
    context.log.info("Data transformation")

    tracking_uri: str = context.resources.mlflow_fraud.tracking_uri
    experiment_name: str = context.resources.mlflow_fraud.experiment_name

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    df_trans = fraud_data.drop(columns="Time")

    dataset = mlflow.data.from_pandas(df_trans, name="transformed_fraud_dataset")  # pyright: ignore

    with mlflow.start_run(run_name=f"input_transform_{time.time():.0f}"):
        mlflow.log_input(dataset=dataset, context="preprocessing")

    return _pd_to_result(df_trans)


@dg.multi_asset(
    description="Train test split",
    compute_kind="python",
    resource_defs={"mlflow_fraud": mlflow_resource},
    outs={"training_data": dg.AssetOut(), "test_data": dg.AssetOut()},
    group_name="ml_fraud_split",
)
def train_test_data(context: dg.AssetExecutionContext, transformed_data: pd.DataFrame):
    context.log.info("Splitting data into train and test")

    tracking_uri: str = context.resources.mlflow_fraud.tracking_uri
    experiment_name: str = context.resources.mlflow_fraud.experiment_name

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    split_idx: list[list[int]] = train_test_split(
        list(range(transformed_data.shape[0])), test_size=0.2, random_state=42, stratify=transformed_data["Class"]
    )

    df_train = transformed_data.iloc[split_idx[0], :]
    df_test = transformed_data.iloc[split_idx[1], :]

    train_dataset = mlflow.data.from_pandas(df_train, name="train_fraud_dataset")  # pyright: ignore
    test_dataset = mlflow.data.from_pandas(df_test, name="test_fraud_dataset")  # pyright: ignore

    with mlflow.start_run(run_name=f"train_test_split_{time.time():.0f}"):
        mlflow.log_input(dataset=train_dataset, context="training")
        mlflow.log_input(dataset=test_dataset, context="testing")

    train_data = _pd_to_result(df_train, dg.AssetKey("training_data"))
    test_data = _pd_to_result(df_test, dg.AssetKey("test_data"))

    return train_data, test_data


@dg.asset(
    description="Model hyperparameter tuning",
    resource_defs={"mlflow_fraud": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_train",
)
def tuned_hyperparameters(
    context: dg.AssetExecutionContext,
    training_data: pd.DataFrame,
) -> dg.MaterializeResult:
    context.log.info("Hyperparameter tuning")

    tracking_uri: str = context.resources.mlflow_fraud.tracking_uri
    experiment_name: str = context.resources.mlflow_fraud.experiment_name

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X_train = training_data.drop(columns="Class")
    y_train = training_data["Class"]

    p_grid = {
        "n_estimators": [10, 50, 100, 200],
        "random_state": [42],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    params_key = list(p_grid.keys())
    params_val = [p_grid[k] for k in params_key]

    best_param, best_score = None, -1.0
    report = []

    with mlflow.start_run(run_name=f"hyperparameter_tuning_rf_{time.time():.0f}"):
        run_num = 0
        for params in product(*params_val):
            train_param: dict = {params_key[i]: params[i] for i in range(len(params_key))}
            run_name = f"gridsearch_{run_num}_params_{'-'.join(map(str, params))}"

            context.log.info(f"Training and evaluating model on hyperparameters: {train_param}")

            with mlflow.start_run(run_name=run_name, nested=True):
                mlflow.log_params(train_param)

                model = RandomForestClassifier(**train_param)
                scores = cross_val_score(
                    estimator=model,
                    X=X_train,
                    y=y_train,
                    scoring="f1",
                    cv=cv,
                )

                score = np.mean(scores)

                mlflow.log_metric("Average f1-score", score)

                if best_score < score:
                    best_param = train_param
                    best_score = score

                report.append({**train_param, "f1-score": score})

    return dg.MaterializeResult(
        value=best_param,
        metadata={
            "report": dg.MetadataValue.md(pd.DataFrame(report).to_markdown() or ""),
            "best parameters": dg.MetadataValue.md(
                pd.DataFrame(best_param, index=pd.Index(["value"])).to_markdown() or ""
            ),
            "best f1 score": f"{best_score:.2f}",
        },
    )


@dg.asset(
    description="Model training",
    resource_defs={"mlflow_fraud": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_train",
)
def tuned_model(
    context: dg.AssetExecutionContext,
    training_data: pd.DataFrame,
    tuned_hyperparameters: dict,
) -> dg.MaterializeResult:
    context.log.info("Train model on best hyperparameters")

    tracking_uri: str = context.resources.mlflow_fraud.tracking_uri
    experiment_name: str = context.resources.mlflow_fraud.experiment_name
    registered_model_name: str = context.resources.mlflow_fraud.registered_model_name

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X_train = training_data.drop(columns="Class")
    y_train = training_data["Class"]

    registered_model_name = "tuned-fraud-detector"

    model = RandomForestClassifier(**tuned_hyperparameters)
    metrics, model_version_info = {}, {}

    with mlflow.start_run(run_name=f"model_training_{time.time():.0f}") as run:
        mlflow.log_params(tuned_hyperparameters)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_train)
        metrics["f1-score"] = f1_score(y_train, y_pred)
        metrics["accuracy"] = accuracy_score(y_train, y_pred)
        metrics["precision"] = precision_score(y_train, y_pred)
        metrics["recall"] = recall_score(y_train, y_pred)
        mlflow.log_metrics(metrics)

        cm = confusion_matrix(y_train, y_pred)

        fig, ax = plt.subplots()
        display = ConfusionMatrixDisplay(cm)
        display.plot(ax=ax)

        image = _dump_figure(fig)

        mlflow.log_image(image, "train_confusion_matrix.png")

        log_model_info = ms.log_model(
            sk_model=tuned_model,
            artifact_path="tuned_fraud_detector",
            input_example=X_train.iloc[: min(5, X_train.shape[0]), :],
            registered_model_name=registered_model_name,
        )
        context.log.info(f"Model logged to MLflow Run ID: {run.info.run_id}")
        context.log.info(f"Logged model artifact URI: {log_model_info.model_uri}")

        model_versions = mlflow.search_model_versions(filter_string=f"name='{registered_model_name}'")
        matching_versions = [mv for mv in model_versions if mv.run_id == run.info.run_id]

        if matching_versions:
            registered_model_version = matching_versions[0]
            model_version_info = {
                "name": registered_model_version.name,
                "version": registered_model_version.version,
                "status": registered_model_version.status,
                "stage": registered_model_version.current_stage,
                "model_uri": f"models:/{registered_model_version.name}/{registered_model_version.version}",
            }
            context.log.info("Successfully retrieved registered model version info from registry.")
        else:
            context.log.error(
                f"Could not find registered model version for run ID {run.info.run_id} "
                f"and name '{registered_model_name}'."
            )
            raise Exception("Failed to retrieve registered model version details after logging.")

    return dg.MaterializeResult(
        value={"tuned_model": model, **model_version_info},
        metadata={
            "f1 score": f"{metrics['f1-score']:.2f}",
            "accuracy score": f"{metrics['accuracy']:.2f}",
            "precision score": f"{metrics['precision']:.2f}",
            "recall score": f"{metrics['recall']:.2f}",
            **model_version_info,
        },
    )


@dg.asset(
    description="Model evaluation",
    resource_defs={"mlflow_fraud": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_eval",
)
def model_evaluation(
    context: dg.AssetExecutionContext,
    test_data: pd.DataFrame,
    tuned_model: dict,
) -> dg.MaterializeResult:
    context.log.info("Evaluating model on test set")

    tracking_uri: str = context.resources.mlflow_fraud.tracking_uri
    experiment_name: str = context.resources.mlflow_fraud.experiment_name

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    X_test = test_data.drop(columns="Class")
    y_test = test_data["Class"]

    if "tuned_model" not in tuned_model:
        raise Exception("No model passed for evaluation")

    model = tuned_model["tuned_model"]

    metrics = {}

    with mlflow.start_run(run_name=f"model_eval_{time.time():.0f}"):
        y_pred = model.predict(X_test)
        metrics["f1-score"] = f1_score(y_test, y_pred)
        metrics["accuracy"] = accuracy_score(y_test, y_pred)
        metrics["precision"] = precision_score(y_test, y_pred)
        metrics["recall"] = recall_score(y_test, y_pred)
        mlflow.log_metrics(metrics)

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        display = ConfusionMatrixDisplay(cm)
        display.plot(ax=ax)

        image = _dump_figure(fig)

        mlflow.log_image(image, "test_confusion_matrix.png")

    return dg.MaterializeResult(
        value=metrics,
        metadata={
            "f1 score": f"{metrics['f1-score']:.2f}",
            "accuracy score": f"{metrics['accuracy']:.2f}",
            "precision score": f"{metrics['precision']:.2f}",
            "recall score": f"{metrics['recall']:.2f}",
        },
    )


@dg.asset(
    description="Slack Report",
    resource_defs={
        "slack_bot": dagster_slack.SlackResource(token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN")),
    },
    compute_kind="python",
    group_name="ml_fraud_eval",
)
def slack_eval_report(
    context: dg.AssetExecutionContext,
    model_evaluation: dict,
):
    context.log.info("Sending slack message")

    slack: dagster_slack.SlackResource = context.resources.slack_bot
    slack.get_client().chat_postMessage(
        channel="aims_course_october2025",
        text=f"""\
Classification Report
{"\n".join(f"- {k}: {v}" for (k, v) in model_evaluation.items())}

Run by: {os.environ.get("GITHUB_USER", "default")}
""",
    )


@dg.asset(
    description="Model promotion to staging",
    resource_defs={"mlflow_fraud": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_promotion",
)
def model_promotion_staging(
    context: dg.AssetExecutionContext,
    config: FraudPromotionConfig,
    tuned_model: dict,
    model_evaluation: dict,
) -> dg.MaterializeResult:
    context.log.info("Evaluating model for staging")

    tracking_uri: str = context.resources.mlflow_fraud.tracking_uri
    experiment_name: str = context.resources.mlflow_fraud.experiment_name
    mlflow_client: mlflow.MlflowClient = context.resources.mlflow_fraud.client

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if ("name" not in tuned_model) or ("version" not in tuned_model):
        raise Exception("No model passed for promotion")

    model_name = tuned_model["name"]
    model_version = tuned_model["version"]

    f1_score = model_evaluation.get("f1-score", 0.0)
    acc_score = model_evaluation.get("accuracy", 0.0)

    F1_THRESH = config.staging_f1_threshold
    ACC_THRESH = config.staging_acc_threshold

    if (f1_score > F1_THRESH) and (acc_score > ACC_THRESH):
        context.log.info(f"Model '{model_name}' (version {model_version}) meets criteria. Promoting to Staging")
        mlflow_client.transition_model_version_stage(name=model_name, version=model_version, stage="Staging")

        tuned_model["stage"] = "Staging"

        return dg.MaterializeResult(
            value={"promote_status": "success", **tuned_model},
            metadata={
                "promote_status": "succes",
                **{k: v for (k, v) in tuned_model.items() if k not in ["tuned_model"]},
            },
        )

    context.log.info(f"Model '{model_name}' (version {model_version}) failed to meet criteria.")
    return dg.MaterializeResult(
        value={"promote_status": "failure", **tuned_model},
        metadata={
            "promote_status": "failure",
            **{k: v for (k, v) in tuned_model.items() if k not in ["tuned_model"]},
        },
    )


@dg.asset(
    description="Model promotion to production",
    resource_defs={"mlflow_fraud": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_promotion",
)
def model_promotion_production(
    context: dg.AssetExecutionContext,
    model_promotion_staging: dict,
) -> dg.MaterializeResult:
    context.log.info("Evaluating model for production")

    tracking_uri: str = context.resources.mlflow_fraud.tracking_uri
    experiment_name: str = context.resources.mlflow_fraud.experiment_name
    mlflow_client: mlflow.MlflowClient = context.resources.mlflow_fraud.client

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if "promote_status" not in model_promotion_staging:
        raise Exception("No staging result")

    if ("name" not in model_promotion_staging) or ("version" not in model_promotion_staging):
        raise Exception("No model passed for promotion")

    model_name = model_promotion_staging["name"]
    model_version = model_promotion_staging["version"]

    if model_promotion_staging["promote_status"] == "failure":
        return dg.MaterializeResult(
            value={**model_promotion_staging},
            metadata={
                "error": "model did not pass staging",
                **{k: v for (k, v) in model_promotion_staging.items() if k not in ["tuned_model"]},
            },
        )

    # Some promotion criteria
    promote = True

    if not promote:
        return dg.MaterializeResult(
            value={**model_promotion_staging, "promote_status": "failure"},
            metadata={
                "error": "model did not pass promotion criteria",
                **{k: v for (k, v) in model_promotion_staging.items() if k not in ["tuned_model"]},
                "promote_status": "failure",
            },
        )

    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        if mv.current_stage == "Production":
            context.log.info(f"Archiving previous Production model '{mv.name}' (version {mv.version})")
            mlflow_client.transition_model_version_stage(name=mv.name, version=mv.version, stage="Archived")

    context.log.info(f"Model '{model_name}' (version {model_version}) meets criteria. Promoting to Production")
    mlflow_client.transition_model_version_stage(name=model_name, version=model_version, stage="Production")

    model_promotion_staging["stage"] = "Production"

    return dg.MaterializeResult(
        value={**model_promotion_staging},
        metadata={
            **{k: v for (k, v) in model_promotion_staging.items() if k not in ["tuned_model"]},
        },
    )
