import dagster as dg
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split

from ..ml.resources import mlflow_resource
from .resources import FraudDataConfig


@dg.asset(
    description="Dowload data for fraud detection",
    compute_kind="python",
    group_name="ml_fraud_ingest",
    resource_defs={"fraud_data": FraudDataConfig(), "mlflow_tracking": mlflow_resource
    }
)
def fraud_detection(
    context: dg.AssetExecutionContext,
) -> pd.DataFrame:

    # Simulate data ingestion
    # data = dg.MaterializeResult(
    #     value={
    #         "TransactionID": [1, 2, 3, 4, 5],
    #         "Amount": [100.0, 250.5, 275.0, 300.0, 150.0],
    #         "isFraud": [0, 1, 0, 1, 0]
    #     }
    # )

    # Load data from csv
    fraud_data_config = context.resources.fraud_data.data_source
    context.log.info(f"Processing file:\n {fraud_data_config}")

    # Convert to pandas DataFrame
    df: pd.DataFrame = pd.read_csv(fraud_data_config)
    context.log.info(f"Pandas DataFrame:\n {df}")

    mlflow_client = context.resources.mlflow_tracking

    dataset = mlflow_client.data.from_pandas(df, name="fraud_detection_data")
    mlflow_client.log_input(dataset=dataset, context="training")

    # columns = [dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]

    return df


# @dg.asset(
#     description="Cleans and renames columns",
#     resource_defs={"mlflow_tracking": mlflow_resource},
#     compute_kind="python",
#     group_name="ml_fraud_transform"
# )
# def clean_fraud_data(
#     context: dg.AssetExecutionContext,
#     fraud_detection: pd.DataFrame
# ) -> pd.DataFrame:  # dg.MaterializeResult
#     mlflow_client = context.resources.mlflow_tracking
#     context.log.info("Starting data cleaning.")

#     # 1. Ensure 'time' column is datetime type
#     fraud_detection["Time"]: pd.Series = pd.to_timedelta(fraud_detection["Time"], unit="s")

#     # Log the cleaned data to MLflow
#     df_dataset = mlflow_client.data.from_pandas(fraud_detection, name="cleaned_fraud_detection_dataset")
#     mlflow_client.log_input(dataset=df_dataset, context="training")

#     # columns = [dg.TableColumn(k, str(v)) for k, v in formatted_df.dtypes.to_dict().items()]

#     return fraud_detection


@dg.asset(
    description="Split data into train and test",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_model"
)
def train_test_split_data(fraud_detection: pd.DataFrame) -> dict:
    """Split data into 80–20 train/test sets."""
    X = fraud_detection.drop(columns=["Class"])
    y = fraud_detection["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


@dg.asset(
    description="Perform cross validation on the data",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_model"
)
def cross_validation_results(train_test_split_data: dict, tuned_random_forest: dict) -> dict:
    """Perform 3-fold cross-validation."""
    X_train = train_test_split_data["X_train"]
    y_train = train_test_split_data["y_train"]
    model = tuned_random_forest["best_model"]  # use the model from tuned_random_forest

    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
    return {"mean_accuracy": scores.mean(), "scores": scores.tolist()}


@dg.asset(
    description="Perform hyperparameter tuning",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_model"
)
def tuned_random_forest(train_test_split_data: dict) -> dict:
    """Perform hyperparameter tuning using GridSearchCV."""
    X_train = train_test_split_data["X_train"]
    y_train = train_test_split_data["y_train"]

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                               param_grid, cv=3,
                               scoring="accuracy",
                               n_jobs=-1,
                               error_score='raise'  # This will show the exact error for the failing fit
    )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    return {
        "best_model": best_model,
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
    }


@dg.asset(
    description="Test model on the test data",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_evaluate"
)
def fraud_test_model(train_test_split_data: dict, tuned_random_forest: dict) -> dict:
    """Evaluate tuned model on test data."""
    X_test = train_test_split_data["X_test"]
    y_test = train_test_split_data["y_test"]
    model = tuned_random_forest["best_model"]

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return {"accuracy": acc, "report": report}
