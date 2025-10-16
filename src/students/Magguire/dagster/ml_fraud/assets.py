import os
import pickle

import dagster as dg
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

    # Load data from csv
    fraud_data_config = context.resources.fraud_data.data_source
    context.log.info(f"Processing file:\n {fraud_data_config}")

    # Convert to pandas DataFrame
    df: pd.DataFrame = pd.read_csv(fraud_data_config)
    context.log.info(f"Pandas DataFrame shape: {df.shape}")

    mlflow_client = context.resources.mlflow_tracking

    dataset = mlflow_client.data.from_pandas(df, name="fraud_detection_data")
    mlflow_client.log_input(dataset=dataset, context="training")

    # Add a light materialization with some metadata so Dagster shows it
    try:
        context.log.info("Materializing fraud_detection dataset metadata.")
        context.log_event(dg.AssetMaterialization(asset_key=dg.AssetKey("fraud_detection"), metadata={
            "rows": df.shape[0],
            "columns": df.shape[1],
            "source": fraud_data_config
        }))
    except Exception:
        # Older/newer dagster APIs differ; fall back to logging
        context.log.info(f"fraud_detection metadata: rows={df.shape[0]}, cols={df.shape[1]}")

    return df


@dg.asset(
    description="Split data into train and test",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_model"
)
def train_test_split_data(context: dg.AssetExecutionContext, fraud_detection: pd.DataFrame) -> dict:
    """Split data into 80–20 train/test sets."""
    # Expect the dataset to use column name 'Class' for fraud label (as in common creditcard datasets)
    if "Class" not in fraud_detection.columns:
        raise ValueError("Input dataframe must contain 'Class' column for labels.")

    X = fraud_detection.drop(columns=["Class"])
    y = fraud_detection["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    mlflow_client = context.resources.mlflow_tracking
    try:
        mlflow_client.log_input(dataset=mlflow_client.data.from_pandas(X_train, name="X_train"), context="training")
        mlflow_client.log_input(dataset=mlflow_client.data.from_pandas(X_test, name="X_test"), context="training")
    except Exception:
        context.log.info("Could not log train/test splits to MLflow "
        "(client may not support log_input for these objects).")

    context.log.info(f"Train/Test split sizes: X_train={X_train.shape}, X_test={X_test.shape}")  # type: ignore

    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


@dg.asset(
    description="Perform cross validation on the data",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_model"
)
def cross_validation_results(context: dg.AssetExecutionContext, train_test_split_data: dict,
tuned_random_forest: dict) -> dict:
    """Perform 3-fold cross-validation."""
    X_train = train_test_split_data["X_train"]
    y_train = train_test_split_data["y_train"]

    # tuned_random_forest is expected to be a dict with keys 'best_model' etc.
    model = tuned_random_forest["best_model"]  # use the model from tuned_random_forest

    scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
    mean_acc = float(scores.mean())

    # Log CV results to mlflow if available
    mlflow_client = context.resources.mlflow_tracking
    try:
        mlflow_client.log_metric("cv_mean_accuracy", mean_acc)
        for i, s in enumerate(scores.tolist()):
            mlflow_client.log_metric(f"cv_fold_{i + 1}_accuracy", float(s))
    except Exception:
        context.log.info("Could not log cross-validation metrics to MLflow.")

    context.log.info(f"CV scores: {scores}")

    return {"mean_accuracy": mean_acc, "scores": scores.tolist()}


@dg.asset(
    description="Perform hyperparameter tuning",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_model"
)
def tuned_random_forest(context: dg.AssetExecutionContext, train_test_split_data: dict) -> dict:
    """Perform hyperparameter tuning using GridSearchCV."""
    X_train = train_test_split_data["X_train"]
    y_train = train_test_split_data["y_train"]

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 10, 20],
        "min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42),
                               param_grid, cv=3,
                               scoring="accuracy",
                               n_jobs=-1,
                               )
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    best_params = grid_search.best_params_
    best_score = float(grid_search.best_score_)

    context.log.info(f"GridSearch best params: {best_params}")
    context.log.info(f"GridSearch best CV score: {best_score}")

    # Attempt to record tuning metadata in MLflow and as a Dagster materialization
    mlflow_client = context.resources.mlflow_tracking
    try:
        for k, v in best_params.items():
            mlflow_client.log_param(k, v)
        mlflow_client.log_metric("tuning_best_cv_accuracy", best_score)
    except Exception:
        context.log.info("Could not log tuning params/metrics to MLflow.")

    try:
        context.log_event(dg.AssetMaterialization(asset_key=dg.AssetKey("tuned_random_forest"), metadata={
            "best_params": best_params,
            "best_cv_score": best_score,
            "n_training_samples": X_train.shape[0]
        }))
    except Exception:
        context.log.info("Could not create AssetMaterialization event; falling back to logging metadata.")

    # Return structured dict for downstream assets (including model object & metadata)
    return {
        "best_model": best_model,
        "best_params": best_params,
        "best_score": best_score,
        "metadata": {"n_training_samples": X_train.shape[0]}
    }


@dg.asset(
    description="Test model on the test data",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_evaluate"
)
def fraud_test_model(context: dg.AssetExecutionContext, train_test_split_data: dict, tuned_random_forest: dict) -> dict:
    """Evaluate tuned model on test data."""
    X_test = train_test_split_data["X_test"]
    y_test = train_test_split_data["y_test"]
    model = tuned_random_forest["best_model"]

    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Log evaluation metrics to MLflow if available
    mlflow_client = context.resources.mlflow_tracking
    try:
        mlflow_client.log_metric("test_accuracy", acc)
        # Log class-level metrics
        for label, metrics in report.items():  # type: ignore
            if label in ("accuracy", "macro avg", "weighted avg"):
                continue
            mlflow_client.log_metric(f"precision_class_{label}", float(metrics.get("precision", 0)))
            mlflow_client.log_metric(f"recall_class_{label}", float(metrics.get("recall", 0)))
            mlflow_client.log_metric(f"f1_class_{label}", float(metrics.get("f1-score", 0)))
    except Exception:
        context.log.info("Could not log evaluation metrics to MLflow.")

    context.log.info(f"Test accuracy: {acc}")
    context.log.info(f"Confusion matrix:\n{cm}")

    return {"accuracy": acc, "report": report, "confusion_matrix": cm}


@dg.asset(
    description="Save the tuned RandomForest model as a pickle file for Gradio use.",
    compute_kind="python",
    group_name="ml_fraud_promote",
    resource_defs={"mlflow_tracking": mlflow_resource}
)
def save_tuned_model(context: dg.AssetExecutionContext, tuned_random_forest: dict) -> str:
    """
    Saves the tuned RandomForest model to a local pickle file
    and returns the absolute file path.
    """
    best_model = tuned_random_forest["best_model"]

    # Make sure models directory exists relative to Dagster project root
    save_dir = "../../gradioapp/utils/"

    model_path = os.path.join(save_dir, "tuned_random_forest.pkl")

    context.log.info(f"Saving model to: {model_path}")

    try:
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        context.log.info(f"✅ Model successfully saved at {model_path}")
    except Exception as e:
        context.log.error(f"❌ Failed to save model: {e}")
        raise

    # Optional: Log file artifact to MLflow
    try:
        mlflow_client = context.resources.mlflow_tracking
        mlflow_client.log_artifact(model_path)
    except Exception as e:
        context.log.warning(f"⚠️ Could not log model to MLflow: {e}")

    # Emit a materialization event with the path so Dagster UI records it
    try:
        context.log_event(dg.AssetMaterialization(asset_key=dg.AssetKey("tuned_random_forest_artifact"), metadata={
            "model_path": model_path
        }))
    except Exception:
        context.log.info(f"Saved model at {model_path}")

    # Return file path so you can use it downstream (e.g. in Gradio)
    return model_path
