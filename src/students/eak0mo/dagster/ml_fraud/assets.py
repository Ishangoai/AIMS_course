"""
Fraud Detection ML Pipeline Assets.

This module contains Dagster assets for a complete machine learning pipeline
that detects credit card fraud. It includes data ingestion, cleaning, transformation,
model training, evaluation, and deployment to staging/production environments.

The pipeline integrates with MLflow for experiment tracking and Slack for notifications.
"""

import os
import time
from typing import Any, Dict, Iterable

import dagster as dg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from .configs import DataConfig, ModelConfig, ModelPromotionConfig
from .constants import (
    ARCHIVED_STAGE_NAME,
    FRAUD_MODEL_NAME,
    MODEL_ARTIFACT_NAME,
    PRODUCTION_STAGE_NAME,
    STAGING_STAGE_NAME,
    STATUS_MODEL_NOT_PROMOTED_TO_PRODUCTION,
    STATUS_MODEL_NOT_PROMOTED_TO_STAGING,
    STATUS_MODEL_PROMOTED_TO_PRODUCTION,
    STATUS_MODEL_PROMOTED_TO_STAGING,
    STATUS_MODEL_SKIPPED,
)
from .utils import _sanitize_report_dict


@dg.asset(
    group_name="ml_fraud_ingestion",
    description="Loads the raw credit card fraud data from a URL.",
    compute_kind="python",
    required_resource_keys={"mlflow", "data_config"},
)
def raw_fraud_data(context: dg.OpExecutionContext) -> dg.MaterializeResult:
    """
    Load raw credit card fraud detection dataset from configured URL.

    This asset downloads the fraud detection dataset, computes metadata about
    the data, and logs all metrics to MLflow for tracking.

    Args:
        context: Dagster execution context providing access to resources and logging.

    Returns:
        MaterializeResult containing the raw DataFrame and comprehensive metadata
        including row counts, fraud ratios, download time, and memory usage.

    Resources:
        data_config: Configuration object containing the dataset URL.
        mlflow: MLflow client for logging parameters and metrics.
    """
    data_config: DataConfig = context.resources.data_config
    mlflow_client = context.resources.mlflow

    context.log.info(f"Loading data from {data_config.dataset_url}")

    # Calculate download time
    start_time = time.time()
    df = pd.read_csv(data_config.dataset_url)
    end_time = time.time()
    download_time = end_time - start_time
    context.log.info(f"Data downloaded in {download_time:.2f} seconds.")

    # Calculate all required metrics
    num_rows, num_columns = df.shape
    memory_usage = df.memory_usage(deep=True).sum()  # In bytes
    fraud_count = int(df["Class"].sum())
    non_fraud_count = num_rows - fraud_count
    fraud_ratio = (fraud_count / num_rows) if num_rows > 0 else 0.0

    context.log.info(f"Dataset dimensions: {num_rows} rows, {num_columns} columns.")
    context.log.info(f"Memory usage: {memory_usage / 1e6:.2f} MB")
    context.log.info(f"Fraud cases: {fraud_count}, Non-fraud cases: {non_fraud_count}")

    # Log all metrics to MLflow
    mlflow_client.log_param("dataset_url", data_config.dataset_url)
    mlflow_client.log_metric("download_time_seconds", download_time)
    mlflow_client.log_metric("memory_usage_bytes", memory_usage)
    mlflow_client.log_metric("num_rows", num_rows)
    mlflow_client.log_metric("num_columns", num_columns)
    mlflow_client.log_metric("fraud_count", fraud_count)
    mlflow_client.log_metric("non_fraud_count", non_fraud_count)
    mlflow_client.log_metric("fraud_ratio", fraud_ratio)
    context.log.info("Logged all raw data parameters and metrics to MLflow.")

    # Return all metrics in the MaterializeResult
    return dg.MaterializeResult(
        value=df,
        metadata={
            "dataset_url": dg.MetadataValue.url(data_config.dataset_url),
            "num_rows": dg.MetadataValue.int(int(num_rows)),
            "num_columns": dg.MetadataValue.int(int(num_columns)),
            "download_time_seconds": dg.MetadataValue.float(float(download_time)),
            "memory_usage_mb": dg.MetadataValue.float(float(memory_usage / 1e6)),
            "fraud_count": dg.MetadataValue.int(int(fraud_count)),
            "non_fraud_count": dg.MetadataValue.int(int(non_fraud_count)),
            "fraud_ratio": dg.MetadataValue.float(fraud_ratio),
            "preview_data": dg.MetadataValue.md(
                df.head().to_markdown() if not df.empty else "No data to preview."  # type: ignore
            ),
        },
    )


@dg.asset(
    group_name="ml_fraud_transformation",
    description="Removes duplicates, missing values, and unnecessary columns from the raw data.",
    compute_kind="python",
    required_resource_keys={"mlflow"},
)
def cleaned_fraud_data(
    context: dg.OpExecutionContext, raw_fraud_data: pd.DataFrame
) -> dg.MaterializeResult:
    """
    Clean fraud detection data by removing duplicates and handling missing values.

    This asset performs data cleaning operations including duplicate removal,
    null value handling, and dropping the 'Time' column which is not needed
    for model training.

    Args:
        context: Dagster execution context providing access to resources and logging.
        raw_fraud_data: Raw DataFrame from the raw_fraud_data asset.

    Returns:
        MaterializeResult containing the cleaned DataFrame and cleaning statistics.

    Resources:
        mlflow: MLflow client for logging cleaning metrics.
    """
    context.log.info("Starting data cleaning process.")

    if raw_fraud_data.empty:
        context.log.warning("Input DataFrame is empty. Skipping cleaning.")
        return dg.MaterializeResult(value=pd.DataFrame())

    df = raw_fraud_data.copy()
    original_row_count = len(df)

    df.drop_duplicates(inplace=True)
    duplicates_removed_count = original_row_count - len(df)
    context.log.info(f"Removed {duplicates_removed_count} duplicate rows.")

    missing_values_count = df.isnull().sum().sum()
    if missing_values_count > 0:
        df.dropna(inplace=True)
        context.log.warning(f"Found and dropped {missing_values_count} missing values.")

    if "Time" in df.columns:
        df.drop(["Time"], axis=1, inplace=True)

    metadata = {
        "original_row_count": original_row_count,
        "cleaned_row_count": len(df),
        "duplicates_removed_count": duplicates_removed_count,
        "missing_values_found": int(missing_values_count),
    }

    context.resources.mlflow.log_metrics({f"clean_{k}": v for k, v in metadata.items()})

    return dg.MaterializeResult(
        value=df,
        metadata={
            "cleaned_row_count": dg.MetadataValue.int(metadata["cleaned_row_count"]),
            "duplicates_removed_count": dg.MetadataValue.int(
                metadata["duplicates_removed_count"]
            ),
            "preview_data": dg.MetadataValue.md(
                df.head().to_markdown() if not df.empty else "No data to preview."  # type: ignore
            ),
        },
    )


@dg.asset(
    group_name="ml_fraud_transformation",
    description="Applies feature scaling to the cleaned data.",
    compute_kind="python",
    required_resource_keys={"mlflow"},
)
def transformed_fraud_data(
    context: dg.OpExecutionContext, cleaned_fraud_data: pd.DataFrame
) -> dg.MaterializeResult:
    """
    Transform fraud detection data by scaling the Amount feature.

    This asset applies StandardScaler to the 'Amount' column to normalize
    transaction amounts, which helps improve model performance.

    Args:
        context: Dagster execution context providing access to resources and logging.
        cleaned_fraud_data: Cleaned DataFrame from the cleaned_fraud_data asset.

    Returns:
        MaterializeResult containing the transformed DataFrame with scaled Amount
        feature and scaler statistics.

    Resources:
        mlflow: MLflow client for logging transformation metrics.
    """
    context.log.info("Starting data transformation process.")

    if cleaned_fraud_data.empty:
        context.log.warning("Input cleaned_fraud_data is empty. Skipping transformation.")
        return dg.MaterializeResult(value=pd.DataFrame())

    df = cleaned_fraud_data.copy()
    amount_mean = 0.0
    amount_std = 0.0

    scaler = StandardScaler()
    df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

    if (
        hasattr(scaler, "mean_")
        and scaler.mean_ is not None
        and hasattr(scaler, "scale_")
        and scaler.scale_ is not None
    ):
        mean_array: np.ndarray = scaler.mean_  # type: ignore[assignment]
        scale_array: np.ndarray = scaler.scale_  # type: ignore[assignment]

        if len(mean_array) > 0 and len(scale_array) > 0:
            amount_mean = float(mean_array[0])
            amount_std = float(scale_array[0])
            context.log.info("Successfully scaled 'Amount' feature and extracted stats.")
        else:
            context.log.warning("StandardScaler arrays are empty. Using default stats.")
    else:
        context.log.warning(
            "StandardScaler did not set attributes correctly. Using default stats."
        )

    df.drop(["Amount"], axis=1, inplace=True)

    metadata = {
        "transformed_row_count": len(df),
        "amount_scaler_mean": amount_mean,
        "amount_scaler_std": amount_std,
    }

    context.resources.mlflow.log_metrics(
        {f"transform_{k}": v for k, v in metadata.items()}
    )

    return dg.MaterializeResult(
        value=df,
        metadata={
            "row_count": dg.MetadataValue.int(metadata["transformed_row_count"]),
            "scaler_mean": dg.MetadataValue.float(metadata["amount_scaler_mean"]),
            "scaler_std": dg.MetadataValue.float(metadata["amount_scaler_std"]),
            "preview_data": dg.MetadataValue.md(
                df.head().to_markdown() if not df.empty else "No data to preview."  # type: ignore
            ),
        },
    )


@dg.asset(
    group_name="ml_fraud_data_split",
    description="Splits transformed data into train and test sets.",
    compute_kind="python",
    required_resource_keys={"mlflow", "data_config"},
)
def data_splits(
    context: dg.OpExecutionContext,
    transformed_fraud_data: pd.DataFrame,
) -> dg.MaterializeResult:
    """
    Split transformed data into training and testing sets.

    This asset separates features (X) from the target variable (y) and performs
    a stratified train-test split to maintain fraud ratio in both sets.

    Args:
        context: Dagster execution context providing access to resources and logging.
        transformed_fraud_data: Transformed DataFrame from transformed_fraud_data asset.

    Returns:
        MaterializeResult containing a dictionary with X_train, X_test, y_train,
        and y_test DataFrames/Series, along with split statistics.

    Resources:
        data_config: Configuration object containing test_size and random_state.
        mlflow: MLflow client for logging split parameters and metrics.
    """
    data_config: DataConfig = context.resources.data_config
    mlflow_client = context.resources.mlflow

    if transformed_fraud_data.empty:
        context.log.warning("Transformed data is empty. Returning empty splits.")
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype="int64")

        return dg.MaterializeResult(
            value={
                "X_train": empty_df,
                "X_test": empty_df,
                "y_train": empty_series,
                "y_test": empty_series,
            },
            metadata={"status": dg.MetadataValue.text("Skipped, input was empty.")},
        )

    context.log.info("Splitting data into train and test sets...")
    X = transformed_fraud_data.drop("Class", axis=1)
    y = transformed_fraud_data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_config.test_size,
        random_state=data_config.random_state,
        stratify=y,
    )

    train_fraud_rate = (
        float((y_train.sum() / len(y_train)) * 100) if len(y_train) > 0 else 0.0  # type: ignore[arg-type]
    )
    test_fraud_rate = (
        float((y_test.sum() / len(y_test)) * 100) if len(y_test) > 0 else 0.0  # type: ignore[arg-type]
    )

    context.log.info(f"Train set: {len(X_train)} samples, {train_fraud_rate:.2f}% fraud.")
    context.log.info(f"Test set: {len(X_test)} samples, {test_fraud_rate:.2f}% fraud.")

    mlflow_client.log_param("test_size", data_config.test_size)
    mlflow_client.log_param("random_state", data_config.random_state)
    mlflow_client.log_metric("train_set_size", len(X_train))
    mlflow_client.log_metric("test_set_size", len(X_test))
    mlflow_client.log_metric("train_set_fraud_rate", train_fraud_rate)
    mlflow_client.log_metric("test_set_fraud_rate", test_fraud_rate)

    output_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }

    return dg.MaterializeResult(
        value=output_data,
        metadata={
            "train_set_size": dg.MetadataValue.int(len(X_train)),
            "test_set_size": dg.MetadataValue.int(len(X_test)),
            "train_fraud_rate_percent": dg.MetadataValue.float(train_fraud_rate),
            "test_fraud_rate_percent": dg.MetadataValue.float(test_fraud_rate),
        },
    )


@dg.asset(
    group_name="ml_fraud_main_model",
    description="Tunes Random Forest hyperparameters using 3-fold cross-validation.",
    compute_kind="python",
    required_resource_keys={"mlflow", "model_config"},
)
def tune_hyperparameters(
    context: dg.OpExecutionContext, data_splits: Dict[str, Any]
) -> dg.MaterializeResult:
    """
    Perform hyperparameter tuning for RandomForestClassifier.

    This asset uses stratified k-fold cross-validation to find the best
    max_depth parameter for the Random Forest model. Each trial is logged
    as a nested MLflow run.

    Args:
        context: Dagster execution context providing access to resources and logging.
        data_splits: Dictionary containing X_train and y_train from data_splits asset.

    Returns:
        MaterializeResult containing a dictionary with the best hyperparameters
        and cross-validation score.

    Resources:
        model_config: Configuration object with hyperparameter options and CV settings.
        mlflow: MLflow client for logging trials and results.
    """
    model_config: ModelConfig = context.resources.model_config
    mlflow_client = context.resources.mlflow

    X_train = data_splits["X_train"]
    y_train = data_splits["y_train"]

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)

    if X_train.empty:
        context.log.warning("Training data is empty. Skipping hyperparameter tuning.")
        return dg.MaterializeResult(
            value={"best_params": {}},
            metadata={"status": dg.MetadataValue.text("Skipped, input was empty.")},
        )

    context.log.info(
        f"Starting hyperparameter tuning with {model_config.cv_folds}-fold CV."
    )

    best_score = -1.0
    best_params: Dict[str, int] = {}

    mlflow_client.log_param("cv_folds", model_config.cv_folds)
    mlflow_client.log_param("scoring_metric", model_config.scoring_metric)

    for max_depth in model_config.max_depth_options:
        run_name = f"trial_max_depth_{max_depth}"
        with mlflow_client.start_run(run_name=run_name, nested=True):
            context.log.info(f"Running Trial: max_depth={max_depth}")

            params = {
                "n_estimators": int(model_config.n_estimators),
                "max_depth": int(max_depth),
                "random_state": int(model_config.random_state),
            }
            mlflow_client.log_params(params)

            clf = RandomForestClassifier(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=params["random_state"],
            )
            cv = StratifiedKFold(
                n_splits=model_config.cv_folds,
                shuffle=True,
                random_state=model_config.random_state,
            )

            scores = cross_val_score(
                clf, X_train, y_train, cv=cv, scoring=model_config.scoring_metric
            )
            avg_score = float(np.mean(scores))

            mlflow_client.log_metric(f"avg_{model_config.scoring_metric}", avg_score)
            context.log.info(f"Avg {model_config.scoring_metric}: {avg_score:.4f}")

            if avg_score > best_score:
                best_score = avg_score
                best_params = params

    mlflow_client.log_metric(f"best_cv_{model_config.scoring_metric}", float(best_score))
    mlflow_client.log_params({f"best_{k}": v for k, v in best_params.items()})
    context.log.info(f"Best params found: {best_params} with score: {best_score:.4f}")

    return dg.MaterializeResult(
        value={"best_params": best_params},
        metadata={
            "best_cv_score": dg.MetadataValue.float(float(best_score)),
            "best_max_depth": dg.MetadataValue.int(int(best_params.get("max_depth", 0))),
        },
    )


@dg.asset(
    group_name="ml_fraud_main_model",
    description="Trains the final model with the best hyperparameters.",
    compute_kind="python",
    required_resource_keys={"mlflow"},
)
def train_model(
    context: dg.OpExecutionContext,
    tune_hyperparameters: Dict[str, Any],
    data_splits: Dict[str, Any],
) -> dg.MaterializeResult:
    """
    Train the final RandomForestClassifier using best hyperparameters.

    This asset trains a Random Forest model on the full training dataset using
    the optimal hyperparameters found during tuning, and logs the model to MLflow.

    Args:
        context: Dagster execution context providing access to resources and logging.
        tune_hyperparameters: Dictionary containing best_params from tuning.
        data_splits: Dictionary containing X_train and y_train.

    Returns:
        MaterializeResult containing the trained RandomForestClassifier model
        and training metadata.

    Resources:
        mlflow: MLflow client for logging the trained model.
    """
    mlflow_client = context.resources.mlflow
    best_params = tune_hyperparameters["best_params"]

    if not best_params:
        context.log.warning("No best parameters found. Skipping model training.")
        return dg.MaterializeResult(
            value=RandomForestClassifier(),
            metadata={
                "status": dg.MetadataValue.text("Skipped, no best parameters found.")
            },
        )

    X_train = data_splits["X_train"]
    y_train = data_splits["y_train"]

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)

    context.log.info(f"Training final model with params: {best_params}")

    final_model = RandomForestClassifier(**best_params)
    final_model.fit(X_train, y_train)

    context.log.info("Final model trained successfully.")
    mlflow_client.sklearn.log_model(final_model, "random_forest_model")
    context.log.info("Logged trained model to MLflow.")

    return dg.MaterializeResult(
        value=final_model,
        metadata={
            "model_class": dg.MetadataValue.text(final_model.__class__.__name__),
            "training_parameters": dg.MetadataValue.json(best_params),
        },
    )


@dg.asset(
    group_name="ml_fraud_evaluate_model",
    description="Evaluates the final model and logs a confusion matrix.",
    compute_kind="python",
    required_resource_keys={"mlflow"},
)
def evaluate_model(
    context: dg.OpExecutionContext,
    train_model: RandomForestClassifier,
    data_splits: Dict[str, Any],
) -> dg.MaterializeResult:
    """
    Evaluate the trained model on test data and generate performance metrics.

    This asset computes accuracy, F1 score, ROC-AUC, precision, and recall metrics.
    It also generates and logs a confusion matrix visualization to MLflow and
    registers the model in the MLflow Model Registry.

    Args:
        context: Dagster execution context providing access to resources and logging.
        train_model: Trained RandomForestClassifier from train_model asset.
        data_splits: Dictionary containing X_test and y_test.

    Returns:
        MaterializeResult containing evaluation metrics, classification report,
        confusion matrix, and registered model information.

    Resources:
        mlflow: MLflow client for logging metrics, artifacts, and registering models.
    """
    mlflow_client = context.resources.mlflow

    X_test: pd.DataFrame = data_splits["X_test"]
    y_test: pd.Series = data_splits["y_test"]

    if X_test.empty:
        context.log.warning("Test data is empty. Skipping model evaluation.")
        return dg.MaterializeResult(
            value={},
            metadata={"status": dg.MetadataValue.text("Skipped, test data was empty.")},
        )

    context.log.info("Evaluating model on test data.")
    predictions = train_model.predict(X_test)
    predictions_proba = train_model.predict_proba(X_test)[:, 1]  # type: ignore[call-overload]

    accuracy = float(accuracy_score(y_test, predictions))
    f1 = float(f1_score(y_test, predictions))
    roc_auc = float(roc_auc_score(y_test, predictions_proba))
    report_dict: dict[str, Any] = classification_report(y_test, predictions, output_dict=True)  # type: ignore[assignment]
    sanitized_report = _sanitize_report_dict(report_dict)

    metrics = {
        "test_accuracy": accuracy,
        "test_f1_score": f1,
        "test_roc_auc": roc_auc,
        "test_precision_class_0": sanitized_report["0"]["precision"],
        "test_recall_class_0": sanitized_report["0"]["recall"],
        "test_precision_class_1": sanitized_report["1"]["precision"],
        "test_recall_class_1": sanitized_report["1"]["recall"],
    }

    # Log metrics to MLflow
    mlflow_client.log_metrics(metrics)

    context.log.info(f"Test Accuracy: {accuracy:.4f}")
    context.log.info(f"Test F1 Score: {f1:.4f}")
    context.log.info(f"Test ROC-AUC: {roc_auc:.4f}")

    # Create confusion matrix plot
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()

    # Save plot temporarily
    plot_dir = os.path.join(os.getcwd(), "outputs")
    os.makedirs(plot_dir, exist_ok=True)
    plot_path = os.path.join(plot_dir, "confusion_matrix.png")

    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Log plot to MLflow as artifact
    mlflow_client.log_artifact(plot_path, artifact_path="plots")
    context.log.info("Logged confusion matrix plot to MLflow artifacts.")

    # Clean up temporary file
    if os.path.exists(plot_path):
        os.remove(plot_path)

    # Get the model info from MLflow
    run_id = mlflow_client.get_run(mlflow_client.active_run().info.run_id).info.run_uuid
    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}"
    model_info = mlflow_client.register_model(model_uri, FRAUD_MODEL_NAME)
    context.log.info(f"Registered model '{model_info.name}' version {model_info.version}")

    output_value = {
        "metrics": metrics,
        "classification_report": _sanitize_report_dict(sanitized_report),
        "confusion_matrix": cm.tolist(),
        "model_info": {
            "name": model_info.name,
            "version": model_info.version,
            "run_id": run_id,
        },
    }

    return dg.MaterializeResult(
        value=output_value,
        metadata={
            "test_accuracy": dg.MetadataValue.float(accuracy),
            "test_f1_score": dg.MetadataValue.float(f1),
            "test_roc_auc": dg.MetadataValue.float(roc_auc),
            "test_precision_class_0": dg.MetadataValue.float(
                sanitized_report["0"]["precision"]
            ),
            "test_recall_class_0": dg.MetadataValue.float(
                sanitized_report["0"]["recall"]
            ),
            "test_precision_class_1": dg.MetadataValue.float(
                sanitized_report["1"]["precision"]
            ),
            "test_recall_class_1": dg.MetadataValue.float(
                sanitized_report["1"]["recall"]
            ),
            "classification_report": dg.MetadataValue.json(sanitized_report),
            "confusion_matrix": dg.MetadataValue.md(
                f"```\n{cm}\n```\nTrue Negatives: {cm[0][0]}, False Positives: {cm[0][1]}\n"
                f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}"
            ),
        },
    )


@dg.asset(
    group_name="ml_fraud_promote_model",
    description="Promotes the model to Staging if it meets performance criteria and sends a Slack notification.",
    compute_kind="python",
    required_resource_keys={"mlflow_api_client", "model_promotion_config", "slack"},
)
def promote_fraud_model_to_staging(
    context: dg.OpExecutionContext, evaluate_model: Dict[str, Any]
) -> dg.MaterializeResult:
    """
    Promote model to Staging stage if performance thresholds are met.

    This asset checks if the model meets both F1 score and ROC-AUC thresholds.
    If criteria are met, the model is transitioned to the Staging stage in MLflow
    and a detailed notification is sent to Slack.

    Args:
        context: Dagster execution context providing access to resources and logging.
        evaluate_model: Dictionary containing evaluation metrics and model info.

    Returns:
        MaterializeResult with promotion status and model version information.

    Resources:
        mlflow_api_client: High-level MLflow client for model stage transitions.
        model_promotion_config: Configuration with promotion thresholds.
        slack: Slack client for sending notifications.
    """
    model_promotion_config: ModelPromotionConfig = (
        context.resources.model_promotion_config
    )
    mlflow_client = context.resources.mlflow_api_client
    slack = context.resources.slack

    metrics = evaluate_model.get("metrics", {})
    model_info = evaluate_model.get("model_info", {})

    if not metrics or not model_info:
        context.log.warning("Upstream evaluation data is missing. Skipping promotion.")
        return dg.MaterializeResult(
            metadata={"status": dg.MetadataValue.text("Skipped, upstream data missing.")}
        )

    # Define performance thresholds
    STAGING_F1_THRESHOLD = model_promotion_config.staging_f1_threshold
    STAGING_ROC_AUC_THRESHOLD = model_promotion_config.staging_roc_auc_threshold

    # Unpack metrics and model info
    test_accuracy = metrics.get("test_accuracy", 0.0)
    test_f1_score = metrics.get("test_f1_score", 0.0)
    test_roc_auc = metrics.get("test_roc_auc", 0.0)
    test_precision_fraud = metrics.get("test_precision_class_1", 0.0)
    test_precision_legit = metrics.get("test_precision_class_0", 0.0)
    test_recall_legit = metrics.get("test_recall_class_0", 0.0)
    test_recall_fraud = metrics.get("test_recall_class_1", 0.0)

    model_name = model_info.get("name")
    model_version = model_info.get("version")

    # Check if the model meets BOTH promotion criteria
    meets_criteria = (test_f1_score >= STAGING_F1_THRESHOLD) and (
        test_roc_auc >= STAGING_ROC_AUC_THRESHOLD
    )

    if meets_criteria:
        context.log.info(
            f"Model '{model_name}' (v{model_version}) meets performance criteria. "
            f"F1: {test_f1_score:.4f} >= {STAGING_F1_THRESHOLD}, "
            f"ROC AUC: {test_roc_auc:.4f} >= {STAGING_ROC_AUC_THRESHOLD}. "
            f"Promoting to Staging."
        )

        try:
            mlflow_client.transition_model_version_stage(
                name=str(model_name), version=str(model_version), stage=STAGING_STAGE_NAME
            )
        except Exception as e:
            context.log.error(f"Error during model promotion: {e}")
            return dg.MaterializeResult(
                metadata={
                    "status": dg.MetadataValue.text(STATUS_MODEL_NOT_PROMOTED_TO_STAGING),
                    "reason": dg.MetadataValue.text(f"Promotion failed: {e}"),
                }
            )

        try:
            # Construct and send the success notification
            slack_message = f""" Elisha and Lionel Staging Model Performance:
- Accuracy: {test_accuracy:.4f}
- F1 Score: {test_f1_score:.4f} (Threshold: {STAGING_F1_THRESHOLD})
- ROC AUC: {test_roc_auc:.4f} (Threshold: {STAGING_ROC_AUC_THRESHOLD})
- Precision (Fraud): {test_precision_fraud:.4f}
- Recall (Fraud): {test_recall_fraud:.4f}
- Precision (Not Fraud): {test_precision_legit:.4f}
- Recall (Not Fraud): {test_recall_legit:.4f}
🫠
            """
            slack.get_client().chat_postMessage(
                channel="aims_course_october2025", text=slack_message
            )
            context.log.info("Successfully promoted model and sent Slack notification.")

            return dg.MaterializeResult(
                value={
                    "status": STATUS_MODEL_PROMOTED_TO_STAGING,
                    "model_name": model_name,
                    "model_version": model_version,
                },
                metadata={
                    "status": dg.MetadataValue.text(STATUS_MODEL_PROMOTED_TO_STAGING),
                    "model_version": dg.MetadataValue.text(f"{model_name} v{model_version}"),
                    "test_f1_score": dg.MetadataValue.float(test_f1_score),
                    "test_roc_auc": dg.MetadataValue.float(test_roc_auc),
                },
            )
        except Exception as e:
            context.log.error(f"Failed to send 'Promoted' Slack notification: {e}")
            return dg.MaterializeResult(
                metadata={
                    "status": dg.MetadataValue.text(STATUS_MODEL_NOT_PROMOTED_TO_STAGING),
                    "model_version": dg.MetadataValue.text(f"{model_name} v{model_version}"),
                    "test_f1_score": dg.MetadataValue.float(test_f1_score),
                    "test_roc_auc": dg.MetadataValue.float(test_roc_auc),
                    "notification_status": dg.MetadataValue.text(
                        f"Failed to send notification: {e}"
                    ),
                }
            )

    else:
        # This block executes if one or both performance thresholds are NOT met
        context.log.warning("Model does not meet performance criteria for Staging promotion.")

        # Determine which check failed for a clearer message
        f1_status = "Passed" if test_f1_score >= STAGING_F1_THRESHOLD else "Failed"
        roc_auc_status = (
            "Passed" if test_roc_auc >= STAGING_ROC_AUC_THRESHOLD else "Failed"
        )

        # Construct and send the failure notification
        slack_message = f"""❌ Model Promotion Failed
Model '{model_name}' v{model_version} did not meet promotion criteria:
- F1 Score: {test_f1_score:.4f} (Threshold: {STAGING_F1_THRESHOLD}) - {f1_status}
- ROC AUC: {test_roc_auc:.4f} (Threshold: {STAGING_ROC_AUC_THRESHOLD}) - {roc_auc_status}
"""
        try:
            slack.get_client().chat_postMessage(
                channel="aims_course_october2025", text=slack_message
            )
            context.log.info(
                "Sent Slack notification for model not meeting promotion criteria."
            )
        except Exception as e:
            context.log.error(f"Failed to send 'Not Promoted' Slack notification: {e}")

        return dg.MaterializeResult(
            metadata={
                "status": dg.MetadataValue.text(STATUS_MODEL_NOT_PROMOTED_TO_STAGING),
                "reason": dg.MetadataValue.text("Performance threshold(s) not met."),
                "test_f1_score": dg.MetadataValue.float(test_f1_score),
                "f1_threshold": dg.MetadataValue.float(STAGING_F1_THRESHOLD),
                "test_roc_auc": dg.MetadataValue.float(test_roc_auc),
                "roc_auc_threshold": dg.MetadataValue.float(STAGING_ROC_AUC_THRESHOLD),
            }
        )


@dg.asset(
    group_name="ml_fraud_promote_model",
    description="Promotes the best model from Staging to Production.",
    compute_kind="python",
    required_resource_keys={"mlflow_api_client", "slack"},
)
def promote_fraud_model_to_production(
    context: dg.OpExecutionContext, promote_fraud_model_to_staging: Dict[str, Any]
) -> dg.MaterializeResult:
    """
    Promote model from Staging to Production stage.

    This asset finds the latest model in Staging, archives any existing Production
    models, and promotes the Staging model to Production. A notification is sent
    to Slack upon successful promotion.

    Args:
        context: Dagster execution context providing access to resources and logging.
        promote_fraud_model_to_staging: Dictionary with staging promotion status.

    Returns:
        MaterializeResult with production promotion status and model version.

    Resources:
        mlflow_api_client: High-level MLflow client for model stage transitions.
        slack: Slack client for sending notifications.
    """
    mlflow_client = context.resources.mlflow_api_client
    context.log.info("Starting model promotion to Production.")

    if (
        promote_fraud_model_to_staging.get("status")
        != STATUS_MODEL_PROMOTED_TO_STAGING
    ):
        context.log.info(
            "No model was promoted to Staging in the previous step. "
            "Skipping production promotion."
        )
        return dg.MaterializeResult(
            metadata={
                "status": dg.MetadataValue.text("Skipped, no new model in Staging.")
            }
        )

    model_name = promote_fraud_model_to_staging.get("model_name") or ""

    try:
        # Find the latest model version in the 'Staging' stage
        staging_versions = [
            mv
            for mv in mlflow_client.search_model_versions(f"name='{model_name}'")
            if mv.current_stage == STAGING_STAGE_NAME
        ]
        if not staging_versions:
            context.log.warning(
                f"No model versions found in Staging for '{model_name}'. "
                f"Skipping promotion."
            )
            return dg.MaterializeResult(
                metadata={
                    "status": dg.MetadataValue.text(STATUS_MODEL_SKIPPED),
                    "reason": dg.MetadataValue.text("No model in Staging."),
                }
            )

        # Get the version with the highest version number
        latest_staging_version = max(staging_versions, key=lambda mv: int(mv.version))
        prod_model_version = latest_staging_version.version

        # Archive any existing models currently in 'Production'
        for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
            if mv.current_stage == PRODUCTION_STAGE_NAME:
                context.log.info(f"Archiving previous Production model version {mv.version}.")
                mlflow_client.transition_model_version_stage(
                    name=model_name, version=mv.version, stage=ARCHIVED_STAGE_NAME
                )

        # Promote the new version to 'Production'
        context.log.info(
            f"Promoting model '{model_name}' version {prod_model_version} to Production."
        )
        mlflow_client.transition_model_version_stage(
            name=model_name, version=prod_model_version, stage=PRODUCTION_STAGE_NAME
        )

        context.log.info("Successfully promoted model to Production notification.")

        return dg.MaterializeResult(
            metadata={
                "status": dg.MetadataValue.text(STATUS_MODEL_PROMOTED_TO_PRODUCTION),
                "model_name": dg.MetadataValue.text(model_name),
                "promoted_version": dg.MetadataValue.text(str(prod_model_version)),
            }
        )
    except Exception as e:
        context.log.error(f"Error promoting model to Production: {e}")
        return dg.MaterializeResult(
            metadata={
                "status": dg.MetadataValue.text(STATUS_MODEL_NOT_PROMOTED_TO_PRODUCTION),
                "reason": dg.MetadataValue.text(f"Promotion failed: {e}"),
            }
        )


@dg.multi_asset_check(
    specs=[
        dg.AssetCheckSpec(name="no_nulls_in_raw_data", asset="raw_fraud_data"),
        dg.AssetCheckSpec(name="valid_class_labels", asset="raw_fraud_data"),
    ],
    description="Checks the quality of the raw fraud detection data.",
)
def raw_data_quality_checks(
    raw_fraud_data: pd.DataFrame,
) -> Iterable[dg.AssetCheckResult]:
    """
    Perform data quality checks on raw fraud detection data.

    This multi-asset check validates two critical properties:
    1. No null values exist in the dataset
    2. The 'Class' column contains only binary values (0 and 1)

    Args:
        raw_fraud_data: Raw DataFrame to validate.

    Yields:
        AssetCheckResult for each quality check performed.
    """
    # Check 1: No null values
    num_nulls = raw_fraud_data.isnull().sum().sum()
    yield dg.AssetCheckResult(
        check_name="no_nulls_in_raw_data",
        passed=bool(num_nulls == 0),
        metadata={"num_null_values": dg.MetadataValue.int(int(num_nulls))},
    )

    # Check 2: 'Class' column has only 0s and 1s
    unique_classes = raw_fraud_data["Class"].unique()
    is_valid = np.all(np.isin(unique_classes, [0, 1]))  # type: ignore[arg-type]
    yield dg.AssetCheckResult(
        check_name="valid_class_labels",
        passed=bool(is_valid),
        metadata={
            "unique_values_found": dg.MetadataValue.text(str(list(unique_classes)))
        },
    )
