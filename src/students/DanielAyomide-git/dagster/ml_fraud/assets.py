import os
import time
from typing import Dict, Iterable

import dagster as dg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
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
    Loads the raw credit card fraud detection dataset from the URL specified in DataConfig,
    logs metadata, and logs parameters to MLFlow.
    """
    data_config: DataConfig = context.resources.data_config
    mlflow_client = context.resources.mlflow

    context.log.info(f"Loading data from {data_config.dataset_url}")

    # 1. Calculate download time
    start_time = time.time()
    df = pd.read_csv(data_config.dataset_url)
    end_time = time.time()
    download_time = end_time - start_time
    context.log.info(f"Data downloaded in {download_time:.2f} seconds.")

    # 2. Calculate all required metrics
    num_rows, num_columns = df.shape
    memory_usage = df.memory_usage(deep=True).sum()  # In bytes
    fraud_count = int(df["Class"].sum())
    non_fraud_count = num_rows - fraud_count
    # Handle division by zero if there are no rows
    fraud_ratio = (fraud_count / num_rows) if num_rows > 0 else 0.0

    context.log.info(f"Dataset dimensions: {num_rows} rows, {num_columns} columns.")
    context.log.info(f"Memory usage: {memory_usage / 1e6:.2f} MB")
    context.log.info(f"Fraud cases: {fraud_count}, Non-fraud cases: {non_fraud_count}")

    # 3. Log all metrics to MLFlow
    mlflow_client.log_param("dataset_url", data_config.dataset_url)
    mlflow_client.log_metric("download_time_seconds", download_time)
    mlflow_client.log_metric("memory_usage_bytes", memory_usage)
    mlflow_client.log_metric("num_rows", num_rows)
    mlflow_client.log_metric("num_columns", num_columns)
    mlflow_client.log_metric("fraud_count", fraud_count)
    mlflow_client.log_metric("non_fraud_count", non_fraud_count)
    mlflow_client.log_metric("fraud_ratio", fraud_ratio)
    context.log.info("Logged all raw data parameters and metrics to MLFlow.")

    # 4. Return all metrics in the MaterializeResult
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
                df.head().to_markdown() or "" if not df.empty else "No data to preview."
            ),
        },
    )


@dg.asset(
    group_name="ml_fraud_transformation",
    description="Removes duplicates, missing values, and unnecessary columns from the raw data.",
    compute_kind="python",
    required_resource_keys={"mlflow"},
)
def cleaned_fraud_data(context: dg.OpExecutionContext, raw_fraud_data: pd.DataFrame) -> dg.MaterializeResult:
    """
    Cleans data by removing duplicates, handling nulls, and dropping the 'Time' column.
    The 'Amount' column is passed through for later transformation.
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
            "duplicates_removed_count": dg.MetadataValue.int(metadata["duplicates_removed_count"]),
            "preview_data": dg.MetadataValue.md(
                df.head().to_markdown() or "" if not df.empty else "No data to preview."
            ),
        },
    )


@dg.asset(
    group_name="ml_fraud_transformation",
    description="Applies feature scaling to the cleaned data.",
    compute_kind="python",
    required_resource_keys={"mlflow"},
)
def transformed_fraud_data(context: dg.OpExecutionContext, cleaned_fraud_data: pd.DataFrame) -> dg.MaterializeResult:
    """
    Transforms the data by scaling the 'Amount' feature using StandardScaler.
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
        mean_array: np.ndarray = scaler.mean_  # type: ignore
        scale_array: np.ndarray = scaler.scale_  # type: ignore

        if len(mean_array) > 0 and len(scale_array) > 0:
            amount_mean = float(mean_array[0])
            amount_std = float(scale_array[0])
            context.log.info("Successfully scaled 'Amount' feature and extracted stats.")
        else:
            context.log.warning("StandardScaler arrays are empty. Using default stats.")
    else:
        context.log.warning("StandardScaler did not set attributes correctly. Using default stats.")

    df.drop(["Amount"], axis=1, inplace=True)

    metadata = {
        "transformed_row_count": len(df),
        "amount_scaler_mean": amount_mean,
        "amount_scaler_std": amount_std,
    }

    context.resources.mlflow.log_metrics({f"transform_{k}": v for k, v in metadata.items()})

    return dg.MaterializeResult(
        value=df,
        metadata={
            "row_count": dg.MetadataValue.int(metadata["transformed_row_count"]),
            "scaler_mean": dg.MetadataValue.float(metadata["amount_scaler_mean"]),
            "scaler_std": dg.MetadataValue.float(metadata["amount_scaler_std"]),
            "preview_data": dg.MetadataValue.md(
                df.head().to_markdown() or "" if not df.empty else "No data to preview."
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
    Splits data into features (X) and target (y) for both training and
    testing sets, using stratification.
    """
    data_config: DataConfig = context.resources.data_config
    mlflow_client = context.resources.mlflow

    if transformed_fraud_data.empty:
        context.log.warning("Transformed data is empty. Returning empty splits.")
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype="int64")

        return dg.MaterializeResult(
            value={"X_train": empty_df, "X_test": empty_df, "y_train": empty_series, "y_test": empty_series},
            metadata={"status": dg.MetadataValue.text("Skipped, input was empty.")},
        )

    context.log.info("Splitting data into train and test sets...")
    X = transformed_fraud_data.drop("Class", axis=1)
    y = transformed_fraud_data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=data_config.test_size, random_state=data_config.random_state, stratify=y
    )

    train_fraud_rate = float((y_train.sum() / len(y_train)) * 100 if len(y_train) > 0 else 0.0)  # type: ignore
    test_fraud_rate = float((y_test.sum() / len(y_test)) * 100 if len(y_test) > 0 else 0.0)  # type: ignore

    context.log.info(f"Train set: {len(X_train)} samples, {train_fraud_rate:.2f}% fraud.")
    context.log.info(f"Test set: {len(X_test)} samples, {test_fraud_rate:.2f}% fraud.")

    mlflow_client.log_param("test_size", data_config.test_size)
    mlflow_client.log_param("random_state", data_config.random_state)
    mlflow_client.log_metric("train_set_size", len(X_train))
    mlflow_client.log_metric("test_set_size", len(X_test))
    mlflow_client.log_metric("train_set_fraud_rate", train_fraud_rate)
    mlflow_client.log_metric("test_set_fraud_rate", test_fraud_rate)

    output_data = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}

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
def tune_hyperparameters(context: dg.OpExecutionContext, data_splits: Dict) -> dg.MaterializeResult:
    """
    Performs hyperparameter tuning for a RandomForestClassifier using nested MLFlow runs.
    """
    model_config: ModelConfig = context.resources.model_config
    mlflow_client = context.resources.mlflow

    X_train = data_splits["X_train"]
    y_train = data_splits["y_train"]

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)

    if X_train.empty:
        context.log.warning("Training data is empty.")
        return dg.MaterializeResult(
            value={"best_params": {}}, metadata={"status": dg.MetadataValue.text("Skipped, input was empty.")}
        )

    context.log.info(f"Starting hyperparameter {model_config.cv_folds}-fold CV.")

    best_score = -1.0
    best_params = {}

    mlflow_client.log_param("cv_folds", model_config.cv_folds)
    mlflow_client.log_param("scoring_metric", model_config.scoring_metric)

    for max_depth in model_config.max_depth_options:
        run_name = f"trial_max_depth_{max_depth}"
        with mlflow_client.start_run(run_name=run_name, nested=True) as _:
            context.log.info(f"Running Trial: max_depth={max_depth}")

            params = {
                "n_estimators": int(model_config.n_estimators),
                "max_depth": int(max_depth),
                "random_state": int(model_config.random_state),
            }
            mlflow_client.log_params(params)

            clf = RandomForestClassifier(
                n_estimators=params["n_estimators"], max_depth=params["max_depth"], random_state=params["random_state"]
            )
            cv = StratifiedKFold(n_splits=model_config.cv_folds, shuffle=True, random_state=model_config.random_state)

            scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring=model_config.scoring_metric)
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
def train_model(context: dg.OpExecutionContext, tune_hyperparameters: Dict, data_splits: Dict) -> dg.MaterializeResult:
    """
    Trains a RandomForestClassifier on the full training data using the best hyperparameters.
    """
    mlflow_client = context.resources.mlflow
    best_params = tune_hyperparameters["best_params"]

    if not best_params:
        context.log.warning("No best parameters found. Skipping model training.")
        return dg.MaterializeResult(
            value=RandomForestClassifier(),
            metadata={"status": dg.MetadataValue.text("Skipped, no best parameters found.")},
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
    context.log.info("Logged trained model to MLFlow.")

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
    context: dg.OpExecutionContext, train_model: RandomForestClassifier, data_splits: Dict
) -> dg.MaterializeResult:
    """
    Evaluates the model on the test set, logs metrics, and logs a confusion matrix plot.
    """
    mlflow_client = context.resources.mlflow

    X_test: pd.DataFrame = data_splits["X_test"]
    y_test: pd.Series = data_splits["y_test"]

    if X_test.empty:
        context.log.warning("Test data is empty. Skipping model evaluation.")
        return dg.MaterializeResult(
            value={}, metadata={"status": dg.MetadataValue.text("Skipped, test data was empty.")}
        )

    context.log.info("Evaluating model on test data.")
    predictions = train_model.predict(X_test)
    predictions_proba = train_model.predict_proba(X_test)[:, 1]  # type: ignore

    accuracy = float(accuracy_score(y_test, predictions))
    f1 = float(f1_score(y_test, predictions))
    roc_auc = float(roc_auc_score(y_test, predictions_proba))
    report_dict: dict = classification_report(y_test, predictions, output_dict=True)  # type: ignore
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
    context.log.info("Logged confusion matrix plot to MLFlow artifacts.")

    # Clean up temporary file
    if os.path.exists(plot_path):
        os.remove(plot_path)

    # Get the model info from MLFlow
    run_id = mlflow_client.get_run(mlflow_client.active_run().info.run_id).info.run_uuid
    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}"
    model_info = mlflow_client.register_model(model_uri, FRAUD_MODEL_NAME)
    context.log.info(f"Registered model '{model_info.name}' version {model_info.version}")

    output_value = {
        "metrics": metrics,
        "classification_report": _sanitize_report_dict(sanitized_report),
        "confusion_matrix": cm.tolist(),
        "model_info": {"name": model_info.name, "version": model_info.version, "run_id": run_id},
    }

    return dg.MaterializeResult(
        value=output_value,
        metadata={
            "test_accuracy": dg.MetadataValue.float(accuracy),
            "test_f1_score": dg.MetadataValue.float(f1),
            "test_roc_auc": dg.MetadataValue.float(roc_auc),
            "test_precision_class_0": dg.MetadataValue.float(sanitized_report["0"]["precision"]),
            "test_recall_class_0": dg.MetadataValue.float(sanitized_report["0"]["recall"]),
            "test_precision_class_1": dg.MetadataValue.float(sanitized_report["1"]["precision"]),
            "test_recall_class_1": dg.MetadataValue.float(sanitized_report["1"]["recall"]),
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
def promote_fraud_model_to_staging(context: dg.OpExecutionContext, evaluate_model: Dict) -> dg.MaterializeResult:
    """
    Checks model performance against F1 and ROC AUC thresholds and promotes to Staging if criteria are met.
    Sends a Slack message with test accuracy, F1, ROC-AUC, and precision metrics 🚀
    """
    model_promotion_config: ModelPromotionConfig = context.resources.model_promotion_config
    mlflow_client = context.resources.mlflow_api_client
    slack = context.resources.slack

    metrics = evaluate_model.get("metrics", {})
    model_info = evaluate_model.get("model_info", {})

    if not metrics or not model_info:
        context.log.warning("Upstream evaluatione data is missing. Skipping promotion.")
        return dg.MaterializeResult(metadata={"status": dg.MetadataValue.text("Skipped, upstream data missing.")})

    # Thresholds
    STAGING_F1_THRESHOLD = model_promotion_config.staging_f1_threshold
    STAGING_ROC_AUC_THRESHOLD = model_promotion_config.staging_roc_auc_threshold

    # Unpack metrics
    test_accuracy = metrics.get("test_accuracy", 0.0)
    test_f1_score = metrics.get("test_f1_score", 0.0)
    test_roc_auc = metrics.get("test_roc_auc", 0.0)
    test_precision_fraud = metrics.get("test_precision_class_1", 0.0)
    test_recall_fraud = metrics.get("test_recall_class_1", 0.0)
    test_precision_legitimate = metrics.get("test_precision_class_0", 0.0)
    test_recall_legitimate = metrics.get("test_recall_class_0", 0.0)

    model_name = model_info.get("name")
    model_version = model_info.get("version")

    meets_criteria = test_f1_score >= STAGING_F1_THRESHOLD and test_roc_auc >= STAGING_ROC_AUC_THRESHOLD

    if meets_criteria:
        context.log.info(
            f"✅ Model '{model_name}' (v{model_version}) meets staging promotion criteria. "
            f"F1: {test_f1_score:.4f}, ROC AUC: {test_roc_auc:.4f}"
        )

        try:
            mlflow_client.transition_model_version_stage(
                name=str(model_name),
                version=str(model_version),
                stage=STAGING_STAGE_NAME,
            )
        except Exception as e:
            context.log.error(f"Error during model promotion: {e}")
            return dg.MaterializeResult(
                metadata={
                    "status": dg.MetadataValue.text(STATUS_MODEL_NOT_PROMOTED_TO_STAGING),
                    "reason": dg.MetadataValue.text(f"Promotion failed: {e}"),
                }
            )

        # 🎯 Send Slack message with metrics
        slack_message = (
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"*🚀 FINAL MODEL REPORT BY DANIEL AND ABDULRASHEED *\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"📊 *Overall Performance*\n"
            f"> ✅ *Accuracy:* `{test_accuracy:.4f}`\n"
            f"> 🎯 *F1-score:* `{test_f1_score:.4f}`\n"
            f"> 📈 *ROC-AUC:* `{test_roc_auc:.4f}`\n"
            f"\n"
            f"💰 *Fraud Detection Metrics*\n"
            f"> 💸 *Precision:* `{test_precision_fraud:.4f}`\n"
            f"> 🔍 *Recall:* `{test_recall_fraud:.4f}`\n"
            f"\n"
            f"🏦 *Legitimate Transaction Metrics*\n"
            f"> 🧾 *Precision:* `{test_precision_legitimate:.4f}`\n"
            f"> 🕵️ *Recall:* `{test_recall_legitimate:.4f}`\n"
            f"\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━"
        )

        try:
            slack.get_client().chat_postMessage(channel="aims_course_october2025", text=slack_message)
            context.log.info("Sent Slack staging promotion notification.")
        except Exception as e:
            context.log.error(f"Failed to send Slack notification: {e}")

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
                "test_accuracy": dg.MetadataValue.float(test_accuracy),
            },
        )

    else:
        context.log.warning("Model does not meet performance criteria for Staging promotion.")

        slack_message = "*⚠️ Model Is not working*\n> *Omooooooo!*\n"

        try:
            slack.get_client().chat_postMessage(channel="aims_course_october2025", text=slack_message)
            context.log.info("Sent Slack notification for non-promotion.")
        except Exception as e:
            context.log.error(f"Failed to send Slack notification: {e}")

        return dg.MaterializeResult(
            metadata={
                "status": dg.MetadataValue.text(STATUS_MODEL_NOT_PROMOTED_TO_STAGING),
                "reason": dg.MetadataValue.text("Performance thresholds not met."),
                "test_f1_score": dg.MetadataValue.float(test_f1_score),
                "test_roc_auc": dg.MetadataValue.float(test_roc_auc),
                "test_accuracy": dg.MetadataValue.float(test_accuracy),
            },
        )


@dg.asset(
    group_name="ml_fraud_promote_model",
    description="Promotes the best model from Staging to Production.",
    compute_kind="python",
    required_resource_keys={"mlflow_api_client", "slack"},
)
def promote_fraud_model_to_production(
    context: dg.OpExecutionContext,
    promote_fraud_model_to_staging: Dict,
) -> dg.MaterializeResult:
    """
    Promotes latest staging model to Production (no Slack messages here).
    """
    mlflow_client = context.resources.mlflow_api_client
    context.log.info("Starting model promotion to Production.")

    if promote_fraud_model_to_staging.get("status") != STATUS_MODEL_PROMOTED_TO_STAGING:
        context.log.info("No model was promoted to Staging in the previous step. Skipping.")
        return dg.MaterializeResult(metadata={"status": dg.MetadataValue.text("Skipped, no new model in Staging.")})

    model_name = promote_fraud_model_to_staging.get("model_name") or ""

    try:
        staging_versions = [
            mv
            for mv in mlflow_client.search_model_versions(f"name='{model_name}'")
            if mv.current_stage == STAGING_STAGE_NAME
        ]
        if not staging_versions:
            context.log.warning(f"No model versions in Staging for '{model_name}'.")
            return dg.MaterializeResult(
                metadata={
                    "status": dg.MetadataValue.text(STATUS_MODEL_SKIPPED),
                    "reason": dg.MetadataValue.text("No model in Staging."),
                }
            )

        latest_staging_version = max(staging_versions, key=lambda mv: int(mv.version))
        prod_model_version = latest_staging_version.version

        for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
            if mv.current_stage == PRODUCTION_STAGE_NAME:
                context.log.info(f"Archiving previous Production version {mv.version}.")
                mlflow_client.transition_model_version_stage(
                    name=model_name, version=mv.version, stage=ARCHIVED_STAGE_NAME
                )

        context.log.info(f"Promoting model '{model_name}' version {prod_model_version} to Production.")
        mlflow_client.transition_model_version_stage(
            name=model_name, version=prod_model_version, stage=PRODUCTION_STAGE_NAME
        )

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
def raw_data_quality_checks(raw_fraud_data: pd.DataFrame) -> Iterable[dg.AssetCheckResult]:
    """
    Performs data quality checks on the raw DataFrame.
    1. Checks for any null values.
    2. Checks that the 'Class' column contains only 0s and 1s.
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
    is_valid = np.all(np.isin(unique_classes, [0, 1]))  # type: ignore
    yield dg.AssetCheckResult(
        check_name="valid_class_labels",
        passed=bool(is_valid),
        metadata={"unique_values_found": dg.MetadataValue.text(str(list(unique_classes)))},
    )
