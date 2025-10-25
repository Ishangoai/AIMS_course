from typing import Any

import dagster as dg
import dagster_slack
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn as sk
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Imports all resources from the ml resources file.
from ..ml.resources import (
    PromotionConfig,
    mlflow_client,
    mlflow_resource,
)


@dg.asset(
    description="Collect data for fraud detection",
    compute_kind="python",
    group_name="Extract",
)
def fraud_data(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:
    """Load fraud detection dataset from remote CSV."""

    data = pd.read_csv(
        "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"
    )

    fraud_count = data[data.columns[-1]].sum()
    fraud_ratio = fraud_count / len(data)

    return dg.MaterializeResult(
        value=data,
        metadata={
            "# of entries": len(data),
            "# of features": dg.MetadataValue.int(len(data.columns) - 1),
            "fraud transactions": dg.MetadataValue.int(int(fraud_count)),
            "fraud_ratio": dg.MetadataValue.float(float(fraud_ratio)),
            "preview": dg.MetadataValue.md(data.head().to_markdown() or ""),
        }
    )


@dg.asset(
    description="Process the data.",
    compute_kind="python",
    group_name="Transform"
)
def process_data(
    context: dg.AssetExecutionContext, fraud_data: pd.DataFrame
) -> dg.MaterializeResult:
    """Process fraud detection data."""

    processed = fraud_data.copy()

    return dg.MaterializeResult(
        value=processed,
        metadata={
            "# of entries": len(processed),
            "# of features": dg.MetadataValue.int(len(processed.columns) - 1),
        }
    )


@dg.asset(
    description="Splits the data into training and testing sets",
    compute_kind="python",
    group_name="Transform",
)
def split_data(
    context: dg.AssetExecutionContext, process_data: pd.DataFrame
) -> dg.MaterializeResult:
    """Split data into 80% training and 20% testing sets with stratification."""

    x = process_data[process_data.columns[:-1]]
    y = process_data[process_data.columns[-1]]

    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    return dg.MaterializeResult(
        value={
            "x_train": x_train,
            "x_test": x_test,
            "y_train": y_train,
            "y_test": y_test,
        },
        metadata={
            "train samples": dg.MetadataValue.int(len(x_train)),
            "test samples": dg.MetadataValue.int(len(x_test)),
            "train fraud count": dg.MetadataValue.int(int(y_train.sum())),
            "test fraud count": dg.MetadataValue.int(int(y_test.sum())),
            "split ratio": dg.MetadataValue.text("80/20 train/test"),
        }
    )


@dg.asset(
    description="Tune hyperparameters with GridSearch",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="sklearn",
    group_name="Model",
)
def tune_hyperparameter(
    context: dg.AssetExecutionContext, split_data: Any
) -> dg.MaterializeResult:
    """Perform hyperparameter tuning with 3-fold CV on n_estimators."""

    x_train = split_data["x_train"]
    y_train = split_data["y_train"]

    param_grid = {"n_estimators": [50, 100, 150, 200]}
    rf = RandomForestClassifier(max_depth=25, random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring="f1",
        refit=True,
        n_jobs=-1,
        verbose=2,
    )

    grid_search.fit(x_train, y_train)
    best_params = grid_search.best_params_
    best_params["max_depth"] = 25
    best_params["random_state"] = 42

    context.log.info(f"Best parameters found: {best_params}")

    return dg.MaterializeResult(
        value=best_params,
        metadata={
            "best n_estimators": dg.MetadataValue.int(best_params["n_estimators"]),
            "best_f1_score": dg.MetadataValue.float(float(grid_search.best_score_)),
            "CV folds": dg.MetadataValue.int(3),
            "param combinations tested": dg.MetadataValue.int(len(param_grid["n_estimators"])),
            "tuning method": dg.MetadataValue.text("GridSearchCV"),
        }
    )


@dg.asset(
    description="Train final model and register it in MLflow",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="sklearn",
    group_name="Model",
    deps=[split_data, tune_hyperparameter]
)
def train_model(
    context: dg.AssetExecutionContext,
    split_data: Any,
    tune_hyperparameter: Any
) -> dg.MaterializeResult:
    """Trains and registers the model, then returns the model object and its run_id."""

    x_train = split_data["x_train"]
    y_train = split_data["y_train"]

    model = RandomForestClassifier(**tune_hyperparameter, n_jobs=-1)
    model.fit(x_train, y_train)
    model_name = "fraud_detection_model"

    mlflow.sklearn.log_model(  # type: ignore[attr-defined]
        sk_model=model, artifact_path="model", registered_model_name=model_name
    )
    active_run = mlflow.active_run()
    if active_run is None:
        raise RuntimeError("No active MLflow run")
    run_id = active_run.info.run_id

    context.log.info(f"Model trained and registered. MLflow Run ID: {run_id}")

    return dg.MaterializeResult(
        value={"model": model, "run_id": run_id},
        metadata={
            "Model type": dg.MetadataValue.text("RandomForestClassifier"),
            "n_estimators": dg.MetadataValue.int(tune_hyperparameter["n_estimators"]),
            "Max depth": dg.MetadataValue.int(tune_hyperparameter["max_depth"]),
            "Mlflow run_id": dg.MetadataValue.text(run_id),
            "Registered model name": dg.MetadataValue.text(model_name),
        }
    )


@dg.asset(
    description="Evaluate trained model on test set with key plots",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="Evaluate",
    deps=[split_data, train_model]
)
def evaluate_model(
    context: dg.AssetExecutionContext,
    split_data: Any,
    train_model: Any
) -> dg.MaterializeResult:
    """Evaluates the model and returns performance metrics."""

    context.log.info("Evaluating trained model on test set.")
    x_train = split_data["x_train"]
    x_test = split_data["x_test"]
    y_test = split_data["y_test"]
    model = train_model["model"]

    y_pred = model.predict(x_test)
    y_pred_proba = model.predict_proba(x_test)[:, 1]

    # Calculate metrics
    test_f1 = f1_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_accuracy = accuracy_score(y_test, y_pred)

    # Log metrics to MLflow
    mlflow.log_metric("test_f1", float(test_f1))
    mlflow.log_metric("test_precision", float(test_precision))
    mlflow.log_metric("test_recall", float(test_recall))
    mlflow.log_metric("test_accuracy", test_accuracy)
    context.log.info(f"Test F1 Score: {test_f1:.4f}")

    log_model_info = mlflow.sklearn.log_model(  # type: ignore[attr-defined]
            sk_model=model,
            artifact_path="tuned_temperature_forecaster",
            input_example=pd.DataFrame(x_test[:min(5, len(x_test))]),
            registered_model_name="fraud_detection_model"
        )
    context.log.info(f"Logged model artifact URI: {log_model_info.model_uri}")

    # 1. CONFUSION MATRIX
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    im = ax.imshow(cm, cmap="Blues", aspect="auto")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Count", rotation=270, labelpad=20)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Fraud (0)", "Fraud (1)"])
    ax.set_yticklabels(["Non-Fraud (0)", "Fraud (1)"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=16,
                fontweight="bold",
            )
    ax.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix - Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    mlflow.log_figure(fig, "test_confusion_matrix.png")
    plt.close(fig)

    # 2. ROC CURVE
    fig, ax = plt.subplots(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    mlflow.log_metric("test_roc_auc", roc_auc)
    ax.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"ROC curve (AUC = {roc_auc:.4f})",
        color="darkorange",
    )
    ax.plot([0, 1], [0, 1], "k--", linewidth=2, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12, fontweight="bold")
    ax.set_ylabel("True Positive Rate", fontsize=12, fontweight="bold")
    ax.set_title("ROC Curve - Test Set", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    mlflow.log_figure(fig, "roc_curve.png")
    plt.close(fig)

    # 3. FEATURE IMPORTANCE (Top 20)
    fig, ax = plt.subplots(figsize=(12, 8))
    indices = np.argsort(model.feature_importances_)[-20:]
    ax.barh(range(len(indices)), model.feature_importances_[indices], color="steelblue")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels(np.array(x_train.columns)[indices])
    ax.set_xlabel("Feature Importance", fontsize=12, fontweight="bold")
    ax.set_ylabel("Feature", fontsize=12, fontweight="bold")
    ax.set_title("Top 20 Feature Importances", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    mlflow.log_figure(fig, "feature_importance.png")
    plt.close(fig)

    evaluation_metrics = {
        "test_f1_score": test_f1,
        "test_precision": test_precision,
        "test_recall": test_recall,
        "test_accuracy": test_accuracy,
        "test_roc_auc": roc_auc,
    }

    # Create confusion matrix summary for metadata
    tn, fp, fn, tp = cm.ravel()

    return dg.MaterializeResult(
        value=evaluation_metrics,
        metadata={
            "test accuracy": dg.MetadataValue.float(float(test_accuracy)),
            "test f1 score": dg.MetadataValue.float(float(test_f1)),
            "test precision": dg.MetadataValue.float(float(test_precision)),
            "test recall": dg.MetadataValue.float(float(test_recall)),
            "test ROC-AUC": dg.MetadataValue.float(float(roc_auc)),
            "confusion matrix": dg.MetadataValue.md(
                f"**True Negatives:** {tn} | **False Positives:** {fp}\n\n"
                f"**False Negatives:** {fn} | **True Positives:** {tp}"
            ),
            "plots logged": dg.MetadataValue.text("Confusion Matrix, ROC Curve, Feature Importance"),
        }
    )


@dg.asset(
    description="Sends slack evaluation message",
    resource_defs={"slack2": dagster_slack.SlackResource(token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN"))},
    compute_kind="python",
    group_name="Evaluate",
)
def send_slack(
    context: dg.AssetExecutionContext,
    evaluate_model: Any
) -> dg.MaterializeResult:
    """Sends a formatted message with model evaluation metrics to a Slack channel."""

    slack: dagster_slack.SlackResource = context.resources.slack2

    message = f"""
🎯 Fraud Detection Model Evaluation Report!
:technologist: Hassan, Nagi & Khalo :technologist:
*Performance Metrics on Test Set*:
- *Accuracy*: `{evaluate_model['test_accuracy']:.4f}`
- *Recall*: `{evaluate_model['test_recall']:.4f}`
- *F1 Score*: `{evaluate_model['test_f1_score']:.4f}`
Great run! :rocket:
    """.strip()

    slack.get_client().chat_postMessage(
        channel="aims_course_october2025", text=message
    )

    context.log.info("Notification sent to Slack successfully.")

    return dg.MaterializeResult(
        value=None,
        metadata={
            "slack channel": dg.MetadataValue.text("aims_course_october2025"),
            "notification status": dg.MetadataValue.text("Success"),
            "metrics sent": dg.MetadataValue.text("Accuracy, Recall, F1 Score"),
        }
    )


@dg.asset(
    description="Promote model to Staging if it meets performance thresholds.",
    resource_defs={"mlflow_client": mlflow_client},
    compute_kind="python",
    group_name="Promote",
    deps=[train_model, evaluate_model]
)
def model_to_staging(
    context: dg.AssetExecutionContext,
    config: PromotionConfig,
    evaluate_model: Any,
    train_model: Any,
) -> dg.MaterializeResult:
    """Promotes the model to Staging if it passes the performance checks."""

    mlflow_client: MlflowClient = context.resources.mlflow_client

    model_name = "fraud_detection_model"
    f1 = evaluate_model.get("test_f1_score", 0.0)
    accuracy = evaluate_model.get("test_accuracy", 0.0)
    run_id = train_model["run_id"]
    versions = mlflow_client.search_model_versions(f"name='{model_name}'")
    model_version = next((v for v in versions if v.run_id == run_id), None)

    if not model_version:
        raise Exception(f"Could not find a registered model version for run ID: {run_id}")

    promotion_decision = (
        f1 > config.staging_f1_threshold
        and accuracy > config.staging_accuracy_threshold
    )

    if promotion_decision:
        context.log.info(f"Model version {model_version.version} passed checks. Promoting to Staging.")
        mlflow_client.transition_model_version_stage(
            name=model_name, version=model_version.version, stage="Staging"
        )

        return dg.MaterializeResult(
            value={"promote_status": "success", "model_name": model_name, "model_version": model_version.version},
            metadata={
                "promotion status": dg.MetadataValue.text("✅ Promoted to Staging"),
                "model version": dg.MetadataValue.text(str(model_version.version)),
                "f1 score": dg.MetadataValue.float(float(f1)),
                "accuracy": dg.MetadataValue.float(float(accuracy)),
                "f1 threshold": dg.MetadataValue.float(float(config.staging_f1_threshold)),
                "accuracy threshold": dg.MetadataValue.float(float(config.staging_accuracy_threshold)),
            }
        )
    else:
        context.log.warning(f"Model version {model_version.version} failed to meet thresholds. Not promoting.")

        return dg.MaterializeResult(
            value={"promote_status": "failure", "model_name": model_name, "model_version": model_version.version},
            metadata={
                "promotion status": dg.MetadataValue.text("❌ Not Promoted"),
                "model version": dg.MetadataValue.text(str(model_version.version)),
                "f1 score": dg.MetadataValue.float(float(f1)),
                "accuracy": dg.MetadataValue.float(float(accuracy)),
                "f1 threshold": dg.MetadataValue.float(float(config.staging_f1_threshold)),
                "accuracy threshold": dg.MetadataValue.float(float(config.staging_accuracy_threshold)),
                "reason": dg.MetadataValue.text(
                    f"F1: {f1:.4f} (need >{config.staging_f1_threshold}), "
                    f"Accuracy: {accuracy:.4f} (need >{config.staging_accuracy_threshold})"
                ),
            }
        )


@dg.asset(
    description="Promote Staged model to Production and archive the old one.",
    resource_defs={"mlflow_client": mlflow_client},
    compute_kind="python",
    group_name="Promote",
)
def model_to_production(
    context: dg.AssetExecutionContext,
    model_to_staging: Any
) -> dg.MaterializeResult:
    """Archives the old Production model and promotes the new Staged model."""

    mlflow_client: MlflowClient = context.resources.mlflow_client

    if model_to_staging.get("promote_status") != "success":
        context.log.warning("Model was not promoted to Staging. Skipping Production promotion.")

        return dg.MaterializeResult(
            value=None,
            metadata={
                "promotion_status": dg.MetadataValue.text("⏭️ Skipped"),
                "reason": dg.MetadataValue.text("Model did not pass Staging promotion criteria"),
            }
        )

    model_name = model_to_staging["model_name"]
    model_version = model_to_staging["model_version"]

    archived_versions = []
    for mv in mlflow_client.get_latest_versions(model_name, stages=["Production"]):
        context.log.info(f"Archiving old production model version: {mv.version}")
        mlflow_client.transition_model_version_stage(
            name=model_name, version=mv.version, stage="Archived"
        )
        archived_versions.append(str(mv.version))

    context.log.info(f"Promoting new model version {model_version} to Production.")
    mlflow_client.transition_model_version_stage(
        name=model_name, version=model_version, stage="Production"
    )

    return dg.MaterializeResult(
        value=None,
        metadata={
            "promotion status": dg.MetadataValue.text("✅ Promoted to Production"),
            "model name": dg.MetadataValue.text(model_name),
            "new production version": dg.MetadataValue.text(str(model_version)),
            "archived versions": dg.MetadataValue.text(", ".join(archived_versions) if archived_versions else "None"),
        }
    )
