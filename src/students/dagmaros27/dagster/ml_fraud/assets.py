import json
import os
from typing import Any, Dict, Optional, Tuple

import dagster as dg
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

MODEL_REGISTRY_NAME = "fraud_detection_model"


@dg.asset(
    description="Loads the credit card fraud detection dataset from a remote source.",
    compute_kind="python",
    group_name="ml_fraud_ingest",
    required_resource_keys={"fraud_data_api"},
)
def raw_fraud_dataset(context: dg.AssetExecutionContext) -> pd.DataFrame:
    """Fetch and load the fraud detection dataset using the FraudDataAPI resource."""
    data_api = context.resources.fraud_data_api
    url = getattr(data_api, "url", None)
    if url is None:
        raise dg.ResourceFunctionError("FraudDataAPI resource did not expose a 'url' property")

    context.log.info(f"Loading fraud detection dataset from {url}...")
    df = pd.read_csv(url)

    fraud_distribution = df["Class"].value_counts().to_dict()
    fraud_rate = (fraud_distribution.get(1, 0) / len(df)) * 100

    context.log.info(f"Dataset shape: {df.shape}")
    context.log.info(f"Fraud distribution: {fraud_distribution}")
    context.log.info(f"Fraud rate: {fraud_rate:.2f}%")

    columns = [dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]

    context.add_output_metadata(
        {
            "num_rows": dg.MetadataValue.int(int(len(df))),
            "num_columns": dg.MetadataValue.int(int(len(df.columns))),
            "fraud_count": dg.MetadataValue.int(int(fraud_distribution.get(1, 0))),
            "non_fraud_count": dg.MetadataValue.int(int(fraud_distribution.get(0, 0))),
            "fraud_rate_percent": dg.MetadataValue.float(float(fraud_rate)),
            "preview": dg.MetadataValue.md(df.head().to_markdown() or ""),
            "dagster/column_schema": dg.TableSchema(columns=columns),
            "source": dg.MetadataValue.text(url),
        }
    )

    return df


@dg.asset(
    description="Preprocesses the fraud data by cleaning and scaling features.",
    compute_kind="python",
    group_name="ml_fraud_preprocess_split",
)
def preprocessed_data(
    context: dg.AssetExecutionContext,
    raw_fraud_dataset: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Preprocess the raw data — handle missing values and scale features."""
    context.log.info("Preprocessing data...")
    df = raw_fraud_dataset

    X: pd.DataFrame = df.drop("Class", axis=1)
    y: pd.Series = df["Class"]  # type: ignore

    missing_before: int = int(X.isna().sum().sum())
    X = X.fillna(X.mean())
    missing_after: int = int(X.isna().sum().sum())

    scaler = StandardScaler()
    X_scaled: pd.DataFrame = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    context.log.info(f"Preprocessed shape: {X_scaled.shape}")
    context.log.info(f"Missing values before: {missing_before}, after: {missing_after}")

    columns = [dg.TableColumn(k, str(v)) for k, v in X_scaled.dtypes.to_dict().items()]

    context.add_output_metadata(
        {
            "num_features": dg.MetadataValue.int(int(X_scaled.shape[1])),
            "num_samples": dg.MetadataValue.int(int(X_scaled.shape[0])),
            "missing_values_filled": dg.MetadataValue.int(missing_before),
            "scaler_type": dg.MetadataValue.text("StandardScaler"),
            "feature_mean": dg.MetadataValue.float(float(X_scaled.mean().mean())),  # type: ignore
            "feature_std": dg.MetadataValue.float(float(X_scaled.std().mean())),  # type: ignore
            "preview": dg.MetadataValue.md(X_scaled.head().to_markdown() or ""),
            "dagster/column_schema": dg.TableSchema(columns=columns),
        }
    )

    return (X_scaled, y)


@dg.asset(
    description="Splits the preprocessed data into training and testing sets with stratification.",
    compute_kind="python",
    group_name="ml_fraud_preprocess_split",
)
def split_dataset(
    context: dg.AssetExecutionContext,
    preprocessed_data: Tuple[pd.DataFrame, pd.Series]
) -> Dict[str, Any]:
    """Split data into training and test sets with stratification."""
    X, y = preprocessed_data

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_fraud_rate: float = float((y_train.sum() / len(y_train)) * 100)  # type: ignore
    test_fraud_rate: float = float((y_test.sum() / len(y_test)) * 100)  # type: ignore

    context.log.info(f"Training set size: {X_train.shape}")  # type: ignore
    context.log.info(f"Test set size: {X_test.shape}")  # type: ignore
    context.log.info(f"Training fraud rate: {train_fraud_rate:.2f}%")
    context.log.info(f"Test fraud rate: {test_fraud_rate:.2f}%")

    context.add_output_metadata(
        {
            "train_size": dg.MetadataValue.int(int(len(X_train))),
            "test_size": dg.MetadataValue.int(int(len(X_test))),
            "train_fraud_count": dg.MetadataValue.int(int(y_train.sum())),  # type: ignore
            "test_fraud_count": dg.MetadataValue.int(int(y_test.sum())),  # type: ignore
            "train_fraud_rate_percent": dg.MetadataValue.float(float(train_fraud_rate)),
            "test_fraud_rate_percent": dg.MetadataValue.float(float(test_fraud_rate)),
            "test_split_ratio": dg.MetadataValue.float(0.2),
            "random_seed": dg.MetadataValue.int(42),
            "stratified": dg.MetadataValue.bool(True),
        }
    )

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


@dg.asset(
    group_name="ml_fraud_tuning",
    description="Tunes RandomForest hyperparameters using GridSearchCV with MLflow tracking.",
    required_resource_keys={"fraud_mlflow"},
)
def tune_hyperparameters(
    context: dg.AssetExecutionContext, split_dataset: Dict[str, Any]
) -> dg.MaterializeResult:
    """
    Tune RandomForest hyperparameters using GridSearchCV with 3-fold cross-validation.
    Uses the FraudMlflow resource for MLflow logging and experiment management.
    """

    # Setup MLflow tracking
    fraud_mlflow = context.resources.fraud_mlflow
    mlflow.set_tracking_uri(fraud_mlflow.tracking_uri)
    mlflow.set_experiment(fraud_mlflow.experiment_name)

    split_data = split_dataset
    X_train = split_data["X_train"]
    y_train = split_data["y_train"]

    context.log.info("🚀 Starting hyperparameter tuning with 3-fold cross-validation...")

    # Define hyperparameter search grid
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [10],
        "min_samples_split": [5],
        "random_state": [42],
    }
    cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    # ✅ Attach to the currently active MLflow run
    active_run = mlflow.active_run()
    if active_run is None:
        context.log.warning("⚠️ No active MLflow run found — metrics will not be logged to MLflow.")
        run_id = None
    else:
        run_id = active_run.info.run_id
        context.log.info(f"Using active MLflow run: {run_id}")

    # Log parameters (only if a run is active)
    if active_run:
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("tuned_parameter", "n_estimators")
        mlflow.log_param("param_grid", json.dumps(param_grid))

    # Initialize model and grid search
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=cv_splitter, scoring="f1", n_jobs=-1, verbose=1
    )

    # Train grid search
    grid_search.fit(X_train, y_train)

    # Extract results
    best_params = grid_search.best_params_
    best_score = float(grid_search.best_score_)

    # Log best results (only if a run is active)
    if active_run:
        mlflow.log_param("best_n_estimators", int(best_params["n_estimators"]))
        mlflow.log_metric("best_cv_f1_score", best_score)

        # Log trial details
        results_df = pd.DataFrame(grid_search.cv_results_)
        for idx, row in results_df.iterrows():
            mlflow.log_metric(f"trial_{idx}_n_estimators", int(row["param_n_estimators"]))
            mlflow.log_metric(f"trial_{idx}_mean_f1_score", float(row["mean_test_score"]))
            mlflow.log_metric(f"trial_{idx}_std_f1_score", float(row["std_test_score"]))

    context.log.info(f"✅ Completed hyperparameter tuning. Best params: {best_params}, Best F1: {best_score:.4f}")

    return dg.MaterializeResult(
        value=(grid_search.best_estimator_, best_params),
        metadata={
            "best_n_estimators": dg.MetadataValue.int(int(best_params["n_estimators"])),
            "best_cv_f1_score": dg.MetadataValue.float(best_score),
            "cv_folds": dg.MetadataValue.int(3),
            "tuned_parameter": dg.MetadataValue.text("n_estimators"),
            "scoring_metric": dg.MetadataValue.text("f1"),
            "mlflow_run_id": dg.MetadataValue.text(run_id or "none"),
        },
    )


@dg.asset(
    group_name="ml_fraud_training",
    description="Trains the final RandomForest model using the best hyperparameters and logs metrics to MLflow.",
    compute_kind="python",
    required_resource_keys={"fraud_mlflow"},
    deps=["split_dataset", "tune_hyperparameters"]
)
def train_model(
    context: dg.AssetExecutionContext,
    tune_hyperparameters: Tuple[RandomForestClassifier, Dict[str, Any]],
    split_dataset: Dict[str, Any],  # <-- typing fixed below
) -> dg.MaterializeResult[Tuple[RandomForestClassifier, Dict[str, float]]]:
    """Train final model with best hyperparameters and log to MLflow."""

    fraud_mlflow = context.resources.fraud_mlflow
    mlflow.set_tracking_uri(fraud_mlflow.tracking_uri)
    mlflow.set_experiment(fraud_mlflow.experiment_name)

    best_model, best_params = tune_hyperparameters
    split_data: Dict[str, Any] = split_dataset

    # Correct typing for Pyright
    X_train: pd.DataFrame = split_data["X_train"]
    y_train: pd.Series = split_data["y_train"]
    X_test: pd.DataFrame = split_data["X_test"]
    y_test: pd.Series = split_data["y_test"]

    # ✅ Attach to the currently active MLflow run
    active_run = mlflow.active_run()
    run_id: Optional[str] = None
    if active_run is None:
        context.log.warning("⚠️ No active MLflow run found — training metrics will not be logged.")
    else:
        run_id = active_run.info.run_id
        context.log.info(f"Using active MLflow run: {run_id}")

    # Log best parameters
    if active_run:
        mlflow.log_params({
            k: (int(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
            for k, v in best_params.items()
        })

    # Train model
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # type: ignore

    # Evaluate
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))
    roc_auc = float(roc_auc_score(y_test, y_pred_proba))

    metrics: Dict[str, float] = {
        "test_precision": precision,
        "test_recall": recall,
        "test_f1_score": f1,
        "test_roc_auc": roc_auc,
    }

    # Log metrics and model to MLflow if active
    if active_run:
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, artifact_path="final_fraud_model")  # type: ignore

    context.log.info(f"✅ Model training complete. F1={f1:.4f}, ROC-AUC={roc_auc:.4f}")

    return dg.MaterializeResult(
        value=(best_model, metrics),
        metadata={
            "model_type": dg.MetadataValue.text("RandomForestClassifier"),
            "test_precision": dg.MetadataValue.float(precision),
            "test_recall": dg.MetadataValue.float(recall),
            "test_f1_score": dg.MetadataValue.float(f1),
            "test_roc_auc": dg.MetadataValue.float(roc_auc),
            "n_estimators": dg.MetadataValue.int(int(best_params.get("n_estimators", 100))),
            "training_samples": dg.MetadataValue.int(len(X_train)),
            "test_samples": dg.MetadataValue.int(len(X_test)),
            "mlflow_run_id": dg.MetadataValue.text(run_id or "none"),
        },
    )


@dg.asset(
    group_name="ml_fraud_evaluation",
    description="Generates confusion matrix and performance plots, then logs them to MLflow.",
    compute_kind="python",
    required_resource_keys={"fraud_mlflow"},
)
def generate_evaluation_plots(
    context: dg.AssetExecutionContext,
    train_model: Tuple[RandomForestClassifier, Dict[str, float]],
    split_dataset: Dict[str, Any],
) -> dg.MaterializeResult:
    """Generate confusion matrix and performance plots, and log them to MLflow."""

    fraud_mlflow = context.resources.fraud_mlflow
    mlflow.set_tracking_uri(fraud_mlflow.tracking_uri)
    mlflow.set_experiment(fraud_mlflow.experiment_name)

    model, metrics = train_model
    split_data = split_dataset
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]

    context.log.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Predictions
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Derived metrics
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # === Visualization ===
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Confusion matrix
    ax = axes[0]
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)  # type: ignore
    ax.set_title("Confusion Matrix - Test Set")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Non-Fraud", "Fraud"])
    ax.set_yticklabels(["Non-Fraud", "Fraud"])
    plt.colorbar(im, ax=ax)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    # Metrics bar chart
    metric_names = ["Precision", "Recall", "F1-Score", "ROC-AUC", "Specificity", "Sensitivity"]
    metric_values: list[float] = [
        metrics["test_precision"], metrics["test_recall"],
        metrics["test_f1_score"], metrics["test_roc_auc"],
        specificity, sensitivity,
    ]

    ax2 = axes[1]
    bars = ax2.bar(metric_names, metric_values)
    ax2.set_title("Model Performance Metrics")
    ax2.set_ylim([0, 1.05])
    ax2.set_ylabel("Score")
    ax2.tick_params(axis="x", rotation=45)
    for bar, val in zip(bars, metric_values):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.01,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()

    # Save plots
    artifact_dir = os.path.join(os.getcwd(), "mlflow_artifacts_fraud")
    os.makedirs(artifact_dir, exist_ok=True)
    plot_path = os.path.join(artifact_dir, "evaluation_plots.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    context.log.info(f"Evaluation plots saved to {plot_path}")

    # Attach to currently active MLflow run
    active_run = mlflow.active_run()
    eval_run_id: Optional[str] = None

    if active_run is not None:
        mlflow.log_artifact(plot_path, artifact_path="evaluation_plots")
        mlflow.log_metrics({
            **{k: float(v) for k, v in metrics.items()},
            "specificity": float(specificity),
            "sensitivity": float(sensitivity),
        })

        # Log input example from test data (first 5 samples)
        num_samples = min(5, len(X_test))
        input_example = X_test[:num_samples]
        context.log.info(f"Input example shape: {input_example.shape}")
        context.log.info(f"Input example:\n{input_example}")

        # Log as artifact (CSV format for reference)
        input_df = pd.DataFrame(input_example)
        input_example_path = os.path.join(artifact_dir, "input_example.csv")
        input_df.to_csv(input_example_path, index=False)
        mlflow.log_artifact(input_example_path, artifact_path="input_examples")
        context.log.info(f"Input example saved to {input_example_path}")

        eval_run_id = active_run.info.run_id
        context.log.info(f"Logged plots, metrics, and input example to MLflow run: {eval_run_id}")
    else:
        context.log.warning("No active MLflow run found — skipping MLflow artifact logging.")

    return dg.MaterializeResult(
        value={
            "confusion_matrix": cm.tolist(),
            "metrics": {
                k: float(v) for k, v in metrics.items()
            },
            "additional_metrics": {
                "specificity": float(specificity),
                "sensitivity": float(sensitivity),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "plot_path": str(plot_path),
            "mlflow_eval_run_id": str(eval_run_id or "none"),
        },
        metadata={
            "confusion_matrix_tn": dg.MetadataValue.int(int(tn)),
            "confusion_matrix_fp": dg.MetadataValue.int(int(fp)),
            "confusion_matrix_fn": dg.MetadataValue.int(int(fn)),
            "confusion_matrix_tp": dg.MetadataValue.int(int(tp)),
            "precision": dg.MetadataValue.float(float(metrics["test_precision"])),
            "recall": dg.MetadataValue.float(float(metrics["test_recall"])),
            "f1_score": dg.MetadataValue.float(float(metrics["test_f1_score"])),
            "roc_auc": dg.MetadataValue.float(float(metrics["test_roc_auc"])),
            "specificity": dg.MetadataValue.float(float(specificity)),
            "sensitivity": dg.MetadataValue.float(float(sensitivity)),
            "plot_artifact_path": dg.MetadataValue.path(str(plot_path)),
            "mlflow_evaluation_run_id": dg.MetadataValue.text(str(eval_run_id or "none")),
        },
    )


@dg.asset(
    group_name="ml_fraud_registry",
    description="Registers the trained model into the MLflow Model Registry and promotes it to Production.",
    compute_kind="python",
    required_resource_keys={"fraud_mlflow", "fraud_mlflow_client"},
)
def register_model_to_registry(
    context: dg.AssetExecutionContext,
    train_model: Tuple[RandomForestClassifier, Dict[str, float]]
) -> dg.MaterializeResult:
    """Register the best model to the MLflow Model Registry."""

    fraud_mlflow = context.resources.fraud_mlflow
    fraud_mlflow_client = context.resources.fraud_mlflow_client

    model, metrics = train_model
    context.log.info(f"✅ Unpacked train_model tuple - model type: {type(model)}, metrics keys: {list(metrics.keys())}")

    mlflow.set_tracking_uri(fraud_mlflow.tracking_uri)
    context.log.info(f"✅ Set tracking URI: {fraud_mlflow.tracking_uri}")

    mlflow.set_experiment(fraud_mlflow.experiment_name)
    context.log.info(f"✅ Set experiment: {fraud_mlflow.experiment_name}")

    experiment = mlflow.get_experiment_by_name(fraud_mlflow.experiment_name)
    if experiment is None:
        context.log.error("❌ Experiment not found")
        return dg.MaterializeResult(
            value=None,
            metadata={"registration_status": dg.MetadataValue.text("failed"),
            "error": dg.MetadataValue.text("Experiment not found")}
        )

    experiment_id = experiment.experiment_id
    context.log.info(f"✅ Got experiment ID: {experiment_id}")

    runs = mlflow.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"], max_results=1)
    context.log.info(
        f"✅ Search runs returned type: {type(runs)}, shape: {runs.shape if hasattr(runs, 'shape') else 'N/A'}")  # type: ignore
    context.log.info(f"✅ Runs data:\n{runs}")

    if runs.empty:  # type: ignore
        context.log.error("❌ No runs found in experiment")
        return dg.MaterializeResult(
            value=None,
            metadata={"registration_status": dg.MetadataValue.text("failed"),
            "error": dg.MetadataValue.text("No runs found")}
        )

    latest_run = runs.iloc[0]  # type: ignore
    run_id: str = latest_run["run_id"]
    context.log.info(f"✅ Latest run ID: {run_id}")

    model_uri = f"runs:/{run_id}/final_fraud_model"
    context.log.info(f"✅ Model URI: {model_uri}")

    try:
        registered_model = mlflow.register_model(model_uri=model_uri, name=MODEL_REGISTRY_NAME)
        context.log.info(f"✅ Model registered - name: {registered_model.name}, version: {registered_model.version}")

        fraud_mlflow_client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=registered_model.version,
            stage="Production"
        )
        context.log.info("✅ Model promoted to Production")

        return dg.MaterializeResult(
            value={
                "name": registered_model.name,
                "version": str(registered_model.version),
                "stage": "Production",
            },
            metadata={
                "registration_status": dg.MetadataValue.text("success"),
                "model_name": dg.MetadataValue.text(MODEL_REGISTRY_NAME),
                "model_version": dg.MetadataValue.text(str(registered_model.version)),
                "model_stage": dg.MetadataValue.text("Production"),
            },
        )
    except Exception as e:
        context.log.error(f"❌ Error: {e}")
        return dg.MaterializeResult(
            value=None,
            metadata={
                "registration_status": dg.MetadataValue.text("failed"),
                "error": dg.MetadataValue.text(str(e)),
            },
        )


@dg.asset(
    group_name="ml_fraud_notification",
    description="Sends model performance summary to Slack after evaluation.",
    required_resource_keys={"fraud_mlflow"},
)
def send_slack_notification(
    context: dg.AssetExecutionContext,
    generate_evaluation_plots: Dict[str, Any]
) -> dg.MaterializeResult:
    """Send model performance metrics to Slack with emojis."""

    metrics: Dict[str, float] = generate_evaluation_plots.get("metrics", {})
    _ = generate_evaluation_plots.get("additional_metrics", {})

    context.log.info("Sending Slack notification...")

    SLACK_BOT_TOKEN = os.getenv("SLACK_AIMS_COURSE_BOT_TOKEN")
    SLACK_CHANNEL = os.getenv("SLACK_CHANNEL", "aims_course_october2025")

    slack_status = "skipped"
    slack_message = ""

    if not SLACK_BOT_TOKEN:
        context.log.warning("Slack bot token not configured — skipping notification.")
        slack_status = "skipped"
        slack_message = "Slack token not configured"
    else:
        try:
            client = WebClient(token=SLACK_BOT_TOKEN)
            message = (
                f"*✅ Fraud Detection Model Training Complete!* \n\n"
                f"*🏃 Model Performance on Test Set:*\n"
                f"• Precision: {metrics.get('test_precision', 0):.4f}\n"
                f"• Recall: {metrics.get('test_recall', 0):.4f}\n"
                f"• F1-Score: {metrics.get('test_f1_score', 0):.4f}\n"
                f"• ROC-AUC: {metrics.get('test_roc_auc', 0):.4f}\n"
                f"💻 Created by dagmaros27 & Chekwube"
            )

            response = client.chat_postMessage(channel=SLACK_CHANNEL, text=message)
            context.log.info(f"Slack message sent successfully (ts: {response['ts']})")
            slack_status = "sent"
            slack_message = f"Message sent to {SLACK_CHANNEL}"

        except SlackApiError as e:
            error_msg = e.response.get('error', str(e)) if hasattr(e, 'response') else str(e)
            context.log.error(f"Slack API Error: {error_msg}")
            slack_status = "failed"
            slack_message = f"Slack API error: {error_msg}"
        except Exception as e:
            context.log.error(f"Unexpected error sending Slack notification: {e}")
            slack_status = "failed"
            slack_message = f"Slack notification failed: {e}"

    return dg.MaterializeResult(
        value=slack_message,
        metadata={
            "slack_status": dg.MetadataValue.text(slack_status),
            "slack_channel": dg.MetadataValue.text(SLACK_CHANNEL or ""),
            "message": dg.MetadataValue.text(slack_message)
        }
    )
