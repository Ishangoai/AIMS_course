import json
import os
from typing import cast

import dagster as dg
import dagster_slack
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from .resources import FRAUD_SQLITE_DB_PATH

# Ensure mlflow writes to the same tracking DB used by the Dagster mlflow resource
mlflow.set_tracking_uri(f"sqlite:///{FRAUD_SQLITE_DB_PATH}")
mlflow.set_experiment("fraud_detection_pipeline")


def safe_start_run(run_name: str):
    """Start an MLflow run, using nested=True if a parent run is active."""
    if mlflow.active_run() is None:
        return mlflow.start_run(run_name=run_name)
    return mlflow.start_run(run_name=run_name, nested=True)


@dg.asset(
    description="Download and load data for the fraud detection",
    compute_kind="python",
    group_name="ml_fraud_ingest",
    required_resource_keys={"config"}
)
def fraud_df(context: dg.AssetExecutionContext):
    config = context.resources.config
    try:
        df = pd.read_csv(config.data_source)
    except Exception as e:
        context.log.error(f"Failed to load CSV: {e}")
        raise

    n_rows = int(len(df))
    context.log.info(f"Loaded {n_rows} rows from {config.data_source}")

    # Log a lightweight MLflow run for this asset (nested-safe)
    with safe_start_run("data_load"):
        mlflow.log_param("data_source", config.data_source)
        mlflow.log_param("n_rows", n_rows)

    metadata = {
        "n_rows": n_rows,
        "data_source": config.data_source,
        "load_report": dg.MetadataValue.md(f"Loaded {n_rows} rows from {config.data_source}")
    }
    return dg.Output(df, metadata=metadata)


@dg.multi_asset(
    outs={
        "X_train": dg.AssetOut(),
        "X_test": dg.AssetOut(),
        "y_train": dg.AssetOut(),
        "y_test": dg.AssetOut(),
    },
    group_name="ml_fraud_transform",
    required_resource_keys={"mlflow_track"}
)
def split_data(context: dg.AssetExecutionContext, fraud_df: pd.DataFrame):
    X = fraud_df.drop(columns=["Class"])
    y = fraud_df["Class"]

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # FIX: Explicitly cast the raw outputs to their expected types
    X_train = cast(pd.DataFrame, X_train_raw)
    X_test = cast(pd.DataFrame, X_test_raw)
    y_train = cast(pd.Series, y_train_raw)
    y_test = cast(pd.Series, y_test_raw)

    n_train = int(X_train.shape[0])
    n_test = int(X_test.shape[0])
    pos_total = int(y.sum())

    context.log.info(f"Split data: {n_train} train rows, {n_test} test rows, positives={pos_total}")

    # Log counts to MLflow in a nested-safe run
    with safe_start_run("data_split"):
        mlflow.log_param("n_train_rows", n_train)
        mlflow.log_param("n_test_rows", n_test)
        mlflow.log_param("n_positive_total", pos_total)

    # Yield each output with numeric metadata so Dagster records materialization and numeric points
    yield dg.Output(
        X_train,
        output_name="X_train",
        metadata={
            "rows": n_train,
            "positive_in_train": int(y_train.sum()),
            "split_report": dg.MetadataValue.md(f"Train rows: {n_train}, positives: {int(y_train.sum())}")
        }
    )
    yield dg.Output(
        X_test,
        output_name="X_test",
        metadata={
            "rows": n_test,
            "positive_in_test": int(y_test.sum()),
            "split_report": dg.MetadataValue.md(f"Test rows: {n_test}, positives: {int(y_test.sum())}")
        }
    )
    yield dg.Output(
        y_train,
        output_name="y_train",
        metadata={"rows": int(y_train.shape[0]), "positives": int(y_train.sum())}
    )
    yield dg.Output(
        y_test,
        output_name="y_test",
        metadata={"rows": int(y_test.shape[0]), "positives": int(y_test.sum())}
    )


@dg.asset(
    description="""Hyperparameter tuning (3-fold CV) with  n_estimators grid. Logs nested MLflow runs.""",
    ins={"X_train": dg.AssetIn(), "y_train": dg.AssetIn()},
    group_name="ml_fraud_model",
    required_resource_keys={"mlflow_track"}
)
def rf_hyperparam_tuning(context: dg.AssetExecutionContext, X_train: pd.DataFrame, y_train: pd.Series):
    """
    Performs a simple inner-loop GridSearchCV (3-fold StratifiedKFold) tuning only n_estimators.
    Logs:
      - Parent run: 'rf_hyperparam_tuning' with best params/score
      - Child runs: 'trial_i' for each grid entry with logged param n_estimators and metrics mean_test_f1 / std_test_f1
    Returns:
      dg.Output(value=best_params, metadata=...) so Dagster records numeric metadata for plotting.
    """

    param_grid = {"n_estimators": [50, 100, 150]}
    clf = RandomForestClassifier(random_state=42, n_jobs=-1)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    context.log.info("Starting RF hyperparameter tuning with 3-fold inner CV on training split.")
    trials = []
    with safe_start_run("rf_hyperparam_tuning"):
        grid = GridSearchCV(
            estimator=clf,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="f1",
            return_train_score=False,
            n_jobs=-1
        )
        grid.fit(X_train, y_train)

        # Log each trial as a nested run and collect trials for Dagster metadata
        for i in range(len(grid.cv_results_['params'])):
            params = grid.cv_results_['params'][i]
            mean_f1 = float(grid.cv_results_['mean_test_score'][i])
            std_f1 = float(grid.cv_results_['std_test_score'][i])
            with safe_start_run(f"trial_{i}"):
                mlflow.log_param("n_estimators", int(params["n_estimators"]))
                mlflow.log_metric("mean_test_f1", mean_f1)
                mlflow.log_metric("std_test_f1", std_f1)
            trials.append({
                "trial": i,
                "n_estimators": int(params["n_estimators"]),
                "mean_test_f1": mean_f1,
                "std_test_f1": std_f1
            })
            context.log.info(f"Logged trial_{i}: n_estimators={params['n_estimators']}, mean_test_f1={mean_f1:.4f}")

        best_params = grid.best_params_
        best_score = float(grid.best_score_)

        # Parent run logs (in parent run context)
        mlflow.log_params({k: int(v) if isinstance(v, (np.integer, int)) else v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_f1_score", best_score)

        context.log.info(f"Best params from GridSearchCV: {best_params}, best_cv_f1_score={best_score:.4f}")

    # Build a markdown summary of trials for Dagit
    summary_md = "### Grid Search Trials\n\n|trial|n_estimators|mean_test_f1|std_test_f1|\n|---:|---:|---:|---:|\n"
    for t in trials:
        summary_md += f"|{t['trial']}|{t['n_estimators']}|{t['mean_test_f1']:.4f}|{t['std_test_f1']:.4f}|\n"

    # Prepare numeric metadata for Dagster to plot (ensure numeric types)
    md = {
        "best_cv_f1_score": float(round(best_score, 6)),
        "best_n_estimators": int(best_params.get("n_estimators", 0)),
        "n_trials": int(len(trials)),
        "trials_summary": dg.MetadataValue.md(summary_md)
    }

    # Return the best_params as a dg.Output with metadata so Dagster will show materialization and numeric metadata
    return dg.Output(value=best_params, metadata=md)


@dg.asset(
    description="Emit a notification summarizing the load & split metadata (logs to mlflow and returns the message).",
    ins={"X_train": dg.AssetIn(), "X_test": dg.AssetIn(), "y_train": dg.AssetIn(), "y_test": dg.AssetIn()},
    group_name="ml_fraud_notification",
    required_resource_keys={"mlflow_track"}
)
def send_split_notification(context: dg.AssetExecutionContext, X_train, X_test, y_train, y_test):
    n_train = int(X_train.shape[0])
    n_test = int(X_test.shape[0])
    pos_train = int(y_train.sum())
    pos_test = int(y_test.sum())

    message = (
        f"📊 Data Load & Split Summary\n"
        f"- Train rows: {n_train}\n"
        f"- Test rows: {n_test}\n"
        f"- Positive (fraud) in train: {pos_train}\n"
        f"- Positive (fraud) in test: {pos_test}\n"
        f"🚀 Ready for modeling!"
    )
    context.log.info(f"Prepared notification message:\n{message}")

    # Log summary metrics to mlflow
    with safe_start_run("split_notification"):
        mlflow.log_metric("n_train_rows", n_train)
        mlflow.log_metric("n_test_rows", n_test)
        mlflow.log_metric("pos_train", pos_train)
        mlflow.log_metric("pos_test", pos_test)
        mlflow.log_param("notified_by", "dagster_asset_send_split_notification")

    # Return the message as an asset materialization with numeric metadata for Dagster plotting
    metadata = {
        "n_train_rows": n_train,
        "n_test_rows": n_test,
        "pos_train": pos_train,
        "pos_test": pos_test,
        "notification": dg.MetadataValue.md(message)
    }
    return dg.Output(value=message, metadata=metadata)


@dg.asset(
    description="Train RandomForest model and extract top 10 important features. Logs them to MLflow.",
    ins={
        "X_train": dg.AssetIn(),
        "y_train": dg.AssetIn(),
        "rf_hyperparam_tuning": dg.AssetIn()
    },
    group_name="ml_fraud_training",
    required_resource_keys={"mlflow_track"}
)
def feature_importance(
    context: dg.AssetExecutionContext,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    rf_hyperparam_tuning: dict
) -> dict:
    """
    Trains RandomForest with best hyperparameters and extracts top 10 important features.
    """

    n_estimators = int(rf_hyperparam_tuning.get("n_estimators", 100))
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_
    feature_names = X_train.columns
    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    top_10_features = feature_importance_df.head(10)
    top_feature_names = top_10_features["feature"].tolist()

    # Log top features to MLflow
    mlflow.log_params({
        "top_10_features": ",".join(top_feature_names)
    })

    context.log.info(f"Top 10 important features: {top_feature_names}")

    return {
        "top_10_features": top_feature_names
    }


@dg.asset(
    description="Train model with best hyperparameters and evaluate on test set. Logs model to MLflow, an10 important features.",  # noqa: E501
    ins={
        "X_train": dg.AssetIn(),
        "X_test": dg.AssetIn(),
        "y_train": dg.AssetIn(),
        "y_test": dg.AssetIn(),
        "rf_hyperparam_tuning": dg.AssetIn(),
        "feature_importance": dg.AssetIn()
    },
    group_name="ml_fraud_training",
    required_resource_keys={"mlflow_track"}
)
def train_evaluate(
    context: dg.AssetExecutionContext,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    rf_hyperparam_tuning: dict,
    feature_importance: dict
):
    # --- Prepare Directories ---
    artifacts_dir = os.path.abspath(os.path.join("artifacts", "ml_fraud"))
    os.makedirs(artifacts_dir, exist_ok=True)
    cm_path = os.path.join(artifacts_dir, "confusion_matrix.png")

    # --- Extract top 10 features ---
    top_10 = feature_importance["top_10_features"]
    # Filter features
    top_10 = feature_importance["top_10_features"]
    X_train = cast(pd.DataFrame, X_train[list(top_10)])
    X_test = cast(pd.DataFrame, X_test[list(top_10)])

    # Save top_10 features to JSON file (to be saved alongside the model)
    model_artifacts_dir = os.path.join(artifacts_dir, "model_artifacts")
    os.makedirs(model_artifacts_dir, exist_ok=True)
    top_features_path = os.path.join(model_artifacts_dir, "top_10_features.json")
    with open(top_features_path, "w") as f:
        json.dump(top_10, f, indent=4)

    # --- Train Model ---
    n_estimators = int(rf_hyperparam_tuning.get("n_estimators", 100))
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=42,
        n_jobs=-1
    )
    context.log.info(f"Training RandomForest with: n_estimators={n_estimators}")
    model.fit(X_train, y_train)

    # --- Predict and Evaluate ---
    y_pred = model.predict(X_test)
    f1 = float(f1_score(y_test, y_pred, zero_division='warn'))
    precision = float(precision_score(y_test, y_pred, zero_division='warn'))
    recall = float(recall_score(y_test, y_pred, zero_division='warn'))
    accuracy = float(accuracy_score(y_test, y_pred))

    # --- MLflow Logging ---
    with safe_start_run("final_model_training"):
        mlflow.log_metrics({
            "test_f1_score": f1,
            "test_precision": precision,
            "test_recall": recall,
            "test_accuracy": accuracy
        })
        mlflow.log_params({
            "n_estimators": n_estimators,
            "top_10_features": ",".join(top_10)
        })

        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        _, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set(
            xticks=[0, 1],
            yticks=[0, 1],
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'],
            xlabel='Predicted',
            ylabel='Actual',
            title='Confusion Matrix'
        )
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black", fontsize=12)
        plt.tight_layout()
        plt.savefig(cm_path)
        plt.close()

        # Log confusion matrix image
        mlflow.log_artifact(cm_path)
        # Log the top_10_features JSON file artifact separately
        mlflow.log_artifact(top_features_path, artifact_path="random_forest_model/artifacts")

        # --- Log Model with top 10 features as artifacts ---
        input_example = pd.DataFrame(X_test[top_10].iloc[:5])
        try:  # Log the sklearn model (without `artifacts` argument)
            mlflow.sklearn.log_model(  # pyright: ignore
                sk_model=model,
                artifact_path="random_forest_model",
                input_example=input_example,
                registered_model_name="fraud_detection_rf"
            )
        except Exception:
            print(".")

        context.log.info(f"Eval metrics - F1: {f1:.4f}, Precision: {precision:.4f}")
        context.log.info(f" Recall: {recall:.4f}, Accuracy: {accuracy:.4f}")
        context.log.info(f"Top 10 important features saved to: {top_features_path}")

    # --- Dagster Metadata for UI ---
    result_data = {
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "confusion_matrix": cm.tolist()
    }

    report_md = f"""
## Model Evaluation Results
- **F1 Score**: {f1:.4f}
- **Precision**: {precision:.4f}
- **Recall**: {recall:.4f}
- **Accuracy**: {accuracy:.4f}

![Confusion Matrix](file://{cm_path})
"""

    return dg.Output(
        value=result_data,
        metadata={
            "test_f1_score": round(f1, 6),
            "test_precision": round(precision, 6),
            "test_recall": round(recall, 6),
            "test_accuracy": round(accuracy, 6),
            "confusion_matrix_plot": dg.MetadataValue.path(cm_path),
            "evaluation_report": dg.MetadataValue.md(report_md),
            "top_10_features_file": dg.MetadataValue.path(top_features_path)
        }
    )


@dg.asset(
    description="Send Slack message with evaluation metrics and upload confusion matrix if possible.",
    ins={
        "train_evaluate": dg.AssetIn(),
        "rf_hyperparam_tuning": dg.AssetIn(),
    },
    group_name="ml_fraud_notification",
    resource_defs={
        # Per-asset Slack resource style (uses env var SLACK_AIMS_COURSE_BOT_TOKEN)
        "slack": dagster_slack.SlackResource(token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN")),
    },
    required_resource_keys={"slack"},
)
def send_results_to_slack(
    context: dg.AssetExecutionContext,
    train_evaluate: dict,
    rf_hyperparam_tuning: dict,
) -> dg.MaterializeResult:
    """
    Sends evaluation summary as a simple Slack message (text only).
    """

    # Extract metrics (fail fast if missing)
    f1 = float(train_evaluate["f1_score"])
    precision = float(train_evaluate["precision"])
    recall = float(train_evaluate["recall"])
    accuracy = float(train_evaluate["accuracy"])

    # Compose message text (your exact requested output)
    header = "Fraud Detection —Model by : Mirindra and TIAO :"
    metrics_text = (
        f"🎯 F1: {f1:.4f}\n"
        f"✨ Precision: {precision:.4f}\n"
        f"🔁 Recall: {recall:.4f}\n"
        f"🚀 Accuracy: {accuracy:.4f}"
    )
    best_text = f"Best params: {rf_hyperparam_tuning}"
    second_text = "😜 Second-best: see MLflow trials"

    message = f"{header}\nEvaluation Summary:\n{metrics_text}\n{best_text}\n{second_text}"

    # Send Slack message (no file uploads, no error handling)
    slack_channel = os.getenv("SLACK_AIMS_CHANNEL", "aims_course_october2025")
    slack_client = context.resources.slack.get_client()
    slack_client.chat_postMessage(channel=slack_channel, text=message)
    context.log.info("Slack notification sent to channel %s", slack_channel)

    # Return the message text as MaterializeResult value
    return dg.MaterializeResult(value=message)
