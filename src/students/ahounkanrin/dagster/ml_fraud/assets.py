import os
import tempfile
from collections import abc

import dagster as dg
import dagster_slack
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn as ms
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import KFold, train_test_split

from ..ml.resources import mlflow_client, mlflow_resource
from .resources import PromotionConfig


@dg.asset(
    description="Download data for fraud detection",
    compute_kind="python",
    resource_defs={"mlflow_tracking": mlflow_resource},
    group_name="download_fraud_data"
)
def fraud_data(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:
    mlflow_client = context.resources.mlflow_tracking

    url = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"
    data = pd.read_csv(url)
    context.log.info(f"Fraud data downloaded with {len(data)} rows.")
    columns = [dg.TableColumn(k, str(v)) for k, v in data.dtypes.to_dict().items()]

    dataset = mlflow_client.data.from_pandas(data, name="ml_fraud_train_data")
    mlflow_client.log_input(dataset=dataset, context="training")

    return dg.MaterializeResult(
        value=data,
        metadata={
            "preview": dg.MetadataValue.md(data.head().to_markdown() or ""),
            "dagster/row_count": len(data),
            "dagster/column_schema": dg.TableSchema(columns=columns)
        }
    )


@dg.asset(
    description="Split data into training and test sets",
    compute_kind="python",
    group_name="split_data"
)
def training_test_data(
    context: dg.AssetExecutionContext,
    fraud_data: pd.DataFrame,
) -> dg.MaterializeResult:

    train_df, test_df = train_test_split(fraud_data,
                                         test_size=0.2,
                                         random_state=42,
                                         shuffle=True)
    context.log.info(f"Data split into training set of size {len(train_df)}\
                     and test set of size {len(test_df)}.")

    return dg.MaterializeResult(
        value=(train_df, test_df),
        metadata={
        "dagster/row_count_train": len(train_df),
        "dagster/row_count_test": len(test_df)
        })


@dg.asset(
    description="Tunes Randdom Forest Classifier hyperparameters.",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="random_forest_model",
)
def tune_rf_hyperparameters(  # noqa: C901
    context: dg.AssetExecutionContext,
    # config: TuningConfig,
    training_test_data: tuple[pd.DataFrame, pd.DataFrame]
) -> dict:
    """
    Perform hyperparameter tuning for a Random Forest model using GridSearchCV.
    Logs results to MLflow and returns the best parameters.
    """

    mlflow_client = context.resources.mlflow_tracking
    context.log.info("Starting hyperparameter tuning for Random Forest model.")

    train_data, test_data = training_test_data
    X_train = train_data.iloc[:, 1:30]
    y_train = train_data.iloc[:, 30].astype(int)
    X_test = test_data.iloc[:, 1:30]
    y_test = test_data.iloc[:, 30].astype(int)

    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

    # MLflow experiment context for nested runs
    # Ensure the experiment exists or is created
    # Ensure experiment exists and is active
    experiment_name = "credit_card_fraud_detection"
    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow_client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    if mlflow.active_run() is not None:
        mlflow.end_run()
    with mlflow.start_run(experiment_id=experiment_id,
                          run_name="hp_tuning_with_CV",
                          ):
        best_f1 = 0
        best_params = {}

        # Example hyperparameter grid
        for n_estimators in [10, 50, 100, 200, 300]:
            inner_fold = 1
            inner_scores = []

            # Inner CV for this hyperparameter config
            for inner_train_idx, inner_val_idx in inner_cv.split(X_train, y_train):
                X_train_inner, X_val_inner = X_train.iloc[inner_train_idx], X_train.iloc[inner_val_idx]
                y_train_inner, y_val_inner = y_train.iloc[inner_train_idx], y_train.iloc[inner_val_idx]

                # Start a child (nested) run
                with mlflow.start_run(
                    experiment_id=experiment_id,
                    run_name=f"n_estimators_{n_estimators}_fold_{inner_fold}",
                    nested=True):
                    model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        random_state=42
                    )
                    model.fit(X_train_inner, y_train_inner)

                    preds = model.predict(X_val_inner)
                    f1 = f1_score(y_val_inner, preds)

                    # Log metrics and params for this fold
                    mlflow_client.log_param("n_estimators", n_estimators)
                    mlflow_client.log_metric("val_f1", f1)

                    inner_scores.append(f1)
                    inner_fold += 1

            # Aggregate inner CV results
            mean_f1 = sum(inner_scores) / len(inner_scores)
            mlflow_client.log_metric(f"mean_f1_n{n_estimators}", mean_f1)

            # Track best params per outer fold
            if mean_f1 > best_f1:
                best_f1 = mean_f1
                best_params = {"n_estimators": n_estimators}

        # Log performance across all hyperparameters
        mlflow_client.log_params(best_params)
        mlflow_client.log_metric("best_f1", best_f1)

    # Return a single dictionary
    return {
        "best_params": best_params,
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
    }


@dg.asset(
    description="Trains a Random Forest classification model using the best hyperparameters",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="random_forest_model"
)
def train_tuned_rf_model(
    context: dg.AssetExecutionContext,
    tune_rf_hyperparameters
) -> dict:
    mlflow_client = context.resources.mlflow_tracking
    context.log.info("Starting final model training with tuned hyperparameters.")

    best_params = tune_rf_hyperparameters["best_params"]
    X_train = tune_rf_hyperparameters["X_train"]
    y_train = tune_rf_hyperparameters["y_train"]
    X_test = tune_rf_hyperparameters["X_test"]
    y_test = tune_rf_hyperparameters["y_test"]

    context.log.info(f"Training Random Forest model with parameters: {best_params}")
    context.log.info(f"Training on {len(X_train)} samples.")

    final_model = RandomForestClassifier(n_estimators=best_params["n_estimators"],
                                         random_state=42)
    final_model.fit(X_train, y_train)
    context.log.info("Final Random Forest model trained.")

    train_params_log = {
        "model_type": "Random Forest Classifier",
        "n_estimators": best_params['n_estimators'],
        "final_train_samples": len(X_train),
        "final_test_samples": len(X_test)
    }
    mlflow_client.log_params(train_params_log)
    context.log.info(f"Logged final training parameters to MLflow: {train_params_log}")

    return {
        "model": final_model,
        "X_test": X_test,
        "y_test": y_test,
    }


@dg.asset(
    description="Evaluates the tuned model and logs model and metrics to MLflow.",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="evaluate_model"
)
def test_rf_model(
    context: dg.AssetExecutionContext,
    train_tuned_rf_model: dict
) -> dg.MaterializeResult:
    mlflow_client = context.resources.mlflow_tracking
    context.log.info("Starting final model evaluation.")

    model = train_tuned_rf_model["model"]
    X_test = train_tuned_rf_model["X_test"]
    y_test = train_tuned_rf_model["y_test"]

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_true=y_test, y_pred=predictions)
    precision = precision_score(y_true=y_test, y_pred=predictions)
    recall = recall_score(y_true=y_test, y_pred=predictions)
    f1 = f1_score(y_true=y_test, y_pred=predictions)

    context.log.info(f"Final Model Evaluation Metrics on Test Set:\
                     Accuracy={accuracy:.4f},\
                     Precision={precision:.4f},\
                     Recall ={recall:.4f}\
                     F1 score = {f1:.4f}")

    eval_metrics = {"test_accuracy": accuracy, "test_precision": precision,
                    "test_recall": recall, "test_f1_score": f1}
    # mlflow_client.log_metrics(eval_metrics)
    context.log.info(f"Logged final evaluation metrics to MLflow: {eval_metrics}")

    registered_model_name = "tuned-fraud-detector"
    model_version_info = None

    experiment_name = "credit_card_fraud_detection"
    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow_client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id,
                          run_name="model_evaluation",
                          nested=True) as current_run:
        context.log.info(f"Starting nested MLflow run for model logging: {current_run.info.run_id}")

        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        with tempfile.TemporaryDirectory(prefix="mlflow_eval_") as tmpdir:
            cm_path = os.path.join(tmpdir, "confusion_matrix.png")
            fig, ax = plt.subplots(figsize=(6, 5))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            disp.plot(ax=ax, values_format='d', colorbar=False)
            ax.set_title("Confusion Matrix")
            plt.tight_layout()
            fig.savefig(cm_path, dpi=150)
            plt.close(fig)

            mlflow_client.log_artifact(
                run_id=current_run.info.run_id,
                local_path=cm_path,
                artifact_path="evaluation_plots"
            )
        context.log.info(f"Confusion matrix logged to MLflow: {cm_path}")

        # ROC curve
        y_probs = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_probs)
        roc_auc = auc(fpr, tpr)

        with tempfile.TemporaryDirectory(prefix="mlflow_eval_") as tmpdir:
            roc_path = os.path.join(tmpdir, "roc_curve.png")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC={roc_auc:.4f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve - Test Set')
            ax.legend(loc='lower right')
            plt.tight_layout()
            fig.savefig(roc_path, dpi=150)
            plt.close(fig)

            mlflow_client.log_artifact(
                run_id=current_run.info.run_id,
                local_path=roc_path,
                artifact_path="evaluation_plots"
            )

        context.log.info(f"ROC curve logged to MLflow: {roc_path}")

        mlflow_client.log_metrics(eval_metrics)
        mlflow_client.log_metric("test_AUC", roc_auc)
        context.log.info(f"Logged final evaluation metrics to MLflow: {eval_metrics}")

        log_model_info = ms.log_model(
            sk_model=model,
            artifact_path="tuned_fraud_detector",
            input_example=pd.DataFrame(X_test[:min(5, len(X_test))],
                                       columns=X_test.columns.tolist()),
            registered_model_name=registered_model_name
        )
        context.log.info(f"Model logged to MLflow Run ID: {current_run.info.run_id}")
        context.log.info(f"Logged model artifact URI: {log_model_info.model_uri}")

        # use search_model_versions with a proper filter string
        model_versions = mlflow_client.search_model_versions(
            filter_string=f"name='{registered_model_name}'"
        )

        # Find the model version registered in this run
        matching_versions = [
            mv for mv in model_versions if mv.run_id == current_run.info.run_id
        ]

        if matching_versions:
            registered_model_version = matching_versions[0]
            model_version_info = {
                "name": registered_model_version.name,
                "version": registered_model_version.version,
                "status": registered_model_version.status,
                "stage": registered_model_version.current_stage,
                "model_uri": f"models:/{registered_model_version.name}/{registered_model_version.version}"
            }
            context.log.info("Successfully retrieved registered model version info from registry.")
        else:
            context.log.error(
                f"Could not find registered model version for run ID {current_run.info.run_id} "
                f"and name '{registered_model_name}'."
            )
            raise Exception("Failed to retrieve registered model version details after logging.")

    output_value_for_downstream = {
        "eval_metrics": eval_metrics,
        "model_version_info": model_version_info,
        "status": "evaluated_successfully"
    }

    return dg.MaterializeResult(
        value=output_value_for_downstream,
        metadata={
            "test_accuracy": dg.MetadataValue.float(float(accuracy)),
            "test_precision": dg.MetadataValue.float(float(precision)),
            "test_recall": dg.MetadataValue.float(float(recall)),
            "test_f1_score": dg.MetadataValue.float(float(f1)),
            "model_name": dg.MetadataValue.text(model_version_info["name"]),
            "model_version": dg.MetadataValue.text(str(model_version_info["version"])),
            "mlflow_run_id": dg.MetadataValue.text(current_run.info.run_id),
            "mlflow_model_uri": dg.MetadataValue.text(model_version_info["model_uri"]),
            "mlflow_stage": dg.MetadataValue.text(model_version_info["stage"])
        }
    )


# Promoting a model to Staging means it passed your quality checks and is ready
# for more thorough testing or limited release
@dg.asset(
    description="Promotes the newly trained model to Staging if it meets performance criteria.",
    resource_defs={"mlflow_tracking": mlflow_resource, "mlflow_client": mlflow_client},
    compute_kind="python",
    group_name="promote_model"
)
def promote_rf_model_to_staging(
    context: dg.AssetExecutionContext,
    config: PromotionConfig,
    test_rf_model: dict
) -> dg.MaterializeResult:
    # Get the MLflow client from the context to interact with the model registry
    mlflow_client = context.resources.mlflow_client
    context.log.info("Starting model promotion to Staging.")

    # If the evaluation step was skipped, we also skip promotion
    if test_rf_model.get("status") == "skipped_evaluation":
        context.log.info("Evaluation was skipped in the previous step. Skipping staging promotion.")
        return dg.MaterializeResult(
            value={"status": "skipped_promotion", "reason": "evaluation_skipped_upstream"},
            metadata={"status": "skipped_promotion_due_to_upstream_skip"}
        )
    # Extract metrics and model version info from evaluation result
    eval_metrics = test_rf_model.get("eval_metrics", {})
    model_version_info = test_rf_model.get("model_version_info")

    # If no model version info was returned, skip promotion.
    # model_version_info might be None due to an upstream failure
    if not model_version_info:
        context.log.warning("No valid model version info found from evaluation. Skipping staging promotion.")
        return dg.MaterializeResult(
            value={"status": "skipped_promotion", "reason": "no_model_version_info_from_evaluation"},
            metadata={"status": "skipped_promotion_no_model_info"}
        )
    # Get performance metrics (default to infinity if missing)
    current_accuracy = eval_metrics.get("test_accuracy", float(0))
    current_precision = eval_metrics.get("test_precision", float(0))
    current_recall = eval_metrics.get("test_recall", float(0))
    current_f1_score = eval_metrics.get("test_f1_score", float(0))

    STAGGING_F1_SCORE_THRESHOLD = config.stagging_f1_score_threshold
    # Log the evaluation metrics and threshold criteria
    context.log.info(f"Model evaluated with Accuracy: {current_accuracy:.4f}, F1 score: {current_f1_score:.4f}")
    context.log.info(f"Staging promotion thresholds: F1 score > {STAGGING_F1_SCORE_THRESHOLD}")

    # Check if model meets promotion criteria
    if current_f1_score >= STAGGING_F1_SCORE_THRESHOLD:
        try:
            # Extract the model name and version for promotion
            model_name = model_version_info["name"]
            model_version = model_version_info["version"]

            # Promote the model to the 'Staging' stage
            context.log.info(f"Model '{model_name}' (version {model_version}) meets criteria. Promoting to Staging")
            mlflow_client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Staging"
            )

            # Return successful result with status and relevant metadata
            context.log.info(f"Model '{model_name}' (version {model_version}) promoted to Staging.")
            return dg.MaterializeResult(
                value={
                    "status": "promoted_to_staging",
                    "model_name": model_name,
                    "model_version": model_version,
                    "metrics": eval_metrics
                },
                metadata={
                    "status": "promoted_to_staging",
                    "model_name": dg.MetadataValue.text(model_name),
                    "model_version": dg.MetadataValue.text(str(model_version)),
                    "accuracy_at_promotion": dg.MetadataValue.float(current_accuracy),
                    "precision_at_promotion": dg.MetadataValue.float(current_precision),
                    "recall_at_promotion": dg.MetadataValue.float(current_recall),
                    "f1_score_at_promotion": dg.MetadataValue.float(current_f1_score)
                }
            )
        except Exception as e:
            # Handle any exception during promotion and log the error
            context.log.error(f"Error promoting model to Staging: {e}")
            return dg.MaterializeResult(
                value={"status": "failed_promotion_to_staging", "error": str(e)},
                metadata={"status": "failed_promotion_to_staging", "error_message": dg.MetadataValue.text(str(e))}
            )
    # If model doesn't meet criteria, log and return "not promoted"
    else:
        context.log.info("Model does not meet performance criteria for Staging promotion. Skipping.")
        return dg.MaterializeResult(
            value={
                "status": "not_promoted_to_staging",
                "reason": "criteria_not_met",
                "metrics": eval_metrics
            },
            metadata={
                "status": "not_promoted_to_staging",
                "accuracy": dg.MetadataValue.float(current_accuracy),
                "precision": dg.MetadataValue.float(current_precision),
                "recall": dg.MetadataValue.float(current_recall),
                "f1_score": dg.MetadataValue.float(current_f1_score)
            }
        )


@dg.asset(
    description="Posts model performance on Slack.",
    resource_defs={"slack_resource": dagster_slack.SlackResource(token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN"))},
    compute_kind="python",
    group_name="post_performance"
)
def slack_post(
    context: dg.AssetExecutionContext,
    promote_rf_model_to_staging: dict
) -> None:
    eval_metrics = promote_rf_model_to_staging["metrics"]
    f1 = eval_metrics["test_f1_score"]
    context.log.info(f"F1 score: {f1:.4f}")
    # slack_resource: dagster_slack.SlackResource = context.resources.slack_resource
    # slack_resource.get_client().chat_postMessage(
    #     channel='aims_course_october2025',
    #     text=f"{os.environ.get("GITHUB_USER", "default")}'s F1 score: {f1:.4f}."
    # )


@dg.multi_asset_check(
    specs=[dg.AssetCheckSpec(name="no_nulls", asset="fraud_data", blocking=False)]
)
def dq_check_ml_fraud(fraud_data) -> abc.Iterable[dg.AssetCheckResult]:
    has_nan = fraud_data.isnull().values.any()
    yield dg.AssetCheckResult(
        check_name="no_nulls",
        passed=not (has_nan),
        asset_key="fraud_data"
    )
