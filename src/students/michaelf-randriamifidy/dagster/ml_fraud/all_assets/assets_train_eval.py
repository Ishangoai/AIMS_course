
import dagster as dg
import mlflow
import mlflow.sklearn as ms
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import (
    cross_val_score,
    train_test_split,  # For robust splitting
)

from ...client_consumers import slack_provider
from ...ml.resources import mlflow_resource
from ..resources import FraudTuningConfig
from ..utils import (
    calculate_false_positive_rate,
    get_experiment,
    log_confusion_matrix,
    log_feature_importance,
    to_native,
)

EXP_FRAUD_DETECTION = "fraud_detection"


@dg.asset(
    description="Tunes Random Forest  hyperparameters using grid search and prepares data splits.",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_model_fraud",
)
def tune_random_forest_hyperparameters(
    context: dg.AssetExecutionContext,
    config: FraudTuningConfig,
    pandas_data_df: pd.DataFrame
) -> dict:
    mlflow_client = context.resources.mlflow_tracking
    context.log.info("Starting hyperparameter tuning for RandomForest model.")

    clean_df = pandas_data_df

    if len(clean_df) < 20:
        msg = "Not enough data points for hyperparameter tuning, training, and testing. Need at least 20."
        context.log.error(msg)
        raise ValueError(msg)

    target_col = "Class"
    y = clean_df[target_col]
    X = clean_df.drop(columns=[target_col])
    feature_names = X.columns
    X = X.values
    y = y.values

    # Split data: 80% for training + grid search validation, 20% for final test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X,
                                                                y,
                                                                test_size=0.2,
                                                                random_state=42,
                                                                shuffle=False)

    if len(X_train_val) < 5 or len(X_test) < 1:  # Need enough for grid search val and at least one test sample
        msg = "Train/validation or test set is too small after initial split."
        context.log.error(msg)
        raise ValueError(msg)

    context.log.info(f"Data split: X_train_val: {X_train_val.shape}, X_test: {X_test.shape}")  # type: ignore

    # Define grid search search space for RandomForest n_estimators
    grid_n_estimators = [10, 20, 30, 40, 50]

    # MLflow experiment context for nested runs

    experiment_id = get_experiment(mlflow_client, EXP_FRAUD_DETECTION)
    k_folds = 3

    best_f1 = -float('inf')
    best_params = {}

    for trial_num, n_estimators in enumerate(grid_n_estimators):
        try:
            # Split train_val further for evaluation
            X_train_h, X_val_h, y_train_h, _ = train_test_split(
                X_train_val, y_train_val, test_size=0.2, random_state=42, shuffle=True
            )

            if len(X_train_h) == 0 or len(X_val_h) == 0 or k_folds >= len(X_train_h):
                context.log.warning(f"Trial {trial_num}: invalid split, skipping")
                continue

            run_name = f"gridsearch_trial_{trial_num}_n_estimators_{n_estimators:.4f}"

            model = RandomForestClassifier(n_estimators=n_estimators,
                                                random_state=42,
                                                n_jobs=-1
                                                )
            model.fit(X_train_h, y_train_h)

            f1 = cross_val_score(model, X_train_h, y_train_h, cv=k_folds, scoring='f1').mean()
            with mlflow_client.start_run(experiment_id=experiment_id,
                            run_name=run_name,
                            nested=True):
                context.log.info(f"Trial {trial_num} successful: params={n_estimators}, f1={f1:.4f}")
            if f1 > best_f1:
                best_f1 = f1
                best_params = {'n_estimators': n_estimators}
        except Exception as e:
            context.log.error(f"Trial {trial_num} failed: {e}")

    context.log.info(f"gridsearch completed, Best params: {best_params}, Best F1: {best_f1:.4f}")

    best_n_estimators_to_log = float('nan')  # Initialize
    best_f1_score_to_log = -float('inf')  # Initialize

    if best_params:
        best_n_estimators_to_log = best_params["n_estimators"]
    else:
        raise ValueError("gridsearch tuning failed: no successful trials found.")

    context.log.info(f"Final best n_estimators: {best_n_estimators_to_log:.4f}"
                    ", Corresponding f1_score: {best_f1_score_to_log:.4f}")

    with mlflow_client.start_run(experiment_id=experiment_id,
                                run_name="best_gridsearch_trial",
                                nested=True):
        mlflow_client.log_param("best_random_forest_n_estimators", best_n_estimators_to_log)
        if best_f1_score_to_log != float('inf'):
            mlflow_client.log_metric("best_gridsearch_validation_f1_score", best_f1_score_to_log)
        else:
            # log something to indicate the metric wasn't reliably found
            mlflow_client.log_metric("best_gridsearch_validation_f1_score_unavailable", 0.0)

    final_best_params_output = {'n_estimators': int(best_n_estimators_to_log)}

    # Return a single dictionary
    return {
        "best_params": final_best_params_output,
        "X_train_val": X_train_val,
        "y_train_val": y_train_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
    }


@dg.asset(
description="Trains a Random Forest model using the best hyperparameters found by grid search.",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_model_fraud"
)
def train_tuned_model_fraud(
    context: dg.AssetExecutionContext,
    tune_random_forest_hyperparameters
) -> dict:
    mlflow_client = context.resources.mlflow_tracking
    experiment_id = get_experiment(mlflow_client, EXP_FRAUD_DETECTION)
    context.log.info("Starting final model training with tuned hyperparameters.")

    best_params = tune_random_forest_hyperparameters["best_params"]
    X_train_val = tune_random_forest_hyperparameters["X_train_val"]
    y_train_val = tune_random_forest_hyperparameters["y_train_val"]
    X_test = tune_random_forest_hyperparameters["X_test"]
    y_test = tune_random_forest_hyperparameters["y_test"]
    feature_names = tune_random_forest_hyperparameters["feature_names"]

    context.log.info(f"Training Random Forest model with parameters: {best_params}")
    context.log.info(f"Training on {len(X_train_val)} samples.")

    final_model = RandomForestClassifier(n_estimators=int(best_params['n_estimators']),
                                        random_state=42,
                                        n_jobs=-1
                                        )
    final_model.fit(X_train_val, y_train_val)
    context.log.info("Final Random Forest model trained.")

    train_params_log = {
        "model_type": "Random Forest",
        "n_estimators": best_params['n_estimators'],
        "feature_used": ", ".join(feature_names),
        "lag_period": 1,  # Assuming lag_period is 1 as per feature eng.
        "final_train_samples": len(X_train_val),
        "final_test_samples": len(X_test)
    }
    with mlflow_client.start_run(experiment_id=experiment_id,
                            nested=True):
        mlflow_client.log_params(train_params_log)
        context.log.info(f"Logged final training parameters to MLflow: {train_params_log}")

    return {
        "model": final_model,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names
    }


@dg.asset(
    description="Evaluates the tuned model and logs model and metrics to MLflow.",
    resource_defs={"mlflow_tracking": mlflow_resource,
                "slack": slack_provider
                },
    compute_kind="python",
    group_name="ml_evaluate_fraud"
)
def test_model_fraud(
    context: dg.AssetExecutionContext,
    train_tuned_model_fraud: dict
) -> dg.MaterializeResult:
    # slack: dagster_slack.SlackResource = context.resources.slack
    mlflow_client = context.resources.mlflow_tracking
    experiment_id = get_experiment(mlflow_client, EXP_FRAUD_DETECTION)
    context.log.info("Starting final model evaluation.")

    model = train_tuned_model_fraud["model"]
    X_test = train_tuned_model_fraud["X_test"]
    y_test = train_tuned_model_fraud["y_test"]
    feature_names = train_tuned_model_fraud.get("feature_names", ["feature"])

    if len(X_test) == 0:
        context.log.warning("Test set is empty. Skipping evaluation and model logging.")
        return dg.MaterializeResult(
            value={
                "status": "skipped_evaluation",
                "reason": "Test set empty, evaluation skipped.",
                "eval_metrics": {"test_accuracy": float('nan'),
                                "test_recall":  float('nan'),
                                "test_fpr":  float('nan')},
                "model_version_info": None
            },
            metadata={
                "status": "skipped_evaluation",
                "reason": dg.MetadataValue.text("Test set was empty, no evaluation performed."),
                "test_accuracy": dg.MetadataValue.float(float('nan')),
                "test_recall": dg.MetadataValue.float(float('nan')),
                "test_fpr": dg.MetadataValue.float(float('nan')),
            }
        )
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    recall = recall_score(y_test, preds)
    fpr = calculate_false_positive_rate(y_test, preds)

    with mlflow_client.start_run(experiment_id=experiment_id,
                            run_name="relevant_model_plots",
                            nested=True):
        context.log.info("Final Model Evaluation Metrics on Test Set"
                ": accuracy={accuracy:.4f}, recall={recall:.4f}, fpr={fpr:.4f}")

        context.log.info("Logging Confusion matrix to Mlflow Artifacts")
        log_confusion_matrix(y_test, preds, labels=[0, 1])
        context.log.info("Confusion matrix logged as Artifact to Mlflow")

        context.log.info("Logging Feature Importance Plot to MLflow Artifacts")
        log_feature_importance(
            feature_names=feature_names,
            importances=model.feature_importances_,
            cumulative_threshold=0.8  # automatically select top 80%
        )
        context.log.info("Feature Importance Plot logged as Artifact to MLflow")

    eval_metrics = {"test_accuracy": accuracy, "test_recall": recall, "test_fpr": fpr}

    mlflow_client.log_metrics(eval_metrics)
    context.log.info(f"Logged final evaluation metrics to MLflow: {eval_metrics}")

    registered_model_name = "tuned-fraud-detector"
    model_version_info = None

    with mlflow.start_run(nested=True) as current_run:
        context.log.info(f"Starting nested MLflow run for model logging: {current_run.info.run_id}")

        log_model_info = ms.log_model(
            sk_model=model,
            artifact_path="tuned_fraud_detector",
            input_example=pd.DataFrame(X_test[:min(5, len(X_test))], columns=feature_names),
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
            "test_accuracy": dg.MetadataValue.float(to_native(accuracy)),
            "test_recall": dg.MetadataValue.float(to_native(recall)),
            "test_fpr": dg.MetadataValue.float(to_native(fpr)),
            "model_name": dg.MetadataValue.text(model_version_info["name"]),
            "model_version": dg.MetadataValue.text(str(model_version_info["version"])),
            "mlflow_run_id": dg.MetadataValue.text(current_run.info.run_id),
            "mlflow_model_uri": dg.MetadataValue.text(model_version_info["model_uri"]),
            "mlflow_stage": dg.MetadataValue.text(model_version_info["stage"])
        }
    )
