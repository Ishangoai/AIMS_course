import os
from collections import abc

import dagster as dg
import dagster_slack
import hyperopt
import mlflow
import mlflow.sklearn as ms
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, average_precision_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import train_test_split  # For robust splitting

from ...ml.resources import mlflow_resource
from ...ml.resources import mlflow_resource, mlflow_client

from ..resources import FraudTuningConfig
from ..utils import (
    calculate_false_positive_rate, to_native, random_forest_summary_message,
    post_message_in_slack
)

from ...client_consumers import slack_provider

@dg.asset(
    description="Tunes Random Forest  hyperparameters using Hyperopt and prepares data splits.",
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
    context.log.info("Starting hyperparameter tuning for Ridge model.")

    clean_df = pandas_data_df
    
    if len(clean_df) < 20:  # Increased threshold for meaningful splits
        msg = "Not enough data points for hyperparameter tuning, training, and testing. Need at least 20."
        context.log.error(msg)
        raise ValueError(msg)

    target_col = "Class"
    y = clean_df[target_col]
    X = clean_df.drop(columns=[target_col])
    feature_names = X.columns
    X = X.values
    y = y.values

    # Split data: 80% for training + hyperopt validation, 20% for final test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X,
                                                                y,
                                                                test_size=0.2,
                                                                random_state=42,
                                                                shuffle=False)

    if len(X_train_val) < 5 or len(X_test) < 1:  # Need enough for hyperopt val and at least one test sample
        msg = "Train/validation or test set is too small after initial split."
        context.log.error(msg)
        raise ValueError(msg)

    context.log.info(f"Data split: X_train_val: {X_train_val.shape}, X_test: {X_test.shape}")  # type: ignore
    
    # Define Hyperopt search space for RandomForest n_estimators
    n_estimators_list = [10, 20, 30]
    search_space = {
        'n_estimators': hyperopt.hp.choice('n_estimators',
                                        n_estimators = n_estimators_list)
    }
    
    # MLflow experiment context for nested runs
    # Ensure the experiment exists or is created
    try:
        experiment = mlflow_client.get_experiment_by_name("fraud_detection")
        if experiment is None:
            experiment = mlflow_client.create_experiment("fraud_detection")
            experiment_id = experiment.experiment_id
        else:
            experiment_id = experiment.experiment_id
    except Exception:  # Handle cases where get_experiment_by_name might raise error if not found
        experiment_id = mlflow_client.create_experiment("fraud_detection")

    trials = hyperopt.Trials()
    k_folds = 3
    # Objective function for Hyperopt
    def objective(params):
        trial_num = len(trials.trials)
        try:
            n_estimators = int(params['n_estimators'])
            # Split train_val further into training_for_hyperopt and validation_for_hyperopt
            X_train_h, X_val_h, y_train_h, y_val_h = train_test_split(X_train_val,
                                                                y_train_val,
                                                                test_size=0.2,
                                                                random_state=42,
                                                                shuffle=True)

            if len(X_train_h) == 0 or len(X_val_h) == 0:
                # This case should be rare given prior checks but good to have
                return {'loss': float('inf'), 'status': hyperopt.STATUS_OK, 'params': params}  # Penalize if split fails
            run_name = f"hyperopt_trial_{trial_num}_n_estimators_{n_estimators:.4f}"
        
            model = RandomForestClassifier(n_estimators=n_estimators,
                                            random_state=42,
                                            n_jobs=-1
                                            )
            model.fit(X_train_h, y_train_h)
            preds = model.predict(X_val_h)
            f1 = cross_val_score(model, X_train_h, y_train_h, cv=k_folds, scoring='f1').mean()
            return {'loss': -f1, 'status': hyperopt.STATUS_OK, 'params': params}
        except Exception as e:
            return {'loss': float('inf'), 'status': hyperopt.STATUS_FAIL, 'params': params, 'error_message': str(e)}

    best_hyperparams = hyperopt.fmin(
        fn=objective,
        space=search_space,
        algo=hyperopt.tpe.suggest,
        # max_evals=2,
        max_evals=config.max_hyperopt_evals,  # Number of iterations
        trials=trials
    )
    context.log.info(f"fmin completed. Returned best_hyperparams: {best_hyperparams}")
    best_trial_info = trials.best_trial
    best_n_estimators_to_log = float('nan')  # Initialize
    best_f1_score_to_log = float('inf')  # Initialize

    if best_trial_info is None:
        # context.log.error("trials.best_trial is None.")
        if best_hyperparams:
            # context.log.warning(f"Using fmin's output: {best_hyperparams}")
            best_index = best_hyperparams.get('n_estimators', None)
            best_n_estimators_to_log = n_estimators_list[best_index] if best_index is None else None
            # best_f1_score_to_log remains float('inf')
        else:
            raise ValueError("Hyperopt tuning failed: fmin returned no parameters and no successful trials found.")
    else:
        best_f1_score_to_log = -best_trial_info['result']['loss']
        n_estimators_from_best_trial = best_trial_info['misc']['vals']['n_estimators'][0]
        best_n_estimators_to_log = n_estimators_list[n_estimators_from_best_trial]
        context.log.info(
            f"Hyperopt best n_estimators (fmin): {best_n_estimators_to_log}, "  # type: ignore
            f"Best validation_f1_score (trials.best_trial): {best_f1_score_to_log:.4f}"
        )

    context.log.info(f"Final best n_estimators: {best_n_estimators_to_log:.4f}, Corresponding f1_score: {best_f1_score_to_log:.4f}")
    mlflow_client.log_param("best_random_forest_n_estimators", best_n_estimators_to_log)
    if best_f1_score_to_log != float('inf'):
        mlflow_client.log_metric("best_hyperopt_validation_f1_score", best_f1_score_to_log)
    else:
        # log something to indicate the metric wasn't reliably found
        mlflow_client.log_metric("best_hyperopt_validation_f1_score_unavailable", 0.0)

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
description="Trains a Random Forest model using the best hyperparameters found by Hyperopt.",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_model_fraud"
)
def train_tuned_model_fraud(
    context: dg.AssetExecutionContext,
    tune_random_forest_hyperparameters
) -> dict:
    mlflow_client = context.resources.mlflow_tracking
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
    slack: dagster_slack.SlackResource = context.resources.slack
    mlflow_client = context.resources.mlflow_tracking
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
    context.log.info(f"Final Model Evaluation Metrics on Test Set: accuracy={accuracy:.4f}, recall={recall:.4f}, fpr={fpr:.4f}")

    eval_metrics = {"test_accuracy": accuracy, "test_recall": recall, "test_fpr": fpr}
    mlflow_client.log_metrics(eval_metrics)
    context.log.info(f"Logged final evaluation metrics to MLflow: {eval_metrics}")

    registered_model_name = "tuned-fraud-identifier"
    model_version_info = None

    with mlflow.start_run(nested=True) as current_run:
        context.log.info(f"Starting nested MLflow run for model logging: {current_run.info.run_id}")

        log_model_info = ms.log_model(
            sk_model=model,
            artifact_path="tuned_fraud_identifier",
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

    authors = "Rado Fitiavana and Michael Fitiavana"
    hyperparameters = model.get_params()
    n_estimators = hyperparameters.n_estimators
    message = random_forest_summary_message(authors, accuracy, recall, fpr, n_estimators)
    channel = "aims_course_october2025"

    try:
        post_message_in_slack(slack, message, channel)
        context.log.info(f"The following message is posted to Slack {channel} \n", message)
    except Exception as e:
        error_msg = f"Error while posting model summary: {e}"
        context.log.error(error_msg)
    
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