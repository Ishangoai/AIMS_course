# pyright: ignore-all
import base64
import io

import dagster as dg
import dagster_slack
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn as ms
import pandas as pd

# import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split

from ..ml.resources import mlflow_client, mlflow_resource
from .resources import FraudDataConfig, promotion_config, tuning_config


@dg.asset(
    description="Load raw fraud detection data from CSV URL",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_ingest"
)
def raw_fraud_data(
    context: dg.AssetExecutionContext,
    config: FraudDataConfig
) -> dg.MaterializeResult:
    """
    Downloads credit card fraud dataset from URL and logs to MLflow.
    Returns a pandas DataFrame.
    """
    mlflow_client = context.resources.mlflow_tracking

    context.log.info(f"Loading fraud detection data from: {config.data_url}")

    # Loading the CSV data
    df = pd.read_csv(config.data_url)

    context.log.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")

    # Log dataset info to MLflow and convert all to python types
    mlflow_client.log_param("data_source", config.data_url)
    mlflow_client.log_param("total_rows", len(df))
    mlflow_client.log_param("total_columns", len(df.columns))
    mlflow_client.log_param("fraud_cases", int(df['Class'].sum()))
    mlflow_client.log_param("normal_cases", int((df['Class'] == 0).sum()))

    fraud_percentage = float((df['Class'].sum() / len(df)) * 100)
    mlflow_client.log_metric("fraud_percentage", fraud_percentage)

    context.log.info(f"Fraud cases: {int(df['Class'].sum())}, Normal cases: {int((df['Class'] == 0).sum())}")
    context.log.info(f"Fraud percentage: {fraud_percentage:.2f}%")

    # Log dataset to MLflow
    dataset = mlflow_client.data.from_pandas(df, name="creditcard_fraud_raw_data")
    mlflow_client.log_input(dataset=dataset, context="training")

    columns = [dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]

    return dg.MaterializeResult(
        value=df,
        metadata={
            "dagster/row_count": len(df),
            "dagster/column_schema": dg.TableSchema(columns=columns),
            "preview": dg.MetadataValue.md(df.head(10).to_markdown() or ""),
            "fraud_count": dg.MetadataValue.int(int(df['Class'].sum())),
            "fraud_percentage": dg.MetadataValue.float(float(fraud_percentage)),
            "description": dg.MetadataValue.text(
                "Raw credit card fraud detection dataset"
            )
        }
    )


@dg.asset(
    description="Split data into 80% training and 20% test",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_transform"
)
def split_train_test(
    context: dg.AssetExecutionContext,
    config: FraudDataConfig,
    raw_fraud_data: pd.DataFrame
) -> dict:
    """
    Splits the fraud dataset into training (80%) and test (20%) sets.
    Uses stratified split to maintain fraud/normal ratio in both sets.
    """
    mlflow_client = context.resources.mlflow_tracking

    context.log.info("Splitting data into train and test sets")

    # Separate features and target
    X: pd.DataFrame = raw_fraud_data.drop('Class', axis=1)
    y: pd.Series = raw_fraud_data['Class']  # pyright: ignore

    context.log.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(  # pyright: ignore
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y  # Maintain fraud/normal ratio in both sets
    )

    context.log.info(f"Training set: {len(X_train)} samples")
    context.log.info(f"Test set: {len(X_test)} samples")

    # type of X_train, X_test, y_train, y_test
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    # Calculate fraud counts in each
    train_fraud = int(y_train.sum())
    test_fraud = int(y_test.sum())
    train_normal = int((y_train == 0).sum())
    test_normal = int((y_test == 0).sum())

    train_fraud_pct = (train_fraud / len(y_train)) * 100
    test_fraud_pct = (test_fraud / len(y_test)) * 100

    context.log.info(f"Training set - Fraud: {train_fraud} ({train_fraud_pct:.2f}%), Normal: {train_normal}")
    context.log.info(f"Test set - Fraud: {test_fraud} ({test_fraud_pct:.2f}%), Normal: {test_normal}")

    # Log split info to MLflow
    mlflow_client.log_param("train_size", len(X_train))
    mlflow_client.log_param("test_size", len(X_test))
    mlflow_client.log_param("test_split_ratio", config.test_size)
    mlflow_client.log_param("random_state", config.random_state)
    mlflow_client.log_param("stratified_split", True)

    mlflow_client.log_metric("train_fraud_count", train_fraud)
    mlflow_client.log_metric("test_fraud_count", test_fraud)
    mlflow_client.log_metric("train_fraud_percentage", train_fraud_pct)
    mlflow_client.log_metric("test_fraud_percentage", test_fraud_pct)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }


@dg.asset(
    description="Tune RandomForest hyperparameters using GridSearchCV with 3-fold cross-validation",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_model"
)
def tune_random_forest(
    context: dg.AssetExecutionContext,
    split_train_test: dict
) -> dict:
    """
    Performs hyperparameter tuning for RandomForest using GridSearchCV.
    Logs all trials to MLflow.
    """
    mlflow_client = context.resources.mlflow_tracking

    X_train = split_train_test["X_train"]
    y_train = split_train_test["y_train"]

    context.log.info("Starting hyperparameter tuning for:")
    context.log.info(f"Training on {len(X_train)} samples with {tuning_config.cv_folds}-fold CV")

    # Define parameter grid - TUNE ALL PARAMETERS TOGETHER
    param_grid = {
        "n_estimators": tuning_config.n_estimators_options,
        "max_depth": tuning_config.max_depth_options,
        # "min_samples_split": tuning_config.min_samples_split_options
    }

    context.log.info(f"Parameter grid: {param_grid}")

    # Calculate total combinations
    total_combinations = (
        len(tuning_config.n_estimators_options) *
        len(tuning_config.max_depth_options)
        )

    total_fits = total_combinations * tuning_config.cv_folds

    context.log.info(f"Total combinations: {total_combinations}")
    context.log.info(f"Total fits (with CV): {total_fits}")
    context.log.info("⚠️  This may take a few minutes...")

    # Log tuning configuration to MLflow
    mlflow_client.log_param("hyperparameters_tuned", "n_estimators, max_depth")
    mlflow_client.log_param("cv_folds", tuning_config.cv_folds)
    mlflow_client.log_param("total_combinations", total_combinations)
    mlflow_client.log_param("total_cv_fits", total_fits)
    mlflow_client.log_param("param_grid", str(param_grid))

    # Initialize RandomForest with class_weight='balanced' for imbalanced data
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # IMPORTANT: Handle class imbalance
    )

    context.log.info("✅ Using class_weight='balanced' to handle imbalanced data")

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tuning_config.cv_folds,
        scoring='f1',  # Use F1 instead of accuracy for imbalanced data
        verbose=2,
        n_jobs=-1,
        return_train_score=True
    )

    context.log.info("Running GridSearchCV...")
    context.log.info("Scoring metric: F1 (better for imbalanced data)")
    grid_search.fit(X_train, y_train)
    context.log.info("GridSearchCV completed!")

    # Log all CV results
    cv_results = grid_search.cv_results_
    context.log.info(f"Total trials executed: {len(cv_results['params'])}")

    # Log top 10 results
    context.log.info("\n" + "=" * 80)
    context.log.info("TOP 10 HYPERPARAMETER COMBINATIONS")
    context.log.info("=" * 80)

    # Sort by mean test score
    sorted_indices = sorted(
        range(len(cv_results['mean_test_score'])),
        key=lambda i: cv_results['mean_test_score'][i],
        reverse=True
    )

    for rank, i in enumerate(sorted_indices[:10], 1):
        param_values = cv_results['params'][i]
        score = cv_results['mean_test_score'][i]
        std = cv_results['std_test_score'][i]
        context.log.info(
            f"Rank {rank:2d}: F1={score:.4f} (±{std:.4f}) | "
            f"n_est={param_values['n_estimators']:3d}, "
            f"depth={param_values['max_depth']:2d}, "
            # f"split={param_values['min_samples_split']:2d}"
        )

    # Create a summary of all trials
    all_trials_summary = []
    for i in range(len(cv_results['params'])):
        trial_info = {
            "trial": i,
            **cv_results['params'][i],
            "mean_cv_f1": float(cv_results['mean_test_score'][i]),
            "std_cv_f1": float(cv_results['std_test_score'][i]),
            "mean_train_f1": float(cv_results['mean_train_score'][i])
        }
        all_trials_summary.append(trial_info)

    # Log summary to MLflow as artifact
    import json
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(all_trials_summary, f, indent=2)
        temp_path = f.name
    mlflow_client.log_artifact(temp_path, "cv_results")

    context.log.info(f"All {len(all_trials_summary)} trials logged to MLflow")

    # Log best parameters and score
    best_params = grid_search.best_params_
    best_score = float(grid_search.best_score_)

    context.log.info("\n" + "=" * 80)
    context.log.info("BEST HYPERPARAMETERS FOUND")
    context.log.info("=" * 80)
    context.log.info(f"  n_estimators: {best_params['n_estimators']}")
    context.log.info(f"  max_depth: {best_params['max_depth']}")
    # context.log.info(f"  min_samples_split: {best_params['min_samples_split']}")
    context.log.info(f"  CV F1 Score: {best_score:.4f}")
    context.log.info("=" * 80)

    mlflow_client.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow_client.log_metric("best_cv_f1_score", best_score)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "best_estimator": grid_search.best_estimator_,
        "X_train": X_train,
        "X_test": split_train_test["X_test"],
        "y_train": y_train,
        "y_test": split_train_test["y_test"]
    }


@dg.asset(
    description="Train final model with best hyperparameters",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_model"
)
def train_fraud_model(
    context: dg.AssetExecutionContext,
    tune_random_forest: dict
) -> dict:
    """
    Trains the final RandomForest model using the best hyperparameters.
    Model is already trained by GridSearchCV.
    """
    mlflow_client = context.resources.mlflow_tracking

    context.log.info("Final model training with best parameters")

    best_model = tune_random_forest["best_estimator"]
    X_train = tune_random_forest["X_train"]
    y_train = tune_random_forest["y_train"]
    X_test = tune_random_forest["X_test"]
    y_test = tune_random_forest["y_test"]
    best_params = tune_random_forest["best_params"]

    context.log.info(f"Model trained on {len(X_train)} samples")

    # Log final model parameters
    train_params_log = {
        "model_type": "RandomForestClassifier",
        **{f"final_{k}": v for k, v in best_params.items()},
        "final_train_samples": len(X_train),
        "final_test_samples": len(X_test)
    }
    mlflow_client.log_params(train_params_log)
    context.log.info(f"Logged final training parameters to MLflow: {train_params_log}")

    return {
        "model": best_model,
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train
    }


@dg.asset(
    description="Evaluate model on test set, generate confusion matrix, and register in MLflow",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_evaluate"
)
def test_fraud_model(
    context: dg.AssetExecutionContext,
    train_fraud_model: dict
) -> dg.MaterializeResult:
    """
    Evaluates the trained model on test data and generates confusion matrix.
    Logs metrics and confusion matrix plot to MLflow as an artifact.
    Registers model in MLflow Model Registry.
    """
    mlflow_client = context.resources.mlflow_tracking

    context.log.info("Starting final model evaluation")

    model = train_fraud_model["model"]
    X_test = train_fraud_model["X_test"]
    y_test = train_fraud_model["y_test"]

    if len(X_test) == 0:
        context.log.warning("Test set is empty. Skipping evaluation and model logging.")
        return dg.MaterializeResult(
            value={
                "status": "skipped_evaluation",
                "reason": "Test set empty, evaluation skipped.",
                "eval_metrics": {"test_accuracy": float('nan'), "test_precision": float('nan')},
                "model_version_info": None
            },
            metadata={
                "status": "skipped_evaluation",
                "reason": dg.MetadataValue.text("Test set was empty, no evaluation performed.")
            }
        )

    context.log.info(f"Test set size: {len(X_test)} samples")

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics - CONVERT TO PYTHON TYPES
    accuracy = float(accuracy_score(y_test, predictions))
    precision = float(precision_score(y_test, predictions, zero_division=0))  # pyright: ignore
    recall = float(recall_score(y_test, predictions, zero_division=0))  # pyright: ignore
    f1 = float(f1_score(y_test, predictions, zero_division=0))  # pyright: ignore

    context.log.info(
        "Final Model Evaluation Metrics on Test Set:"
        "Accuracy={accuracy:.4f},"
        "Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}"
        )

    eval_metrics = {
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1_score": f1
    }
    mlflow_client.log_metrics(eval_metrics)
    context.log.info(f"Logged final evaluation metrics to MLflow: {eval_metrics}")

    # Generate confusion matrix
    cm = confusion_matrix(y_test, predictions)

    context.log.info(f"Confusion Matrix: TN={cm[0, 0]}, FP={cm[0, 1]}, FN={cm[1, 0]}, TP={cm[1, 1]}")

    # Create confusion matrix plot
    plt.figure(figsize=(10, 8))

    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix - Fraud Detection Model\n', fontsize=16, fontweight='bold')
    plt.colorbar()
    tick_marks = [0, 1]
    plt.xticks(tick_marks, ['Normal (0)', 'Fraud (1)'])
    plt.yticks(tick_marks, ['Normal (0)', 'Fraud (1)'])
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)

    # Annoter les cases
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='black')

    # Add metrics text to plot
    metrics_text = f"Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
    plt.figtext(0.5, 0.02, metrics_text, ha='center', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)

    # Log confusion matrix as artifact to MLflow
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        f.write(buf.getvalue())
        temp_path = f.name
    mlflow_client.log_artifact(temp_path, "evaluation_plots")
    context.log.info("Confusion matrix plot logged to MLflow")

    plt.close()

    # Encode image for Dagster metadata
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()

    # Generate classification report
    report = classification_report(y_test, predictions, target_names=['Normal', 'Fraud'])
    context.log.info(f"Classification Report:\n{report}")

    # Log classification report as text artifact
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(report)  # pyright: ignore
        f.write("\n" + "=" * 60 + "\n")
        f.write("\nConfusion Matrix:\n")
        f.write(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}\n")
        f.write(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}\n")
        temp_report_path = f.name
    mlflow_client.log_artifact(temp_report_path, "evaluation_reports")

    # Register model in MLflow Model Registry
    registered_model_name = "fraud-detection-rf"
    model_version_info = None

    with mlflow.start_run(nested=True) as current_run:
        context.log.info(f"Starting nested MLflow run for model logging: {current_run.info.run_id}")

        log_model_info = ms.log_model(
            sk_model=model,
            artifact_path="fraud_detection_model",
            input_example=X_test[:min(5, len(X_test))],
            registered_model_name=registered_model_name
        )
        context.log.info(f"Model logged to MLflow Run ID: {current_run.info.run_id}")
        context.log.info(f"Logged model artifact URI: {log_model_info.model_uri}")

        # Get model version info from registry
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
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1_score": f1,
        "model_version_info": model_version_info,
        "status": "evaluated_successfully"
    }

    return dg.MaterializeResult(
        value=output_value_for_downstream,
        metadata={
            "test_accuracy": dg.MetadataValue.float(accuracy),
            "test_precision": dg.MetadataValue.float(precision),
            "test_recall": dg.MetadataValue.float(recall),
            "test_f1_score": dg.MetadataValue.float(f1),
            "confusion_matrix": dg.MetadataValue.md(
                f"![Confusion Matrix](data:image/png;base64,{img_base64})"
            ),
            "model_name": dg.MetadataValue.text(model_version_info["name"]),
            "model_version": dg.MetadataValue.text(str(model_version_info["version"])),
            "mlflow_run_id": dg.MetadataValue.text(current_run.info.run_id),
            "mlflow_model_uri": dg.MetadataValue.text(model_version_info["model_uri"]),
            "mlflow_stage": dg.MetadataValue.text(model_version_info["stage"])
        }
    )


@dg.asset(
    description="Send ML prediction, performance on the test on slack channel",
    resource_defs={"newslack": dagster_slack.SlackResource(token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN"))},
    compute_kind="python",
    group_name="ml_fraud_evaluate"
)
def send_message(
    context: dg.AssetExecutionContext,
    test_fraud_model: dict,
) -> dg.MaterializeResult:
    """
    Send the performance on the test to slack channel
    """
    # metrics = evaluate_fraud_model["metrics"]
    accuracy = test_fraud_model["test_accuracy"]
    score_f1 = test_fraud_model["test_f1_score"]
    emoji_success = ":rocket:"

    slack: dagster_slack.SlackResource = context.resources.newslack
    slack.get_client().chat_postMessage(
       channel='aims_course_october2025',
       text=(
           f"{emoji_success} Model evaluated: Fraud Detection\n"
           f"👤 Team Ange and James\n"
           f"Accuracy:{accuracy:.4f}\n"
           f"f1_score:{score_f1:.4f}\n"
       )
    )

    context.log.info(f"Slack message sent with accuracy: {accuracy:.4f}")
    context.log.info(f"Slack message sent with f1_score: {score_f1:.4f}")

    return dg.MaterializeResult(
       value={
        "acc": accuracy,
        "f1": score_f1
       },
       metadata={
           "preview": dg.MetadataValue.md(f"Accuracy: {accuracy}"),
           "test_accuracy": dg.MetadataValue.float(accuracy)
        })


@dg.asset(
    description="Promotes the model to Staging if it meets performance criteria",
    resource_defs={"mlflow_tracking": mlflow_resource, "mlflow_client": mlflow_client},
    compute_kind="python",
    group_name="ml_fraud_promote"
)
def promote_fraud_model_to_staging(
    context: dg.AssetExecutionContext,
    test_fraud_model: dict
) -> dg.MaterializeResult:
    """
    Promotes the fraud detection model to Staging if it meets criteria.
    """
    mlflow_client = context.resources.mlflow_client
    context.log.info("Starting model promotion to Staging.")

    # If the evaluation step was skipped, we also skip promotion
    if test_fraud_model.get("status") == "skipped_evaluation":
        context.log.info("Evaluation was skipped in the previous step. Skipping staging promotion.")
        return dg.MaterializeResult(
            value={"status": "skipped_promotion", "reason": "evaluation_skipped_upstream"},
            metadata={"status": "skipped_promotion_due_to_upstream_skip"}
        )

    # Extract metrics and model version info from evaluation result
    eval_metrics = test_fraud_model.get("eval_metrics", {})
    model_version_info = test_fraud_model.get("model_version_info")

    # If no model version info was returned, skip promotion
    if not model_version_info:
        context.log.warning("No valid model version info found from evaluation. Skipping staging promotion.")
        return dg.MaterializeResult(
            value={"status": "skipped_promotion", "reason": "no_model_version_info_from_evaluation"},
            metadata={"status": "skipped_promotion_no_model_info"}
        )

    # Get performance metrics
    current_accuracy = eval_metrics.get("test_accuracy", float('inf'))
    current_precision = eval_metrics.get("test_precision", float('inf'))
    current_recall = eval_metrics.get("test_recall", float('inf'))

    STAGING_ACCURACY_THRESHOLD = promotion_config.staging_accuracy_threshold
    STAGING_PRECISION_THRESHOLD = promotion_config.staging_precision_threshold
    STAGING_RECALL_THRESHOLD = promotion_config.staging_recall_threshold

    # Log the evaluation metrics and threshold criteria
    context.log.info(f"Model evaluated with Accuracy: {current_accuracy:.4f},"
                    "Precision: {current_precision:.4f}, Recall: {current_recall:.4f}")
    context.log.info(f"Staging promotion thresholds: Accuracy > {STAGING_ACCURACY_THRESHOLD},"
                    "Precision > {STAGING_PRECISION_THRESHOLD}, Recall > {STAGING_RECALL_THRESHOLD}")

    # Check if model meets promotion criteria
    if (current_accuracy >= STAGING_ACCURACY_THRESHOLD and
        current_precision >= STAGING_PRECISION_THRESHOLD and
        current_recall >= STAGING_RECALL_THRESHOLD):
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
                    "recall_at_promotion": dg.MetadataValue.float(current_recall)
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
                "recall": dg.MetadataValue.float(current_recall)
            }
        )


@dg.asset(
    description="Promotes the best model from Staging to Production",
    resource_defs={"mlflow_tracking": mlflow_resource, "mlflow_client": mlflow_client},
    compute_kind="python",
    group_name="ml_fraud_promote"
)
def promote_fraud_model_to_production(
    context: dg.AssetExecutionContext,
    promote_fraud_model_to_staging: dict
) -> dg.MaterializeResult:
    """
    Promotes the fraud detection model from Staging to Production.
    """
    mlflow_client = context.resources.mlflow_client
    context.log.info("Starting model promotion to Production.")

    # Check if a model was promoted to Staging previously
    if promote_fraud_model_to_staging.get("status") != "promoted_to_staging":
        context.log.info("No model was promoted to Staging in the previous step. Skipping production promotion.")
        return dg.MaterializeResult(
            value={"status": "skipped_production_promotion", "reason": "no_model_in_staging_from_previous_step"},
            metadata={"status": "skipped_production_promotion"}
        )

    # Get the model name from the previous promotion step
    model_name = promote_fraud_model_to_staging.get("model_name", "fraud-detection-rf")

    # Simulate manual approval
    manual_approval_granted = True

    # Proceed with promotion only if manual approval is granted
    if manual_approval_granted:
        try:
            # Find the latest model version in Staging for the given model_name
            latest_staging_version = None
            for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
                if mv.current_stage == "Staging":
                    if latest_staging_version is None or mv.version > latest_staging_version.version:
                        latest_staging_version = mv

            # If no model is found in staging, log a warning and skip promotion
            if not latest_staging_version:
                context.log.warning(f"No model found in Staging stage for '{model_name}'. Skipping prod promotion.")
                return dg.MaterializeResult(
                    value={"status": "skipped_production_promotion", "reason": "no_staging_model_found_for_prod"},
                    metadata={"status": "skipped_production_promotion_no_staging_model"}
                )

            # Extract the model name and version to promote
            prod_model_name = latest_staging_version.name
            prod_model_version = latest_staging_version.version

            # Archive all existing models in Production
            for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
                if mv.current_stage == "Production":
                    context.log.info(f"Archiving previous Production model '{mv.name}' (version {mv.version})")
                    mlflow_client.transition_model_version_stage(
                        name=mv.name,
                        version=mv.version,
                        stage="Archived"
                    )

            # Promote the new version to Production
            context.log.info(f"Promoting model '{prod_model_name}' (version {prod_model_version}) to Production")
            mlflow_client.transition_model_version_stage(
                name=prod_model_name,
                version=prod_model_version,
                stage="Production"
            )

            # Return success with metadata about the promoted model
            context.log.info(f"Model '{prod_model_name}' (version {prod_model_version}) promoted to Production.")
            return dg.MaterializeResult(
                value={
                    "status": "promoted_to_production",
                    "model_name": prod_model_name,
                    "model_version": prod_model_version,
                    "previous_metrics": promote_fraud_model_to_staging.get("metrics")
                },
                metadata={
                    "status": "promoted_to_production",
                    "model_name": dg.MetadataValue.text(prod_model_name),
                    "model_version": dg.MetadataValue.text(str(prod_model_version))
                }
            )
        except Exception as e:
            # Catch and log any error that occurs during the promotion process
            context.log.error(f"Error promoting model to Production: {e}")
            return dg.MaterializeResult(
                value={"status": "failed_production_promotion", "error": str(e)},
                metadata={"status": "failed_production_promotion", "error_message": dg.MetadataValue.text(str(e))}
            )

    # If manual approval was denied, skip promotion and return reason
    else:
        context.log.info("Manual approval not granted. Skipping production promotion")
        return dg.MaterializeResult(
            value={"status": "not_promoted_to_production", "reason": "manual_approval_denied"},
            metadata={"status": "not_promoted_to_production"}
        )
