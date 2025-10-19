import os

import dagster as dg
import dagster_slack
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn as ms
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# Import mlflow resources from ml.resources (as instructed)
from ..ml.resources import mlflow_fraud_resource


@dg.asset(
    description="Fetches raw credit card fraud detection dataset from GitHub URL.",
    compute_kind="python",
    group_name="ml_fraud_ingest"
)
def raw_fraud_data(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:
    """
    Loads the credit card fraud dataset from GitHub URL.
    """
    # Dataset URL
    dataset_url = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"

    context.log.info(f"Loading fraud detection dataset from: {dataset_url}")

    # Load the dataset from URL
    df = pd.read_csv(dataset_url)

    # Log dataset info
    context.log.info(f"Dataset loaded successfully. Shape: {df.shape}")
    context.log.info(f"Fraud cases: {df['Class'].sum()}, Normal cases: {(df['Class'] == 0).sum()}")

    columns = [dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]

    return dg.MaterializeResult(
        value=df,
        metadata={
            "preview": dg.MetadataValue.md(df.head(10).to_markdown() or ""),
            "dagster/row_count": len(df),
            "dagster/column_schema": dg.TableSchema(columns=columns),
            "fraud_count": dg.MetadataValue.int(int(df['Class'].sum())),
            "normal_count": dg.MetadataValue.int(int((df['Class'] == 0).sum())),
            "fraud_percentage": dg.MetadataValue.float(float(df['Class'].sum() / len(df) * 100)),
            "description": dg.MetadataValue.text(
                "Raw credit card fraud detection dataset with anonymized features from GitHub."
            )
        }
    )


@dg.asset(
    description="Transforms fraud data by removing null values and preparing for further processing.",
    compute_kind="python",
    group_name="ml_fraud_transform"
)
def transformed_fraud_data(
    context: dg.AssetExecutionContext,
    raw_fraud_data: pd.DataFrame
) -> dg.MaterializeResult:
    """
    Transforms the raw data by deleting any rows with null values.
    """
    context.log.info("Starting data transformation: removing null values.")

    df = raw_fraud_data.copy()

    # Check for missing values before removal
    missing_before = df.isnull().sum().sum()
    rows_before = len(df)
    context.log.info(f"Missing values found: {missing_before}")
    context.log.info(f"Rows before removing nulls: {rows_before}")

    # Delete any rows with null values
    df = df.dropna()

    rows_after = len(df)
    rows_removed = rows_before - rows_after
    context.log.info(f"Rows after removing nulls: {rows_after}")
    context.log.info(f"Rows removed: {rows_removed}")

    columns = [dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]

    return dg.MaterializeResult(
        value=df,
        metadata={
            "dagster/row_count": len(df),
            "dagster/column_schema": dg.TableSchema(columns=columns),
            "preview": dg.MetadataValue.md(df.head(10).to_markdown() or ""),
            "rows_removed": dg.MetadataValue.int(rows_removed),
            "missing_values_before": dg.MetadataValue.int(int(missing_before))
        }
    )


@dg.asset(
    description="Cleans fraud data by standardizing Time and Amount columns.",
    compute_kind="python",
    group_name="ml_fraud_transform"
)
def clean_fraud_data(
    context: dg.AssetExecutionContext,
    transformed_fraud_data: pd.DataFrame
) -> dg.MaterializeResult:
    """
    Cleans the fraud dataset by standardizing Time and Amount columns.
    """
    context.log.info("Starting data cleaning: standardizing Time and Amount columns.")

    df = transformed_fraud_data.copy()

    # Initialize StandardScaler
    scaler = StandardScaler()

    # Standardize Time column if it exists
    if 'Time' in df.columns:
        df['Time'] = scaler.fit_transform(df[['Time']])
        context.log.info("Standardized 'Time' column.")
    else:
        context.log.warning("'Time' column not found in dataset.")

    # Standardize Amount column if it exists
    if 'Amount' in df.columns:
        df['Amount'] = scaler.fit_transform(df[['Amount']])
        context.log.info("Standardized 'Amount' column.")
    else:
        context.log.warning("'Amount' column not found in dataset.")

    columns = [dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]

    return dg.MaterializeResult(
        value=df,
        metadata={
            "dagster/row_count": len(df),
            "dagster/column_schema": dg.TableSchema(columns=columns),
            "preview": dg.MetadataValue.md(df.head(10).to_markdown() or ""),
            "standardized_columns": dg.MetadataValue.text("Time, Amount")
        }
    )


@dg.asset(
    description="Applies Random Undersampling to balance the fraud dataset using scikit-learn.",
    compute_kind="python",
    group_name="ml_fraud_transform"
)
def undersampled_fraud_data(
    context: dg.AssetExecutionContext,
    clean_fraud_data: pd.DataFrame
) -> dg.MaterializeResult:
    """
    Applies Random Undersampling to handle class imbalance in the fraud dataset.
    Uses scikit-learn's resample utility for random undersampling.
    """
    context.log.info("Starting Random Undersampling to balance classes.")

    df = clean_fraud_data.copy()

    # Separate by class
    df_fraud = df[df['Class'] == 1]
    df_normal = df[df['Class'] == 0]

    # Log original class distribution
    fraud_count_before = len(df_fraud)
    normal_count_before = len(df_normal)
    context.log.info(f"Before undersampling - Normal: {normal_count_before}, Fraud: {fraud_count_before}")

    # Undersample majority class (normal transactions) to match minority class (fraud)
    df_normal_undersampled = resample(
        df_normal,
        replace=False,  # Sample without replacement
        n_samples=len(df_fraud),  # Match minority class size
        random_state=42
    )

    # Ensure types are DataFrames for type checker
    assert isinstance(df_fraud, pd.DataFrame)
    assert isinstance(df_normal_undersampled, pd.DataFrame)

    # Combine minority class with undersampled majority class
    df_list: list[pd.DataFrame] = [df_fraud, df_normal_undersampled]
    resampled_df = pd.concat(df_list, ignore_index=False)

    # Shuffle the dataset
    resampled_df = resampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Log new class distribution
    fraud_count_after = int((resampled_df['Class'] == 1).sum())
    normal_count_after = int((resampled_df['Class'] == 0).sum())
    context.log.info(f"After undersampling - Normal: {normal_count_after}, Fraud: {fraud_count_after}")

    columns = [dg.TableColumn(k, str(v)) for k, v in resampled_df.dtypes.to_dict().items()]

    return dg.MaterializeResult(
        value=resampled_df,
        metadata={
            "dagster/row_count": len(resampled_df),
            "dagster/column_schema": dg.TableSchema(columns=columns),
            "preview": dg.MetadataValue.md(resampled_df.head(10).to_markdown() or ""),
            "normal_cases_before": dg.MetadataValue.int(normal_count_before),
            "fraud_cases_before": dg.MetadataValue.int(fraud_count_before),
            "normal_cases_after": dg.MetadataValue.int(normal_count_after),
            "fraud_cases_after": dg.MetadataValue.int(fraud_count_after),
            "undersampling_method": dg.MetadataValue.text("sklearn.utils.resample (Random Undersampling)")
        }
    )


@dg.asset(
    description="Splits undersampled fraud data into 80% training and 20% test sets.",
    compute_kind="python",
    group_name="ml_fraud_transform"
)
def split_fraud_data(
    context: dg.AssetExecutionContext,
    undersampled_fraud_data: pd.DataFrame
) -> dg.MaterializeResult:
    """
    Splits the undersampled data into training (80%) and test (20%) sets.
    """
    context.log.info("Splitting data into train (80%) and test (20%) sets.")

    # Separate features and target
    X = undersampled_fraud_data.drop('Class', axis=1)
    y = undersampled_fraud_data['Class']

    # Split with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Cast to pandas Series to ensure type correctness
    y_train_series = pd.Series(y_train)
    y_test_series = pd.Series(y_test)

    context.log.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
    context.log.info(f"Train fraud cases: {int(y_train_series.sum())}, Test fraud cases: {int(y_test_series.sum())}")

    # Return dictionary with all splits
    split_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": list(X.columns)
    }

    return dg.MaterializeResult(
        value=split_data,
        metadata={
            "train_samples": dg.MetadataValue.int(len(X_train)),
            "test_samples": dg.MetadataValue.int(len(X_test)),
            "train_fraud_cases": dg.MetadataValue.int(int(y_train_series.sum())),
            "test_fraud_cases": dg.MetadataValue.int(int(y_test_series.sum())),
            "num_features": dg.MetadataValue.int(len(X.columns)),
            "split_ratio": dg.MetadataValue.text("80/20 train/test")
        }
    )


@dg.asset(
    description=(
    "Tunes RandomForest hyperparameters using 3-fold CV with GridSearch. "
    "Logs all trials as MLflow nested runs."
),
    compute_kind="python",
    resource_defs={"mlflow_fraud_resource": mlflow_fraud_resource},
    group_name="ml_fraud_train"
)
def tune_random_forest(
    context: dg.AssetExecutionContext,
    split_fraud_data: dict
) -> dg.MaterializeResult:
    """
    Performs hyperparameter tuning for RandomForest using GridSearchCV with 3-fold cross-validation.
    Logs all trials as nested runs in MLflow.
    """
    mlflow_client = context.resources.mlflow_fraud_resource

    context.log.info("Starting RandomForest hyperparameter tuning with 3-fold CV.")

    X_train = split_fraud_data["X_train"]
    y_train = split_fraud_data["y_train"]
    feature_names = split_fraud_data["feature_names"]

    # Define hyperparameter grid - tuning n_estimators as requested (one hyperparameter)
    param_grid = {
        'n_estimators': [50, 100, 150, 200],  # Number of trees in the forest
    }

    context.log.info(f"Hyperparameter grid: {param_grid}")

    # Initialize RandomForest with fixed parameters
    base_model = RandomForestClassifier(
        random_state=42,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        n_jobs=-1
    )

    if mlflow.active_run() is not None:
        mlflow.end_run()

    try:
        experiment = mlflow_client.get_experiment_by_name("ml_fraud_detection_1")
        if experiment is None:
            experiment = mlflow_client.create_experiment("ml_fraud_detection_1")
            experiment_id = experiment.experiment_id
        else:
            experiment_id = experiment.experiment_id
    except Exception:  # Handle cases where get_experiment_by_name might raise error if not found
        experiment_id = mlflow_client.create_experiment("ml_fraud_detection_1")

    # Start MLflow run for logging
    with mlflow.start_run(run_name="fraud_detection_tuning", experiment_id=experiment_id) as parent_run:
        context.log.info(f"Started MLflow parent run: {parent_run.info.run_id}")

        # Log base parameters
        mlflow.log_param("max_depth", 10)
        mlflow.log_param("min_samples_split", 10)
        mlflow.log_param("min_samples_leaf", 4)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("cv_folds", 3)

        # Perform GridSearchCV with 3-fold cross-validation
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,  # 3-fold cross-validation as required
            scoring='f1',
            n_jobs=-1,
            verbose=2,
            return_train_score=True
        )

        context.log.info("Starting GridSearchCV fitting...")
        grid_search.fit(X_train, y_train)
        context.log.info("GridSearchCV fitting completed.")

        # Log all trials as nested runs
        cv_results = grid_search.cv_results_

        for i in range(len(cv_results['params'])):
            trial_params = cv_results['params'][i]
            mean_test_score = cv_results['mean_test_score'][i]
            std_test_score = cv_results['std_test_score'][i]
            mean_train_score = cv_results['mean_train_score'][i]

            run_name = f"trial_{i}_n_estimators_{trial_params['n_estimators']}"

            with mlflow.start_run(run_name=run_name, nested=True):
                # Log parameters for this trial
                mlflow.log_params(trial_params)
                mlflow.log_param("max_depth", 10)
                mlflow.log_param("min_samples_split", 10)
                mlflow.log_param("min_samples_leaf", 4)
                mlflow.log_param("random_state", 42)

                # Log metrics for this trial
                mlflow.log_metric("mean_cv_f1_score", mean_test_score)
                mlflow.log_metric("std_cv_f1_score", std_test_score)
                mlflow.log_metric("mean_train_f1_score", mean_train_score)

                # Log individual fold scores
                for fold_idx in range(3):
                    fold_score = cv_results[f'split{fold_idx}_test_score'][i]
                    mlflow.log_metric(f"fold_{fold_idx}_f1_score", fold_score)

                context.log.info(
                    f"Trial {i}: n_estimators={trial_params['n_estimators']}, "
                    f"mean_cv_f1={mean_test_score:.4f}, std={std_test_score:.4f}"
                )

        # Get best parameters
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        context.log.info(f"Best parameters: {best_params}")
        context.log.info(f"Best cross-validation F1 score: {best_score:.4f}")

        # Log best parameters to parent run
        mlflow.log_params({"best_" + k: v for k, v in best_params.items()})
        mlflow.log_metric("best_cv_f1_score", best_score)

        # Train final model with best parameters on full training set
        best_model = grid_search.best_estimator_
        context.log.info("Training final model with best parameters on full training set.")

    # Return model and data for evaluation
    result_data = {
        "model": best_model,
        "best_params": best_params,
        "best_cv_score": best_score,
        "X_test": split_fraud_data["X_test"],
        "y_test": split_fraud_data["y_test"],
        "feature_names": feature_names,
        "grid_search_results": cv_results
    }

    return dg.MaterializeResult(
        value=result_data,
        metadata={
            "best_n_estimators": dg.MetadataValue.int(best_params['n_estimators']),
            "best_cv_f1_score": dg.MetadataValue.float(float(best_score)),
            "num_trials": dg.MetadataValue.int(len(cv_results['params'])),
            "cv_folds": dg.MetadataValue.int(3),
            "tuned_hyperparameter": dg.MetadataValue.text("n_estimators")
        }
    )


@dg.asset(
    description="Evaluates the tuned RandomForest model on test set and logs confusion matrix to MLflow.",
    compute_kind="python",
    resource_defs={"mlflow_fraud_resource": mlflow_fraud_resource},
    group_name="ml_fraud_evaluate"
)
def evaluate_model(
    context: dg.AssetExecutionContext,
    tune_random_forest: dict
) -> dg.MaterializeResult:
    """
    Evaluates the trained model on the test set and logs metrics and confusion matrix to MLflow.
    """
    context.log.info("Starting model evaluation on test set.")

    model = tune_random_forest["model"]
    X_test = tune_random_forest["X_test"]
    y_test = tune_random_forest["y_test"]
    feature_names = tune_random_forest["feature_names"]

    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    context.log.info(f"Test Accuracy: {accuracy:.4f}")
    context.log.info(f"Test Precision: {precision:.4f}")
    context.log.info(f"Test Recall: {recall:.4f}")
    context.log.info(f"Test F1 Score: {f1:.4f}")
    context.log.info(f"Test ROC-AUC: {roc_auc:.4f}")

    # Generate classification report
    class_report = classification_report(y_test, y_pred)
    context.log.info(f"Classification Report:\n{class_report}")

    # Create confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create a custom colormap visualization
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    # Set labels
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=['Normal (0)', 'Fraud (1)'],
           yticklabels=['Normal (0)', 'Fraud (1)'],
           title='Confusion Matrix - Fraud Detection Model',
           ylabel='True Label',
           xlabel='Predicted Label')

    # Rotate the tick labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations
    thresh = cm.max() / 2.
    total = cm.sum()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percentage = (cm[i, j] / total) * 100
            ax.text(j, i, f'{cm[i, j]}\n({percentage:.2f}%)',
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black",
                   fontsize=12)

    fig.tight_layout()

    # Save confusion matrix plot
    cm_path = "confusion_matrix_fraud_detection.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()

    context.log.info(f"Saved confusion matrix plot to: {cm_path}")

    # Log to MLflow
    registered_model_name = "fraud-detection-random-forest"

    mlflow_client = context.resources.mlflow_fraud_resource

    if mlflow.active_run() is not None:
        mlflow.end_run()

    try:
        experiment = mlflow_client.get_experiment_by_name("ml_fraud_detection_1")
        if experiment is None:
            experiment = mlflow_client.create_experiment("ml_fraud_detection_1")
        #     experiment_id = experiment.experiment_id
        # else:
        #     experiment_id = experiment.experiment_id
    except Exception:  # Handle cases where get_experiment_by_name might raise error if not found
        # experiment_id =
        mlflow_client.create_experiment("ml_fraud_detection_1")

    with mlflow.start_run(run_name="fraud_model_evaluation") as run:
        context.log.info(f"Logging model to MLflow run: {run.info.run_id}")

        # Log evaluation metrics
        mlflow.log_metric("test_accuracy", float(accuracy))
        mlflow.log_metric("test_precision", float(precision))
        mlflow.log_metric("test_recall", float(recall))
        mlflow.log_metric("test_f1_score", float(f1))
        mlflow.log_metric("test_roc_auc", float(roc_auc))

        # Log confusion matrix as artifact
        mlflow.log_artifact(cm_path)
        context.log.info("Logged confusion matrix plot to MLflow")

        # Create input example
        input_example = pd.DataFrame(
            X_test.iloc[:5].values,
            columns=feature_names
        )

        # Log the model
        log_model_info = ms.log_model(
            sk_model=model,
            artifact_path="fraud_detection_model",
            input_example=input_example,
            registered_model_name=registered_model_name
        )

        context.log.info(f"Model logged with URI: {log_model_info.model_uri}")

        model_version_info = {
            "name": registered_model_name,
            "run_id": run.info.run_id,
            "model_uri": log_model_info.model_uri
        }

    # Clean up local file
    if os.path.exists(cm_path):
        os.remove(cm_path)

    # Prepare output
    eval_metrics = {
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1_score": f1,
        "test_roc_auc": roc_auc
    }

    output_data = {
        "eval_metrics": eval_metrics,
        "model_version_info": model_version_info,
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report,
        "status": "evaluated_successfully"
    }

    return dg.MaterializeResult(
        value=output_data,
        metadata={
            "test_accuracy": dg.MetadataValue.float(float(accuracy)),
            "test_precision": dg.MetadataValue.float(float(precision)),
            "test_recall": dg.MetadataValue.float(float(recall)),
            "test_f1_score": dg.MetadataValue.float(float(f1)),
            "test_roc_auc": dg.MetadataValue.float(float(roc_auc)),
            "model_name": dg.MetadataValue.text(model_version_info["name"]),
            "confusion_matrix_logged": dg.MetadataValue.text("confusion_matrix_fraud_detection.png"),
            "mlflow_run_id": dg.MetadataValue.text(model_version_info["run_id"])
        }
    )


@dg.asset(
    description="Sends notification with model evaluation results.",
    resource_defs={"slack2": dagster_slack.SlackResource(token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN"))},
    compute_kind="python",
    group_name="ml_fraud_promote"
)
def send_slack_notification(
    context: dg.AssetExecutionContext,
    evaluate_model: dict
) -> dg.MaterializeResult:
    """
    Sends Slack notification with the evaluation results.
    """
    slack: dagster_slack.SlackResource = context.resources.slack2

    context.log.info("Preparing notification for model evaluation results.")

    if evaluate_model.get("status") != "evaluated_successfully":
        context.log.warning("Model evaluation was not successful. Skipping notification.")
        return dg.MaterializeResult(
            value={"status": "skipped_notification"},
            metadata={"reason": "evaluation_not_successful"}
        )

    eval_metrics = evaluate_model["eval_metrics"]
    model_info = evaluate_model["model_version_info"]

    # Create notification message
    message = f"""
🎯 *Fraud Detection Model Evaluation Complete!*

:new_moon_with_face: Ethar & :relieved: Ghaida
*Model : {model_info['name']}*

*Performance Metrics:*
- Accuracy: {eval_metrics['test_accuracy']:.4f}
- Precision: {eval_metrics['test_precision']:.4f}
- Recall: {eval_metrics['test_recall']:.4f}
- F1 Score: {eval_metrics['test_f1_score']:.4f}
- ROC-AUC: {eval_metrics['test_roc_auc']:.4f}

    """.strip()

    # Send Slack notification
    slack.get_client().chat_postMessage(
        channel='aims_course_october2025',
        text=message
    )

    context.log.info(f"Notification sent to Slack:\n{message}")

    return dg.MaterializeResult(
        value={"status": "notification_sent", "message": message},
        metadata={
            "notification_status": "sent_to_slack",
            "f1_score": dg.MetadataValue.float(eval_metrics['test_f1_score'])
        }
    )


@dg.asset(
    name="promote_fraud_model_to_staging",
    description="Promote fraud detection model to Staging if it meets performance thresholds.",
    resource_defs={"mlflow_fraud_resource": mlflow_fraud_resource},
    compute_kind="python",
    group_name="ml_fraud_promote",
    config_schema={
        "staging_f1_threshold": dg.Field(float, default_value=0.80),
        "staging_accuracy_threshold": dg.Field(float, default_value=0.85),
        "staging_roc_auc_threshold": dg.Field(float, default_value=0.85),
    }
)
def promote_fraud_model_to_staging(
    context: dg.AssetExecutionContext,
    evaluate_model: dict,
) -> dg.MaterializeResult:
    from mlflow.tracking import MlflowClient

    context.log.info("Starting model promotion to Staging.")

    # --- Safety check for resource type ---
    resource_obj = context.resources.mlflow_fraud_resource
    if isinstance(resource_obj, MlflowClient):
        mlflow_client = resource_obj
    else:
        context.log.warning("mlflow_fraud_resource is not an MlflowClient. Creating new MlflowClient().")
        mlflow_client = MlflowClient()

    staging_f1_threshold = context.op_config.get("staging_f1_threshold", 0.80)
    staging_accuracy_threshold = context.op_config.get("staging_accuracy_threshold", 0.85)
    staging_roc_auc_threshold = context.op_config.get("staging_roc_auc_threshold", 0.85)

    model_name = "fraud-detection-random-forest"

    try:
        # --- Fetch versions safely ---
        try:
            versions = mlflow_client.get_latest_versions(model_name, stages=["None", "Staging", "Production"])
            context.log.info(f"Fetched {len(versions)} model versions via MlflowClient.get_latest_versions().")
        except Exception as e:
            context.log.warning(f"Could not fetch model versions via get_latest_versions(): {e}")
            versions = []

        if not versions:
            raise Exception(f"No model versions found for {model_name}")

        versions_sorted = sorted(versions, key=lambda v: int(v.version), reverse=True)
        latest_version = versions_sorted[0]
        context.log.info(f"Using model version {latest_version.version}, stage: {latest_version.current_stage}")

        run = mlflow_client.get_run(str(latest_version.run_id))
        metrics = run.data.metrics or {}
        context.log.info(f"Retrieved metrics: {metrics}")

        f1_score = float(metrics.get("test_f1_score", 0.0))
        accuracy = float(metrics.get("test_accuracy", 0.0))
        roc_auc = float(metrics.get("test_roc_auc", 0.0))

        promotion_criteria_met = (
            f1_score >= staging_f1_threshold and
            accuracy >= staging_accuracy_threshold and
            roc_auc >= staging_roc_auc_threshold
        )

        if promotion_criteria_met:
            mlflow_client.transition_model_version_stage(
                name=model_name,
                version=latest_version.version,
                stage="Staging"
            )
            promote_status = "success"
            context.log.info(f"✓ Promoted model version {latest_version.version} to Staging.")
        else:
            promote_status = "failure"
            context.log.warning(f"Model version {latest_version.version} did not meet thresholds.")

        promotion_result = {
            "promote_status": promote_status,
            "model_name": model_name,
            "model_version": latest_version.version,
            "f1_score": f1_score,
            "accuracy": accuracy,
            "roc_auc": roc_auc
        }

    except Exception as e:
        context.log.error(f"Error during promotion: {str(e)}", exc_info=True)
        promotion_result = {
            "promote_status": "error",
            "model_name": model_name,
            "model_version": "N/A",
            "f1_score": 0.0,
            "accuracy": 0.0,
            "roc_auc": 0.0,
            "error": str(e)
        }

    return dg.MaterializeResult(
        value=promotion_result,
        metadata={
            "promote_status": dg.MetadataValue.text(promotion_result.get("promote_status", "unknown")),
            "model_version": dg.MetadataValue.text(str(promotion_result.get("model_version", "N/A"))),
            "f1_score": dg.MetadataValue.float(float(promotion_result.get("f1_score", 0.0))),
            "accuracy": dg.MetadataValue.float(float(promotion_result.get("accuracy", 0.0))),
            "roc_auc": dg.MetadataValue.float(float(promotion_result.get("roc_auc", 0.0))),
        }
    )


@dg.asset(
    name="promote_fraud_model_to_production",
    description="Promote Staged fraud detection model to Production and archive the old one.",
    resource_defs={"mlflow_fraud_resource": mlflow_fraud_resource},
    compute_kind="python",
    group_name="ml_fraud_promote",
)
def promote_fraud_model_to_production(
    context: dg.AssetExecutionContext,
    promote_fraud_model_to_staging: dict,
) -> dg.MaterializeResult:
    """
    Promotes the Staged model to Production and archives previous Production versions.
    Includes safety check for MlflowClient resource.
    """
    from mlflow.tracking import MlflowClient

    context.log.info("Starting Production promotion process.")

    # --- Safety check for resource type ---
    resource_obj = context.resources.mlflow_fraud_resource
    if isinstance(resource_obj, MlflowClient):
        mlflow_client = resource_obj
    else:
        context.log.warning("mlflow_fraud_resource is not an MlflowClient. Creating new MlflowClient().")
        mlflow_client = MlflowClient()

    promote_status = promote_fraud_model_to_staging.get("promote_status")
    context.log.info(f"Staging promotion status: {promote_status}")

    if promote_status != "success":
        context.log.warning("Skipping Production promotion – model not in Staging.")
        return dg.MaterializeResult(
            value={"promote_status": "skipped", "reason": "model_not_in_staging"},
            metadata={"promote_status": dg.MetadataValue.text("skipped")}
        )

    try:
        model_name = promote_fraud_model_to_staging.get("model_name")
        model_version = promote_fraud_model_to_staging.get("model_version")

        if not model_name or not model_version:
            raise ValueError(f"Missing model info: name={model_name}, version={model_version}")

        context.log.info(f"Promoting {model_name} version {model_version} to Production.")

        # --- Archive any old Production versions safely ---
        archived_versions = []
        try:
            old_versions = mlflow_client.get_latest_versions(model_name, stages=["Production"])
            for mv in old_versions:
                context.log.info(f"Archiving old Production version: {mv.version}")
                mlflow_client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Archived"
                )
                archived_versions.append(mv.version)
        except Exception as e:
            context.log.warning(f"No existing Production versions to archive: {str(e)}")

        # --- Promote the new version ---
        mlflow_client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage="Production"
        )

        context.log.info(f"✓ Successfully promoted model version {model_version} to Production.")

        promotion_result = {
            "promote_status": "success",
            "model_name": model_name,
            "model_version": model_version,
            "archived_versions": archived_versions
        }

    except Exception as e:
        context.log.error(f"Error during Production promotion: {str(e)}", exc_info=True)
        promotion_result = {
            "promote_status": "error",
            "error": str(e)
        }

    return dg.MaterializeResult(
        value=promotion_result,
        metadata={
            "promote_status": dg.MetadataValue.text(promotion_result.get("promote_status", "unknown")),
            "model_version": dg.MetadataValue.text(str(promotion_result.get("model_version", "N/A"))),
        }
    )
