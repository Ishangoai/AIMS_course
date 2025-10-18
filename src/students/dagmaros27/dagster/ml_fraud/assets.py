
import json
import os
from typing import Any, Dict, Tuple

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

# =============================================================================
# CONFIGURATION
# =============================================================================
DATASET_URL = (
    "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"
)
MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
MODEL_REGISTRY_NAME = "fraud_detection_model"


# =============================================================================
# ASSETS: DATA LOADING
# =============================================================================
@dg.asset(
    description="Loads the credit card fraud detection dataset from a remote source.",
    compute_kind="python",
    group_name="ml_fraud_ingest",
)
def raw_fraud_dataset(context: dg.AssetExecutionContext) -> dg.MaterializeResult:
    """Fetch and load the fraud detection dataset."""
    context.log.info("Loading fraud detection dataset...")
    
    df = pd.read_csv(DATASET_URL)
    
    fraud_distribution = df['Class'].value_counts().to_dict()
    fraud_rate = (fraud_distribution.get(1, 0) / len(df)) * 100
    
    context.log.info(f"Dataset shape: {df.shape}")
    context.log.info(f"Fraud distribution: {fraud_distribution}")
    context.log.info(f"Fraud rate: {fraud_rate:.2f}%")
    
    columns = [dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]
    
    return dg.MaterializeResult(
        value=df,
        metadata={
            "num_rows": dg.MetadataValue.int(len(df)),
            "num_columns": dg.MetadataValue.int(len(df.columns)),
            "fraud_count": dg.MetadataValue.int(fraud_distribution.get(1, 0)),
            "non_fraud_count": dg.MetadataValue.int(fraud_distribution.get(0, 0)),
            "fraud_rate_percent": dg.MetadataValue.float(fraud_rate),
            "preview": dg.MetadataValue.md(df.head().to_markdown() or ""),
            "dagster/column_schema": dg.TableSchema(columns=columns),
            "source": dg.MetadataValue.text(DATASET_URL),
        }
    )


# =============================================================================
# ASSETS: PREPROCESSING AND SPLITTING
# =============================================================================
@dg.asset(
    description="Preprocesses the fraud data by cleaning and scaling features.",
    compute_kind="python",
    group_name="ml_fraud_preprocess_split",
)
def preprocessed_data(
    context: dg.AssetExecutionContext,
    # FIX 1: The input type should be the data type you expect, 
    # not MaterializeResult. Assuming pandas DataFrame.
    raw_fraud_dataset: pd.DataFrame 
) -> dg.MaterializeResult:
    """Preprocess the raw data — handle missing values and scale features."""
    context.log.info("Preprocessing data...")
    
    df = raw_fraud_dataset
    
    # Separate features and target
    X = df.drop("Class", axis=1)
    y = df["Class"]
    
    # Handle missing values
    missing_before = X.isna().sum().sum() # This is type np.int64
    X = X.fillna(X.mean())
    missing_after = X.isna().sum().sum()
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    
    context.log.info(f"Preprocessed shape: {X_scaled.shape}")
    context.log.info(f"Missing values before: {missing_before}, after: {missing_after}")
    
    columns = [dg.TableColumn(k, str(v)) for k, v in X_scaled.dtypes.to_dict().items()]
    
    return dg.MaterializeResult(
        value=(X_scaled, y),
        metadata={
            # FIX 2: Cast pandas/numpy numbers to standard Python types
            "num_features": dg.MetadataValue.int(int(X_scaled.shape[1])),
            "num_samples": dg.MetadataValue.int(int(X_scaled.shape[0])),
            "missing_values_filled": dg.MetadataValue.int(int(missing_before)),
            "scaler_type": dg.MetadataValue.text("StandardScaler"),
            "feature_mean": dg.MetadataValue.float(float(X_scaled.mean().mean())),
            "feature_std": dg.MetadataValue.float(float(X_scaled.std().mean())),
            "preview": dg.MetadataValue.md(X_scaled.head().to_markdown() or ""),
            "dagster/column_schema": dg.TableSchema(columns=columns),
        }
    )


@dg.asset(
    description="Splits the preprocessed data into training and testing sets with stratification.",
    compute_kind="python",
    group_name="ml_fraud_preprocess_split",
)
def split_dataset(
    context: dg.AssetExecutionContext,
    # FIX 1: The input type should be the value yielded by the
    # upstream asset, not MaterializeResult itself.
    preprocessed_data: Tuple[pd.DataFrame, pd.Series]
) -> dg.MaterializeResult:
import json
import os
from typing import Dict, Tuple, Any

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
    """Split data into training and test sets with stratification."""
    X, y = preprocessed_data
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # These calculations produce numpy float types
    train_fraud_rate = (y_train.sum() / len(y_train)) * 100
    test_fraud_rate = (y_test.sum() / len(y_test)) * 100
    
    context.log.info(f"Training set size: {X_train.shape}")
    context.log.info(f"Test set size: {X_test.shape}")
    context.log.info(f"Training fraud rate: {train_fraud_rate:.2f}%")
    context.log.info(f"Test fraud rate: {test_fraud_rate:.2f}%")
    
    return dg.MaterializeResult(
        value={
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test
        },
        metadata={
            "train_size": dg.MetadataValue.int(int(len(X_train))),
            "test_size": dg.MetadataValue.int(int(len(X_test))),
            "train_fraud_count": dg.MetadataValue.int(int(y_train.sum())),
            "test_fraud_count": dg.MetadataValue.int(int(y_test.sum())),
            "train_fraud_rate_percent": dg.MetadataValue.float(float(train_fraud_rate)),
            "test_fraud_rate_percent": dg.MetadataValue.float(float(test_fraud_rate)),
            "test_split_ratio": dg.MetadataValue.float(0.2),
            "random_seed": dg.MetadataValue.int(42),
            "stratified": dg.MetadataValue.bool(True),
        }
    )
# =============================================================================
# ASSETS: HYPERPARAMETER TUNING
# =============================================================================
@dg.asset(
    description="Performs hyperparameter tuning for the RandomForest model using GridSearchCV with 3-fold cross-validation.",
    compute_kind="python",
    group_name="ml_fraud_tuning",
)
def tune_hyperparameters(
    context: dg.AssetExecutionContext,
    # FIX 1: The input type is the dictionary returned by the
    # upstream asset, not MaterializeResult
    split_dataset: Dict[str, Any] 
) -> dg.MaterializeResult:
    """
    Tune RandomForest hyperparameters using GridSearchCV with 3-fold cross-validation.
    Logs trials and best results to MLflow.
    """
    split_data = split_dataset
    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    
    context.log.info("Starting hyperparameter tuning with 3-fold cross-validation...")
    
    # Configure MLflow
    # Make sure MLFLOW_TRACKING_URI is accessible
    # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI) 
    mlflow.set_experiment("fraud_detection_hypertuning")
    
    # Define parameter grid (tuning n_estimators only)
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [10],
        "min_samples_split": [5],
        "random_state": [42]
    }
    
    cv_splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    with mlflow.start_run(run_name="hyperparameter_tuning") as parent_run:
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("tuned_parameter", "n_estimators")
        mlflow.log_param("param_grid", json.dumps({k: v for k, v in param_grid.items()}, default=str))
        
        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(
            rf,
            param_grid,
            cv=cv_splitter,
            scoring="f1",
            n_jobs=-1,
            verbose=1
        )
        
        # Log per-fold runs as nested runs
        fold_results = []
        for fold_idx, (train_idx, val_idx) in enumerate(cv_splitter.split(X_train, y_train)):
            with mlflow.start_run(run_name=f"fold_{fold_idx}", nested=True):
                mlflow.log_param("fold_index", fold_idx)
                mlflow.log_param("validation_size", len(val_idx))
                mlflow.log_param("training_size", len(train_idx))
                
                fold_results.append({
                    "fold": fold_idx,
                    "train_size": len(train_idx),
                    "val_size": len(val_idx)
                })
        
        context.log.info(f"Logged {len(fold_results)} fold runs as nested MLflow runs")
        
        # Execute grid search
        grid_search.fit(X_train, y_train)
        
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_ # This is np.float64
        
        context.log.info(f"Best parameters: {best_params}")
        context.log.info(f"Best CV F1 score: {best_score:.4f}")
        
        # FIX 2: Cast numpy types when logging to MLflow
        mlflow.log_param("best_n_estimators", int(best_params["n_estimators"]))
        mlflow.log_metric("best_cv_f1_score", float(best_score))
        
        # Log all trial results
        results_df = pd.DataFrame(grid_search.cv_results_)
        trial_metrics = {}
        
        for idx, row in results_df.iterrows():
            trial_num = idx
            n_est = row["param_n_estimators"] # np.int64
            mean_score = row["mean_test_score"] # np.float64
            std_score = row["std_test_score"] # np.float64
            
            # FIX 3: Cast numpy types before adding to dict for JSON metadata
            trial_metrics[f"trial_{trial_num}"] = {
                "n_estimators": int(n_est),
                "mean_f1_score": float(mean_score),
                "std_f1_score": float(std_score)
            }
            
            # (Logging to mlflow also benefits from casting, though it's often more permissive)
            mlflow.log_metric(f"trial_{trial_num}_n_estimators", int(n_est))
            mlflow.log_metric(f"trial_{trial_num}_mean_f1_score", float(mean_score))
            mlflow.log_metric(f"trial_{trial_num}_std_f1_score", float(std_score))
        
        context.log.info(f"Logged {len(trial_metrics)} trial results to MLflow")
    
    return dg.MaterializeResult(
        value=(grid_search.best_estimator_, best_params),
        metadata={
            # FIX 4: Cast numpy types before passing to Dagster metadata
            "num_trials": dg.MetadataValue.int(len(trial_metrics)),
            "best_n_estimators": dg.MetadataValue.int(int(best_params["n_estimators"])),
            "best_cv_f1_score": dg.MetadataValue.float(float(best_score)),
            "cv_folds": dg.MetadataValue.int(3),
            "tuned_parameter": dg.MetadataValue.text("n_estimators"),
            "scoring_metric": dg.MetadataValue.text("f1"),
            "mlflow_parent_run_id": dg.MetadataValue.text(parent_run.info.run_id),
            "trial_results": dg.MetadataValue.json(trial_metrics), # This is safe now
        }
    )
# ============================================================================
# ASSET: MODEL TRAINING & VALIDATION
# ============================================================================
@dg.asset(
    group_name="ml_fraud_training",
    description="Trains the final RandomForest model using the best hyperparameters and logs metrics to MLflow.",
)
def train_model(
    context: dg.AssetExecutionContext,
    # FIX 1: Corrected input type hints
    tune_hyperparameters: Tuple[RandomForestClassifier, Dict[str, Any]],
    split_dataset: Dict[str, Any]
) -> dg.MaterializeResult:
    """Train final model with best hyperparameters and log to MLflow."""
    best_model, best_params = tune_hyperparameters
    split_data = split_dataset
    
    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]
    
    context.log.info("Training final model with best hyperparameters...")
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("fraud_detection_training")
    
    with mlflow.start_run(run_name="final_model_training") as run:
        # Log hyperparameters (cast to standard types for safety)
        mlflow.log_params({k: (int(v) if isinstance(v, (int, float)) else v) for k, v in best_params.items()})
        
        # Train model
        best_model.fit(X_train, y_train)
        context.log.info("Model training completed")
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics (these are np.float64)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log metrics
        metrics = {
            "test_precision": precision,
            "test_recall": recall,
            "test_f1_score": f1,
            "test_roc_auc": roc_auc
        }
        # mlflow.log_metrics is generally fine with numpy types, 
        # but casting is safer.
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        context.log.info(f"Test metrics - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        
        # Log model artifact
        mlflow.sklearn.log_model(best_model, "final_fraud_model")
        
        run_id = run.info.run_id
    
    return dg.MaterializeResult(
        value=(best_model, metrics),
        metadata={
            # FIX 2: Cast all numpy types to standard python types
            "model_type": dg.MetadataValue.text("RandomForestClassifier"),
            "test_precision": dg.MetadataValue.float(float(precision)),
            "test_recall": dg.MetadataValue.float(float(recall)),
            "test_f1_score": dg.MetadataValue.float(float(f1)),
            "test_roc_auc": dg.MetadataValue.float(float(roc_auc)),
            "n_estimators": dg.MetadataValue.int(int(best_params["n_estimators"])),
            "max_depth": dg.MetadataValue.int(int(best_params.get("max_depth", 0))), 
            "training_samples": dg.MetadataValue.int(int(len(X_train))),
            "test_samples": dg.MetadataValue.int(int(len(X_test))),
            "mlflow_run_id": dg.MetadataValue.text(run_id),
        }
    )

# ============================================================================
# ASSET: MODEL EVALUATION & PLOTS
# ============================================================================
@dg.asset(
    group_name="ml_fraud_evaluation",
    description="Generates confusion matrix and performance plots, then logs them to MLflow.",
)
def generate_evaluation_plots(
    context: dg.AssetExecutionContext,
    # FIX 1: Corrected input type hints
    train_model: Tuple[RandomForestClassifier, Dict[str, float]],
    split_dataset: Dict[str, Any]
) -> dg.MaterializeResult:
    """Generate confusion matrix and performance plots, and log them to MLflow."""
    model, metrics = train_model
    split_data = split_dataset
    
    X_test = split_data["X_test"]
    y_test = split_data["y_test"]
    
    context.log.info("Generating evaluation plots...")
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    
    # Calculate additional metrics from confusion matrix
    tn, fp, fn, tp = cm.ravel() # These are np.int64
    # These will be np.float64
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Confusion Matrix Heatmap (Uncommented)
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=True)
    # axes[0].set_title('Confusion Matrix - Test Set')
    # axes[0].set_ylabel('True Label')
    # axes[0].set_xlabel('Predicted Label')
    # axes[0].set_xticklabels(['Non-Fraud', 'Fraud'])
    # axes[0].set_yticklabels(['Non-Fraud', 'Fraud'])

    # Performance Metrics Bar Chart
    # FIX 3: Keys in 'metrics' dict are 'test_precision', 'test_recall', etc.
    metric_names = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Specificity', 'Sensitivity']
    metric_values = [
        metrics['test_precision'],
        metrics['test_recall'],
        metrics['test_f1_score'],
        metrics['test_roc_auc'],
        specificity,
        sensitivity
    ]
    axes[1].bar(metric_names, metric_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    axes[1].set_title('Model Performance Metrics')
    axes[1].set_ylim([0, 1.05]) # Increased ylim slightly
    axes[1].set_ylabel('Score')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    plot_path = os.path.join(os.getcwd(), "evaluation_plots.png")
    plt.savefig(plot_path, dpi=100, bbox_inches='tight')
    plt.close()

    context.log.info(f"Evaluation plots saved to {plot_path}")

    # Log plots & metrics to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("fraud_detection_evaluation")
    
    with mlflow.start_run(run_name="model_evaluation") as eval_run:
        mlflow.log_artifact(plot_path, artifact_path="evaluation_plots")
        # Cast for safety
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.log_metric("specificity", float(specificity))
        mlflow.log_metric("sensitivity", float(sensitivity))
        
        eval_run_id = eval_run.info.run_id
    
    context.log.info("Evaluation plots logged to MLflow")
    
    return dg.MaterializeResult(
        value={
            "confusion_matrix": cm.tolist(),
            "metrics": metrics,
            "additional_metrics": {
                "specificity": float(specificity),
                "sensitivity": float(sensitivity),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp)
            },
            "plot_path": plot_path,
        },
        metadata={
            # FIX 2: Cast all numpy types to standard python types
            "confusion_matrix_tn": dg.MetadataValue.int(int(tn)),
            "confusion_matrix_fp": dg.MetadataValue.int(int(fp)),
            "confusion_matrix_fn": dg.MetadataValue.int(int(fn)),
            "confusion_matrix_tp": dg.MetadataValue.int(int(tp)),
            # FIX 3: Use correct keys from the metrics dictionary
            "precision": dg.MetadataValue.float(float(metrics['test_precision'])),
            "recall": dg.MetadataValue.float(float(metrics['test_recall'])),
            "f1_score": dg.MetadataValue.float(float(metrics['test_f1_score'])),
            "roc_auc": dg.MetadataValue.float(float(metrics['test_roc_auc'])),
            "specificity": dg.MetadataValue.float(float(specificity)),
            "sensitivity": dg.MetadataValue.float(float(sensitivity)),
            "plot_artifact_path": dg.MetadataValue.path(plot_path),
            "mlflow_evaluation_run_id": dg.MetadataValue.text(eval_run_id),
        }
    )

# ============================================================================
# ASSET: SLACK NOTIFICATION
# ============================================================================
@dg.asset(
    group_name="ml_fraud_notification",
    description="Sends model performance summary to Slack after evaluation."
)
def send_slack_notification(
    context: dg.AssetExecutionContext,
    generate_evaluation_plots: dg.MaterializeResult
) -> dg.MaterializeResult:
    """Send model performance metrics to Slack with emojis."""
    eval_data = generate_evaluation_plots
    metrics = eval_data["metrics"]
    additional_metrics = eval_data["additional_metrics"]
    
    context.log.info("Sending Slack notification...")
    
    # Load Slack credentials from environment
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
            context.log.info(additional_metrics)
            message = (
                f"Test: *Fraud Detection Model Training Complete!* 🚨\n\n"
                f"*Model Performance on Test Set:*\n"
                f"• Precision: {metrics['test_precision']:.4f}\n"
                f"• Recall: {metrics['test_recall']:.4f}\n"
                f"• F1-Score: {metrics['test_f1_score']:.4f}\n"
                f"• ROC-AUC: {metrics['test_roc_auc']:.4f}\n"
                f"Created by dagmaros27 & Chekwube"
            )
            
            response = client.chat_postMessage(channel=SLACK_CHANNEL, text=message)
            context.log.info(f"Slack message sent successfully (ts: {response['ts']})")
            slack_status = "sent"
            slack_message = f"Message sent to {SLACK_CHANNEL}"
            
        except SlackApiError as e:
            context.log.error(f"Slack API Error: {e.response.get('error', str(e))}")
            slack_status = "failed"
            slack_message = f"Slack API error: {e.response.get('error', str(e))}"
        except Exception as e:
            context.log.error(f"Unexpected error sending Slack notification: {e}")
            slack_status = "failed"
            slack_message = f"Slack notification failed: {e}"
    
    return dg.MaterializeResult(
        value=slack_message,
        metadata={
            "slack_status": dg.MetadataValue.text(slack_status),
            "slack_channel": dg.MetadataValue.text(SLACK_CHANNEL),
            "message": dg.MetadataValue.text(slack_message),
        }
    )


# ============================================================================
# ASSET: MODEL REGISTRY
# ============================================================================
@dg.asset(
    group_name="ml_fraud_registry",
    description="Registers the trained model into the MLflow Model Registry and promotes it to Production."
)
def register_model_to_registry(
    context: dg.AssetExecutionContext,
    train_model: dg.MaterializeResult
) -> dg.MaterializeResult:
    """Register the best model to the MLflow Model Registry."""
    model, metrics = train_model
    context.log.info("Registering model to MLflow Model Registry...")
    
    registration_status = "failed"
    model_version_info = None
    error_message = ""
    
    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        # Get latest run from training experiment
        runs = mlflow.search_runs(
            experiment_names=["fraud_detection_training"],
            order_by=["start_time DESC"],
            max_results=1
        )
        
        if runs.empty:
            context.log.warning("No MLflow runs found for 'fraud_detection_training'.")
            error_message = "No runs found in fraud_detection_training experiment"
            return dg.MaterializeResult(
                value=None,
                metadata={
                    "status": dg.MetadataValue.text("failed"),
                    "error": dg.MetadataValue.text(error_message),
                }
            )
        
        latest_run = runs.iloc[0]
        model_uri = f"runs:/{latest_run['run_id']}/final_fraud_model"
        context.log.info(f"Found model artifact at: {model_uri}")
        
        # Register model
        registered_model = mlflow.register_model(
            model_uri=model_uri,
            name=MODEL_REGISTRY_NAME
        )
        context.log.info(f"Model registered with version: {registered_model.version}")
        
        # Promote model to production
        client = mlflow.tracking.MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        client.transition_model_version_stage(
            name=MODEL_REGISTRY_NAME,
            version=registered_model.version,
            stage="Production"
        )

        registration_status = "success"
        model_version_info = {
            "name": registered_model.name,
            "version": str(registered_model.version),
            "stage": "Production",
            "model_uri": model_uri
        }

        context.log.info(f"Model promoted to Production: {MODEL_REGISTRY_NAME} v{registered_model.version}")

    except Exception as e:
        context.log.error(f"Error registering model: {e}")
        error_message = str(e)

    return dg.MaterializeResult(
        value=model_version_info,
        metadata={
            "registration_status": dg.MetadataValue.text(registration_status),
            "model_name": dg.MetadataValue.text(MODEL_REGISTRY_NAME),
            "model_version": dg.MetadataValue.text(model_version_info["version"] if model_version_info else "N/A"),
            "model_stage": dg.MetadataValue.text(model_version_info["stage"] if model_version_info else "N/A"),
            "error": dg.MetadataValue.text(error_message) if error_message else None,
        }
    )
