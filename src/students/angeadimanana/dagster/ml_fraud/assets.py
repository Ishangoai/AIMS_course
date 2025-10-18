
import base64
import io

import dagster as dg
import dagster_slack
import matplotlib.pyplot as plt
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

from ..ml.resources import mlflow_resource
from .resources import FraudDataConfig, tuning_config


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
    y: pd.Series = raw_fraud_data['Class']

    context.log.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
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

    context.log.info(f"Starting hyperparameter tuning for: {tuning_config.hyperparameter_to_tune}")
    context.log.info(f"Training on {len(X_train)} samples with {tuning_config.cv_folds}-fold CV")

    # Define parameter grid based on config
    param_grid = {}
    if tuning_config.hyperparameter_to_tune == "n_estimators":
        param_grid = {"n_estimators": tuning_config.n_estimators_options}
    elif tuning_config.hyperparameter_to_tune == "max_depth":
        param_grid = {"max_depth": tuning_config.max_depth_options}
    elif tuning_config.hyperparameter_to_tune == "min_samples_split":
        param_grid = {"min_samples_split": tuning_config.min_samples_split_options}
    else:
        raise ValueError(f"Unknown hyperparameter: {tuning_config.hyperparameter_to_tune}")

    context.log.info(f"Parameter grid: {param_grid}")

    # Log tuning configuration to MLflow
    mlflow_client.log_param("hyperparameter_tuned", tuning_config.hyperparameter_to_tune)
    mlflow_client.log_param("cv_folds", tuning_config.cv_folds)
    mlflow_client.log_param("param_grid", str(param_grid))

    # Initialize RandomForest
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)

    # Perform GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tuning_config.cv_folds,
        scoring='accuracy',
        verbose=2,
        n_jobs=-1,
        return_train_score=True
    )

    context.log.info("Running GridSearchCV...")
    grid_search.fit(X_train, y_train)
    context.log.info("GridSearchCV completed!")

    # Log all CV results
    cv_results = grid_search.cv_results_
    context.log.info(f"Logging {len(cv_results['params'])} trials")

    for i in range(len(cv_results['params'])):
        param_values = cv_results['params'][i]
        context.log.info(
            f"Trial {i}: {param_values} -> "
            f"CV accuracy={cv_results['mean_test_score'][i]:.4f} "
            f"(±{cv_results['std_test_score'][i]:.4f})"
        )

    # Create a summary of all trials
    all_trials_summary = []
    for i in range(len(cv_results['params'])):
        trial_info = {
            "trial": i,
            **cv_results['params'][i],
            "mean_cv_accuracy": float(cv_results['mean_test_score'][i]),
            "std_cv_accuracy": float(cv_results['std_test_score'][i])
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

    context.log.info("=" * 60)
    context.log.info("BEST HYPERPARAMETERS FOUND:")
    context.log.info(f"  Parameters: {best_params}")
    context.log.info(f"  CV Accuracy: {best_score:.4f}")
    context.log.info("=" * 60)

    mlflow_client.log_params({f"best_{k}": v for k, v in best_params.items()})
    mlflow_client.log_metric("best_cv_accuracy", best_score)

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
    description="Evaluate model on test set and generate confusion matrix",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_evaluate"
)
def evaluate_fraud_model(
    context: dg.AssetExecutionContext,
    tune_random_forest: dict
) -> dg.MaterializeResult:
    """
    Evaluates the trained model on test data and generates confusion matrix.
    Logs metrics and confusion matrix plot to MLflow as an artifact.
    """
    mlflow_client = context.resources.mlflow_tracking

    context.log.info("Evaluating model on test set")

    model = tune_random_forest["best_estimator"]
    X_test = tune_random_forest["X_test"]
    y_test = tune_random_forest["y_test"]

    context.log.info(f"Test set size: {len(X_test)} samples")

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics - CONVERT TO PYTHON TYPES
    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    context.log.info("=" * 60)
    context.log.info("TEST SET EVALUATION METRICS:")
    context.log.info(f"  Accuracy:  {accuracy:.4f}")
    context.log.info(f"  Precision: {precision:.4f}")
    context.log.info(f"  Recall:    {recall:.4f}")
    context.log.info(f"  F1-Score:  {f1:.4f}")
    context.log.info("=" * 60)

    # Log metrics to MLflow
    metrics = {
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1_score": f1
    }
    mlflow_client.log_metrics(metrics)

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    context.log.info("Confusion Matrix:")
    context.log.info(f"  True Negatives:  {cm[0, 0]}")
    context.log.info(f"  False Positives: {cm[0, 1]}")
    context.log.info(f"  False Negatives: {cm[1, 0]}")
    context.log.info(f"  True Positives:  {cm[1, 1]}")

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

    # plt.title('Confusion Matrix - Fraud Detection Model\n', fontsize=16, fontweight='bold')
    # plt.ylabel('True Label', fontsize=12)
    # plt.xlabel('Predicted Label', fontsize=12)

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
    report = classification_report(y_test, y_pred, target_names=['Normal', 'Fraud'])
    context.log.info(f"Classification Report:\n{report}")

    # Log classification report as text artifact
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 60 + "\n")
        f.write(report)
        f.write("\n" + "=" * 60 + "\n")
        f.write("\nConfusion Matrix:\n")
        f.write(f"TN: {cm[0, 0]}, FP: {cm[0, 1]}\n")
        f.write(f"FN: {cm[1, 0]}, TP: {cm[1, 1]}\n")
        temp_report_path = f.name
    mlflow_client.log_artifact(temp_report_path, "evaluation_reports")

    return dg.MaterializeResult(
        value={
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "best_params": tune_random_forest["best_params"],
            "status": "evaluated_successfully",
            "accuracy_value": accuracy,
            "f1_score": f1
        },
        metadata={
            "test_accuracy": dg.MetadataValue.float(accuracy),
            "test_precision": dg.MetadataValue.float(precision),
            "test_recall": dg.MetadataValue.float(recall),
            "test_f1_score": dg.MetadataValue.float(f1),
            "confusion_matrix": dg.MetadataValue.md(
                f"![Confusion Matrix](data:image/png;base64,{img_base64})"
            ),
            "true_negatives": dg.MetadataValue.int(int(cm[0, 0])),
            "false_positives": dg.MetadataValue.int(int(cm[0, 1])),
            "false_negatives": dg.MetadataValue.int(int(cm[1, 0])),
            "true_positives": dg.MetadataValue.int(int(cm[1, 1])),
            "description": dg.MetadataValue.text(
                "Model evaluation results with confusion matrix on test set"
            )
        }
    )


@dg.asset(
    description="Send ML prediction, performance on the test on slack channel",
    resource_defs={"newslack": dagster_slack.SlackResource(token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN"))},
    compute_kind="python",
    group_name="ml_fraud_message"
)
def send_message(
    context: dg.AssetExecutionContext,
    evaluate_fraud_model: dict,
) -> dg.MaterializeResult:
    """
    Send the performance on the test to slack channel
    """
    # metrics = evaluate_fraud_model["metrics"]
    accuracy = evaluate_fraud_model["accuracy_value"]
    score_f1 = evaluate_fraud_model["f1_score"]
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
