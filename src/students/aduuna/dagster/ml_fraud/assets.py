import os
from collections import abc

import dagster as dg
import dagster_slack
import matplotlib.pyplot as plt
import pandas as pd
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
from .resources import RandomForestConfig, fraud_data_resource


@dg.asset(
    description="Loads credit card fraud detection dataset from URL.",
    resource_defs={
        "mlflow_tracking": mlflow_resource,
        "fraud_data": fraud_data_resource
    },
    compute_kind="python",
    group_name="fraud_ingest"
)
def raw_fraud_data(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:
    """
    Loads the credit card fraud detection dataset and logs basic information to MLflow.
    """
    mlflow_client = context.resources.mlflow_tracking
    fraud_data_resource = context.resources.fraud_data

    context.log.info(f"Loading fraud detection data from: {fraud_data_resource.data_url}")

    # Load the dataset
    df: pd.DataFrame = pd.read_csv(fraud_data_resource.data_url)
    # Sample the data while preserving the fraud cases
    fraud_df = df[df['Class'] == 1]
    normal_df = df[df['Class'] == 0].sample(n=1000 - len(fraud_df), random_state=42)
    df = pd.DataFrame(pd.concat([fraud_df, normal_df]).sample(frac=1, random_state=42).reset_index(drop=True))

    context.log.info(f"Loaded dataset with shape: {df.shape}")

    # Log basic dataset information to MLflow
    mlflow_client.log_params({
        "dataset_rows": len(df),
        "dataset_columns": len(df.columns),
        "fraud_cases": int(df['Class'].sum()),
        "normal_cases": int(len(df) - df['Class'].sum()),
        "fraud_percentage": float(df['Class'].mean() * 100)
    })

    # Log the dataset to MLflow
    dataset = mlflow_client.data.from_pandas(df, name="raw_fraud_detection_data")
    mlflow_client.log_input(dataset=dataset, context="training")

    columns = [dg.TableColumn(str(col), str(dtype)) for col, dtype in df.dtypes.items()]

    return dg.MaterializeResult(
        value=df,
        metadata={
            "preview": dg.MetadataValue.md(df.head().to_markdown() or ""),
            "dagster/row_count": len(df),
            "dagster/column_schema": dg.TableSchema(columns=columns),
            "fraud_cases": dg.MetadataValue.int(int(df['Class'].sum())),
            "normal_cases": dg.MetadataValue.int(int(len(df) - df['Class'].sum())),
            "fraud_percentage": dg.MetadataValue.float(float(df['Class'].mean() * 100))
        }
    )


@dg.asset(
    description="Preprocesses fraud data and split into training and test sets.",
    resource_defs={
        "mlflow_tracking": mlflow_resource,
        "fraud_data": fraud_data_resource
    },
    compute_kind="python",
    group_name="fraud_transform"
)
def preprocessed_fraud_data(
    context: dg.AssetExecutionContext,
    raw_fraud_data: pd.DataFrame
) -> dg.MaterializeResult:
    """
    Preprocesses the fraud data and split into training and test sets.
    """
    mlflow_client = context.resources.mlflow_tracking
    fraud_data_resource = context.resources.fraud_data

    context.log.info("Starting data preprocessing and splitting")

    # Separate features and target
    X = raw_fraud_data.drop('Class', axis=1)
    y = raw_fraud_data['Class']

    # Split data into train (80%) and test (20%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=fraud_data_resource.test_size,
        random_state=fraud_data_resource.random_state,
        stratify=y  # Maintain class distribution
    )

    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)

    context.log.info(f"Training set size: {len(X_train)}")
    context.log.info(f"Test set size: {len(X_test)}")
    context.log.info(f"Training fraud rate: {y_train.mean():.4f}")
    context.log.info(f"Test fraud rate: {y_test.mean():.4f}")

    # Log split information to MLflow
    mlflow_client.log_params({
        "test_size": fraud_data_resource.test_size,
        "random_state": fraud_data_resource.random_state,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "train_fraud_rate": float(y_train.mean()),
        "test_fraud_rate": float(y_test.mean())
    })

    # Create the split data dictionary
    split_data = {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }

    return dg.MaterializeResult(
        value=split_data,
        metadata={
            "train_samples": dg.MetadataValue.int(len(X_train)),
            "test_samples": dg.MetadataValue.int(len(X_test)),
            "train_fraud_rate": dg.MetadataValue.float(float(y_train.mean())),
            "test_fraud_rate": dg.MetadataValue.float(float(y_test.mean())),
            "feature_count": dg.MetadataValue.int(len(X.columns))
        }
    )


@dg.asset(
    description="Tunes RandomForest hyperparameters using GridSearchCV with 3-fold cross-validation.",
    resource_defs={
        "mlflow_tracking": mlflow_resource,
    },
    compute_kind="python",
    group_name="fraud_model"
)
def tune_random_forest(
    context: dg.AssetExecutionContext,
    config: RandomForestConfig,
    preprocessed_fraud_data: dict
) -> dg.MaterializeResult:
    """
    Performs hyperparameter tuning using GridSearchCV and logs all trials as nested MLflow runs.
    """
    mlflow_client = context.resources.mlflow_tracking

    context.log.info("Starting RandomForest hyperparameter tuning with GridSearchCV")

    X_train = preprocessed_fraud_data["X_train"]
    y_train = preprocessed_fraud_data["y_train"]

    # Ensure the experiment exists
    try:
        experiment = mlflow_client.get_experiment_by_name("fraud_detection_analysis")
        if experiment is None:
            mlflow_client.create_experiment("fraud_detection_analysis")
    except Exception:
        mlflow_client.create_experiment("fraud_detection_analysis")

    # Create RandomForest classifier
    rf = RandomForestClassifier(
        random_state=config.random_state,
        n_jobs=-1,  # Use all available cores
        class_weight="balanced"
    )

    # Set up GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=config.param_grid,
        cv=config.cv_folds,
        scoring=config.scoring,
        n_jobs=-1,
        verbose=2
    )

    # Log hyperparameter tuning configuration
    mlflow_client.log_params({
        "cv_folds": config.cv_folds,
        "scoring_metric": config.scoring,
        "param_grid_keys": list(config.param_grid.keys())
    })

    # Perform grid search
    context.log.info("Performing grid search...")
    grid_search.fit(X_train, y_train)

    # Log best parameters and score
    mlflow_client.log_params({f"best_{k}": v for k, v in grid_search.best_params_.items()})
    mlflow_client.log_metric("best_cv_score", grid_search.best_score_)

    context.log.info(f"Best parameters: {grid_search.best_params_}")
    context.log.info(f"Best CV score: {grid_search.best_score_:.4f}")

    # Log all CV results metrics (without nested runs for simplicity)
    cv_results = grid_search.cv_results_
    for i in range(len(cv_results['params'])):
        trial_params = cv_results['params'][i]
        # Log individual trial metrics with prefixes
        mlflow_client.log_metric(f"trial_{i}_mean_test_score", cv_results['mean_test_score'][i])
        mlflow_client.log_metric(f"trial_{i}_std_test_score", cv_results['std_test_score'][i])
        context.log.info(f"Trial {i}: {trial_params}, Score: {cv_results['mean_test_score'][i]:.4f}")

    return dg.MaterializeResult(
        value={
            "best_model": grid_search.best_estimator_,
            "best_params": grid_search.best_params_,
            "best_score": grid_search.best_score_,
            "cv_results": cv_results
        },
        metadata={
            "best_params": dg.MetadataValue.json(grid_search.best_params_),
            "best_score": dg.MetadataValue.float(float(grid_search.best_score_)),
            "cv_trials": dg.MetadataValue.int(len(cv_results['params']))
        }
    )


@dg.asset(
    description="Evaluates the tuned model on test data and generates evaluation plots.",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="fraud_model"
)
def evaluate_fraud_model(
    context: dg.AssetExecutionContext,
    tune_random_forest: dict,
    preprocessed_fraud_data: dict
) -> dg.MaterializeResult:
    """
    Evaluates the best model on test data and logs confusion matrix and other metrics to MLflow.
    """
    mlflow_client = context.resources.mlflow_tracking

    context.log.info("Starting model evaluation on test data")

    # Get data and model
    best_model = tune_random_forest["best_model"]
    X_test = preprocessed_fraud_data["X_test"]
    y_test = preprocessed_fraud_data["y_test"]

    # Make predictions
    y_pred = best_model.predict(X_test)

    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred, output_dict=True)

    context.log.info(f"Test F1-score: {f1:.4f}")

    # Log test metrics directly (no manual run context needed)
    mlflow_client.log_metric("test_f1_score", f1)
    # We'll use individual metrics calculations
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow_client.log_metric("test_precision", precision)
    mlflow_client.log_metric("test_recall", recall)
    mlflow_client.log_metric("test_accuracy", accuracy)

    # Create and log confusion matrix plot
    plt.figure(figsize=(8, 6))

    # Create confusion matrix heatmap using matplotlib only
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar()

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14)

    plt.xticks([0, 1], ['Normal', 'Fraud'])
    plt.yticks([0, 1], ['Normal', 'Fraud'])
    plt.title('Confusion Matrix - Fraud Detection')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save and log the plot
    plot_path = "confusion_matrix.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    mlflow_client.log_artifact(plot_path)
    plt.close()

    # Create feature importance plot
    feature_importance = best_model.feature_importances_
    feature_names = preprocessed_fraud_data["X_test"].columns

    # Get top 20 most important features
    indices = feature_importance.argsort()[::-1][:20]

    plt.figure(figsize=(12, 8))
    plt.title("Top 20 Feature Importances")
    plt.bar(range(20), feature_importance[indices])
    plt.xticks(range(20), [feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()

    # Save and log feature importance plot
    importance_path = "feature_importance.png"
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    mlflow_client.log_artifact(importance_path)
    plt.close()

    # Register the model using the pattern from ml/assets.py
    model_name = "fraud_detection_random_forest"

    # Clean up temporary files
    if os.path.exists(plot_path):
        os.remove(plot_path)
    if os.path.exists(importance_path):
        os.remove(importance_path)

    evaluation_results = {
        "f1_score": f1,
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_rep,
        "model_name": model_name
    }

    return dg.MaterializeResult(
        value=evaluation_results,
        metadata={
            "test_f1_score": dg.MetadataValue.float(float(f1)),
            "test_precision": dg.MetadataValue.float(float(precision)),
            "test_recall": dg.MetadataValue.float(float(recall)),
            "test_accuracy": dg.MetadataValue.float(float(accuracy)),
            "model_name": dg.MetadataValue.text(model_name)
        }
    )


@dg.asset(
    description="Sends a Slack notification after the fraud model pipeline finishes.",
    resource_defs={
        "slack_notifier": dagster_slack.SlackResource(token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN")),
    },
    compute_kind="python",
    group_name="fraud_model"
)
def notify_fraud_pipeline(
    context: dg.AssetExecutionContext,
    evaluate_fraud_model: dict
) -> dg.MaterializeResult:
    """Notify the Slack channel that the pipeline completed, including the GitHub handle and an emoji."""
    slack: dagster_slack.SlackResource = context.resources.slack_notifier
    github_user = "adduna"

    # Optional: include the f1 score from the evaluation in the message
    f1 = evaluate_fraud_model.get("f1_score") if isinstance(evaluate_fraud_model, dict) else None
    f1_text = f" with F1={f1:.3f}" if isinstance(f1, (int, float)) else ""

    text = f":sunglasses: {github_user}'s fraud Dagster pipeline successfully ran{f1_text}"
    slack.get_client().chat_postMessage(
        channel='aims_course_october2025',
        text=text,
    )

    context.log.info("Slack notification sent.")

    return dg.MaterializeResult(
        value={"message": text},
        metadata={"message": dg.MetadataValue.text(text)}
    )


# Define data quality checks for fraud detection pipeline
@dg.multi_asset_check(
    specs=[
        dg.AssetCheckSpec(name="no_missing_values", asset="raw_fraud_data", blocking=False),
        dg.AssetCheckSpec(name="valid_class_labels", asset="raw_fraud_data", blocking=False),
        dg.AssetCheckSpec(name="balanced_splits", asset="preprocessed_fraud_data", blocking=False),
    ]
)
def fraud_data_quality_checks(
    raw_fraud_data: pd.DataFrame,
    preprocessed_fraud_data: dict,
) -> abc.Iterable[dg.AssetCheckResult]:
    """
    Data quality checks for fraud detection pipeline.
    """
    # Check for missing values
    missing_values = raw_fraud_data.isnull().sum().sum()
    yield dg.AssetCheckResult(
        check_name="no_missing_values",
        passed=bool(missing_values == 0),
        asset_key="raw_fraud_data",
        metadata={"missing_count": dg.MetadataValue.int(int(missing_values))}
    )

    # Check for valid class labels (should be 0 or 1)
    valid_classes = raw_fraud_data['Class'].isin([0, 1]).all()
    yield dg.AssetCheckResult(
        check_name="valid_class_labels",
        passed=bool(valid_classes),
        asset_key="raw_fraud_data",
        metadata={"unique_classes": raw_fraud_data['Class'].unique().tolist()}
    )

    # Check if train/test splits maintain similar class distributions
    train_fraud_rate = preprocessed_fraud_data["y_train"].mean()
    test_fraud_rate = preprocessed_fraud_data["y_test"].mean()
    rate_difference = abs(train_fraud_rate - test_fraud_rate)

    yield dg.AssetCheckResult(
        check_name="balanced_splits",
        passed=bool(rate_difference < 0.01),  # Allow 1% difference
        asset_key="preprocessed_fraud_data",
        metadata={
            "train_fraud_rate": float(train_fraud_rate),
            "test_fraud_rate": float(test_fraud_rate),
            "rate_difference": float(rate_difference)
        }
    )
