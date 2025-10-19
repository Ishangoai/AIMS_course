import os
import pickle
import time

import dagster as dg
import dagster_slack
import joblib
import matplotlib.pyplot as plt
import mlflow.sklearn as mlflow_sklearn

# Import mlflow for type checking/attribute access resolution
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
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.tree import plot_tree

from ..ml.resources import mlflow_resource
from .resources import FraudDataConfig


@dg.asset(
    description="Download data for fraud detection",
    compute_kind="python",
    group_name="ml_fraud_ingest",
    resource_defs={
        "fraud_data": FraudDataConfig(),
        "mlflow_tracking": mlflow_resource
    }
)
def fraud_detection(
    context: dg.AssetExecutionContext,
) -> pd.DataFrame:
    """Loads the fraud data and logs initial stats to MLflow."""

    # Load data from csv
    fraud_data_config = context.resources.fraud_data.data_source
    context.log.info(f"Processing file:\n {fraud_data_config}")

    # Convert to pandas DataFrame
    df: pd.DataFrame = pd.read_csv(fraud_data_config)
    context.log.info(f"Pandas DataFrame:\n {df}")

    mlflow_client = context.resources.mlflow_tracking

    # Log dataset info to MLflow
    mlflow_client.log_param("dataset_rows", len(df))
    mlflow_client.log_param("dataset_columns", len(df.columns))
    mlflow_client.log_param("fraud_cases", df['Class'].sum())
    mlflow_client.log_param("legitimate_cases", len(df) - df['Class'].sum())

    # Note: Assuming mlflow_client.data is correctly configured in your resource
    dataset = mlflow_client.data.from_pandas(df, name="fraud_detection_data")
    mlflow_client.log_input(dataset=dataset, context="training")

    return df


@dg.asset(
    description="Split data into train and test",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_model"
)
def train_test_split_data(
    context: dg.AssetExecutionContext,
    fraud_detection: pd.DataFrame
) -> dict:
    """Split data into 80–20 train/test sets."""
    X = fraud_detection.drop(columns=["Class"])
    y = fraud_detection["Class"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow_client = context.resources.mlflow_tracking
    mlflow_client.log_param("test_size", 0.2)
    mlflow_client.log_param("train_samples", len(X_train))
    mlflow_client.log_param("test_samples", len(X_test))

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }


@dg.asset(
    description="Perform cross validation on the data",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_model"
)
def cross_validation_results(
    context: dg.AssetExecutionContext,
    train_test_split_data: dict,
    tuned_random_forest: dict
) -> dict:
    """Perform 3-fold cross-validation."""
    X_train = train_test_split_data["X_train"]
    y_train = train_test_split_data["y_train"]
    model = tuned_random_forest["best_model"]

    scores = cross_val_score(
        model, X_train, y_train, cv=3, scoring="accuracy"
    )

    mlflow_client = context.resources.mlflow_tracking
    mlflow_client.log_metric("cv_mean_accuracy", scores.mean())
    mlflow_client.log_metric("cv_std_accuracy", scores.std())

    for i, score in enumerate(scores):
        mlflow_client.log_metric(f"cv_fold_{i + 1}_accuracy", score)

    return {"mean_accuracy": scores.mean(), "scores": scores.tolist()}


@dg.asset(
    description="Perform hyperparameter tuning",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_model"
)
def tuned_random_forest(
    context: dg.AssetExecutionContext,
    train_test_split_data: dict
) -> dict:
    """Perform hyperparameter tuning using GridSearchCV."""
    X_train = train_test_split_data["X_train"]
    y_train = train_test_split_data["y_train"]

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid,
        cv=3,
        scoring="accuracy",
        n_jobs=-1,
        error_score=np.nan  # FIX 1: Corrected
    )

    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    # Log best parameters to MLflow
    mlflow_client = context.resources.mlflow_tracking
    for param, value in grid_search.best_params_.items():
        mlflow_client.log_param(f"best_{param}", value)

    mlflow_client.log_metric("best_cv_score", grid_search.best_score_)

    # Save model to file
    os.makedirs("models", exist_ok=True)
    model_path = "models/model.pkl"
    joblib.dump(best_model, model_path)

    context.log.info(f"Best parameters: {grid_search.best_params_}")
    context.log.info(f"Best CV score: {grid_search.best_score_:.4f}")

    return {
        "best_model": best_model,
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
    }


@dg.asset(
    description="Generate and save confusion matrix visualization",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_model"
)
def confusion_matrix_plot(
    context: dg.AssetExecutionContext,
    train_test_split_data: dict,
    tuned_random_forest: dict
) -> str:
    """Generate confusion matrix and save as artifact."""
    X_test = train_test_split_data["X_test"]
    y_test = train_test_split_data["y_test"]
    model = tuned_random_forest["best_model"]

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Create confusion matrix plot using matplotlib
    # Renamed fig to _fig to avoid unused variable warning (good practice)
    _fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap manually
    im = ax.imshow(cm, cmap='Blues', aspect='auto')

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Set ticks and labels
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Legitimate', 'Fraud'])
    ax.set_yticklabels(['Legitimate', 'Fraud'])

    # Add text annotations
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=20, fontweight='bold'
            )

    ax.set_title(
        'Confusion Matrix - Fraud Detection Model\nBy: Koto & Melvin',
        fontsize=14, fontweight='bold', pad=20
    )
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=12)

    plt.tight_layout()

    # Save confusion matrix
    os.makedirs("visualizations", exist_ok=True)
    cm_path = "visualizations/confusion_matrix.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Log the confusion matrix artifact to MLflow
    mlflow_client = context.resources.mlflow_tracking
    try:
        # You can organize artifacts in folders within MLflow
        artifact_folder = "artifacts/confusion_matrices"
        mlflow_client.log_artifact(
            cm_path, artifact_path=artifact_folder
        )
        context.log.info(
            f"Confusion matrix saved to MLflow at {artifact_folder}/"
            f"{os.path.basename(cm_path)}"
        )
    except Exception as e:
        context.log.error(f"Error logging artifact to MLflow: {e}")

    context.log.info(f"Confusion matrix saved to {cm_path}")

    return cm_path


@dg.asset(
    description="Generate and save Random Forest tree visualization",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_model"
)
def random_forest_tree_plot(
    context: dg.AssetExecutionContext,
    train_test_split_data: dict,
    tuned_random_forest: dict
) -> str:
    """Generate visualization of first tree in Random Forest."""
    X_train = train_test_split_data["X_train"]
    model = tuned_random_forest["best_model"]

    # Plot the first tree (index 0)
    plt.figure(figsize=(20, 10))
    plot_tree(
        model.estimators_[0],
        feature_names=X_train.columns.tolist(),
        class_names=['Legitimate', 'Fraud'],
        filled=True,
        rounded=True,
        fontsize=10,
        max_depth=3  # Limit depth for visibility
    )
    plt.title(
        'Random Forest - First Decision Tree (Depth Limited to 3)\n'
        'By: Koto & Melvin',
        fontsize=16, fontweight='bold'
    )

    # Save tree plot
    os.makedirs("visualizations", exist_ok=True)
    tree_path = "visualizations/random_forest_tree.png"
    plt.savefig(tree_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Log to MLflow
    mlflow_client = context.resources.mlflow_tracking
    try:
        # You can organize artifacts in folders within MLflow
        artifact_folder = "artifacts/random_forest_trees"
        mlflow_client.log_artifact(
            tree_path, artifact_path=artifact_folder
        )
        context.log.info(
            f"Random Forest tree saved to MLflow at {artifact_folder}/"
            f"{os.path.basename(tree_path)}"
        )
    except Exception as e:
        context.log.error(f"Error logging artifact to MLflow: {e}")

    return tree_path


@dg.asset(
    description="Generate and save Feature Importance plot for Random Forest model",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_model"
)
def feature_importance_plot(
    context: dg.AssetExecutionContext,
    train_test_split_data: dict,
    tuned_random_forest: dict
) -> str:
    """Generate and save Feature Importance plot for Random Forest."""
    X_train = train_test_split_data["X_train"]
    model = tuned_random_forest["best_model"]

    # Get feature importances from the Random Forest model
    feature_importances = model.feature_importances_

    # Create a DataFrame for better visualization
    feature_names = X_train.columns.tolist()
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })

    # Sort features by importance (highest to lowest)
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plot the feature importances
    plt.figure(figsize=(12, 8))
    plt.barh(
        importance_df['Feature'], importance_df['Importance'], color='teal'
    )
    plt.xlabel('Importance', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.title(
        'Feature Importance - Random Forest Model\nBy: Koto & Melvin',
        fontsize=16, fontweight='bold'
    )
    plt.tight_layout()

    # Save the plot
    os.makedirs("visualizations", exist_ok=True)
    feature_importance_path = "visualizations/feature_importance.png"
    plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Log the feature importance plot to MLflow
    mlflow_client = context.resources.mlflow_tracking
    try:
        # Log the artifact under a specific folder in MLflow
        artifact_folder = "artifacts/feature_importances"
        mlflow_client.log_artifact(
            feature_importance_path, artifact_path=artifact_folder
        )
        context.log.info(
            f"Feature Importance plot saved to MLflow at {artifact_folder}/"
            f"{os.path.basename(feature_importance_path)}"
        )
    except Exception as e:
        context.log.error(f"Error logging feature importance artifact to MLflow: {e}")

    return feature_importance_path


@dg.asset(
    description="Evaluate tuned model on test data and post metrics to Slack.",
    resource_defs={
        "mlflow_tracking": mlflow_resource,
        "slack_messenger": dagster_slack.SlackResource(
            token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN")
        )
    },
    compute_kind="python",
    group_name="ml_fraud_model"
)
def fraud_test_model(
    context: dg.AssetExecutionContext,
    train_test_split_data: dict,
    tuned_random_forest: dict,
    confusion_matrix_plot: str,
    random_forest_tree_plot: str
) -> dict:
    """Evaluate tuned model on test data and send comprehensive metrics to Slack."""

    if not tuned_random_forest or "best_model" not in tuned_random_forest:
        raise ValueError("Missing 'best_model' in tuned_random_forest asset.")

    X_test = train_test_split_data["X_test"]
    y_test = train_test_split_data["y_test"]
    model = tuned_random_forest["best_model"]

    y_pred = model.predict(X_test)

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)

    # FIX 2, 3, 4: Using '0' as a string literal to satisfy strict type hints.
    precision = precision_score(y_test, y_pred, zero_division='0')
    recall = recall_score(y_test, y_pred, zero_division='0')
    f1 = f1_score(y_test, y_pred, zero_division='0')
    report = classification_report(y_test, y_pred, output_dict=True)

    # Log to Dagster
    context.log.info(f"Test Accuracy: {acc:.4f}")
    context.log.info(f"Test Precision: {precision:.4f}")
    context.log.info(f"Test Recall: {recall:.4f}")
    context.log.info(f"Test F1-Score: {f1:.4f}")

    # Log all metrics to MLflow
    mlflow_client = context.resources.mlflow_tracking
    mlflow_client.log_metric("test_accuracy", acc)
    mlflow_client.log_metric("test_precision", precision)
    mlflow_client.log_metric("test_recall", recall)
    mlflow_client.log_metric("test_f1_score", f1)

    # Log per-class and summary metrics from the classification report
    # This section incorporates the necessary type checks to handle the mixed structure
    # (dicts for classes, floats for summaries) of the report.
    for label, metrics in report.items():  # type: ignore
        if isinstance(metrics, dict):
            # Handle per-class metrics (e.g., '0', '1', 'macro avg')
            for metric_name, value in metrics.items():
                # Ensure the value is numeric before logging to MLflow
                if isinstance(value, (int, float)):
                    mlflow_client.log_metric(f"test_{label}_{metric_name}", value)

        # Handle top-level summary metrics (e.g., 'accuracy')
        elif isinstance(metrics, (int, float)):
            mlflow_client.log_metric(f"test_{label}", metrics)

    # Post comprehensive results to Slack
    slack = context.resources.slack_messenger
    slack_message = (
        f"🎯 *Fraud Detection Model Evaluation*\n"
        f"👥 *Team:* Koto & Melvin\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"📊 *Performance Metrics:*\n"
        f"  • Accuracy:  `{acc:.4f}` ({acc * 100:.2f}%)\n"
        f"  • Precision: `{precision:.4f}` ({precision * 100:.2f}%)\n"
        f"  • Recall:    `{recall:.4f}` ({recall * 100:.2f}%)\n"
        f"  • F1-Score:  `{f1:.4f}` ({f1 * 100:.2f}%)\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"✅ Confusion matrix and tree visualization saved!\n"
        f"📈 All metrics logged to MLflow"
    )

    slack.get_client().chat_postMessage(
        channel="aims_course_october2025",
        text=slack_message
    )

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "report": report
    }


@dg.asset(
    description="Promote model to Staging environment in MLflow.",
    compute_kind="python",
    group_name="ml_fraud_promote",
    resource_defs={
        "mlflow_tracking": mlflow_resource,
    }
)
def promote_to_staging(
    context: dg.AssetExecutionContext,
    tuned_random_forest: dict,
    fraud_test_model: dict
) -> dict:
    """
    Promote model to Staging if accuracy threshold is met.
    """
    model = tuned_random_forest["best_model"]
    accuracy = fraud_test_model["accuracy"]

    # Define accuracy threshold for staging
    STAGING_THRESHOLD = 0.65

    mlflow_client = context.resources.mlflow_tracking

    if accuracy >= STAGING_THRESHOLD:
        # Log model to MLflow with staging tag
        try:
            # Using mlflow_client.sklearn.log_model works if the MLflow client
            # resource exposes it. If it fails, switch to global mlflow.sklearn.log_model(model, ...)
            mlflow_client.sklearn.log_model(
                model,
                "random_forest_model",
                registered_model_name="fraud_detection_rf"
            )

            # Add staging tag
            mlflow_client.set_tag("stage", "staging")
            mlflow_client.set_tag("promoted_by", "Koto & Melvin")

            context.log.info(
                f"✅ Model promoted to STAGING (Accuracy: {accuracy:.4f})"
            )

            return {"status": "promoted", "stage": "staging", "accuracy": accuracy}

        except Exception as e:
            context.log.error(f"❌ Failed to promote to staging: {e}")
            return {"status": "failed", "error": str(e)}
    else:
        context.log.warning(
            f"⚠️ Model accuracy {accuracy:.4f} below threshold {STAGING_THRESHOLD}"
        )
        return {
            "status": "rejected",
            "reason": "accuracy_too_low",
            "accuracy": accuracy
        }


@dg.asset(
    description="Promote model from Staging to Production in MLflow.",
    compute_kind="python",
    group_name="ml_fraud_promote",
    resource_defs={
        "mlflow_tracking": mlflow_resource,
    }
)
def promote_to_production(
    context: dg.AssetExecutionContext,
    promote_to_staging: dict,
    tuned_random_forest: dict,
    fraud_test_model: dict
) -> dict:
    """
    Promote model to Production if staging validation passes.
    """
    if promote_to_staging.get("status") != "promoted":
        context.log.warning(
            "⚠️ Model not in staging, skipping production promotion"
        )
        return {"status": "skipped", "reason": "not_in_staging"}

    model = tuned_random_forest["best_model"]
    accuracy = fraud_test_model["accuracy"]

    # Define accuracy threshold for production
    PRODUCTION_THRESHOLD = 0.80

    mlflow_client = context.resources.mlflow_tracking

    if accuracy >= PRODUCTION_THRESHOLD:
        try:
            # Update model stage to production
            mlflow_client.set_tag("stage", "production")
            mlflow_client.set_tag("production_promoted_by", "Koto & Melvin")

            # Save production model
            save_dir = os.path.abspath(os.path.join(os.getcwd(), "fraud_model"))
            os.makedirs(save_dir, exist_ok=True)
            model_path = os.path.join(
                save_dir, "tuned_random_forest_production.pkl"
            )

            with open(model_path, "wb") as f:
                pickle.dump(model, f)

            mlflow_client.log_artifact(model_path)

            context.log.info(
                f"✅ Model promoted to PRODUCTION (Accuracy: {accuracy:.4f})"
            )

            return {
                "status": "promoted",
                "stage": "production",
                "accuracy": accuracy,
                "model_path": model_path
            }

        except Exception as e:
            context.log.error(f"❌ Failed to promote to production: {e}")
            return {"status": "failed", "error": str(e)}
    else:
        context.log.warning(
            f"⚠️ Model accuracy {accuracy:.4f} below production threshold "
            f"{PRODUCTION_THRESHOLD}"
        )

        return {
            "status": "rejected",
            "reason": "accuracy_too_low",
            "accuracy": accuracy
        }


@dg.asset(
    description="Save the tuned RandomForest model as a pickle file for Gradio use.",
    compute_kind="python",
    group_name="ml_fraud_promote",
    resource_defs={"mlflow_tracking": mlflow_resource}
)
def save_tuned_model(
    context: dg.AssetExecutionContext,
    tuned_random_forest: dict,
    promote_to_production: dict
) -> str:
    """
    Saves the best model as a pickle file and logs it to the MLflow
    Model Registry.
    """
    best_model = tuned_random_forest["best_model"]

    # Save pickle (keep this for backup)
    save_dir = os.path.abspath(os.path.join(os.getcwd(), "fraud_model"))
    os.makedirs(save_dir, exist_ok=True)
    version = int(time.time())
    model_filename = f"tuned_random_forest_v{version}.pkl"
    model_path = os.path.join(save_dir, model_filename)

    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)

    # Log model to MLflow properly
    # FIX 6: Using the globally imported mlflow.sklearn resolves the private import error.

    mlflow_sklearn.log_model(
        sk_model=best_model,
        artifact_path="model",  # This is important!
        registered_model_name="fraud_detection_model"  # Register in Model Registry
    )

    context.log.info("✅ Model logged to MLflow Model Registry")

    return model_path
