import os
import time
from datetime import datetime
from typing import Dict, Iterable, List

import dagster as dg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from .configs import DataConfig, ModelConfig, ModelPromotionConfig
from .constants import (
    ARCHIVED_STAGE_NAME,
    FRAUD_MODEL_NAME,
    GROUP_EMOJI,
    MODEL_ARTIFACT_NAME,
    PRODUCTION_STAGE_NAME,
    STAGING_STAGE_NAME,
    STATUS_MODEL_NOT_PROMOTED_TO_PRODUCTION,
    STATUS_MODEL_NOT_PROMOTED_TO_STAGING,
    STATUS_MODEL_PROMOTED_TO_PRODUCTION,
    STATUS_MODEL_PROMOTED_TO_STAGING,
    STATUS_MODEL_SKIPPED,
)
from .utils import _sanitize_report_dict


@dg.asset(
    group_name="ml_fraud_ingest",
    description="Loads the raw credit card fraud data from a URL.",
    compute_kind="python",
    required_resource_keys={"mlflow", "data_config"}
)
def raw_fraud_data(
    context: dg.OpExecutionContext
) -> dg.MaterializeResult:
    """
    Loads the raw credit card fraud detection dataset from the URL specified in DataConfig,
    logs metadata, and logs parameters to MLFlow.
    """
    data_config: DataConfig = context.resources.data_config
    mlflow_client = context.resources.mlflow

    context.log.info(f"Loading data from {data_config.dataset_url}")

    # 1. Calculate download time
    start_time = time.time()
    df = pd.read_csv(data_config.dataset_url)
    end_time = time.time()
    download_time = end_time - start_time
    context.log.info(f"Data downloaded in {download_time:.2f} seconds.")

    # 2. Calculate all required metrics
    num_rows, num_columns = df.shape
    memory_usage = df.memory_usage(deep=True).sum()  # In bytes
    fraud_count = int(df["Class"].sum())
    non_fraud_count = num_rows - fraud_count
    # Handle division by zero if there are no rows
    fraud_ratio = (fraud_count / num_rows) if num_rows > 0 else 0.0

    context.log.info(f"Dataset dimensions: {num_rows} rows, {num_columns} columns.")
    context.log.info(f"Memory usage: {memory_usage / 1e6:.2f} MB")
    context.log.info(f"Fraud cases: {fraud_count}, Non-fraud cases: {non_fraud_count}")

    # 3. Log all metrics to MLFlow
    mlflow_client.log_param("dataset_url", data_config.dataset_url)
    mlflow_client.log_metric("download_time_seconds", download_time)
    mlflow_client.log_metric("memory_usage_bytes", memory_usage)
    mlflow_client.log_metric("num_rows", num_rows)
    mlflow_client.log_metric("num_columns", num_columns)
    mlflow_client.log_metric("fraud_count", fraud_count)
    mlflow_client.log_metric("non_fraud_count", non_fraud_count)
    mlflow_client.log_metric("fraud_ratio", fraud_ratio)
    context.log.info("Logged all raw data parameters and metrics to MLFlow.")

    # 4. Return all metrics in the MaterializeResult
    return dg.MaterializeResult(
        value=df,
        metadata={
            "dataset_url": dg.MetadataValue.url(data_config.dataset_url),
            "num_rows": dg.MetadataValue.int(int(num_rows)),
            "num_columns": dg.MetadataValue.int(int(num_columns)),
            "download_time_seconds": dg.MetadataValue.float(float(download_time)),
            "memory_usage_mb": dg.MetadataValue.float(float(memory_usage / 1e6)),
            "fraud_count": dg.MetadataValue.int(int(fraud_count)),
            "non_fraud_count": dg.MetadataValue.int(int(non_fraud_count)),
            "fraud_ratio": dg.MetadataValue.float(fraud_ratio),
            "preview_data": dg.MetadataValue.md(
                df.head().to_markdown() or "" if not df.empty else "No data to preview."
            ),
        }
    )


@dg.asset(
    group_name="ml_fraud_transform",
    description="Removes duplicates, missing values, and unnecessary columns from the raw data.",
    compute_kind="python",
    required_resource_keys={"mlflow"}
)
def cleaned_fraud_data(
    context: dg.OpExecutionContext,
    raw_fraud_data: pd.DataFrame
) -> dg.MaterializeResult:
    """
    Cleans data by removing duplicates, handling nulls, and dropping the 'Time' column.
    The 'Amount' column is passed through for later transformation.
    """
    context.log.info("Starting data cleaning process.")

    if raw_fraud_data.empty:
        context.log.warning("Input DataFrame is empty. Skipping cleaning.")
        return dg.MaterializeResult(value=pd.DataFrame())

    df = raw_fraud_data.copy()
    original_row_count = len(df)

    df.drop_duplicates(inplace=True)
    duplicates_removed_count = original_row_count - len(df)
    context.log.info(f"Removed {duplicates_removed_count} duplicate rows.")

    missing_values_count = df.isnull().sum().sum()
    if missing_values_count > 0:
        df.dropna(inplace=True)
        context.log.warning(f"Found and dropped {missing_values_count} missing values.")

    if 'Time' in df.columns:
        df.drop(['Time'], axis=1, inplace=True)

    metadata = {
        'original_row_count': original_row_count,
        'cleaned_row_count': len(df),
        'duplicates_removed_count': duplicates_removed_count,
        'missing_values_found': int(missing_values_count),
    }

    context.resources.mlflow.log_metrics({f"clean_{k}": v for k, v in metadata.items()})

    return dg.MaterializeResult(
        value=df,
        metadata={
            "cleaned_row_count": dg.MetadataValue.int(metadata['cleaned_row_count']),
            "duplicates_removed_count": dg.MetadataValue.int(metadata['duplicates_removed_count']),
            "preview_data": dg.MetadataValue.md(
                df.head().to_markdown() or "" if not df.empty else "No data to preview."
            ),
        },
    )


@dg.asset(
    group_name="ml_fraud_transform",
    description="Performs feature selection based on Random Forest feature importance.",
    compute_kind="python",
    required_resource_keys={"mlflow", "model_config"}
)
def feature_selection(
    context: dg.OpExecutionContext,
    cleaned_fraud_data: pd.DataFrame
) -> dg.MaterializeResult:
    """Selects important features using Random Forest feature importance.
    Always includes 'Amount' feature and logs feature importance plot to MLFlow.

    Returns:
        Dictionary containing:
        - 'selected_features': List of selected feature names
        - 'feature_importances': DataFrame with all feature importances
    """
    model_config: ModelConfig = context.resources.model_config
    mlflow_client = context.resources.mlflow

    if cleaned_fraud_data.empty:
        context.log.warning("Input cleaned_fraud_data is empty. Skipping feature selection.")
        return dg.MaterializeResult(
            value={'selected_features': [], 'feature_importances': pd.DataFrame()},
            metadata={"status": dg.MetadataValue.text("Skipped, input was empty.")}
        )

    context.log.info("Starting feature selection process...")

    df = cleaned_fraud_data.copy()

    # Separate features and target
    if 'Class' not in df.columns:
        context.log.error("'Class' column not found in data.")
        return dg.MaterializeResult(
            value={'selected_features': [], 'feature_importances': pd.DataFrame()},
            metadata={"status": dg.MetadataValue.text("Error: No 'Class' column found.")}
        )

    X = df.drop('Class', axis=1)
    y = df['Class']

    initial_feature_count = len(X.columns)
    context.log.info(f"Initial feature count: {initial_feature_count}")

    # Get importance threshold from config or use default
    importance_threshold = getattr(model_config, 'feature_importance_threshold', 0.02)

    # Train baseline Random Forest for feature importance
    context.log.info("Training baseline Random Forest to assess feature importance...")
    baseline_model = RandomForestClassifier(
        n_estimators=200,
        random_state=model_config.random_state,
        max_depth=None,
        n_jobs=-1
    )
    baseline_model.fit(X, y)

    # Extract feature importances
    importances = baseline_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False)

    context.log.info("Top 10 most important features:")
    for _, row in feature_importance_df.head(10).iterrows():
        context.log.info(f"  {row['feature']:30s}: {row['importance']:.6f}")

    # Select features based on importance threshold
    selected_features: List[str] = feature_importance_df[  # type: ignore
        feature_importance_df['importance'] > importance_threshold
    ]['feature'].tolist()

    # ALWAYS include 'Amount' if it exists and isn't already selected
    if 'Amount' in X.columns and 'Amount' not in selected_features:
        selected_features.append('Amount')
        context.log.info("Added 'Amount' feature (mandatory inclusion).")

    # Fallback: if no features meet threshold, use top 10
    if not selected_features:
        context.log.warning(
            f"No features met importance threshold of {importance_threshold}. "
            "Using top 10 features as fallback."
        )
        selected_features = feature_importance_df.head(10)['feature'].tolist()  # type: ignore

        # Ensure Amount is in the fallback selection
        if 'Amount' in X.columns and 'Amount' not in selected_features:
            selected_features.append('Amount')

    context.log.info(f"Selected {len(selected_features)} features out of {initial_feature_count}")
    context.log.info(f"Selected features: {selected_features}")

    # Create feature importance plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(feature_importance_df) * 0.3)))

    # Color selected features differently
    colors = ['#2ecc71' if feat in selected_features else '#95a5a6'
              for feat in feature_importance_df['feature']]

    sns.barplot(
        data=feature_importance_df,
        y='feature',
        x='importance',
        palette=colors,
        ax=ax
    )

    ax.set_title('Feature Importances from Baseline Random Forest', fontsize=14, fontweight='bold')
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Features', fontsize=12)
    ax.axvline(x=importance_threshold, color='red', linestyle='--',
               label=f'Threshold ({importance_threshold})')
    ax.legend()

    # Add text annotation for selected features
    textstr = f'Selected: {len(selected_features)}/{initial_feature_count} features'
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot temporarily
    plot_path = "/tmp/feature_importance.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Log plot to MLflow
    mlflow_client.log_artifact(plot_path, artifact_path="feature_selection")
    context.log.info("Logged feature importance plot to MLFlow artifacts.")

    # Clean up temporary file
    if os.path.exists(plot_path):
        os.remove(plot_path)

    # Log metrics to MLFlow
    mlflow_client.log_param("feature_importance_threshold", importance_threshold)
    mlflow_client.log_metric("initial_feature_count", initial_feature_count)
    mlflow_client.log_metric("selected_feature_count", len(selected_features))
    mlflow_client.log_metric("feature_reduction_pct",
                            (1 - len(selected_features) / initial_feature_count) * 100)

    # Log top feature importances
    for idx, row in feature_importance_df.head(10).iterrows():
        mlflow_client.log_metric(f"importance_{row['feature']}", row['importance'])

    # Prepare output
    output_value = {
        'selected_features': selected_features,
        'feature_importances': feature_importance_df
    }

    # Create markdown table for metadata
    top_features_md = feature_importance_df.head(15).to_markdown(index=False) or ""

    return dg.MaterializeResult(
        value=output_value,
        metadata={
            "initial_feature_count": dg.MetadataValue.int(initial_feature_count),
            "selected_feature_count": dg.MetadataValue.int(len(selected_features)),
            "feature_reduction_percent": dg.MetadataValue.float(
                (1 - len(selected_features) / initial_feature_count) * 100
            ),
            "importance_threshold": dg.MetadataValue.float(importance_threshold),
            "selected_features": dg.MetadataValue.json(selected_features),
            "top_15_features": dg.MetadataValue.md(top_features_md),
            "amount_included": dg.MetadataValue.bool('Amount' in selected_features)
        }
    )


@dg.asset(
    group_name="ml_fraud_split",
    description="Splits data with selected features into train and test sets.",
    compute_kind="python",
    required_resource_keys={"mlflow", "data_config"},
)
def data_splits(
    context: dg.OpExecutionContext,
    cleaned_fraud_data: pd.DataFrame,
    feature_selection: Dict
) -> dg.MaterializeResult:
    """
    Splits data into train/test sets using only the selected features.
    """
    data_config: DataConfig = context.resources.data_config
    mlflow_client = context.resources.mlflow

    selected_features = feature_selection['selected_features']

    if cleaned_fraud_data.empty or not selected_features:
        context.log.warning("Data is empty or no features selected. Returning empty splits.")
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype='int64')

        return dg.MaterializeResult(
            value={'X_train': empty_df, 'X_test': empty_df,
                   'y_train': empty_series, 'y_test': empty_series,
                   'selected_features': []},
            metadata={"status": dg.MetadataValue.text("Skipped, input was empty.")}
        )

    context.log.info(f"Splitting data with {len(selected_features)} selected features...")

    # Use only selected features
    X = cleaned_fraud_data[selected_features]
    y = cleaned_fraud_data['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=data_config.test_size,
        random_state=data_config.random_state,
        stratify=y
    )

    train_fraud_rate = float((np.sum(y_train) / len(y_train)) * 100 if len(y_train) > 0 else 0.0)
    test_fraud_rate = float((np.sum(y_test) / len(y_test)) * 100 if len(y_test) > 0 else 0.0)

    context.log.info(f"Train set: {len(X_train)} samples, {train_fraud_rate:.2f}% fraud.")
    context.log.info(f"Test set: {len(X_test)} samples, {test_fraud_rate:.2f}% fraud.")
    context.log.info(f"Features used: {list(X_train.columns)}")  # type: ignore

    # Log to MLFlow
    mlflow_client.log_param("test_size", data_config.test_size)
    mlflow_client.log_param("random_state", data_config.random_state)
    mlflow_client.log_param("num_features_used", len(selected_features))
    mlflow_client.log_metric("train_set_size", len(X_train))
    mlflow_client.log_metric("test_set_size", len(X_test))
    mlflow_client.log_metric("train_set_fraud_rate", train_fraud_rate)
    mlflow_client.log_metric("test_set_fraud_rate", test_fraud_rate)

    output_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'selected_features': selected_features
    }

    return dg.MaterializeResult(
        value=output_data,
        metadata={
            "train_set_size": dg.MetadataValue.int(len(X_train)),
            "test_set_size": dg.MetadataValue.int(len(X_test)),
            "num_features": dg.MetadataValue.int(len(selected_features)),
            "train_fraud_rate_percent": dg.MetadataValue.float(train_fraud_rate),
            "test_fraud_rate_percent": dg.MetadataValue.float(test_fraud_rate),
            "features_used": dg.MetadataValue.json(selected_features)
        }
    )


@dg.asset(
    group_name="ml_fraud_model",
    description="Tunes Random Forest hyperparameters using 3-fold cross-validation.",
    compute_kind="python",
    required_resource_keys={"mlflow", "model_config"},
)
def tune_hyperparameters(
    context: dg.OpExecutionContext,
    data_splits: Dict
) -> dg.MaterializeResult:
    """
    Performs hyperparameter tuning for a RandomForestClassifier using nested MLFlow runs.
    """
    model_config: ModelConfig = context.resources.model_config
    mlflow_client = context.resources.mlflow

    X_train = data_splits['X_train']
    y_train = data_splits['y_train']

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)

    if X_train.empty:
        context.log.warning("Training data is empty. Skipping hyperparameter tuning.")
        return dg.MaterializeResult(
            value={'best_params': {}},
            metadata={"status": dg.MetadataValue.text("Skipped, input was empty.")}
        )

    context.log.info(f"Starting hyperparameter tuning with {model_config.cv_folds}-fold CV.")

    best_score = -1.0
    best_params = {}

    mlflow_client.log_param("cv_folds", model_config.cv_folds)
    mlflow_client.log_param("scoring_metric", model_config.scoring_metric)

    for max_depth in model_config.max_depth_options:
        run_name = f"trial_max_depth_{max_depth}"
        with mlflow_client.start_run(run_name=run_name, nested=True) as _:
            context.log.info(f"Running Trial: max_depth={max_depth}")

            params = {
                'n_estimators': int(model_config.n_estimators),
                'max_depth': int(max_depth),
                'random_state': int(model_config.random_state)
            }
            mlflow_client.log_params(params)

            clf = RandomForestClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                random_state=params['random_state']
            )
            cv = StratifiedKFold(
                n_splits=model_config.cv_folds,
                shuffle=True,
                random_state=model_config.random_state
            )

            scores = cross_val_score(
                clf,
                X_train,
                y_train,
                cv=cv,
                scoring=model_config.scoring_metric
            )
            avg_score = float(np.mean(scores))

            mlflow_client.log_metric(f"avg_{model_config.scoring_metric}", avg_score)
            context.log.info(f"Avg {model_config.scoring_metric}: {avg_score:.4f}")

            if avg_score > best_score:
                best_score = avg_score
                best_params = params

    mlflow_client.log_metric(f"best_cv_{model_config.scoring_metric}", float(best_score))
    mlflow_client.log_params({f"best_{k}": v for k, v in best_params.items()})
    context.log.info(f"Best params found: {best_params} with score: {best_score:.4f}")

    return dg.MaterializeResult(
        value={'best_params': best_params},
        metadata={
            "best_cv_score": dg.MetadataValue.float(float(best_score)),
            "best_max_depth": dg.MetadataValue.int(int(best_params.get('max_depth', 0))),
        }
    )


@dg.asset(
    group_name="ml_fraud_model",
    description="Trains the final model with the best hyperparameters.",
    compute_kind="python",
    required_resource_keys={"mlflow"},
)
def train_model(
    context: dg.OpExecutionContext,
    tune_hyperparameters: Dict,
    data_splits: Dict
) -> dg.MaterializeResult:
    """
    Trains a RandomForestClassifier on the full training data using the best hyperparameters.
    """
    mlflow_client = context.resources.mlflow
    best_params = tune_hyperparameters['best_params']

    if not best_params:
        context.log.warning("No best parameters found. Skipping model training.")
        return dg.MaterializeResult(
            value=RandomForestClassifier(),
            metadata={"status": dg.MetadataValue.text("Skipped, no best parameters found.")}
        )

    X_train = data_splits['X_train']
    y_train = data_splits['y_train']

    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(y_train, pd.Series)

    context.log.info(f"Training final model with params: {best_params}")

    final_model = RandomForestClassifier(**best_params)
    final_model.fit(X_train, y_train)

    context.log.info("Final model trained successfully.")
    mlflow_client.sklearn.log_model(final_model, "random_forest_model")
    context.log.info("Logged trained model to MLFlow.")

    return dg.MaterializeResult(
        value=final_model,
        metadata={
            "model_class": dg.MetadataValue.text(final_model.__class__.__name__),
            "training_parameters": dg.MetadataValue.json(best_params),
        }
    )


@dg.asset(
    group_name="ml_fraud_evaluate",
    description="Evaluates the final model and logs a confusion matrix.",
    compute_kind="python",
    required_resource_keys={"mlflow"},
)
def evaluate_model(
    context: dg.OpExecutionContext,
    train_model: RandomForestClassifier,
    data_splits: Dict
) -> dg.MaterializeResult:
    """
    Evaluates the model on the test set, logs metrics, and logs a confusion matrix plot.
    """
    mlflow_client = context.resources.mlflow

    X_test: pd.DataFrame = data_splits['X_test']
    y_test: pd.Series = data_splits['y_test']

    if X_test.empty:
        context.log.warning("Test data is empty. Skipping model evaluation.")
        return dg.MaterializeResult(
            value={},
            metadata={"status": dg.MetadataValue.text("Skipped, test data was empty.")}
        )

    context.log.info("Evaluating model on test data.")
    predictions = train_model.predict(X_test)
    predictions_proba = train_model.predict_proba(X_test)[:, 1]  # type: ignore

    accuracy = float(accuracy_score(y_test, predictions))
    f1 = float(f1_score(y_test, predictions))
    roc_auc = float(roc_auc_score(y_test, predictions_proba))
    report_dict: dict = classification_report(y_test, predictions, output_dict=True)  # type: ignore
    sanitized_report = _sanitize_report_dict(report_dict)  # assumes helper exists in module

    metrics = {
        "test_accuracy": accuracy,
        "test_f1_score": f1,
        "test_roc_auc": roc_auc,
        "test_precision_class_0": sanitized_report['0']['precision'],
        "test_recall_class_0": sanitized_report['0']['recall'],
        "test_precision_class_1": sanitized_report['1']['precision'],
        "test_recall_class_1": sanitized_report['1']['recall'],
    }

    # Log metrics to MLflow
    mlflow_client.log_metrics(metrics)

    context.log.info(f"Test Accuracy: {accuracy:.4f}")
    context.log.info(f"Test F1 Score: {f1:.4f}")
    context.log.info(f"Test ROC-AUC: {roc_auc:.4f}")

    # Create confusion matrix plot
    cm = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    plt.tight_layout()

    # Save plot temporarily
    plot_path = "/tmp/confusion_matrix.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Log plot to MLflow as artifact
    mlflow_client.log_artifact(plot_path, artifact_path="plots")
    context.log.info("Logged confusion matrix plot to MLFlow artifacts.")

    # Clean up temporary file
    if os.path.exists(plot_path):
        os.remove(plot_path)

    # Register model in MLflow (preserves previous behavior)
    try:
        run_id = mlflow_client.get_run(mlflow_client.active_run().info.run_id).info.run_uuid
    except Exception:
        run_id = None
    model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}" if run_id else MODEL_ARTIFACT_NAME
    try:
        model_info = mlflow_client.register_model(model_uri, FRAUD_MODEL_NAME)
        context.log.info(f"Registered model '{model_info.name}' version {model_info.version}")
    except Exception as e:
        context.log.warning(f"Model registration skipped/failed: {e}")
        model_info = None

    output_value = {
        "metrics": metrics,
        "classification_report": sanitized_report,
        "confusion_matrix": cm.tolist(),
        "model_info": {
            "name": getattr(model_info, "name", None),
            "version": getattr(model_info, "version", None),
            "run_id": run_id
        }
    }

    meta = {
        "test_accuracy": dg.MetadataValue.float(accuracy),
        "test_f1_score": dg.MetadataValue.float(f1),
        "test_roc_auc": dg.MetadataValue.float(roc_auc),
        "classification_report": dg.MetadataValue.json(sanitized_report),
        "confusion_matrix": dg.MetadataValue.md(
            f"```\n{cm}\n```\nTrue Negatives: {cm[0][0]}, False Positives: {cm[0][1]}\n"
            f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}"
        )
    }

    # add per-class precision/recall to metadata
    meta.update({
        "test_precision_class_0": dg.MetadataValue.float(sanitized_report['0']['precision']),
        "test_recall_class_0": dg.MetadataValue.float(sanitized_report['0']['recall']),
        "test_precision_class_1": dg.MetadataValue.float(sanitized_report['1']['precision']),
        "test_recall_class_1": dg.MetadataValue.float(sanitized_report['1']['recall']),
    })

    return dg.MaterializeResult(
        value=output_value,
        metadata=meta
    )


@dg.asset(
    group_name="ml_fraud_promote",
    description="Promotes the model to Staging if it meets performance criteria and sends a Slack notification.",
    compute_kind="python",
    required_resource_keys={"mlflow_api_client", "model_promotion_config", "slack"},
)
def promote_fraud_model_to_staging(
    context: dg.OpExecutionContext,
    evaluate_model: Dict
) -> dg.MaterializeResult:
    """
    Checks model performance against F1 and ROC AUC thresholds and promotes to Staging if criteria are met.
    Sends a detailed notification to a Slack channel for both success and failure cases.
    """
    model_promotion_config: ModelPromotionConfig = context.resources.model_promotion_config
    mlflow_client = context.resources.mlflow_api_client  # high-level client
    slack = context.resources.slack

    metrics = evaluate_model.get("metrics", {})
    model_info = evaluate_model.get("model_info", {})

    if not metrics or not model_info:
        context.log.warning("Upstream evaluation data is missing. Skipping promotion.")
        return dg.MaterializeResult(
            metadata={"status": dg.MetadataValue.text("Skipped, upstream data missing.")}
        )

    # Define performance thresholds from your config for both metrics
    STAGING_F1_THRESHOLD = model_promotion_config.staging_f1_threshold
    STAGING_ROC_AUC_THRESHOLD = model_promotion_config.staging_roc_auc_threshold

    # Unpack metrics and model info
    test_f1_score = metrics.get("test_f1_score", 0.0)
    test_roc_auc = metrics.get("test_roc_auc", 0.0)
    test_precision_fraud = metrics.get("test_precision_class_1", 0.0)
    test_recall_fraud = metrics.get("test_recall_class_1", 0.0)
    test_precision_normal = metrics.get("test_precision_class_0", 0.0)
    test_recall_normal = metrics.get("test_recall_class_0", 0.0)

    model_name = model_info.get("name")
    model_version = model_info.get("version")

    # Check if the model meets BOTH promotion criteria
    meets_criteria = (test_f1_score >= STAGING_F1_THRESHOLD) and (test_roc_auc >= STAGING_ROC_AUC_THRESHOLD)

    if meets_criteria:
        context.log.info(
            f"Model '{model_name}' (v{model_version}) meets performance criteria. "
            f"F1: {test_f1_score:.4f} >= {STAGING_F1_THRESHOLD}, "
            f"ROC AUC: {test_roc_auc:.4f} >= {STAGING_ROC_AUC_THRESHOLD}. Promoting to Staging."
        )

        try:
            mlflow_client.transition_model_version_stage(
                name=str(model_name),
                version=str(model_version),
                stage=STAGING_STAGE_NAME
            )
        except Exception as e:
            context.log.error(f"Error during model promotion: {e}")
            return dg.MaterializeResult(
                metadata={
                    "status": dg.MetadataValue.text(STATUS_MODEL_NOT_PROMOTED_TO_STAGING),
                    "reason": dg.MetadataValue.text(f"Promotion failed: {e}")
                }
            )

        try:
            # Construct and send the success notification
            slack_message = f"""✅ *Model Promoted to Staging: Fraud Detection* ✅

            {GROUP_EMOJI} *GROUP MEMBERS:* Khadija EDARZI, Similoluwa OKUNOWO

            *Performance Summary:*
            - *F1 Score:* `{test_f1_score:.4f}` (Threshold: `> {STAGING_F1_THRESHOLD}`) -> *Passed*
            - *ROC AUC:* `{test_roc_auc:.4f}` (Threshold: `> {STAGING_ROC_AUC_THRESHOLD}`) -> *Passed*

            *Detailed Metrics:*
            - *Precision (Fraud Class):* `{test_precision_fraud:.4f}`
            - *Recall (Fraud Class):* `{test_recall_fraud:.4f}`
            - *Precision (Normal Class):* `{test_precision_normal:.4f}`
            - *Recall (Normal Class):* `{test_recall_normal:.4f}`

            -----
            Run Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
            """
            slack.get_client().chat_postMessage(channel='aims_course_october2025', text=slack_message)
            context.log.info("Successfully promoted model and sent Slack notification.")

            return dg.MaterializeResult(
                value={
                    "status": STATUS_MODEL_PROMOTED_TO_STAGING,
                    "model_name": model_name,
                    "model_version": model_version
                },
                metadata={
                    "status": dg.MetadataValue.text(STATUS_MODEL_PROMOTED_TO_STAGING),
                    "model_version": dg.MetadataValue.text(f"{model_name} v{model_version}"),
                    "test_f1_score": dg.MetadataValue.float(test_f1_score),
                    "test_roc_auc": dg.MetadataValue.float(test_roc_auc)
                }
            )
        except Exception as e:
            context.log.error(f"Failed to send 'Promoted' Slack notification: {e}")
            return dg.MaterializeResult(
                metadata={
                    "status": dg.MetadataValue.text(STATUS_MODEL_NOT_PROMOTED_TO_STAGING),
                    "model_version": dg.MetadataValue.text(f"{model_name} v{model_version}"),
                    "test_f1_score": dg.MetadataValue.float(test_f1_score),
                    "test_roc_auc": dg.MetadataValue.float(test_roc_auc),
                    "notification_status": dg.MetadataValue.text(f"Failed to send notification: {e}")
                }
            )

    else:
        # This block executes if one or both performance thresholds are NOT met
        context.log.warning("Model does not meet performance criteria for Staging promotion.")

        # Determine which check failed for a clearer message
        f1_status = "Passed" if test_f1_score >= STAGING_F1_THRESHOLD else "Failed"
        roc_auc_status = "Passed" if test_roc_auc >= STAGING_ROC_AUC_THRESHOLD else "Failed"

        # Construct and send the failure notification
        slack_message = f"""⚠️ *Model NOT Promoted: Performance Threshold Not Met* ⚠️

        *Performance Summary:*
        - *F1 Score:* `{test_f1_score:.4f}` (Threshold: `> {STAGING_F1_THRESHOLD}`) -> *{f1_status}*
        - *ROC AUC:* `{test_roc_auc:.4f}` (Threshold: `> {STAGING_ROC_AUC_THRESHOLD}`) -> *{roc_auc_status}*

        This model was *not* promoted to Staging.

        -----
        Run Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
        """
        try:
            slack.get_client().chat_postMessage(channel='aims_course_october2025', text=slack_message)
            context.log.info("Sent Slack notification for model not meeting promotion criteria.")
        except Exception as e:
            context.log.error(f"Failed to send 'Not Promoted' Slack notification: {e}")

        return dg.MaterializeResult(
            metadata={
                "status": dg.MetadataValue.text(STATUS_MODEL_NOT_PROMOTED_TO_STAGING),
                "reason": dg.MetadataValue.text("Performance threshold(s) not met."),
                "test_f1_score": dg.MetadataValue.float(test_f1_score),
                "f1_threshold": dg.MetadataValue.float(STAGING_F1_THRESHOLD),
                "test_roc_auc": dg.MetadataValue.float(test_roc_auc),
                "roc_auc_threshold": dg.MetadataValue.float(STAGING_ROC_AUC_THRESHOLD),
            }
        )


@dg.asset(
    group_name="ml_fraud_promote",
    description="Promotes the best model from Staging to Production.",
    compute_kind="python",
    required_resource_keys={"mlflow_api_client", "slack"},
)
def promote_fraud_model_to_production(
    context: dg.OpExecutionContext,
    promote_fraud_model_to_staging: Dict
) -> dg.MaterializeResult:
    """
    Finds the latest model in Staging, archives the current Production model,
    and promotes the Staging model to Production. Sends a Slack notification.
    """
    mlflow_client = context.resources.mlflow_api_client
    slack = context.resources.slack
    context.log.info("Starting model promotion to Production.")

    if promote_fraud_model_to_staging.get("status") != STATUS_MODEL_PROMOTED_TO_STAGING:
        context.log.info("No model was promoted to Staging in the previous step. Skipping production promotion.")
        return dg.MaterializeResult(
            metadata={"status": dg.MetadataValue.text("Skipped, no new model in Staging.")}
        )

    model_name = promote_fraud_model_to_staging.get("model_name") or ""

    try:
        # Find the latest model version in the 'Staging' stage
        staging_versions = [
            mv for mv in mlflow_client.search_model_versions(f"name='{model_name}'")
            if mv.current_stage == STAGING_STAGE_NAME
        ]
        if not staging_versions:
            context.log.warning(f"No model versions found in Staging for '{model_name}'. Skipping promotion.")
            return dg.MaterializeResult(
                metadata={
                    "status": dg.MetadataValue.text(STATUS_MODEL_SKIPPED),
                    "reason": dg.MetadataValue.text("No model in Staging.")
                }
            )

        # Get the version with the highest version number
        latest_staging_version = max(staging_versions, key=lambda mv: int(mv.version))
        prod_model_version = latest_staging_version.version

        # Archive any existing models currently in 'Production'
        for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
            if mv.current_stage == PRODUCTION_STAGE_NAME:
                context.log.info(f"Archiving previous Production model version {mv.version}.")
                mlflow_client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage=ARCHIVED_STAGE_NAME
                )

        # Promote the new version to 'Production'
        context.log.info(f"Promoting model '{model_name}' version {prod_model_version} to Production.")
        mlflow_client.transition_model_version_stage(
            name=model_name,
            version=prod_model_version,
            stage=PRODUCTION_STAGE_NAME
        )

        # Send Slack notification
        slack_message = f"""🚀 *Model Promoted to Production: Fraud Detection* 🚀

        {GROUP_EMOJI} *GROUP MEMBERS:* Khadija EDARZI, Similoluwa OKUNOWO

        - *Model Name:* `{model_name}`
        - *Version:* `{prod_model_version}`

        This model is now live in the production environment.

        -----
        Run Timestamp: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}
        """
        slack.get_client().chat_postMessage(channel='aims_course_october2025', text=slack_message)
        context.log.info("Successfully promoted model to Production and sent Slack notification.")

        return dg.MaterializeResult(
            metadata={
                "status": dg.MetadataValue.text(STATUS_MODEL_PROMOTED_TO_PRODUCTION),
                "model_name": dg.MetadataValue.text(model_name),
                "promoted_version": dg.MetadataValue.text(str(prod_model_version))
            }
        )
    except Exception as e:
        context.log.error(f"Error promoting model to Production: {e}")
        return dg.MaterializeResult(
            metadata={
                "status": dg.MetadataValue.text(STATUS_MODEL_NOT_PROMOTED_TO_PRODUCTION),
                "reason": dg.MetadataValue.text(f"Promotion failed: {e}")
            }
        )


@dg.multi_asset_check(
    specs=[
        dg.AssetCheckSpec(name="no_nulls_in_raw_data", asset="raw_fraud_data"),
        dg.AssetCheckSpec(name="valid_class_labels", asset="raw_fraud_data"),
    ],
    description="Checks the quality of the raw fraud detection data.",
)
def raw_data_quality_checks(raw_fraud_data: pd.DataFrame) -> Iterable[dg.AssetCheckResult]:
    """
    Performs data quality checks on the raw DataFrame.
    1. Checks for any null values.
    2. Checks that the 'Class' column contains only 0s and 1s.
    """
    # Check 1: No null values
    num_nulls = raw_fraud_data.isnull().sum().sum()
    yield dg.AssetCheckResult(
        check_name="no_nulls_in_raw_data",
        passed=bool(num_nulls == 0),
        metadata={"num_null_values": dg.MetadataValue.int(int(num_nulls))},
    )

    # Check 2: 'Class' column has only 0s and 1s
    unique_classes = raw_fraud_data['Class'].unique()
    is_valid = np.all(np.isin(unique_classes, [0, 1]))  # type: ignore
    yield dg.AssetCheckResult(
        check_name="valid_class_labels",
        passed=bool(is_valid),
        metadata={"unique_values_found": dg.MetadataValue.text(str(list(unique_classes)))},
    )
