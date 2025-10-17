import os
import tempfile
import threading
import time
from typing import Any, Dict, List

import dagster as dg
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

# from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from .resources import mlflow_resource, slack_resource


def categorize_time(time_in_seconds: int) -> int:
    """Categorizes time as 'day' or 'night' based on seconds.

    Args:
        time_in_seconds: Time in seconds.

    Returns:
        1 if between 6 AM and 6 PM, 0 otherwise.
    """
    if 21600 <= time_in_seconds <= 64800:
        return 1
    else:
        return 0


@dg.asset(
    description="Import data for fraud detection",
    compute_kind="python",
    group_name="ml_fraud_ingest"
)
def fraud_data(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:

    csv_path = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        context.log.error(f"Could not find the file at {csv_path}.")
        raise

    row_count = len(df)
    context.log.info(f"Raw data ingested with {row_count} rows.")
    column_schema = [dg.TableColumn(name, str(dtype)) for name, dtype in df.dtypes.items()]

    return dg.MaterializeResult(
        value=df,
        metadata={
            "preview": dg.MetadataValue.md(df.head().to_markdown() or ""),
            "dagster/row_count": row_count,
            "dagster/column_schema": dg.TableSchema(columns=column_schema)
        }
    )


@dg.asset_check(
    asset="fraud_data",
    description="Checks for null values and negative amounts in fraud_data"
)
def check_fraud_data(
    context: dg.AssetCheckExecutionContext,
    fraud_data: pd.DataFrame
    ) -> dg.AssetCheckResult:
    # Check for nulls
    num_nulls = fraud_data.isnull().sum().sum()

    # Check for negative amounts if 'Amount' column exists
    if 'Amount' in fraud_data.columns:
        negative_amounts = (fraud_data['Amount'] < 0).sum()
    else:
        negative_amounts = 0  # Consider 0 if column doesn't exist

    num_nulls = int(num_nulls)
    negative_amounts = int(negative_amounts)

    passed = (num_nulls == 0) and (negative_amounts == 0)
    metadata = {
        "num_nulls": dg.MetadataValue.int(num_nulls),
        "num_negative_amounts": dg.MetadataValue.int(negative_amounts)
    }

    return dg.AssetCheckResult(
        passed=passed,
        metadata=metadata,
        description=(
            "Passed" if passed else
            f"{'Nulls present. ' if num_nulls > 0 else ''}{'Negative Amounts found.' if negative_amounts > 0 else ''}"
        ),
        asset_key="fraud_data",
    )


@dg.asset(
    description="Preprocess fraud data with feature engineering, normalization, and correlation analysis",
    compute_kind="python",
    group_name="ml_fraud_transform"
)
def preprocessed_fraud_data(
    context: dg.AssetExecutionContext,
    fraud_data: pd.DataFrame
) -> dg.MaterializeResult:
    """Preprocess fraud detection data"""

    df: pd.DataFrame = fraud_data.copy()
    context.log.info(f"Starting preprocessing with {len(df)} rows")

    # Feature engineering: create Time_OfDay column
    df['Time_OfDay'] = df['Time'].apply(categorize_time)
    context.log.info("Created Time_OfDay feature")

    # Data subsampling: balance fraudulent and non-fraudulent transactions
    df_fraud = df[df['Class'] == 1]
    df_non_fraud = df[df['Class'] == 0]
    num_fraud = len(df_fraud)
    df_non_fraud_subsample = df_non_fraud.sample(num_fraud, random_state=42)
    df_subsample = pd.concat([df_fraud, df_non_fraud_subsample])
    df_subsample = df_subsample.sample(frac=1, random_state=42).reset_index(drop=True)

    context.log.info(f"Created balanced subsample with {len(df_subsample)} rows")
    class_distribution = (
        pd.Series(df_subsample['Class']).value_counts()
        if hasattr(df_subsample['Class'], 'value_counts')
        else None
    )
    context.log.info(
        f"Class distribution: {dict(class_distribution) if class_distribution is not None else 'N/A'}"
    )

    # Data normalization
    scaler = StandardScaler()
    columns_to_normalize = df_subsample.drop(['Class', 'Time_OfDay'], axis=1).columns
    df_subsample[columns_to_normalize] = scaler.fit_transform(df_subsample[columns_to_normalize])
    context.log.info("Applied feature normalization")

    # Correlation analysis and feature selection
    correlation_matrix = df_subsample.corr()
    class_correlations = correlation_matrix['Class'].drop('Class')
    sorted_class_correlations = class_correlations.abs().sort_values(ascending=False)
    correlation_threshold = 0.2
    top_features: List[str] = sorted_class_correlations[sorted_class_correlations > correlation_threshold].index.tolist()

    context.log.info(f"Selected {len(top_features)} features with correlation > {correlation_threshold}")
    context.log.info(f"Top features: {top_features}")

    # Output: Use the original df with selected columns and 'Class'
    features_to_output = [col for col in top_features if col in df.columns]
    df_processed = df[features_to_output + ['Class']].copy()

    context.log.info(f"Final processed dataset shape: {df_processed.shape}")

    return dg.MaterializeResult(
        value=df_processed,
        metadata={
            "preview": dg.MetadataValue.md(df_processed.head().to_markdown() or ""),
            "shape": dg.MetadataValue.text(f"{df_processed.shape[0]} rows, {df_processed.shape[1]} columns"),
            "class_distribution": dg.MetadataValue.text(str(dict(df_processed['Class'].value_counts()))),
            "selected_features": dg.MetadataValue.text(", ".join(features_to_output)),
            "correlation_threshold": dg.MetadataValue.float(correlation_threshold)
        }
    )


@dg.asset_check(
    asset="preprocessed_fraud_data",
    description="Checks that preprocessing produced no nulls and selected features"
)
def check_preprocessed_fraud_data(
    context: dg.AssetCheckExecutionContext,
    preprocessed_fraud_data: pd.DataFrame
) -> dg.AssetCheckResult:
    # Check for nulls
    num_nulls = int(preprocessed_fraud_data.isnull().sum().sum())

    # Check that we have features
    has_features = len(preprocessed_fraud_data.columns) > 1

    passed = bool((num_nulls == 0) and has_features)

    metadata = {
        "num_nulls": dg.MetadataValue.int(num_nulls),
        "has_features": dg.MetadataValue.bool(bool(has_features)),
        "num_features": dg.MetadataValue.int(int(len(preprocessed_fraud_data.columns) - 1))
    }

    return dg.AssetCheckResult(
        passed=passed,
        metadata=metadata,
        description=(
            "Passed" if passed else
            f"{'Nulls present. ' if num_nulls > 0 else ''}"
            f"{'No features selected. ' if not has_features else ''}"
        ),
        asset_key="preprocessed_fraud_data",
    )


@dg.asset(
    description="Split preprocessed data into 80% train and 20% test",
    compute_kind="python",
    group_name="ml_fraud_transform"
)
def train_test_split_data(
    context: dg.AssetExecutionContext,
    preprocessed_fraud_data: pd.DataFrame
) -> dg.MaterializeResult:
    """Split the preprocessed fraud data into train and test sets"""

    df: pd.DataFrame = preprocessed_fraud_data
    context.log.info(f"Splitting data with shape: {df.shape}")

    # Prepare features and target
    X = df.drop(['Class'], axis=1).to_numpy()
    y = df['Class'].to_numpy()

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Log split information
    context.log.info(f"Train set: {X_train.shape[0]} samples")
    context.log.info(f"Test set: {X_test.shape[0]} samples")

    # Log class distributions
    train_class_dist = np.bincount(y_train)
    test_class_dist = np.bincount(y_test)

    context.log.info(f"Train class distribution: {dict(zip(range(len(train_class_dist)), train_class_dist))}")
    context.log.info(f"Test class distribution: {dict(zip(range(len(test_class_dist)), test_class_dist))}")

    split_data = {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': df.drop(['Class'], axis=1).columns.tolist()
    }

    return dg.MaterializeResult(
        value=split_data,
        metadata={
            "train_shape": dg.MetadataValue.text(f"{X_train.shape[0]} samples, {X_train.shape[1]} features"),
            "test_shape": dg.MetadataValue.text(f"{X_test.shape[0]} samples, {X_test.shape[1]} features"),
            "train_class_dist": dg.MetadataValue.text(str(dict(zip(range(len(train_class_dist)), train_class_dist)))),
            "test_class_dist": dg.MetadataValue.text(str(dict(zip(range(len(test_class_dist)), test_class_dist)))),
            "split_ratio": dg.MetadataValue.text("80/20"),
            "stratified": dg.MetadataValue.bool(True)
        }
    )


@dg.asset_check(
    asset="train_test_split_data",
    description="Verifies that stratification preserved class balance in train/test split"
)
def check_train_test_split_data(
    context: dg.AssetCheckExecutionContext,
    train_test_split_data: Dict[str, Any]
) -> dg.AssetCheckResult:
    y_train = train_test_split_data['y_train']
    y_test = train_test_split_data['y_test']

    # Calculate class distributions
    train_class_dist = np.bincount(y_train)
    test_class_dist = np.bincount(y_test)

    # Check that both sets have both classes
    has_both_classes_train = len(train_class_dist) == 2
    has_both_classes_test = len(test_class_dist) == 2

    # Check that class ratios are similar (stratification worked)
    if has_both_classes_train and has_both_classes_test:
        train_ratio = train_class_dist[1] / train_class_dist[0]
        test_ratio = test_class_dist[1] / test_class_dist[0]
        ratio_similarity = abs(train_ratio - test_ratio) < 0.1  # Within 10%
    else:
        ratio_similarity = False

    passed = bool(has_both_classes_train and has_both_classes_test and ratio_similarity)

    metadata = {
        "train_both_classes": dg.MetadataValue.bool(bool(has_both_classes_train)),
        "test_both_classes": dg.MetadataValue.bool(bool(has_both_classes_test)),
        "ratio_similarity": dg.MetadataValue.bool(bool(ratio_similarity)),
        "train_ratio": dg.MetadataValue.float(
            float(train_class_dist[1] / train_class_dist[0] if has_both_classes_train else 0.0)
        ),
        "test_ratio": dg.MetadataValue.float(
            float(test_class_dist[1] / test_class_dist[0] if has_both_classes_test else 0.0)
        )
    }

    return dg.AssetCheckResult(
        passed=passed,
        metadata=metadata,
        description=(
            "Passed" if passed else
            f"{'Train set missing classes. ' if not has_both_classes_train else ''}"
            f"{'Test set missing classes. ' if not has_both_classes_test else ''}"
            f"{'Class ratios too different. ' if not ratio_similarity else ''}"
        ),
        asset_key="train_test_split_data",
    )


@dg.asset(
    description="Train RandomForest model with 3-fold CV, hyperparameter tuning",
    compute_kind="python",
    group_name="ml_fraud_model",
    resource_defs={"mlflow_fraud": mlflow_resource}
)
def trained_fraud_model(
    context: dg.AssetExecutionContext,
    train_test_split_data: Dict[str, Any]
) -> dg.MaterializeResult:
    """Train RandomForest classifier with hyperparameter tuning"""

    mlflow_client = context.resources.mlflow_fraud
    mlflow.set_experiment("fraud_detection_ml")

    active_run = mlflow.active_run()
    if active_run:
        context.log.info("Found active MLflow run, ending it before starting new one")
        mlflow.end_run()
    with mlflow.start_run(run_name=f"fraud_detection_training_{context.run_id[:8]}"):
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("task", "fraud_detection")

        X_train = train_test_split_data['X_train']
        y_train = train_test_split_data['y_train']
        feature_names = train_test_split_data['feature_names']
        context.log.info(f"Training model with {X_train.shape[0]} samples, {X_train.shape[1]} features")

        # Apply RandomOverSampler to training data
        # oversampler = RandomOverSampler(sampling_strategy={1: 800, 0: 22390}, random_state=42)
        # X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

        X_train_resampled = X_train
        y_train_resampled = y_train

        # context.log.info(f"After oversampling: {X_train_resampled.shape[0]} samples")
        # context.log.info(f"Class distribution after oversampling: {np.unique(y_train_resampled, return_counts=True)}")

        # Hyperparameter tuning with cross-validation
        n_estimators_values = [50, 100, 200, 300, 500]
        best_score = 0
        best_params = None
        best_model = None

        for n_estimators in n_estimators_values:
            context.log.info(f"Testing n_estimators={n_estimators}")

            # Start nested run for each hyperparameter
            with mlflow.start_run(nested=True, run_name=f"n_estimators_{n_estimators}"):
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("random_state", 42)
                mlflow.log_param("cv_folds", 3)

                # Perform 3-fold cross-validation
                rf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=42
                )

                # Manual CV to log each fold
                kfold = KFold(n_splits=3, shuffle=True, random_state=42)
                cv_scores = []

                for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train_resampled)):
                    X_fold_train, X_fold_val = X_train_resampled[train_idx], X_train_resampled[val_idx]
                    y_fold_train, y_fold_val = y_train_resampled[train_idx], y_train_resampled[val_idx]

                    rf.fit(X_fold_train, y_fold_train)
                    y_fold_pred = rf.predict(X_fold_val)
                    fold_recall = float(recall_score(y_fold_val, y_fold_pred))
                    cv_scores.append(fold_recall)

                    # Log fold results
                    mlflow.log_metric(f"fold_{fold + 1}_recall", fold_recall)

                mean_cv_score = float(np.mean(cv_scores))
                std_cv_score = float(np.std(cv_scores))

                mlflow.log_metric("mean_cv_recall", mean_cv_score)
                mlflow.log_metric("std_cv_recall", std_cv_score)

                context.log.info(f"n_estimators={n_estimators}: CV Recall = {mean_cv_score:.4f} ± {std_cv_score:.4f}")

                # Track best model
                if mean_cv_score > best_score:
                    best_score = mean_cv_score
                    best_params = {"n_estimators": n_estimators}
                    # Train final model on full resampled data
                    best_model = RandomForestClassifier(
                        n_estimators=n_estimators,
                        random_state=42
                    )
                    best_model.fit(X_train_resampled, y_train_resampled)

        # Log best parameters and model
        if best_params is not None:
            mlflow.log_params(best_params)
        mlflow.log_metric("best_cv_recall", float(best_score))

        # Log feature importance
        if best_model is not None:
            feature_importance = best_model.feature_importances_
            for i, (feature, importance) in enumerate(zip(feature_names, feature_importance)):
                mlflow.log_metric(f"feature_importance_{feature}", float(importance))

            # Log model
            import mlflow.sklearn
            mlflow.sklearn.log_model(
                best_model,
                "model",
                signature=mlflow.models.infer_signature(
                    X_train_resampled[:10],  # Sample for signature
                    best_model.predict(X_train_resampled[:10])
                )
            )

        if best_params is not None:
            context.log.info(f"Best model: n_estimators={best_params['n_estimators']}, CV Recall={best_score:.4f}")

        if best_model is None:
            raise ValueError("No model was trained successfully")

        return dg.MaterializeResult(
            value=best_model,
            metadata={
                "best_params": dg.MetadataValue.text(str(best_params)),
                "best_cv_recall": dg.MetadataValue.float(float(best_score)),
                "n_estimators": dg.MetadataValue.int(best_params['n_estimators'] if best_params else 0),
                "training_samples": dg.MetadataValue.int(X_train_resampled.shape[0]),
                "num_features": dg.MetadataValue.int(X_train_resampled.shape[1]),
                "feature_names": dg.MetadataValue.text(", ".join(feature_names))
            }
        )


@dg.asset_check(
    asset="trained_fraud_model",
    description="Verifies that the model was trained successfully with reasonable performance"
)
def check_trained_fraud_model(
    context: dg.AssetCheckExecutionContext,
    trained_fraud_model: RandomForestClassifier
) -> dg.AssetCheckResult:
    # Check that model is a RandomForest
    is_random_forest = hasattr(trained_fraud_model, 'estimators_')

    # Check that model has been fitted (has feature_importances_)
    is_fitted = hasattr(trained_fraud_model, 'feature_importances_')

    # Check that we have reasonable number of trees
    if is_fitted:
        n_estimators = trained_fraud_model.n_estimators
        reasonable_trees = 50 <= n_estimators <= 500
    else:
        n_estimators = 0
        reasonable_trees = False

    passed = bool(is_random_forest and is_fitted and reasonable_trees)

    metadata = {
        "is_random_forest": dg.MetadataValue.bool(bool(is_random_forest)),
        "is_fitted": dg.MetadataValue.bool(bool(is_fitted)),
        "n_estimators": dg.MetadataValue.int(int(n_estimators)),
        "reasonable_trees": dg.MetadataValue.bool(bool(reasonable_trees))
    }

    return dg.AssetCheckResult(
        passed=passed,
        metadata=metadata,
        description=(
            "Passed" if passed else
            f"{'Not a RandomForest. ' if not is_random_forest else ''}"
            f"{'Model not fitted. ' if not is_fitted else ''}"
            f"{'Unreasonable number of trees. ' if not reasonable_trees else ''}"
        ),
        asset_key="trained_fraud_model",
    )


@dg.asset(
    description="Evaluate model on test set with confusion matrix and MLflow image logging",
    compute_kind="python",
    group_name="ml_fraud_model",
    resource_defs={"mlflow_fraud": mlflow_resource}
)
def model_evaluation(
    context: dg.AssetExecutionContext,
    trained_fraud_model: RandomForestClassifier,
    train_test_split_data: Dict[str, Any]
) -> dg.MaterializeResult:
    """Evaluate the trained model on test set and log results to MLflow"""

    import mlflow

    mlflow_client = context.resources.mlflow_fraud
    mlflow.set_experiment("fraud_detection_ml")

    active_run = mlflow.active_run()
    if active_run:
        context.log.info("Found active MLflow run, ending it before starting new one")
        mlflow.end_run()
    with mlflow.start_run(run_name=f"fraud_detection_evaluation_{context.run_id[:8]}"):
        mlflow.set_tag("model_type", "RandomForest")
        mlflow.set_tag("task", "fraud_detection")
        mlflow.set_tag("phase", "evaluation")

        X_test = train_test_split_data['X_test']
        y_test = train_test_split_data['y_test']

        context.log.info(f"Evaluating model on {X_test.shape[0]} test samples")

        # Make predictions
        y_pred = trained_fraud_model.predict(X_test)
        y_pred_proba = trained_fraud_model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        recall = float(recall_score(y_test, y_pred))
        precision = float(precision_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred))
        roc_auc = float(roc_auc_score(y_test, y_pred_proba))

        context.log.info(f"Test Recall: {recall:.4f}")
        context.log.info(f"Test Precision: {precision:.4f}")
        context.log.info(f"Test F1: {f1:.4f}")
        context.log.info(f"Test ROC-AUC: {roc_auc:.4f}")

        # Log metrics to MLflow
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_f1", f1)
        mlflow.log_metric("test_roc_auc", roc_auc)

        # Create confusion matrix
        cm = confusion_matrix(y_test, y_pred)

        # Create and save confusion matrix plot
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            plt.figure(figsize=(8, 6))
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=['Not Fraud', 'Fraud']
            )
            disp.plot(cmap='Blues', values_format='d')
            plt.title('Fraud Detection Model - Confusion Matrix')
            plt.tight_layout()
            plt.savefig(tmp_file.name, dpi=300, bbox_inches='tight')
            plt.close()

            # Log image to MLflow
            mlflow.log_artifact(tmp_file.name, "confusion_matrix")

            # Clean up temporary file
            os.unlink(tmp_file.name)

        context.log.info("Confusion matrix saved and logged to MLflow")

        # Log classification report
        class_report = classification_report(y_test, y_pred, target_names=['Not Fraud', 'Fraud'])
        context.log.info(f"Classification Report:\n{class_report}")

        # Log detailed metrics to MLflow
        mlflow.log_text(str(class_report), "classification_report.txt")

        evaluation_results = {
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm.tolist(),
            'classification_report': class_report
        }

        return dg.MaterializeResult(
            value=evaluation_results,
            metadata={
                "test_recall": dg.MetadataValue.float(recall),
                "test_precision": dg.MetadataValue.float(precision),
                "test_f1": dg.MetadataValue.float(f1),
                "test_roc_auc": dg.MetadataValue.float(roc_auc),
                "test_samples": dg.MetadataValue.int(len(y_test)),
                "confusion_matrix": dg.MetadataValue.text(f"TN:{cm[0, 0]}, FP:{cm[0, 1]}, FN:{cm[1, 0]}, TP:{cm[1, 1]}")
            }
        )


@dg.asset_check(
    asset="model_evaluation",
    description="Verifies that model evaluation produced reasonable recall score"
)
def check_model_evaluation(
    context: dg.AssetCheckExecutionContext,
    model_evaluation: Dict[str, Any]
) -> dg.AssetCheckResult:
    recall = model_evaluation['recall']
    f1 = model_evaluation['f1']
    roc_auc = model_evaluation['roc_auc']

    # Check that recall meets minimum threshold
    recall_threshold = 0.7
    recall_acceptable = recall >= recall_threshold

    # Check that F1 and ROC-AUC are reasonable
    f1_reasonable = 0.0 <= f1 <= 1.0
    roc_auc_reasonable = 0.0 <= roc_auc <= 1.0

    passed = bool(recall_acceptable and f1_reasonable and roc_auc_reasonable)

    metadata = {
        "recall": dg.MetadataValue.float(float(recall)),
        "recall_threshold": dg.MetadataValue.float(float(recall_threshold)),
        "recall_acceptable": dg.MetadataValue.bool(bool(recall_acceptable)),
        "f1_score": dg.MetadataValue.float(float(f1)),
        "roc_auc": dg.MetadataValue.float(float(roc_auc))
    }

    return dg.AssetCheckResult(
        passed=passed,
        metadata=metadata,
        description=(
            "Passed" if passed else
            f"{'Recall below threshold. ' if not recall_acceptable else ''}"
            f"{'Invalid F1 score. ' if not f1_reasonable else ''}"
            f"{'Invalid ROC-AUC. ' if not roc_auc_reasonable else ''}"
        ),
        asset_key="model_evaluation",
    )


def start_model_server(model, port: int = 5001):
    """Start a Flask server to serve the model (runs in background)"""
    from flask import Flask, jsonify, request

    app = Flask(__name__)

    @app.route('/predict', methods=['POST'])
    def predict():
        try:
            data = request.get_json()
            if 'features' not in data:
                return jsonify({'error': 'features field required'}), 400

            features = np.array(data['features']).reshape(1, -1)
            prediction = model.predict(features)[0]
            probability = model.predict_proba(features)[0]

            return jsonify({
                'prediction': int(prediction),
                'probability_fraud': float(probability[1]),
                'probability_not_fraud': float(probability[0])
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({'status': 'healthy', 'model_type': 'RandomForest'})

    # Run in a separate thread
    def run_server():
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    return server_thread


@dg.asset(
    description="Post recall metric to Slack and start model server on port 5001",
    compute_kind="python",
    group_name="ml_fraud_deploy",
    resource_defs={"slack": slack_resource, "mlflow_fraud": mlflow_resource}
)
def notify_and_serve(
    context: dg.AssetExecutionContext,
    model_evaluation: Dict[str, Any],
    trained_fraud_model: RandomForestClassifier
) -> dg.MaterializeResult:
    """Send notification to Slack and start model serving server"""

    recall = model_evaluation['recall']
    precision = model_evaluation['precision']
    f1 = model_evaluation['f1']
    roc_auc = model_evaluation['roc_auc']

    context.log.info(f"Recall: {recall:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")

    # Send notification to Slack
    try:
        slack = context.resources.slack
        github_user = os.environ.get("GITHUB_USER", "unknown")

        message = f"""🎯 Fraud Detection Model Training Complete!

        📊 **Model Performance:**
        • Recall: {recall:.4f}
        • Precision: {precision:.4f}
        • F1 Score: {f1:.4f}
        • ROC-AUC: {roc_auc:.4f}

        👤 Trained by: {github_user}
        🔧 Model: RandomForest Classifier
        📈 Experiment: fraud_detection_ml

        {'✅ Model meets recall threshold!' if recall >= 0.7 else '⚠️ Model below recall threshold'}
        """

        response = slack.get_client().chat_postMessage(
            channel='aims_course_october2025',
            text=message
        )
        context.log.info(f"Slack API response: {response}")
        context.log.info("Successfully posted notification to Slack")

    except Exception as e:
        context.log.error(f"Failed to post to Slack: {e}")

    # Save model using MLflow
    try:
        import mlflow

        mlflow.set_tracking_uri(
            f"sqlite:///{os.path.join(os.path.abspath(os.getenv('DAGSTER_HOME', '.')),
            'mlflow_fraud_detection.db')}"
            )
        mlflow.set_experiment("fraud_detection_ml")

        active_run = mlflow.active_run()
        if active_run:
            context.log.info("Found active MLflow run, ending it before starting new one")
            mlflow.end_run()
        with mlflow.start_run(run_name=f"fraud_detection_serving_{context.run_id[:8]}"):
            mlflow.set_tag("model_type", "RandomForest")
            mlflow.set_tag("task", "fraud_detection")
            mlflow.set_tag("phase", "serving")

            # Log final metrics
            mlflow.log_metric("final_recall", recall)
            mlflow.log_metric("final_precision", precision)
            mlflow.log_metric("final_f1", f1)
            mlflow.log_metric("final_roc_auc", roc_auc)

            # Save model for serving
            model_path = f"./models/fraud_detection_model_{context.run_id[:8]}"
            import mlflow.sklearn
            mlflow.sklearn.save_model(
                trained_fraud_model,
                model_path
            )
            context.log.info(f"Model saved to {model_path}")

    except Exception as e:
        context.log.error(f"Failed to save model: {e}")

    # Start model server
    port = 5001
    try:
        server_thread = start_model_server(trained_fraud_model, port)

        # Give server a moment to start
        time.sleep(2)

        context.log.info(f"Model server started on port {port}")
        context.log.info(f"Health check: http://localhost:{port}/health")
        context.log.info(f"Predict endpoint: http://localhost:{port}/predict")

    except Exception as e:
        context.log.error(f"Failed to start model server: {e}")
        server_thread = None

    return dg.MaterializeResult(
        value={
            'recall': recall,
            'precision': precision,
            'f1': f1,
            'roc_auc': roc_auc,
            'server_port': port,
            'server_running': server_thread is not None
        },
        metadata={
            "final_recall": dg.MetadataValue.float(recall),
            "final_precision": dg.MetadataValue.float(precision),
            "final_f1": dg.MetadataValue.float(f1),
            "final_roc_auc": dg.MetadataValue.float(roc_auc),
            "server_port": dg.MetadataValue.int(port),
            "server_status": dg.MetadataValue.text("running" if server_thread else "failed"),
            "slack_notification": dg.MetadataValue.bool(True),
            "model_saved": dg.MetadataValue.bool(True)
        }
    )


@dg.asset_check(
    asset="notify_and_serve",
    description="Verifies that notification was sent and server is running"
)
def check_notify_and_serve(
    context: dg.AssetCheckExecutionContext,
    notify_and_serve: Dict[str, Any]
) -> dg.AssetCheckResult:
    server_running = notify_and_serve['server_running']
    recall = notify_and_serve['recall']

    # Check that server is running
    server_ok = server_running

    # Check that recall is reasonable
    recall_ok = 0.0 <= recall <= 1.0

    passed = bool(server_ok and recall_ok)

    metadata = {
        "server_running": dg.MetadataValue.bool(bool(server_running)),
        "recall": dg.MetadataValue.float(float(recall)),
        "recall_ok": dg.MetadataValue.bool(bool(recall_ok))
    }

    return dg.AssetCheckResult(
        passed=passed,
        metadata=metadata,
        description=(
            "Passed" if passed else
            f"{'Server not running. ' if not server_ok else ''}"
            f"{'Invalid recall score. ' if not recall_ok else ''}"
        ),
        asset_key="notify_and_serve",
    )
