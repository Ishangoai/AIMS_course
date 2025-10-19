"""
Utility functions for fraud detection Gradio app.
Handles model loading, predictions, and visualizations.
"""

import os
import tempfile
from typing import Any, List, Optional, Tuple, cast

import matplotlib.pyplot as plt
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd


class FraudDetectionModel:
    """Wrapper for fraud detection model with robust predict_proba support."""

    def __init__(self, model_name: str = "fraud_detection_rf",
                 model_stage: str = "latest",
                 db_path: str = "mlflow_artifacts/mlflow_fraud_tracking.db"):
        """
        Initialize the fraud detection model.

        Args:
            model_name: Name of the registered model in MLflow
            model_stage: Model stage/version to load
            db_path: Path to MLflow tracking database
        """
        self.model_name = model_name
        self.model_stage = model_stage
        self.db_path = os.path.abspath(db_path)

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(f"sqlite:///{self.db_path}")

        # Load model
        model_uri = f"models:/{model_name}/{model_stage}"
        self.model = mlflow.pyfunc.load_model(model_uri)

        # Try to get the underlying sklearn model for predict_proba
        self._sklearn_model = None
        if hasattr(self.model, '_model_impl'):
            self._sklearn_model = getattr(self.model._model_impl, 'sklearn_model', None)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels."""
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        Falls back to binary predictions if predict_proba not available.
        """
        # Try sklearn model first
        if self._sklearn_model is not None and hasattr(self._sklearn_model, 'predict_proba'):
            return self._sklearn_model.predict_proba(X)

        # Try PyFuncModel directly
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)  # pyright: ignore

        # Fallback: use predict and create pseudo-probabilities
        predictions = self.predict(X)
        n_samples = len(predictions)
        proba = np.zeros((n_samples, 2))

        for i, pred in enumerate(predictions):
            if pred == 1:
                proba[i] = [0.05, 0.95]  # High confidence fraud
            else:
                proba[i] = [0.95, 0.05]  # High confidence non-fraud

        return proba


def load_default_dataset(data_url: str) -> Tuple[pd.DataFrame, dict]:
    """
    Load default dataset and compute feature ranges.

    Args:
        data_url: URL to the default dataset

    Returns:
        Tuple of (dataframe, feature_ranges_dict)
    """
    df = pd.read_csv(data_url)

    # Define feature names (exclude 'Class' label)
    # This still loads all features since default dataset has all 30,
    # but downstream we use only the selected features.
    feature_names = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                     'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                     'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

    # Compute ranges for sliders for all features in the dataset (if needed)
    feature_ranges = {
        col: (float(df[col].min()), float(df[col].max()))
        for col in feature_names if col in df.columns
    }

    return df, feature_ranges


def predict_single_transaction(model: FraudDetectionModel,
                               slider_values: List[float],
                               feature_names: List[str]) -> Tuple:
    """
    Predict fraud for a single transaction.

    Args:
        model: FraudDetectionModel instance
        slider_values: List of feature values from sliders
        feature_names: List of feature names

    Returns:
        Tuple of (matplotlib figure, prediction label)
    """
    # Create input dataframe
    input_data = {name: val for name, val in zip(feature_names, slider_values)}
    input_df = pd.DataFrame([input_data]).astype(float)

    # Get prediction and probability
    pred = model.predict(input_df)[0]
    prob_array = model.predict_proba(input_df)[0]
    fraud_prob = prob_array[1]  # Probability of fraud (class 1)

    # Determine prediction text and color
    prediction_text = "FRAUD" if pred == 1 else "Not Fraud"
    color = "#F44336" if pred == 1 else "#4CAF50"

    # Create probability gauge
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.barh([0], [fraud_prob * 100], color=color, height=0.4)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Fraud Probability (%)", fontsize=12)
    ax.set_title(f"{prediction_text} - {fraud_prob * 100:.1f}%",
                fontsize=14, fontweight='bold', color=color)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.tight_layout()

    return fig, prediction_text


def predict_batch_transactions(model: FraudDetectionModel,
                               file: Optional[object],
                               use_default: bool,
                               default_df: pd.DataFrame,
                               feature_names: List[str]) -> Tuple:
    """
    Predict fraud for a batch of transactions.

    Args:
        model: FraudDetectionModel instance
        file: Uploaded file object (or None)
        use_default: Whether to use default dataset
        default_df: Default dataset dataframe
        feature_names: List of feature names (should be the 10 selected features)

    Returns:
        Tuple of (5 matplotlib figures, csv_file_path)
    """
    # Load data
    if use_default:
        df = default_df.copy()
    elif file is not None:
        # FIX: Cast the file object to 'Any' to satisfy the type checker
        # that it has a .name attribute (which it does at runtime via Gradio/tempfile).
        file_path = cast(Any, file).name
        df = pd.read_csv(file_path)
    else:
        return None, None, None, None, None, None

    for col in feature_names:
        df[col] = df[col].astype(float)

    # Select features for prediction
    features_to_predict = df[feature_names]
    features_to_predict = cast(pd.DataFrame, features_to_predict)

    # Make predictions
    preds = model.predict(features_to_predict)
    probs = model.predict_proba(features_to_predict)
    fraud_prob = probs[:, 1]

    # Add predictions to dataframe
    df["Fraud_Prediction"] = preds
    df["Fraud_Probability"] = fraud_prob
    df["Fraud_Prediction_Label"] = df["Fraud_Prediction"].map({0: "Non-Fraud", 1: "Fraud"})  # pyright: ignore [reportArgumentType]

    # Create visualizations
    figures = []

    # Figure 1: Amount distribution by prediction (if 'Amount' exists in data)
    if "Amount" in df.columns:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        for label in df["Fraud_Prediction_Label"].unique():
            subset = df[df["Fraud_Prediction_Label"] == label]
            ax1.hist(subset["Amount"], bins=30, alpha=0.6, label=label, edgecolor='black')
        ax1.set_title("Distribution of Transaction Amounts by Prediction", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Amount", fontsize=12)
        ax1.set_ylabel("Count", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)
        plt.tight_layout()
        figures.append(fig1)
    else:
        figures.append(plt.figure())  # placeholder empty figure

    # Figure 2: Fraud proportion pie chart
    fig2, ax2 = plt.subplots(figsize=(8, 8))
    counts = df["Fraud_Prediction_Label"].value_counts()
    colors = ["#4CAF50", "#F44336"]
    ax2.pie(counts, labels=list(counts.index), autopct='%1.1f%%', colors=colors,
            textprops={'fontsize': 12, 'fontweight': 'bold'}, startangle=90)
    ax2.set_title("Fraud vs Non-Fraud Proportion", fontsize=14, fontweight='bold')
    plt.tight_layout()
    figures.append(fig2)

    # Figures 3-5: Feature distributions for first 3 features in the selected list
    features_to_plot = feature_names[:3]
    for feature in features_to_plot:
        fig, ax = plt.subplots(figsize=(10, 6))
        for label in df["Fraud_Prediction_Label"].unique():
            subset = df[df["Fraud_Prediction_Label"] == label]
            ax.hist(subset[feature], bins=30, alpha=0.6, label=label, edgecolor='black')
        ax.set_title(f"Distribution of {feature} by Prediction", fontsize=14, fontweight='bold')
        ax.set_xlabel(feature, fontsize=12)
        ax.set_ylabel("Count", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        figures.append(fig)

    # Ensure exactly 5 figures (pad with empty if needed)
    while len(figures) < 5:
        figures.append(plt.figure())

    # Save predictions to CSV
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode='w') as tmp_file:
        df.to_csv(tmp_file.name, index=False)
        csv_path = tmp_file.name

    return (*figures, csv_path)


def create_summary_statistics(df: pd.DataFrame) -> str:
    """
    Create a markdown summary of batch prediction results.

    Args:
        df: Dataframe with predictions

    Returns:
        Markdown formatted string
    """
    total = len(df)
    fraud_count = (df["Fraud_Prediction"] == 1).sum()
    non_fraud_count = total - fraud_count
    fraud_pct = (fraud_count / total * 100) if total > 0 else 0

    avg_fraud_prob = df[df["Fraud_Prediction"] == 1]["Fraud_Probability"].mean() if fraud_count > 0 else 0
    avg_amount_fraud = df[df["Fraud_Prediction"] == 1]["Amount"].mean() if fraud_count > 0 else 0
    avg_amount_legit = df[df["Fraud_Prediction"] == 0]["Amount"].mean() if non_fraud_count > 0 else 0

    summary = f"""
    ## 📊 Batch Prediction Summary

    - **Total Transactions**: {total:,}
    - **Predicted Fraud**: {fraud_count:,} ({fraud_pct:.2f}%)
    - **Predicted Non-Fraud**: {non_fraud_count:,} ({100 - fraud_pct:.2f}%)

    ### Average Values
    - **Avg Fraud Probability (Fraud Cases)**: {avg_fraud_prob:.2%}
    - **Avg Amount (Fraud)**: ${avg_amount_fraud:,.2f}
    - **Avg Amount (Non-Fraud)**: ${avg_amount_legit:,.2f}
    """

    return summary
