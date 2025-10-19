"""
Fraud Detection Gradio Application

This app provides a user-friendly interface for fraud detection using a trained RandomForest model.
Users can input transaction features and get real-time fraud predictions.
"""

import os
import pickle
from typing import Dict, Tuple

import gradio as gr
import numpy as np
import pandas as pd

# ============================================================================
# MODEL LOADING
# ============================================================================


def load_model(model_path: str) -> object:
    """Load the trained RandomForest model from pickle file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

FEATURE_NAMES = ["V14", "V4", "V11", "V12", "V10", "V16", "V3", "V9", "V17", "V2", "V7", "V18", "V1", "V6", "V5", "V19"]

FEATURE_DESCRIPTIONS = {
    "V1": "PCA Component 1",
    "V2": "PCA Component 2",
    "V3": "PCA Component 3",
    "V4": "PCA Component 4",
    "V5": "PCA Component 5",
    "V6": "PCA Component 6",
    "V7": "PCA Component 7",
    "V9": "PCA Component 9",
    "V10": "PCA Component 10",
    "V11": "PCA Component 11",
    "V12": "PCA Component 12",
    "V14": "PCA Component 14",
    "V16": "PCA Component 16",
    "V17": "PCA Component 17",
    "V18": "PCA Component 18",
    "V19": "PCA Component 19",
}


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================


def predict_fraud(model: object, features: np.ndarray) -> Tuple[str, float, Dict]:
    """
    Make fraud prediction using the loaded model.

    Args:
        model: Trained RandomForest model
        features: Feature array (16 features)

    Returns:
        Tuple containing (prediction_label, fraud_probability, details_dict)
    """
    try:
        # Ensure features are in correct shape
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)[0]  # type: ignore
        probabilities = model.predict_proba(features)[0]  # type: ignore

        # Extract probabilities
        not_fraud_prob = probabilities[0]
        fraud_prob = probabilities[1]

        # Determine label
        label = "🚨 FRAUD DETECTED" if prediction == 1 else "✅ LEGITIMATE"

        # Create details dictionary
        details = {
            "prediction": int(prediction),
            "fraud_probability": float(fraud_prob),
            "legitimate_probability": float(not_fraud_prob),
            "confidence": float(max(probabilities)),
        }

        return label, fraud_prob, details

    except Exception as e:
        return f"❌ Error: {str(e)}", 0.0, {"error": str(e)}


def process_csv_upload(file_path: str, model: object) -> pd.DataFrame:
    """
    Process uploaded CSV file and make predictions for all rows.

    Args:
        file_path: Path to uploaded CSV file
        model: Trained model

    Returns:
        DataFrame with predictions added
    """
    try:
        df = pd.read_csv(file_path)

        # Extract features (assuming CSV has the 16 required features)
        feature_cols = [col for col in FEATURE_NAMES if col in df.columns]
        if len(feature_cols) != 16:
            return None, f"❌ CSV must contain all 16 features. Found: {len(feature_cols)}"  # type: ignore

        X = df[feature_cols].values
        predictions = model.predict(X)  # type: ignore
        probabilities = model.predict_proba(X)[:, 1]    # type: ignore

        # Add results to dataframe
        df["fraud_prediction"] = ["FRAUD" if p == 1 else "LEGITIMATE" for p in predictions]
        df["fraud_probability"] = probabilities

        return df, "✅ Predictions completed successfully"  # type: ignore

    except Exception as e:
        return None, f"❌ Error processing CSV: {str(e)}"   # type: ignore


# ============================================================================
# GRADIO INTERFACE
# ============================================================================


def create_fraud_detection_app(model_path: str) -> gr.Blocks:
    """
    Create and configure the Gradio fraud detection interface.

    Args:
        model_path: Path to the trained model pickle file

    Returns:
        Configured Gradio Blocks interface
    """

    # Load model
    model = load_model(model_path)

    with gr.Blocks(title="Fraud Detection System", theme="dark") as demo:
        # ====================================================================
        # HEADER
        # ====================================================================

        gr.Markdown("# 🔍 Credit Card Fraud Detection System")
        gr.Markdown(
            """
            This application uses a trained RandomForest model to detect fraudulent transactions.
            **How to use:**
            1. Enter transaction feature values (V1-V19 PCA components)
            2. Click "Predict" to get real-time fraud prediction
            3. Or upload a CSV file for batch predictions
            **Features:** 16 pre-selected PCA components optimized for fraud detection
            """
        )

        # ====================================================================
        # SECTION 1: SINGLE PREDICTION
        # ====================================================================

        with gr.Tab("Single Transaction Prediction"):
            gr.Markdown("### Enter Transaction Features")

            # Create input fields for each feature
            feature_inputs = []
            with gr.Group():
                # Arrange features in 2 columns
                for i in range(0, len(FEATURE_NAMES), 2):
                    with gr.Row():
                        # First feature
                        feature_name = FEATURE_NAMES[i]
                        description = FEATURE_DESCRIPTIONS.get(feature_name, "")
                        slider = gr.Slider(
                            minimum=-10, maximum=10, step=0.01, value=0, label=f"{feature_name}: {description}", scale=1
                        )
                        feature_inputs.append(slider)

                        # Second feature (if exists)
                        if i + 1 < len(FEATURE_NAMES):
                            feature_name = FEATURE_NAMES[i + 1]
                            description = FEATURE_DESCRIPTIONS.get(feature_name, "")
                            slider = gr.Slider(
                                minimum=-10,
                                maximum=10,
                                step=0.01,
                                value=0,
                                label=f"{feature_name}: {description}",
                                scale=1,
                            )
                            feature_inputs.append(slider)

            # Prediction button and outputs
            with gr.Group():
                predict_btn = gr.Button("🔮 Predict Fraud Status", size="lg", variant="primary")

            with gr.Group():
                prediction_output = gr.Textbox(label="Prediction Result", interactive=False, lines=1, container=True)

                fraud_prob_output = gr.Number(label="Fraud Probability (%)", interactive=False, precision=2)

                details_output = gr.JSON(label="Detailed Results")

            # Prediction function
            def predict_single(*feature_values):
                features = np.array(feature_values, dtype=np.float32)
                label, fraud_prob, details = predict_fraud(model, features)

                # Convert to percentage
                fraud_prob_percent = fraud_prob * 100

                return label, fraud_prob_percent, details

            predict_btn.click(
                fn=predict_single, inputs=feature_inputs, outputs=[prediction_output, fraud_prob_output, details_output]
            )

            # Example transactions button
            def load_example_1():
                example_data = [
                    -0.046468703,
                    -3.729531338,
                    -1.597280924,
                    0.390819598,
                    -2.753013935,
                    0.055103291,
                    1.333622859,
                    1.964857828,
                    -0.716117978,
                    -0.022656245,
                    -0.003951346,
                    0.760979148,
                    -1.570066573,
                    -0.884352136,
                    -0.546070188,
                    -0.227290718,
                ]
                return example_data

            def load_example_2():
                example_data = [
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                    10,
                ]
                return example_data

            with gr.Row():
                example_btn_1 = gr.Button("📋 Load Example 1", size="sm")
                example_btn_2 = gr.Button("📋 Load Example 2", size="sm")

            example_btn_1.click(fn=load_example_1, outputs=feature_inputs)

            example_btn_2.click(fn=load_example_2, outputs=feature_inputs)

        # ====================================================================
        # SECTION 2: BATCH PREDICTION
        # ====================================================================

        with gr.Tab("Batch Prediction (CSV)"):
            gr.Markdown("### Upload CSV File for Batch Predictions")
            gr.Markdown(
                """
                Upload a CSV file containing transactions with all 16 required features (V1-V19).
                The app will generate fraud predictions for all rows.
                """
            )

            with gr.Group():
                file_input = gr.File(label="Upload CSV File", file_types=[".csv"], file_count="single")
                batch_predict_btn = gr.Button("🚀 Process Batch", size="lg", variant="primary")

            with gr.Group():
                batch_output = gr.Dataframe(label="Predictions", interactive=False)
                status_output = gr.Textbox(label="Status", interactive=False, lines=1)

            def process_batch(file):
                if file is None:
                    return None, "❌ Please upload a CSV file"

                df, message = process_csv_upload(file.name, model)
                return df, message

            batch_predict_btn.click(fn=process_batch, inputs=file_input, outputs=[batch_output, status_output])

        # ====================================================================
        # SECTION 3: MODEL INFORMATION
        # ====================================================================

        with gr.Tab("Model Information"):
            gr.Markdown("### Model Details")

            model_info = f"""
            **Model Type:** RandomForest Classifier

            **Features Used:** {len(FEATURE_NAMES)} PCA-transformed features

            **Feature List:**
            - {", ".join(FEATURE_NAMES)}

            **Input Range:** Each feature should be approximately between -10 and 10

            **Output:**
            - Binary classification (Fraud / Legitimate)
            - Fraud probability score (0-100%)

            **Typical Performance:**
            - Precision: High
            - Recall: Optimized for fraud detection
            - ROC-AUC: Strong discriminative power
            """

            gr.Markdown(model_info)

            gr.Markdown("### Feature Descriptions")
            gr.Markdown(
                """
                All features (V1-V19) are PCA (Principal Component Analysis) transformed
                from original credit card transaction data. The specific features selected
                (shown above) were chosen based on their correlation with fraud cases.

                **Why PCA?**
                - Protects customer privacy
                - Reduces dimensionality
                - Improves model efficiency
                - Captures variance in the data
                """
            )

    return demo


# ============================================================================
# MAIN EXECUTION
# ============================================================================

# # Model path
# MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "utils", "model.pkl")

# # Alternative path
# if not os.path.exists(MODEL_PATH):
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
# Create app at module level for import
app = create_fraud_detection_app(MODEL_PATH)

if __name__ == "__main__":
    print(f"Loading model from: {MODEL_PATH}")
    app.launch()
