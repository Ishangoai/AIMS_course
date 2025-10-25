"""
Gradio Application for Fraud Detection Model
Interactive UI to test the fraud detection model from MLflow Model Registry
"""
import os

import gradio as gr
import joblib
import pandas as pd

# Configuration
MODEL_NAME = "fraud-detection-rf"
MODEL_STAGE = "Production"

# Construct an absolute path to the database file
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# DB_PATH = os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..", "mlflow_local_tracking.db")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_v5.pkl")
model = joblib.load(MODEL_PATH)

# Set the tracking URI to find the local MLflow database
# mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(DB_PATH)}")

# Construct the model URI for the registry
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

print("=" * 70)
print("🚀 FRAUD DETECTION GRADIO APP")
print("=" * 70)
print(f"Model Name: {MODEL_NAME}")
print(f"Model Stage: {MODEL_STAGE}")
print(f"Model URI: {model_uri}")
# print(f"MLflow DB: {os.path.abspath(DB_PATH)}")
print("=" * 70)

# Load the model into memory


def predict_fraud(time, amount, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                   v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
                   v21, v22, v23, v24, v25, v26, v27, v28):
    """
    Prediction function called by Gradio when user submits the form.
    Args:
        time: Time elapsed since first transaction
        amount: Transaction amount
        v1-v28: PCA-transformed features (anonymized)
        Returns:
        Tuple of (prediction_text, confidence_text, risk_color)
    """
    try:
        # Create a DataFrame matching the model's expected input format
        input_df = pd.DataFrame({
            'Time': [float(time)],
            'V1': [float(v1)], 'V2': [float(v2)], 'V3': [float(v3)], 'V4': [float(v4)], 'V5': [float(v5)],
            'V6': [float(v6)], 'V7': [float(v7)], 'V8': [float(v8)], 'V9': [float(v9)], 'V10': [float(v10)],
            'V11': [float(v11)], 'V12': [float(v12)], 'V13': [float(v13)], 'V14': [float(v14)], 'V15': [float(v15)],
            'V16': [float(v16)], 'V17': [float(v17)], 'V18': [float(v18)], 'V19': [float(v19)], 'V20': [float(v20)],
            'V21': [float(v21)], 'V22': [float(v22)], 'V23': [float(v23)], 'V24': [float(v24)], 'V25': [float(v25)],
            'V26': [float(v26)], 'V27': [float(v27)], 'V28': [float(v28)],
            'Amount': [float(amount)]
        })

        # Make prediction
        input_df = input_df.astype(float)
        prediction_result = model.predict(input_df)
        prediction = prediction_result[0]

        # Format the output
        if prediction == 1:
            result_text = "🚨 FRAUD DETECTED"
            confidence_text = "This transaction shows signs of fraudulent activity."
            risk_html = '<div style="background-color: #ff4444;'
            'color: white; padding: 20px; border-radius: 10px; text-align: center; font-size: 24px;'
            'font-weight: bold;">⚠️ HIGH RISK - FRAUD</div>'
        else:
            result_text = "✅ LEGITIMATE TRANSACTION"
            confidence_text = "This transaction appears to be legitimate."
            risk_html = (
                '<div style="background-color: #44ff44; color: black; padding: 20px; border-radius: 10px;'
                'text-align: center; font-size: 24px; font-weight: bold;">✓ LOW RISK - NORMAL</div>'
            )
        # Additional info
        details = f"""
        ### Transaction Details
        - **Amount**: ${amount:.2f}
        - **Time**: {time:.0f} seconds
        - **Prediction**: Class {int(prediction)}
        - **Model**: {MODEL_NAME} ({MODEL_STAGE})
        """

        return risk_html, result_text, confidence_text, details

    except Exception as e:
        error_html = '<div style="background-color: #ff8800; color: white; padding: 20px;'
        'border-radius: 10px; text-align: center;">⚠️ ERROR: {str(e)}</div>'
        return error_html, "Error during prediction", str(e), ""


def load_sample_normal():
    """Load a sample normal transaction"""
    return (
        150.0,  # time
        50.00,  # amount
        -1.2, 0.8, 0.3, -0.5, 0.6, -0.3, 0.4, -0.2, 0.7, -0.4,
        0.5, -0.6, 0.3, -0.4, 0.2, -0.5, 0.4, -0.3, 0.5, -0.2,
        0.3, -0.4, 0.6, -0.2, 0.3, -0.5, 0.4, -0.3
    )


def load_sample_fraud():
    """Load a sample fraudulent transaction"""
    return (
        30000.0,  # time (unusual time)
        5000.00,  # amount (high amount)
        3.5, -2.8, 4.2, -3.1, 2.9, -3.5, 3.8, -2.5, 4.1, -3.2,
        3.3, -2.9, 3.7, -3.4, 2.8, -3.6, 3.9, -2.7, 3.4, -3.1,
        2.9, -3.3, 3.5, -2.8, 3.2, -3.4, 3.6, -2.9
    )


# Gradio Interface Definition
with gr.Blocks(title="Fraud Detection System", theme=gr.themes.Soft()) as fraud_detect:  # pyright: ignore

    gr.Markdown(
        """
        # 🔒 Credit Card Fraud Detection System
        Enter transaction details below to check if a transaction is fraudulent.
        This system uses a **RandomForest model** trained on credit card transactions and deployed via **MLflow**.
        **Features V1-V28** are anonymized PCA-transformed features from the original dataset.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Transaction Information")

            time_input = gr.Number(
                label="Time (seconds since first transaction)",
                value=150.0,
                info="Time elapsed since the first transaction in the dataset"
            )

            amount_input = gr.Number(
                label="Transaction Amount ($)",
                value=50.00,
                info="Amount of the transaction in dollars"
            )

            gr.Markdown("### 🔐 Anonymized Features (V1-V28)")
            gr.Markdown("*These are PCA-transformed features from the original transaction data*")

            # Create inputs for V1-V28 in a grid layout
            with gr.Row():
                with gr.Column():
                    v1 = gr.Number(label="V1", value=-1.2)
                    v2 = gr.Number(label="V2", value=0.8)
                    v3 = gr.Number(label="V3", value=0.3)
                    v4 = gr.Number(label="V4", value=-0.5)
                    v5 = gr.Number(label="V5", value=0.6)
                    v6 = gr.Number(label="V6", value=-0.3)
                    v7 = gr.Number(label="V7", value=0.4)

                with gr.Column():
                    v8 = gr.Number(label="V8", value=-0.2)
                    v9 = gr.Number(label="V9", value=0.7)
                    v10 = gr.Number(label="V10", value=-0.4)
                    v11 = gr.Number(label="V11", value=0.5)
                    v12 = gr.Number(label="V12", value=-0.6)
                    v13 = gr.Number(label="V13", value=0.3)
                    v14 = gr.Number(label="V14", value=-0.4)

                with gr.Column():
                    v15 = gr.Number(label="V15", value=0.2)
                    v16 = gr.Number(label="V16", value=-0.5)
                    v17 = gr.Number(label="V17", value=0.4)
                    v18 = gr.Number(label="V18", value=-0.3)
                    v19 = gr.Number(label="V19", value=0.5)
                    v20 = gr.Number(label="V20", value=-0.2)
                    v21 = gr.Number(label="V21", value=0.3)

                with gr.Column():
                    v22 = gr.Number(label="V22", value=-0.4)
                    v23 = gr.Number(label="V23", value=0.6)
                    v24 = gr.Number(label="V24", value=-0.2)
                    v25 = gr.Number(label="V25", value=0.3)
                    v26 = gr.Number(label="V26", value=-0.5)
                    v27 = gr.Number(label="V27", value=0.4)
                    v28 = gr.Number(label="V28", value=-0.3)

            with gr.Row():
                predict_btn = gr.Button("🔍 Analyze Transaction", variant="primary", size="lg")
                clear_btn = gr.ClearButton(
                    components=[time_input, amount_input, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                               v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24,
                               v25, v26, v27, v28],
                    value="🗑️ Clear All"
                )

            with gr.Row():
                sample_normal_btn = gr.Button("📝 Load Sample Normal Transaction", size="sm")
                sample_fraud_btn = gr.Button("⚠️ Load Sample Fraud Transaction", size="sm")

        with gr.Column(scale=1):
            gr.Markdown("### 📋 Prediction Results")

            risk_output = gr.HTML(
                value=(
                    '<div style="background-color: #dddddd; color: black; padding: 20px;'
                    'border-radius: 10px; text-align: center;">Waiting for prediction...</div>'
                ),
                label="Risk Assessment"
            )

            result_output = gr.Textbox(
                label="Prediction",
                interactive=False,
                lines=1
            )

            confidence_output = gr.Textbox(
                label="Analysis",
                interactive=False,
                lines=2
            )

            details_output = gr.Markdown(
                value="*Enter transaction details and click 'Analyze Transaction'*"
            )

    gr.Markdown(
        """
        ---
        ### ℹ️ About This System
        - **Model**: RandomForest Classifier
        - **Dataset**: Credit Card Transactions (Anonymized)
        - **Features**: 30 features (Time, Amount, V1-V28)
        - **MLflow Registry**: Models tracked and versioned
        - **Deployment**: Production-ready model from MLflow
        **Note**: The V1-V28 features are the result of PCA transformation to protect sensitive information.
        In a real scenario, these would be derived from actual transaction data.
        """
    )

    # Connect the predict button
    predict_btn.click(
                fn=predict_fraud,
                inputs=[time_input, amount_input, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                    v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28],
                outputs=[risk_output, result_output, confidence_output, details_output]
                )

    # Connect sample buttons
    sample_normal_btn.click(
                fn=load_sample_normal,
                inputs=[],
                outputs=[time_input, amount_input, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28]
                )

    sample_fraud_btn.click(
                    fn=load_sample_fraud,
                    inputs=[],
                    outputs=[time_input, amount_input, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                            v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28]
                )


# Launch the App
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 Launching Gradio Fraud Detection Interface...")
    print("=" * 70)
    fraud_detect.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        show_error=True
    )
