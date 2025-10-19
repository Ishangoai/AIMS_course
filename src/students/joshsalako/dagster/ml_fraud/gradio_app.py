import gradio as gr
import pandas as pd
import requests

# --- Configuration ---
# The Gradio app now calls this URL to get predictions.
# This server is started by deploy_local_model.sh
MODEL_SERVER_URL = "http://localhost:5001/invocations"

# Credit Card Fraud Detection Features (V1-V28, Amount, Time)
FEATURE_NAMES = [
    "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount", "Time"
]


# --- Prediction Function ---
def predict_fraud(
    v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
    v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
    v21, v22, v23, v24, v25, v26, v27, v28, amount, time_seconds
):
    """Sends transaction features to the MLflow model server for a fraud prediction."""

    try:
        # Create a dictionary with feature names and values
        input_data = {
            "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5,
            "V6": v6, "V7": v7, "V8": v8, "V9": v9, "V10": v10,
            "V11": v11, "V12": v12, "V13": v13, "V14": v14, "V15": v15,
            "V16": v16, "V17": v17, "V18": v18, "V19": v19, "V20": v20,
            "V21": v21, "V22": v22, "V23": v23, "V24": v24, "V25": v25,
            "V26": v26, "V27": v27, "V28": v28, "Amount": amount, "Time": time_seconds
        }

        # Create a DataFrame (model server expects this structure)
        df = pd.DataFrame([input_data], columns=FEATURE_NAMES)

        # Convert DataFrame to the JSON format required by the MLflow server
        json_data = df.to_json(orient="split")

        # Send the POST request to the model server
        headers = {"Content-Type": "application/json"}
        response = requests.post(MODEL_SERVER_URL, data=json_data, headers=headers)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        # Parse the prediction from the response
        result = response.json()
        prediction = result['predictions'][0]

        # Format the result for display
        result_text = f"""
        **Prediction:** {'🚨 FRAUD DETECTED' if prediction == 1 else '✅ Legitimate Transaction'}

        **Risk Level:** {'HIGH' if prediction == 1 else 'LOW'}

        *(Note: Confidence scores are not available from the standard model server endpoint.)*
        """

        return result_text

    except requests.exceptions.RequestException as e:
        return f"Error connecting to model server: {e}. Is the server running?"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"


# --- Gradio Interface ---
with gr.Blocks(title="Credit Card Fraud Detection") as iface:
    gr.Markdown(
        """
        # 🚨 Credit Card Fraud Detection System
        This application uses a machine learning model served via MLflow to detect fraudulent credit card transactions.
        Enter the transaction details below to get a fraud prediction.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Transaction Details")

            # Amount and Time inputs
            amount = gr.Number(
                label="Transaction Amount ($)",
                value=100.0,
                minimum=0.0,
                maximum=10000.0,
                step=0.01
            )

            time_seconds = gr.Number(
                label="Time (seconds since start of day)",
                value=36000,  # 10 AM
                minimum=0,
                maximum=86399,  # 23:59:59
                step=1
            )

            gr.Markdown("### PCA Features (V1-V28)")
            gr.Markdown("*These are transformed features. Use sample values for testing.*")

        with gr.Column(scale=2):
            # Create input fields for V1-V28 features
            with gr.Row():
                v1 = gr.Number(label="V1", value=0.0, step=0.01)
                v2 = gr.Number(label="V2", value=0.0, step=0.01)
                v3 = gr.Number(label="V3", value=0.0, step=0.01)
                v4 = gr.Number(label="V4", value=0.0, step=0.01)
                v5 = gr.Number(label="V5", value=0.0, step=0.01)
                v6 = gr.Number(label="V6", value=0.0, step=0.01)
                v7 = gr.Number(label="V7", value=0.0, step=0.01)
                v8 = gr.Number(label="V8", value=0.0, step=0.01)

            with gr.Row():
                v9 = gr.Number(label="V9", value=0.0, step=0.01)
                v10 = gr.Number(label="V10", value=0.0, step=0.01)
                v11 = gr.Number(label="V11", value=0.0, step=0.01)
                v12 = gr.Number(label="V12", value=0.0, step=0.01)
                v13 = gr.Number(label="V13", value=0.0, step=0.01)
                v14 = gr.Number(label="V14", value=0.0, step=0.01)
                v15 = gr.Number(label="V15", value=0.0, step=0.01)
                v16 = gr.Number(label="V16", value=0.0, step=0.01)

            with gr.Row():
                v17 = gr.Number(label="V17", value=0.0, step=0.01)
                v18 = gr.Number(label="V18", value=0.0, step=0.01)
                v19 = gr.Number(label="V19", value=0.0, step=0.01)
                v20 = gr.Number(label="V20", value=0.0, step=0.01)
                v21 = gr.Number(label="V21", value=0.0, step=0.01)
                v22 = gr.Number(label="V22", value=0.0, step=0.01)
                v23 = gr.Number(label="V23", value=0.0, step=0.01)
                v24 = gr.Number(label="V24", value=0.0, step=0.01)

            with gr.Row():
                v25 = gr.Number(label="V25", value=0.0, step=0.01)
                v26 = gr.Number(label="V26", value=0.0, step=0.01)
                v27 = gr.Number(label="V27", value=0.0, step=0.01)
                v28 = gr.Number(label="V28", value=0.0, step=0.01)

    with gr.Row():
        with gr.Column():
            # Sample data buttons
            gr.Markdown("### Sample Data")
            with gr.Row():
                sample_legitimate_btn = gr.Button("Load Legitimate Sample", variant="secondary")
                sample_fraud_btn = gr.Button("Load Fraud Sample", variant="secondary")
                clear_btn = gr.Button("Clear All", variant="stop")

        with gr.Column():
            # Prediction button
            predict_btn = gr.Button("🔍 Analyze Transaction", variant="primary", size="lg")

    # Output
    output = gr.Markdown(
        label="Fraud Analysis Result",
        value="Enter transaction details and click 'Analyze Transaction' to get a fraud prediction."
    )

    # Event handlers
    predict_btn.click(
        fn=predict_fraud,
        inputs=[
            v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
            v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
            v21, v22, v23, v24, v25, v26, v27, v28, amount, time_seconds
        ],
        outputs=output
    )

    # Sample data functions
    def load_legitimate_sample():
        """Load sample data for a legitimate transaction."""
        return [
            1.0, -0.5, 0.8, -1.2, 0.3, -0.7, 0.9, -0.4, 0.6, -0.8,
            1.1, -0.3, 0.7, -0.9, 0.4, -0.6, 0.8, -0.2, 0.5, -0.7,
            0.9, -0.1, 0.6, -0.5, 0.3, -0.4, 0.7, -0.3, 50.0, 36000
        ]

    def load_fraud_sample():
        """Load sample data for a fraudulent transaction."""
        return [
            -2.5, 3.1, -1.8, 2.7, -2.2, 1.9, -3.1, 2.4, -1.7, 2.8,
            -2.9, 1.6, -2.3, 2.1, -1.9, 2.5, -2.7, 1.8, -2.1, 2.3,
            -2.6, 1.4, -1.8, 2.0, -2.4, 1.7, -2.8, 1.9, 5000.0, 72000
        ]

    def clear_all():
        """Clear all input fields."""
        return [0.0] * 30  # 28 V features + Amount + Time

    sample_legitimate_btn.click(
        fn=load_legitimate_sample,
        outputs=[
            v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
            v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
            v21, v22, v23, v24, v25, v26, v27, v28, amount, time_seconds
        ]
    )

    sample_fraud_btn.click(
        fn=load_fraud_sample,
        outputs=[
            v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
            v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
            v21, v22, v23, v24, v25, v26, v27, v28, amount, time_seconds
        ]
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[
            v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
            v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
            v21, v22, v23, v24, v25, v26, v27, v28, amount, time_seconds
        ]
    )

# --- Launch the App ---
if __name__ == "__main__":
    print("🚀 Launching Gradio App Client")
    print(f"📡 Connecting to model server at: {MODEL_SERVER_URL}")
    iface.launch(server_name="0.0.0.0", server_port=7860)
