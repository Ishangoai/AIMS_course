import os
from datetime import datetime, time
from typing import Any

import gradio as gr
import joblib
import pandas as pd

# --- Configuration ---
MODEL_PATH = os.path.join(os.path.dirname(__file__), "utils", "fraud_model.pkl")

try:
    _fraud_model: Any = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please ensure the model is in the correct location.")
    _fraud_model = None
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    _fraud_model = None

FEATURE_NAMES = [
    "V1",
    "V2",
    "V3",
    "V4",
    "V5",
    "V6",
    "V7",
    "V8",
    "V9",
    "V10",
    "V11",
    "V12",
    "V13",
    "V14",
    "V15",
    "V16",
    "V17",
    "V18",
    "V19",
    "V20",
    "V21",
    "V22",
    "V23",
    "V24",
    "V25",
    "V26",
    "V27",
    "V28",
    "Amount",
    "Time",
]

SELECTED_FEATURE_NAMES = [
    "V14",
    "V4",
    "V12",
    "V11",
    "V10",
    "V16",
    "V9",
    "V3",
    "V17",
    "V2",
    "V7",
    "V18",
    "V1",
    "V6",
    "V5",
    "V19",
]


def time_to_seconds(t):
    """Convert a datetime.time object to seconds since midnight."""
    if isinstance(t, str):
        try:
            t_obj = datetime.strptime(t, "%H:%M:%S").time()
        except Exception:
            t_obj = datetime.strptime(t, "%H:%M").time()
        t = t_obj
    return t.hour * 3600 + t.minute * 60 + t.second


# --- Prediction Function ---
def predict_fraud(
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7,
    v8,
    v9,
    v10,
    v11,
    v12,
    v13,
    v14,
    v15,
    v16,
    v17,
    v18,
    v19,
    v20,
    v21,
    v22,
    v23,
    v24,
    v25,
    v26,
    v27,
    v28,
    amount,
    tx_time,
):
    """Predicts credit card fraud using in-app ML model."""
    if _fraud_model is None:
        return ("The prediction service is currently unavailable. "
        "Please try again later or contact the administrator if the issue persists.")

    try:
        # Convert Gradio Time input (datetime.time) to seconds since midnight
        if tx_time is not None:
            time_seconds = time_to_seconds(tx_time)
        else:
            time_seconds = 0

        input_feature_values = {
            "V1": v1,
            "V2": v2,
            "V3": v3,
            "V4": v4,
            "V5": v5,
            "V6": v6,
            "V7": v7,
            "V8": v8,
            "V9": v9,
            "V10": v10,
            "V11": v11,
            "V12": v12,
            "V13": v13,
            "V14": v14,
            "V15": v15,
            "V16": v16,
            "V17": v17,
            "V18": v18,
            "V19": v19,
            "V20": v20,
            "V21": v21,
            "V22": v22,
            "V23": v23,
            "V24": v24,
            "V25": v25,
            "V26": v26,
            "V27": v27,
            "V28": v28,
            "Amount": amount,
            "Time": time_seconds,
        }

        selected_input = {k: input_feature_values[k] for k in SELECTED_FEATURE_NAMES}
        df_input = pd.DataFrame([selected_input], columns=SELECTED_FEATURE_NAMES)  # type: ignore[reportArgumentType]

        prediction = _fraud_model.predict(df_input)[0]

        result_text = f"""
        **Prediction:** {"🚨 FRAUD DETECTED" if prediction == 1 else "✅ Legitimate Transaction"}

        **Risk Level:** {"HIGH" if prediction == 1 else "LOW"}
        """

        return result_text

    except Exception as e:
        return f"An unexpected error occurred during prediction: {str(e)}"


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

            amount = gr.Number(label="Transaction Amount ($)", value=100.0, minimum=0.0, maximum=10000.0, step=0.01)

            tx_time = gr.Textbox(label="Time (hh:mm:ss)", value="10:00:00", interactive=True)

            gr.Markdown("### Features (V1-V28)")
            gr.Markdown("*These are transformed features. Use sample values for testing.*")

        with gr.Column(scale=2):
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

    with gr.Column():
        gr.Markdown("### Control Panel")
        with gr.Row():
            sample_legitimate_btn = gr.Button("Load Legitimate Sample", variant="secondary")
            sample_fraud_btn = gr.Button("Load Fraud Sample", variant="secondary")
            clear_btn = gr.Button("Clear All", variant="stop")
            predict_btn = gr.Button("🔍 Analyze Transaction", variant="primary", size="lg")

    output = gr.Markdown(
        label="Fraud Analysis Result",
        value="Enter transaction details and click 'Analyze Transaction' to get a fraud prediction.",
    )

    predict_btn.click(
        fn=predict_fraud,
        inputs=[
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            v8,
            v9,
            v10,
            v11,
            v12,
            v13,
            v14,
            v15,
            v16,
            v17,
            v18,
            v19,
            v20,
            v21,
            v22,
            v23,
            v24,
            v25,
            v26,
            v27,
            v28,
            amount,
            tx_time,
        ],
        outputs=output,
    )

    def load_legitimate_sample():
        """Load sample data for a legitimate transaction."""
        return [
            1.0,
            -0.5,
            0.8,
            -1.2,
            0.3,
            -0.7,
            0.9,
            -0.4,
            0.6,
            -0.8,
            1.1,
            -0.3,
            0.7,
            -0.9,
            0.4,
            -0.6,
            0.8,
            -0.2,
            0.5,
            -0.7,
            0.9,
            -0.1,
            0.6,
            -0.5,
            0.3,
            -0.4,
            0.7,
            -0.3,
            50.0,
            time(10, 0, 0),
        ]

    def load_fraud_sample():
        """Load sample data for a fraudulent transaction."""
        return [
            -2.857169754,  # V1
            4.045601381,  # V2
            -4.197298786,  # V3
            5.487198631,  # V4
            -3.070776168,  # V5
            -1.422686338,  # V6
            -5.65131374,  # V7
            2.019657395,  # V8
            -5.015490987,  # V9
            -6.319707509,  # V10
            3.779602208,  # V11
            -8.077093646,  # V12
            1.440888856,  # V13
            -7.891908779,  # V14
            0.530452788,  # V15
            -7.954070045,  # V16
            -14.26505595,  # V17
            -5.771064426,  # V18
            2.892169685,  # V19
            0.981608722,  # V20
            1.080322734,  # V21
            -0.56138415,  # V22
            0.102678427,  # V23
            -0.067194723,  # V24
            -0.476931194,  # V25
            -0.103716356,  # V26
            1.166961492,  # V27
            0.663632067,  # V28
            1.0,  # Amount
            time(9, 58, 19),  # Time derived from 35899.0 (in seconds since midnight = 9:58:19)
        ]

    def clear_all():
        """Clear all input fields."""
        cleared = [0.0] * 28 + [0.0, time(0, 0, 0)]
        return cleared

    sample_legitimate_btn.click(
        fn=load_legitimate_sample,
        outputs=[
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            v8,
            v9,
            v10,
            v11,
            v12,
            v13,
            v14,
            v15,
            v16,
            v17,
            v18,
            v19,
            v20,
            v21,
            v22,
            v23,
            v24,
            v25,
            v26,
            v27,
            v28,
            amount,
            tx_time,
        ],
    )

    sample_fraud_btn.click(
        fn=load_fraud_sample,
        outputs=[
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            v8,
            v9,
            v10,
            v11,
            v12,
            v13,
            v14,
            v15,
            v16,
            v17,
            v18,
            v19,
            v20,
            v21,
            v22,
            v23,
            v24,
            v25,
            v26,
            v27,
            v28,
            amount,
            tx_time,
        ],
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            v8,
            v9,
            v10,
            v11,
            v12,
            v13,
            v14,
            v15,
            v16,
            v17,
            v18,
            v19,
            v20,
            v21,
            v22,
            v23,
            v24,
            v25,
            v26,
            v27,
            v28,
            amount,
            tx_time,
        ],
    )

# --- Launch the App ---
if __name__ == "__main__":
    print("🚀 Launching Gradio App Client")
    iface.launch(server_name="0.0.0.0", server_port=7860)
