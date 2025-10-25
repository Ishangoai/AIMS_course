import os
import sys

import gradio as gr
import mlflow
import mlflow.sklearn as ms
import pandas as pd

# MLflow configuration
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..", "mlflow_local_tracking.db")
SQLITE_DB_PATH = os.path.abspath(DB_PATH)
MLFLOW_TRACKING_URI = f"sqlite:///{SQLITE_DB_PATH}"

# Set MLflow tracking URI
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load model from Production stage
MODEL_NAME = "tuned-fraud-detector"
MODEL_STAGE = "production"

# Construct the model URI for the registry
model_uri = f"models:/{MODEL_NAME}@{MODEL_STAGE}"

print(MLFLOW_TRACKING_URI)
print(model_uri)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "..")))

# Load model
# model = ms.load_model(logged_model)


def predict_fraud(*inputs):
    """Make fraud prediction"""
    try:
        # load model
        # import sys

        # # Add project root (where 'src' folder is) to sys.path

        # print("Loading model...")
        model = ms.load_model(model_uri)
        # model = mlflow.pyfunc.load_model(model_uri)
        print(type(model))

        # Create input dataframe with correct column order
        input_data = pd.DataFrame(
            [list(inputs)],
            columns=pd.Index([f"V{i}" for i in range(1, 29)] + ["Amount"]),
        )

        if model is None:
            raise Exception("Model not loaded")

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0]

        fraud_prob = probability[1]
        legit_prob = probability[0]

        # Create detailed result
        result = """
### 🔍 Prediction Result
"""

        if prediction == 1:
            result += f"""
### 🚨 **FRAUDULENT TRANSACTION DETECTED**

**Fraud Probability:** {fraud_prob:.2%}
**Legitimate Probability:** {legit_prob:.2%}
![Suspicious](https://media.tenor.com/NpIRDtlpMRAAAAAM/the-rock-rock-meme.gif)
"""
        else:
            result += f"""
### ✅ **LEGITIMATE TRANSACTION**

**Legitimate Probability:** {legit_prob:.2%}
**Fraud Probability:** {fraud_prob:.2%}
![Nice](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRtBkIscyGh0m8SdMffurtFasM9OI_IQiXZ3tmnRXiQTbYMfMJF5a1FQuKA8Okb-bMhn74&usqp=CAU)
"""

        result += f"\n---\n\n*Model: {model_uri}*"

        return result

    except Exception as e:
        return f"Error making prediction: {str(e)}"


gr.Markdown("# 💳 Fraud Detection System")
# Create Gradio interface with custom theme
with gr.Blocks(
    title="Fraud Detection System",
    css="""

    .gradio-container {
        min-width: 80% !important;
        margin: auto !important;
        padding: 20px !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1) !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
    }
""",
) as iface:
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 💰 Transaction Details")
            amount = gr.Slider(
                label="Transaction Amount ($)",
                minimum=0,
                maximum=30000,
                value=100,
                step=0.01,
                info="The transaction amount in dollars",
            )

            gr.Markdown("### 📊 Variables (V1-V14)")

            V1 = gr.Slider(label="V1", minimum=-50, maximum=50, step=0.01)
            V2 = gr.Slider(label="V2", minimum=-50, maximum=50, step=0.01)
            V3 = gr.Slider(label="V3", minimum=-50, maximum=50, step=0.01)
            V4 = gr.Slider(label="V4", minimum=-50, maximum=50, step=0.01)
            V5 = gr.Slider(label="V5", minimum=-50, maximum=50, step=0.01)
            V6 = gr.Slider(label="V6", minimum=-50, maximum=50, step=0.01)
            V7 = gr.Slider(label="V7", minimum=-50, maximum=50, step=0.01)
            V8 = gr.Slider(label="V8", minimum=-50, maximum=50, step=0.01)
            V9 = gr.Slider(label="V9", minimum=-50, maximum=50, step=0.01)
            V10 = gr.Slider(label="V10", minimum=-50, maximum=50, step=0.01)
            V11 = gr.Slider(label="V11", minimum=-50, maximum=50, step=0.01)
            V12 = gr.Slider(label="V12", minimum=-50, maximum=50, step=0.01)
            V13 = gr.Slider(label="V13", minimum=-50, maximum=50, step=0.01)
            V14 = gr.Slider(label="V14", minimum=-50, maximum=50, step=0.01)

        with gr.Column(scale=1):
            gr.Markdown("### 📊 Variables (V15-V28)")

            V15 = gr.Slider(label="V15", minimum=-50, maximum=50, step=0.01)
            V16 = gr.Slider(label="V16", minimum=-50, maximum=50, step=0.01)
            V17 = gr.Slider(label="V17", minimum=-50, maximum=50, step=0.01)
            V18 = gr.Slider(label="V18", minimum=-50, maximum=50, step=0.01)
            V19 = gr.Slider(label="V19", minimum=-50, maximum=50, step=0.01)
            V20 = gr.Slider(label="V20", minimum=-50, maximum=50, step=0.01)
            V21 = gr.Slider(label="V21", minimum=-50, maximum=50, step=0.01)
            V22 = gr.Slider(label="V22", minimum=-50, maximum=50, step=0.01)
            V23 = gr.Slider(label="V23", minimum=-50, maximum=50, step=0.01)
            V24 = gr.Slider(label="V24", minimum=-50, maximum=50, step=0.01)
            V25 = gr.Slider(label="V25", minimum=-50, maximum=50, step=0.01)
            V26 = gr.Slider(label="V26", minimum=-50, maximum=50, step=0.01)
            V27 = gr.Slider(label="V27", minimum=-50, maximum=50, step=0.01)
            V28 = gr.Slider(label="V28", minimum=-50, maximum=50, step=0.01)

    with gr.Row():
        with gr.Column():
            predict_btn = gr.Button("🔍 Analyze Transaction", variant="primary", size="lg")
            output = gr.Markdown(label="Prediction Result", elem_classes="output-text")

    # Connect prediction button
    all_inputs = [
        V1,
        V2,
        V3,
        V4,
        V5,
        V6,
        V7,
        V8,
        V9,
        V10,
        V11,
        V12,
        V13,
        V14,
        V15,
        V16,
        V17,
        V18,
        V19,
        V20,
        V21,
        V22,
        V23,
        V24,
        V25,
        V26,
        V27,
        V28,
        amount,
    ]

    predict_btn.click(fn=predict_fraud, inputs=all_inputs, outputs=output)


# Launch the App
if __name__ == "__main__":
    print("Launching Gradio app to interact with the model")
    iface.launch(server_name="0.0.0.0", server_port=7861)
