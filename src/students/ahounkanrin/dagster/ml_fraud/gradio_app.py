import os

import gradio as gr
import mlflow
import pandas as pd

# Configuration
MODEL_NAME = "tuned-fraud-detector"
MODEL_STAGE = "Staging"


# Construct an absolute path to the database file.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..", "mlflow_local_tracking.db")

# Set the tracking URI to find the local MLflow database
mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(DB_PATH)}")

# Construct the model URI for the registry
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

# Load the model into memory
model = mlflow.pyfunc.load_model(model_uri)


def predict_fraud(*input_features):
    input_features = list(input_features)
    input_features = [float(x) for x in input_features]
    feature_names = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                                     "V11", "V12", "V13", "V14", "V15", "V16", "V17",
                                     "V18", "V19", "V20", "V21", "V22", "V23", "V24",
                                     "V25", "V26", "V27", "V28", "Amount"]
    # input_df = pd.DataFrame([input_features], columns=feature_names)
    input_df = pd.DataFrame()
    for i in range(len(feature_names)):
        input_df[feature_names[i]] = input_features[i]

    prediction = model.predict(input_df)
    predicted_value = int(prediction[0])
    if predicted_value == 1:
        return "Fraudulent transaction!"
    return "Normal transaction"


with gr.Blocks(css="body {background: #f2f7ff;}") as fraud_detection_app:
    gr.Markdown("# Credit card fraud detector")
    gr.Markdown("Fill in the transaction features to make a prediction.")

    with gr.Row():
        with gr.Column():
            amount = gr.Number(value=10.0, label="Amount")
            v1 = gr.Slider(-10., 10., value=0., label="V1")
            v2 = gr.Slider(-10., 10., value=0., label="V2")
            v3 = gr.Slider(-10., 10., value=0., label="V3")
            v4 = gr.Slider(-10., 10., value=0., label="V4")
            v5 = gr.Slider(-10., 10., value=0., label="V5")
            v6 = gr.Slider(-10., 10., value=0., label="V6")
            v7 = gr.Slider(-10., 10., value=0., label="V7")
            v8 = gr.Slider(-10., 10., value=0., label="V8")
            v9 = gr.Slider(-10., 10., value=0., label="V9")
            v10 = gr.Slider(-10., 10., value=0., label="V10")
            v11 = gr.Slider(-10., 10., value=0., label="V11")
            v12 = gr.Slider(-10., 10., value=0., label="V12")
            v13 = gr.Slider(-10., 10., value=0., label="V13")
            v14 = gr.Slider(-10., 10., value=0., label="V14")

        with gr.Column():
            v15 = gr.Slider(-10., 10., value=0., label="V15")
            v16 = gr.Slider(-10., 10., value=0., label="V16")
            v17 = gr.Slider(-10., 10., value=0., label="V17")
            v18 = gr.Slider(-10., 10., value=0., label="V18")
            v19 = gr.Slider(-10., 10., value=0., label="V19")
            v20 = gr.Slider(-10., 10., value=0., label="V20")
            v21 = gr.Slider(-10., 10., value=0., label="V21")
            v22 = gr.Slider(-10., 10., value=0., label="V22")
            v23 = gr.Slider(-10., 10., value=0., label="V23")
            v24 = gr.Slider(-10., 10., value=0., label="V24")
            v25 = gr.Slider(-10., 10., value=0., label="V25")
            v26 = gr.Slider(-10., 10., value=0., label="V26")
            v27 = gr.Slider(-10., 10., value=0., label="V27")
            v28 = gr.Slider(-10., 10., value=0., label="V28")

    predict_btn = gr.Button("Predict")
    result = gr.Textbox(label="Result")

    predict_btn.click(fn=predict_fraud,
                      inputs=[v1, v2, v3, v4, v5, v6, v7, v8, v9, v10,
                              v11, v12, v13, v14, v15, v16, v17, v18, v19,
                              v20, v21, v22, v23, v24, v25, v26, v27, v28, amount],
                              outputs=result
                              )
