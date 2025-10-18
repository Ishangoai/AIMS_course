import gradio as gr
import pandas as pd
from gradioapp.utils.fraud_utils import predict_fraud
import plotly.graph_objects as go
import requests

# Example transaction values for testing
EXAMPLE_TRANSACTION = [
    -1.508147666, 1.853128013, 0.220932254, -0.146817811, -0.528975655, -0.626681837,
    -0.166862547, 0.999946773, -0.49952308, -0.156840906, 0.847755891, 0.538299087,
    -0.354515904, 0.35237478, 0.23433853, 0.906489408, -0.010299228, 0.576837916,
    0.256061611, 0.130144351, -0.218249367, -0.752299949, 0.059790389, -0.071737144,
    -0.017378138, 0.090616835, 0.204201631, 0.075687247, 8.99
]


COLUMN_NAMES = [f"V{i+1}" for i in range(28)] + ["Amount"]

def wrapped_predict(*args):
    inputs = [arg if arg is not None else EXAMPLE_TRANSACTION[i] for i, arg in enumerate(args)]
    payload = {"features": inputs}
    print("Payload sent:", payload)  # <--- check
    try:
        response = requests.post("http://127.0.0.1:8080/predict", json=payload)
        response.raise_for_status()   # <--- raises if API returns error
        result_json = response.json()
        result_text = result_json.get("prediction", str(result_json))
        prob = result_json.get("probability", 0.0)
    except Exception as e:
        result_text = f"Error: {str(e)}"
        prob = 0.0

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob,
        title={'text': "Fraud Probability"},
        gauge={'axis': {'range': [0, 1]}}
    ))
    prob_html = f"<b>{prob:.4f}</b>"

    return result_text, prob_html, fig



def fill_example():
    return EXAMPLE_TRANSACTION

def reset_inputs():
    return [None]*29

def batch_predict(csv_file):
    df = pd.read_csv(csv_file.name)
    predictions = []
    for _, row in df.iterrows():
        inputs = [row.get(f"V{i}", EXAMPLE_TRANSACTION[i-1]) for i in range(1,29)]
        inputs.append(row.get("Amount", EXAMPLE_TRANSACTION[28]))
        result = predict_fraud(inputs)
        try:
            prob = float(result.split("Probability of Fraud:")[1].strip())
        except:
            prob = 0.0
        predictions.append({"TransactionID": row.get("Time", None), "Prediction": result, "Probability": prob})
    return pd.DataFrame(predictions)

with gr.Blocks(css="body {background: #f2f7ff;}") as fraud_app:
    gr.Markdown("# AIMS Course: Fraud Detection Predictor")
    gr.Markdown("Fill in the transaction features below or upload a CSV file for batch prediction. Leave empty for example values.")

    with gr.Row():
        with gr.Column():
            sliders_1 = [gr.Slider(-5, 5, value=EXAMPLE_TRANSACTION[i], label=f"V{i+1}", info=f"Feature V{i+1}") for i in range(14)]
        with gr.Column():
            sliders_2 = [gr.Slider(-5, 5, value=EXAMPLE_TRANSACTION[i], label=f"V{i+1}", info=f"Feature V{i+1}") for i in range(14, 28)]
            Amount = gr.Number(label="Transaction Amount", value=EXAMPLE_TRANSACTION[28], precision=2, info="Transaction amount in USD")

    predict_btn = gr.Button("Predict Single Transaction")
    example_btn = gr.Button("Fill Example Transaction")
    reset_btn = gr.Button("Reset Inputs")

    result = gr.Textbox(label="Prediction Result")
    prob_bar = gr.HTML(label="Fraud Probability")
    prob_chart = gr.Plot(label="Real-time Fraud Probability Gauge")

    inputs = sliders_1 + sliders_2 + [Amount]

    predict_btn.click(fn=wrapped_predict, inputs=inputs, outputs=[result, prob_bar, prob_chart])
    example_btn.click(fn=fill_example, inputs=[], outputs=inputs)
    reset_btn.click(fn=reset_inputs, inputs=[], outputs=inputs)

    # Batch CSV Upload Section
    gr.Markdown("### Batch Prediction via CSV Upload")
    csv_input = gr.File(label="Upload CSV", file_types=[".csv"])
    batch_output = gr.Dataframe(headers=["TransactionID", "Prediction", "Probability"])
    csv_btn = gr.Button("Run Batch Prediction")
    csv_btn.click(fn=batch_predict, inputs=csv_input, outputs=batch_output)
