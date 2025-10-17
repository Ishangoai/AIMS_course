import gradio as gr
from gradioapp.utils.fraud_detection import predict_fraud


def wrapped_predict(*args):
    return predict_fraud(list(args))


with gr.Blocks(css="body {background: #f2f7ff;}") as fraud_app:
    gr.Markdown("# AIMS Course: Heart Disease Risk Predictor")
    gr.Markdown("Fill in the patient info below to predict heart disease risk.")

    with gr.Row():
        # Input Column 1
        with gr.Column():
            time = gr.Slider(20, 80, value=50, label="Time")
            v1 = gr.Slider(20, 80, value=50, label="V1")
            v2 = gr.Slider(20, 80, value=50, label="V2")
            v3 = gr.Slider(20, 80, value=50, label="V3")
            v4 = gr.Slider(20, 80, value=50, label="V4")
            v5 = gr.Slider(20, 80, value=50, label="V5")
            v6 = gr.Slider(20, 80, value=50, label="V6")
            v7 = gr.Slider(20, 80, value=50, label="V7")
            v8 = gr.Slider(20, 80, value=50, label="V8")
            v9 = gr.Slider(20, 80, value=50, label="V9")
            v10 = gr.Slider(20, 80, value=50, label="V10")
            v11 = gr.Slider(20, 80, value=50, label="V11")
            v12 = gr.Slider(20, 80, value=50, label="V12")
            v13 = gr.Slider(20, 80, value=50, label="V13")
            v14 = gr.Slider(20, 80, value=50, label="V14")

        # Input Column 2
        with gr.Column():
            v15 = gr.Slider(20, 80, value=50, label="V15")
            v16 = gr.Slider(20, 80, value=50, label="V16")
            v17 = gr.Slider(20, 80, value=50, label="V17")
            v18 = gr.Slider(20, 80, value=50, label="V18")
            v19 = gr.Slider(20, 80, value=50, label="V19")
            v20 = gr.Slider(20, 80, value=50, label="V20")
            v21 = gr.Slider(20, 80, value=50, label="V21")
            v22 = gr.Slider(20, 80, value=50, label="V22")
            v23 = gr.Slider(20, 80, value=50, label="V23")
            v24 = gr.Slider(20, 80, value=50, label="V24")
            v25 = gr.Slider(20, 80, value=50, label="V25")
            v26 = gr.Slider(20, 80, value=50, label="V26")
            v27 = gr.Slider(20, 80, value=50, label="V27")
            v28 = gr.Slider(20, 80, value=50, label="V28")
            amount = gr.Slider(20, 80, value=50, label="amount")

    predict_btn = gr.Button("Predict")
    result = gr.Textbox(label="Prediction")

    predict_btn.click(fn=wrapped_predict,
                      inputs=[time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14,
                      v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amount],
                      outputs=result)
