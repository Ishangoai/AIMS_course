import gradio as gr
from gradioapp.utils.heart_disease_utils import predict_heart_disease


def wrapped_predict(*args):
    return predict_heart_disease(list(args))


with gr.Blocks(css="body {background: #f2f7ff;}") as heart_app:
    gr.Markdown("# AIMS Course: Heart Disease Risk Predictor")
    gr.Markdown("Fill in the patient info below to predict heart disease risk.")

    with gr.Row():
        # Input Column 1
        with gr.Column():
            age = gr.Slider(20, 80, value=50, label="Age")
            sex = gr.Radio(choices=[0, 1], value=1, label="Sex", info="0 = Female, 1 = Male")
            cp = gr.Radio(choices=[0, 1, 2, 3], value=1, label="Chest Pain Type")
            trestbps = gr.Slider(90, 200, value=120, label="Resting BP")
            chol = gr.Slider(100, 600, value=250, label="Cholesterol")
            fbs = gr.Radio(choices=[0, 1], value=0, label="Fasting Blood Sugar > 120 mg/dl")
            restecg = gr.Radio(choices=[0, 1, 2], value=1, label="Resting ECG")

        # Input Column 2
        with gr.Column():
            thalach = gr.Slider(70, 210, value=150, label="Max Heart Rate")
            exang = gr.Radio(choices=[0, 1], value=0, label="Exercise Induced Angina")
            oldpeak = gr.Slider(0.0, 6.0, value=1.0, step=0.1, label="ST Depression")
            slope = gr.Slider(0, 2, value=1, label="Slope")
            ca = gr.Slider(0, 4, value=0, label="Major Vessels")
            thal = gr.Slider(0, 7, value=3, label="Thal")

    predict_btn = gr.Button("Predict")
    result = gr.Textbox(label="Result")

    predict_btn.click(fn=wrapped_predict,
                      inputs=[age, sex, cp, trestbps, chol, fbs, restecg,
                              thalach, exang, oldpeak, slope, ca, thal],
                      outputs=result)
