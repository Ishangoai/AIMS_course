import gradio as gr

with gr.Blocks() as iface:
    gr.Markdown(
        """
        # Fraud Detection Predictor
        Enter transaction details to predict the likelihood of fraud.
        This model is loaded directly from the 'Production' stage in the MLflow Model Registry.
        """
    )
    with gr.Column():
        with gr.Row():
            val = gr.Slider(label="Transaction Amount", minimum=0, maximum=200000)
        with gr.Row():
            V1 = gr.Slider(label="V1", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V2 = gr.Slider(label="V2", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V3 = gr.Slider(label="V3", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V4 = gr.Slider(label="V4", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V5 = gr.Slider(label="V5", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V6 = gr.Slider(label="V6", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V7 = gr.Slider(label="V7", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V8 = gr.Slider(label="V8", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V9 = gr.Slider(label="V9", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V10 = gr.Slider(label="V10", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V11 = gr.Slider(label="V11", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V12 = gr.Slider(label="V12", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V13 = gr.Slider(label="V13", minimum=-50, maximum=50, value=0)

    with gr.Column():
        with gr.Row():
            V14 = gr.Slider(label="V14", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V15 = gr.Slider(label="V15", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V16 = gr.Slider(label="V16", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V17 = gr.Slider(label="V17", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V18 = gr.Slider(label="V18", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V19 = gr.Slider(label="V19", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V20 = gr.Slider(label="V20", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V21 = gr.Slider(label="V21", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V22 = gr.Slider(label="V22", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V23 = gr.Slider(label="V23", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V24 = gr.Slider(label="V24", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V25 = gr.Slider(label="V25", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V26 = gr.Slider(label="V26", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V27 = gr.Slider(label="V27", minimum=-50, maximum=50, value=0)
        with gr.Row():
            V28 = gr.Slider(label="V28", minimum=-50, maximum=50, value=0)

    predict = gr.Button("Predict Fraud")
    output = gr.Textbox(label="Fraud Prediction")
    predict.click(
        inputs=[val, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10, V11, V12, V13, V14,
                V15, V16, V17, V18, V19, V20, V21, V22, V23, V24, V25, V26, V27, V28],
        outputs=output
)

if __name__ == "__main__":
    print("Launching Gradio app to interact with the fraud detection model")
    iface.launch()
