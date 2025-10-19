import os

import gradio as gr
from gradioapp.styles.fraud_app import custom_css
from gradioapp.utils.fraud_detection_utils import predict_credit_card_fraud


# Wrap model predictor
def wrapped_predict(*args):
    return predict_credit_card_fraud(list(args))


ABOUT_FILE = os.path.join(os.path.dirname(__file__), "docs/about_fraud_app.md")


def load_about_markdown():
    if os.path.exists(ABOUT_FILE):
        with open(ABOUT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return (
            "⚠️ *About file not found."
            "Please ensure `about.md` exists in the same directory.*"
        )


# UI wrapper to format Gradio callouts
def wrapped_predict_ui(*args):
    result = wrapped_predict(*args)
    if not result.get("status"):
        error_msg = result.get('error', 'Unknown error')
        return f"""
        <div style="
            padding: 20px;
            border-radius: 12px;
            background: linear-gradient(135deg, #fee 0%, #fdd 100%);
            border-left: 5px solid #c33;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 32px;">❌</span>
                <div>
                    <div style="
                        font-size: 18px;
                        font-weight: bold;
                        color: #c33;
                        margin-bottom: 5px;
                    ">Prediction Failed</div>
                    <div style="color: #733; font-size: 14px;">{error_msg}</div>
                </div>
            </div>
        </div>
        """

    pred = result.get("prediction")
    msg = result.get("message", "")

    if pred == 1:
        return f"""
        <div style="
            padding: 20px;
            border-radius: 12px;
            background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%);
            border-left: 5px solid #ff9800;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 32px;">⚠️</span>
                <div>
                    <div style="
                        font-size: 20px;
                        font-weight: bold;
                        color: #e65100;
                        margin-bottom: 5px;
                    ">High Risk Transaction</div>
                    <div style="color: #7d4f00; font-size: 15px; line-height: 1.5;">{msg}</div>
                </div>
            </div>
        </div>
        """
    else:
        return f"""
        <div style="
            padding: 20px;
            border-radius: 12px;
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            border-left: 5px solid #28a745;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        ">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 32px;">✅</span>
                <div>
                    <div style="
                        font-size: 20px;
                        font-weight: bold;
                        color: #155724;
                        margin-bottom: 5px;
                    ">Low Risk Transaction</div>
                    <div style="color: #155724; font-size: 15px; line-height: 1.5;">{msg}</div>
                </div>
            </div>
        </div>
        """


# Sample data
SAMPLE_NORMAL = {
    "V14": 0.35237478, "V17": -0.010299228, "V10": -0.156840906, "V12": 0.538299087,
    "V11": 0.847755891, "V16": 0.906489408, "V4": -0.146817811, "V9": -0.49952308,
    "V18": 0.576837916, "V7": -0.166862547, "V3": 0.220932254, "Amount": 8.99
}

SAMPLE_FRAUD = {
    "V14": -10.5, "V17": -15.0, "V10": -5.0, "V12": -7.5,
    "V11": 5.0, "V16": -8.0, "V4": 5.0, "V9": -3.0,
    "V18": -4.0, "V7": -1.0, "V3": -2.0, "Amount": 500.0
}

with gr.Blocks(css=custom_css) as fraud_app:
    gr.Markdown("# Credit Card Fraud Detector")

    with gr.Tabs():
        with gr.Tab("🔍 Fraud Detection"):
            gr.Markdown("### Enter the transaction features below to predict fraud risk using the trained model.")

            result = gr.HTML(label="Prediction Result")

            with gr.Row():
                with gr.Column():
                    v14 = gr.Slider(minimum=-20, maximum=20, value=SAMPLE_NORMAL["V14"],
                                   label="V14", elem_classes="orange-slider")
                    v17 = gr.Slider(minimum=-20, maximum=20, value=SAMPLE_NORMAL["V17"],
                                   label="V17", elem_classes="orange-slider")
                    v10 = gr.Slider(minimum=-20, maximum=20, value=SAMPLE_NORMAL["V10"],
                                   label="V10", elem_classes="orange-slider")
                    v12 = gr.Slider(minimum=-20, maximum=20, value=SAMPLE_NORMAL["V12"],
                                   label="V12", elem_classes="orange-slider")
                    v11 = gr.Slider(minimum=-20, maximum=20, value=SAMPLE_NORMAL["V11"],
                                   label="V11", elem_classes="orange-slider")
                    v16 = gr.Slider(minimum=-20, maximum=20, value=SAMPLE_NORMAL["V16"],
                                   label="V16", elem_classes="orange-slider")
                with gr.Column():
                    v4 = gr.Slider(minimum=-20, maximum=20, value=SAMPLE_NORMAL["V4"],
                                  label="V4", elem_classes="orange-slider")
                    v9 = gr.Slider(minimum=-20, maximum=20, value=SAMPLE_NORMAL["V9"],
                                  label="V9", elem_classes="orange-slider")
                    v18 = gr.Slider(minimum=-20, maximum=20, value=SAMPLE_NORMAL["V18"],
                                   label="V18", elem_classes="orange-slider")
                    v7 = gr.Slider(minimum=-20, maximum=20, value=SAMPLE_NORMAL["V7"],
                                  label="V7", elem_classes="orange-slider")
                    v3 = gr.Slider(minimum=-20, maximum=20, value=SAMPLE_NORMAL["V3"],
                                  label="V3", elem_classes="orange-slider")
                    amount = gr.Slider(minimum=0, maximum=1000, value=SAMPLE_NORMAL["Amount"],
                                      label="Amount", elem_classes="orange-slider")

            with gr.Row():
                predict_btn = gr.Button("🔍 Predict", variant="primary")
                load_normal_btn = gr.Button("Load Sample Normal Case")
                load_fraud_btn = gr.Button("Load Sample Fraud Case")
                clear_btn = gr.Button("Clear")

            gr.Markdown("### Group Members: Khadija Edarzi & Similoluwa Okunowo")

        with gr.Tab("📖 About"):
            gr.Markdown(load_about_markdown())

    predict_btn.click(
        fn=wrapped_predict_ui,
        inputs=[v14, v17, v10, v12, v11, v16, v4, v9, v18, v7, v3, amount],
        outputs=result
    )

    def _load_sample_normal():
        return tuple(SAMPLE_NORMAL.values())

    def _load_sample_fraud():
        return tuple(SAMPLE_FRAUD.values())

    def _clear_all():
        return (0.0,) * 12

    load_normal_btn.click(fn=_load_sample_normal, inputs=[], outputs=[
        v14, v17, v10, v12, v11, v16, v4, v9, v18, v7, v3, amount
    ])

    load_fraud_btn.click(
        fn=_load_sample_fraud,
        inputs=[],
        outputs=[
            v14, v17, v10, v12,
            v11, v16, v4, v9,
            v18, v7, v3, amount
        ]
    )

    clear_btn.click(fn=_clear_all, inputs=[], outputs=[
        v14, v17, v10, v12, v11, v16, v4, v9, v18, v7, v3, amount
    ])

if __name__ == "__main__":
    fraud_app.launch(server_name="0.0.0.0", share=False, debug=True)
