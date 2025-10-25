import json
import os

import gradio as gr
import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "./utils/fraud_model.pkl")
print(MODEL_PATH)
try:
    fraud_model = joblib.load(MODEL_PATH)
except Exception as e:
    fraud_model = None
    print(f"[ERROR] Could not load fraud detection model: {e}")

# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------


def predict_fraud(features: list):
    """
    Predict fraud risk using the loaded model.
    :param features: list of 30 numerical values in the correct order
    :return: tuple of (prediction_text, fraud_probability, risk_level)
    """
    if fraud_model is None:
        return "Error: Model not loaded", 0, "error"

    try:
        features_np = np.array(features).reshape(1, -1)
        prediction = fraud_model.predict(features_np)[0]
        fraud_prob = fraud_model.predict_proba(features_np)[0][1]

        if prediction == 1:
            return "⚠️ FRAUDULENT", fraud_prob, "high"
        else:
            return "✅ LEGITIMATE", fraud_prob, "low"
    except Exception as e:
        return f"Error: {str(e)}", 0, "error"

# ---------------------------------------------------------------------------
# Wrapper functions
# ---------------------------------------------------------------------------


def wrapped_predict_direct(*args):
    features = list(args)
    text, prob, risk = predict_fraud(features)
    return text, prob, risk


def predict_from_json(json_input):
    try:
        data = json.loads(json_input)

        if not isinstance(data, dict):
            return "Invalid JSON format. Expected object.", 0, "error"

        required_keys = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        features = []

        for key in required_keys:
            if key not in data:
                return f"Missing required field: {key}", 0, "error"
            try:
                features.append(float(data[key]))
            except (ValueError, TypeError):
                return f"Invalid value for {key}. Must be numeric.", 0, "error"

        text, prob, risk = predict_fraud(features)
        return text, prob, risk
    except json.JSONDecodeError:
        return "Invalid JSON format. Please check your input.", 0, "error"


def get_risk_color(risk_level):
    if risk_level == "high":
        return "#ff4444"
    elif risk_level == "low":
        return "#44ff44"
    else:
        return "#cccccc"


# Sample JSON
SAMPLE_JSON = json.dumps({
    "Time": 0,
    "V1": -1.3598,
    "V2": -0.0747,
    "V3": 2.3601,
    "V4": 1.3986,
    "V5": -0.1536,
    "V6": -0.2258,
    "V7": -0.6378,
    "V8": 0.4615,
    "V9": 0.2060,
    "V10": 0.2167,
    "V11": 0.1000,
    "V12": -0.3481,
    "V13": -0.1357,
    "V14": 1.8495,
    "V15": -0.4291,
    "V16": -0.4680,
    "V17": -0.2090,
    "V18": 0.0211,
    "V19": 0.0245,
    "V20": 0.0381,
    "V21": 0.0339,
    "V22": -0.0355,
    "V23": 0.0356,
    "V24": 0.0183,
    "V25": -0.0899,
    "V26": -0.0327,
    "V27": -0.0232,
    "V28": 0.0105,
    "Amount": 149.62
}, indent=2)

# ---------------------------------------------------------------------------
# Build Gradio interface
# ---------------------------------------------------------------------------
with gr.Blocks(
    theme=gr.themes.Soft(  # type: ignore
        primary_hue="red",
        secondary_hue="slate",
        neutral_hue="slate"
    ),
    css="""
    body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .fraud-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 30px; color: white; border-radius: 15px; text-align: center; }
    .fraud-header h1 { margin: 0; font-size: 2.5em; }
    .fraud-header p { margin: 10px 0 0 0; font-size: 1.1em; opacity: 0.9; }
    .result-box { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                  padding: 25px; border-radius: 15px; border-left: 5px solid #667eea; }
    .prob-meter { width: 100%; height: 60px; background: #f0f0f0;
                  border-radius: 10px; position: relative; overflow: hidden; color:black; }
    .prob-fill { height: 100%; background: linear-gradient(90deg, #44ff44 0%, #ffaa00 50%, #ff4444 100%);
                 display: flex; align-items: center; justify-content: center;
                 color: white; font-weight: bold; font-size: 18px; }
    """
) as fraud_app:
    with gr.Group(elem_classes="fraud-header"):
        gr.Markdown("# 💳 Fraud Transaction Risk Predictor")
        gr.Markdown("Advanced ML-powered fraud detection for credit card transactions")

    with gr.Tabs():
        # Tab 1: Direct Input
        with gr.Tab("📝 Direct Input", id="direct"):
            with gr.Row():
                with gr.Column(scale=3):
                    gr.Markdown("### Transaction Features")

                    with gr.Row():
                        time = gr.Number(
                            label="⏱️ Time (seconds)",
                            value=0,
                            scale=1
                        )
                        amount = gr.Number(
                            label="💰 Amount",
                            value=100,
                            scale=1
                        )

                    # V features in grid
                    v_inputs = []
                    for i in range(1, 29, 4):
                        with gr.Row():
                            for j in range(i, min(i + 4, 29)):
                                v_input = gr.Number(
                                    label=f"V{j}",
                                    value=0,
                                    scale=1
                                )
                                v_inputs.append(v_input)

                    predict_btn_direct = gr.Button(
                        "🔍 Analyze Transaction",
                        size="lg",
                        variant="primary"
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Analysis Result")
                    prediction_text = gr.Textbox(
                        label="Status",
                        interactive=False,
                        scale=1
                    )

                    prob_value = gr.Number(
                        label="Risk Probability",
                        interactive=False,
                        scale=1
                    )

                    prob_display = gr.HTML()

            def update_direct(*args):
                text, prob, _risk = wrapped_predict_direct(*args)
                prob_html = f"""
                <div style="background: #f0f0f0; border-radius: 10px; overflow: hidden; height: 50px;">
                    <div style="width: {prob * 100}%; height: 100%;
                               background: linear-gradient(90deg, #44ff44 0%, #ffaa00 50%, #ff4444 100%);
                               display: flex; align-items: center; justify-content: center;
                               color: white; font-weight: bold; font-size: 16px;">
                        {prob * 100:.1f}%
                    </div>
                </div>
                """
                return text, prob, prob_html

            predict_btn_direct.click(
                fn=update_direct,
                inputs=[time] + v_inputs + [amount],
                outputs=[prediction_text, prob_value, prob_display]
            )

        # Tab 2: JSON Input
        with gr.Tab("📋 JSON Input", id="json"):
            with gr.Row():
                with gr.Column(scale=3):
                    json_input = gr.Textbox(
                        label="Paste your JSON transaction data here",
                        lines=20,
                        placeholder="Paste your JSON here...",
                        value=SAMPLE_JSON
                    )

                    predict_btn_json = gr.Button(
                        "🔍 Analyze Transaction",
                        size="lg",
                        variant="primary"
                    )

                with gr.Column(scale=2):
                    gr.Markdown("### 📋 JSON Format")
                    gr.Code(SAMPLE_JSON, language="json")

                    gr.Markdown("### 📊 Analysis Result")
                    prediction_text_json = gr.Textbox(
                        label="Status",
                        interactive=False,
                        scale=1
                    )

                    prob_value_json = gr.Number(
                        label="Risk Probability",
                        interactive=False,
                        scale=1
                    )

                    prob_display_json = gr.HTML()

            def update_json(json_str):
                text, prob, _risk = predict_from_json(json_str)
                prob_html = f"""
                <div style="background: #f0f0f0; border-radius: 10px; overflow: hidden; height: 50px;">
                    <div style="width: {prob * 100}%; height: 100%;
                               background: linear-gradient(90deg, #44ff44 0%, #ffaa00 50%, #ff4444 100%);
                               display: flex; align-items: center; justify-content: center;
                               color: white; font-weight: bold; font-size: 16px;">
                        {prob * 100:.1f}%
                    </div>
                </div>
                """
                return text, prob, prob_html

            predict_btn_json.click(
                fn=update_json,
                inputs=json_input,
                outputs=[prediction_text_json, prob_value_json, prob_display_json]
            )

    gr.Markdown("""
    ---
    ### ℹ️ About This Tool
    This fraud detection system analyzes 30 transaction features using machine learning.
    The probability meter shows the risk level: **Green (Low) → Yellow (Medium) → Red (High)**
    """)

if __name__ == "__main__":
    fraud_app.launch(server_name="0.0.0.0", server_port=7861)
