"""
Gradio application for Fraud Detection Model Deployment
Loads pre-trained model from pickle file
"""

import os

import gradio as gr
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# ============================================================================
# Model Loading
# ============================================================================

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
try:
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {str(e)}")
    model = None

MODEL_NAME = "fraud-detection-random-forest"
PRODUCTION_STAGE = "Production"

# ============================================================================
# Preprocessing
# ============================================================================


def preprocess_input(time_val, amount_val, *v_vals):
    input_data = np.array([[time_val, amount_val] + list(v_vals)])
    scaler = StandardScaler()
    input_data[:, 0] = scaler.fit_transform(input_data[:, [0]])
    input_data[:, 1] = scaler.fit_transform(input_data[:, [1]])
    return input_data

# ============================================================================
# Dashboard Statistics
# ============================================================================


prediction_stats = {"total_predictions": 0, "fraud_detected": 0, "normal_detected": 0, "recent_transactions": []}


def update_stats(is_fraud, fraud_prob, timestamp):
    prediction_stats["total_predictions"] += 1
    if is_fraud:
        prediction_stats["fraud_detected"] += 1
    else:
        prediction_stats["normal_detected"] += 1
    prediction_stats["recent_transactions"].insert(0, {
        "time": timestamp,
        "result": "🚨 Fraud" if is_fraud else "✅ Normal",
        "probability": f"{fraud_prob:.1f}%"
    })
    if len(prediction_stats["recent_transactions"]) > 10:
        prediction_stats["recent_transactions"].pop()


def get_dashboard_stats():
    total = prediction_stats["total_predictions"]
    fraud_rate = (prediction_stats["fraud_detected"] / total * 100) if total > 0 else 0
    stats_text = f"""
📊 **Session Statistics**
- Total Predictions: {total}
- Frauds Detected: {prediction_stats['fraud_detected']} ({fraud_rate:.1f}%)
- Normal Transactions: {prediction_stats['normal_detected']}
"""
    recent_text = "\n📋 **Recent Transactions (Last 10)**\n"
    if prediction_stats["recent_transactions"]:
        recent_text += "| Time | Result | Fraud Prob |\n|------|--------|------------|\n"
        for tx in prediction_stats["recent_transactions"]:
            recent_text += f"| {tx['time']} | {tx['result']} | {tx['probability']} |\n"
    else:
        recent_text += "*No predictions yet*"
    return stats_text + recent_text


def get_feature_stats(time_val, amount_val, *v_vals):
    v_mean, v_std, v_min, v_max = np.mean(v_vals), np.std(v_vals), np.min(v_vals), np.max(v_vals)
    stats = f"""
📈 **Input Feature Statistics**
- Time: {time_val:.2f} seconds
- Amount: ${amount_val:.2f}
- V Features Mean: {v_mean:.4f}
- V Features Std Dev: {v_std:.4f}
- V Features Range: [{v_min:.4f}, {v_max:.4f}]
"""
    return stats

# ============================================================================
# Prediction Function
# ============================================================================


def predict_fraud(*inputs):
    from datetime import datetime
    if model is None:
        return (
            {"Error": "Model not loaded.", "Status": "Model unavailable"},
            get_dashboard_stats(),
            get_feature_stats(*inputs),
        )

    try:
        processed_data = preprocess_input(*inputs)
        prediction = model.predict(processed_data)[0]
        probabilities = model.predict_proba(processed_data)[0]
        is_fraud = bool(prediction)
        fraud_prob, normal_prob = float(probabilities[1]) * 100, float(probabilities[0]) * 100
        timestamp = datetime.now().strftime("%H:%M:%S")
        update_stats(is_fraud, fraud_prob, timestamp)
        result = {
            "🎯 Prediction": "🚨 FRAUD DETECTED" if is_fraud else "✅ Normal Transaction",
            "📊 Fraud Probability": f"{fraud_prob:.2f}%",
            "📊 Normal Probability": f"{normal_prob:.2f}%",
            "🔢 Model Version": "2",
            "✨ Confidence": f"{max(fraud_prob, normal_prob):.2f}%",
            "🕐 Timestamp": timestamp
        }
        return result, get_dashboard_stats(), get_feature_stats(*inputs)
    except Exception as e:
        return {"Error": f"Prediction failed: {str(e)}"}, get_dashboard_stats(), get_feature_stats(*inputs)

# ============================================================================
# Gradio Interface
# ============================================================================


if model is not None:
    model_status = "✅ Model Loaded"
else:
    model_status = "❌ Model Not Loaded"


css_code = """
body {
    background-color: #e6f0fa;
    font-family: 'Segoe UI', Arial, sans-serif;
    color: #1b2838;
}

.gr-row {
    gap: 15px;
}

.gr-column {
    gap: 15px;
}

.gr-button.primary {
    background-color: #1e70bf;
    color: white;
    border-radius: 8px;
}

.gr-button.primary:hover {
    background-color: #145a9c;
}

.gr-button.secondary {
    background-color: #a0c4ff;
    color: #1b2838;
    border-radius: 8px;
}

.section-header {
    background-color: #a0c4ff;
    color: #1b2838;
    font-weight: bold;
    font-size: 18px;
    border-radius: 8px;
    padding: 8px;
}

.main-header {
    background-color: #1e70bf;
    padding: 25px;
    border-radius: 15px;
    margin-bottom: 20px;
    color: white;
    text-align: center;
}

.main-header h1 {
    margin: 0;
    font-size: 32px;
    font-weight: bold;
}

.status-badge {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 12px;
    background-color: #80b3e6;
    color: white;
    font-weight: bold;
    margin-top: 10px;
}

.result-card {
    background-color: #d6e6f5;
    border-radius: 12px;
    border: 1px solid #90c1e3;
    font-size: 16px;
    color: #1b2838;
    padding: 10px;
}

.dashboard-card {
    background-color: #1e5a8a;
    border-radius: 12px;
    border: 1px solid #4a7ba7;
    padding: 10px;
    color: white;
}

.stats-card {
    background-color: #264a6e;
    border-radius: 12px;
    border: 1px solid #4a7ba7;
    padding: 10px;
    color: white;
}

.footer-section {
    background-color: #a0c4ff;
    padding: 12px;
    border-radius: 12px;
    color: #1b2838;
}
"""

with gr.Blocks(
    css=css_code,
    title="Fraud Detection System",
    theme=gr.Theme(primary_hue="blue")
) as fraud_app:

    with gr.Row():
        with gr.Column():
            gr.Markdown(f"""
                <div class="main-header">
                    <h1>💳 Credit Card Fraud Detection System</h1>
                    <div class="status-badge">{model_status} | Model: {MODEL_NAME}</div>
                </div>
            """, elem_classes="main-header")

    if model is None:
        gr.Markdown("⚠️ **Warning**: Model file not found at `model.pkl`.")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 💰 Transaction Details", elem_classes="section-header")
            time_input = gr.Number(label="⏱️ Time (seconds elapsed)", value=0)
            amount_input = gr.Number(label="💵 Amount ($)", value=1.0)
        with gr.Column(scale=2):
            gr.Markdown("### 🔢 PCA Features (V1—V28)", elem_classes="section-header")
            v_features = []
            for i in range(0, 28, 7):
                with gr.Row():
                    for j in range(1, 8):
                        idx = i + j
                        if idx <= 28:
                            v_input = gr.Number(label=f"V{idx}", value=0.0)
                            v_features.append(v_input)

    with gr.Row():
        predict_button = gr.Button("🔍 Predict Fraud", variant="primary")
        clear_button = gr.Button("🗑️ Clear All", variant="secondary")

    gr.Markdown("### 📊 Prediction Results", elem_classes="section-header")
    output = gr.JSON(
    value={"Status": "👆 Enter transaction details and click 'Predict Fraud'"},
    elem_classes="result-card"
)

    with gr.Row():
        with gr.Column(scale=1):
            dashboard_output = gr.Markdown(
                get_dashboard_stats(),
                elem_classes="dashboard-card"
            )
        with gr.Column(scale=1):
            feature_stats_output = gr.Markdown(
                get_feature_stats(0, 0, *[0] * 28),
                elem_classes="stats-card"
            )

    def wrapped_predict(*args):
        return predict_fraud(*args)

    predict_button.click(
    fn=wrapped_predict,
    inputs=[
        time_input,
        amount_input,
        *v_features,
    ],
    outputs=[
        output,
        dashboard_output,
        feature_stats_output,
    ],
)

    def clear_all():
        return (
            [0] * (2 + len(v_features))
            + [
                {"Status": "✨ Cleared!"},
                get_dashboard_stats(),
                get_feature_stats(0, 0, *[0] * 28),
            ]
        )

    clear_button.click(
    fn=clear_all,
    outputs=[
        time_input,
        amount_input,
        *v_features,
        output,
        dashboard_output,
        feature_stats_output,
    ]
)

    gr.Markdown("""
    <div class="footer-section">
        <h3>📖 How It Works</h3>
        <ul>
            <li><strong>Input:</strong> Transaction features (Time, Amount, V1—V28)</li>
            <li><strong>Model:</strong> Random Forest classifier trained with 3-fold cross-validation</li>
            <li><strong>Output:</strong> Fraud prediction with probabilities</li>
            <li><strong>Note:</strong> Demo model for educational purposes</li>
        </ul>
    </div>
    """)
