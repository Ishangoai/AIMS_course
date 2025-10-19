"""
Professional Gradio Application for a Fraud Detection Model.

This script launches a Gradio web interface for real-time and batch-based
credit card fraud detection. It features a rich UI with visual confidence
indicators, session analytics, and file-based processing.

To run this application, please ensure the required packages are installed:
gradio
joblib
pandas
numpy
scikit-learn
"""

import os
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import joblib
import numpy as np
import pandas as pd

# ==================== CONFIGURATION ====================
# Use an absolute path for robustness.
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "fraud_model_v4.pkl"
)

print("=" * 70)
print("🚀 FRAUD DETECTION MODEL - GRADIO APP")
print("=" * 70)
print(f"Attempting to load model from: {MODEL_PATH}")

# --- Robust Model Loading ---
if not os.path.exists(MODEL_PATH):
    error_msg = (
        f"FATAL ERROR: Model file not found at '{MODEL_PATH}'\n"
        "Please ensure 'fraud_model_v4.pkl' is in the same directory as this script."
    )
    print("=" * 70)
    print(error_msg)
    print("=" * 70)
    raise FileNotFoundError(error_msg)

model = joblib.load(MODEL_PATH)
print("✅ Model loaded successfully!")
print(f"Model type: {type(model).__name__}")
print("=" * 70)


# ==================== PREDICTION HISTORY ====================
prediction_history: List[Dict[str, Any]] = []


def log_prediction(inputs: Dict[str, float], prediction: int, timestamp: str) -> None:
    """
    Log a prediction to an in-memory list.

    Note: This global list is suitable for single-user demos. For production or
    multi-user apps, use gr.State or a dedicated database for session management.
    """
    prediction_history.append(
        {"timestamp": timestamp, "prediction": prediction, "inputs": inputs}
    )
    # Keep history capped at a reasonable number to avoid memory issues.
    if len(prediction_history) > 200:
        prediction_history.pop(0)


# ==================== VISUAL COMPONENTS ====================
def create_gauge_html(confidence: float, prediction_class: int) -> str:
    """Create a professional, compact gauge/speedometer visualization using HTML."""
    confidence_pct = confidence * 100

    # Determine visual elements based on prediction.
    if prediction_class == 0:  # Legitimate
        color, bg_color, icon, label, risk_level = (
            "#10b981",
            "#d1fae5",
            "✓",
            "LEGITIMATE",
            "LOW RISK",
        )
    else:  # Fraud
        color, bg_color, icon, label, risk_level = (
            "#ef4444",
            "#fee2e2",
            "⚠",
            "FRAUDULENT",
            "HIGH RISK",
        )

    rotation = -90 + (confidence_pct * 1.8)

    return f"""
    <div style="background: linear-gradient(135deg, {bg_color} 0%, white 100%);
                padding: 20px; border-radius: 20px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1); height: 100%;">
        <!-- Main Result Badge -->
        <div style="text-align: center; margin-bottom: 20px;">
            <div style="background-color: {color}; color: white;
                        padding: 15px 30px; border-radius: 15px;
                        display: inline-block;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                <div style="font-size: 36px; margin-bottom: 8px;">{icon}</div>
                <div style="font-size: 22px; font-weight: bold;
                            letter-spacing: 1.5px;">{label}</div>
                <div style="font-size: 14px; margin-top: 6px;
                            opacity: 0.9;">{risk_level}</div>
            </div>
        </div>
        <!-- Confidence Gauge -->
        <div style="text-align: center; margin: 20px 0;">
            <div style="font-size: 16px; color: #666; margin-bottom: 10px;
                        font-weight: 600;">CONFIDENCE LEVEL</div>
            <div style="position: relative; width: 220px; height: 125px; margin: 0 auto;">
                <svg width="220" height="125" style="position: absolute; top: 0; left: 0;">
                    <path d="M 30 110 A 80 80 0 0 1 190 110" stroke="#e5e7eb"
                        stroke-width="24" fill="none" stroke-linecap="round"/>
                </svg>
                <svg width="220" height="125" style="position: absolute; top: 0; left: 0;">
                    <path d="M 30 110 A 80 80 0 0 1 190 110" stroke="{color}"
                        stroke-width="24" fill="none" stroke-linecap="round"
                        stroke-dasharray="251.2"
                        stroke-dashoffset="{251.2 - (confidence_pct * 2.512)}"
                        style="transition: stroke-dashoffset 1s ease-out;"/>
                </svg>
                <div style="position: absolute; bottom: 15px; left: 50%;
                            width: 3px; height: 70px; background: #333;
                            transform-origin: bottom center;
                            transform: translateX(-50%) rotate({rotation}deg);
                            transition: transform 1s ease-out; border-radius: 2px;"></div>
                <div style="position: absolute; bottom: 10px; left: 50%;
                            transform: translateX(-50%); width: 16px; height: 16px;
                            background: #333; border-radius: 50%;
                            box-shadow: 0 2px 6px rgba(0,0,0,0.3);"></div>
            </div>
            <div style="margin-top: 15px;">
                <div style="font-size: 42px; font-weight: bold;
                            color: {color};">{confidence_pct:.1f}%</div>
            </div>
        </div>
    </div>
    """


def create_result_summary(
    prediction: int,
    confidence: float,
    time_val: float,
    amount_val: float,
    timestamp: str,
) -> str:
    """Create a detailed result summary in Markdown format."""
    interpretation = (
        "⚠️ **HIGH RISK**: This transaction shows strong indicators of "
        "fraudulent activity. Recommend immediate review."
        if prediction == 1
        else "✅ **LOW RISK**: This transaction appears legitimate based on learned patterns."
    )
    return f"""
### 📊 Transaction Analysis

| Metric | Value |
|--- |--- |
| **Prediction** | `Class {int(prediction)}` ({'Fraud' if prediction == 1 else 'Legitimate'}) |
| **Confidence** | `{confidence * 100:.2f}%` |
| **Transaction Amount** | `${amount_val:,.2f}` |
| **Time (seconds)** | `{time_val:,.0f}` |
| **Analysis Timestamp** | `{timestamp}` |

---

#### 💡 Interpretation
{interpretation}
    """


# ==================== PREDICTION FUNCTIONS ====================
def make_prediction(
    time: float, amount: float, *v_features: float
) -> Tuple[str, str]:
    """Make a prediction for a single transaction and log it to history."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        feature_names = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        feature_values = [float(time)] + list(v_features) + [float(amount)]

        input_dict = dict(zip(feature_names, feature_values))
        input_df = pd.DataFrame([input_dict])

        # The model object is loaded from joblib, so its type is dynamic.
        prediction = model.predict(input_df)[0]  # type: ignore

        confidence = 0.5
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]  # type: ignore
            confidence = float(np.max(probabilities))

        log_prediction(input_dict, prediction, timestamp)

        gauge_html = create_gauge_html(confidence, prediction)
        summary = create_result_summary(
            prediction, confidence, time, amount, timestamp
        )

        return gauge_html, summary

    except Exception as e:
        error_html = f"""
        <div style="background-color: #fee2e2; color: #991b1b; padding: 20px;
                    border-radius: 15px; text-align: center;">
            <div style="font-size: 24px; font-weight: bold;
                        margin-bottom: 10px;">⚠️ Prediction Error</div>
            <code style="background-color: #fca5a5; padding: 5px 10px;
                         border-radius: 5px;">{e}</code>
        </div>
        """
        return error_html, f"**Error**: {e}"


def batch_predict(
    file_obj: Optional[tempfile._TemporaryFileWrapper],
) -> Tuple[pd.DataFrame, Optional[str], str]:
    """Make predictions on a batch of data and log the results to history."""
    try:
        if file_obj is None:
            raise ValueError("No file uploaded. Please upload a CSV file.")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df = pd.read_csv(file_obj.name)

        model_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        required_cols = set(model_cols)

        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(
                f"CSV is missing required columns: {', '.join(sorted(list(missing)))}"
            )

        predictions = model.predict(df[model_cols])  # type: ignore
        df["prediction"] = predictions

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(df[model_cols])  # type: ignore
            df["confidence"] = probabilities.max(axis=1)

        for record in df.to_dict("records"):
            input_features = {col: record[col] for col in model_cols}
            log_prediction(
                inputs=input_features,
                prediction=record["prediction"],
                timestamp=timestamp,
            )

        with tempfile.NamedTemporaryFile(
            delete=False, mode="w", suffix=".csv", newline=""
        ) as tmpfile:
            df.to_csv(tmpfile.name, index=False)
            output_path: Optional[str] = tmpfile.name

        pred_counts = df["prediction"].value_counts()
        total = len(df)
        fraud_count = int(pred_counts.get(1, 0))  # type: ignore
        legit_count = int(pred_counts.get(0, 0))  # type: ignore

        alert_msg = ""
        if fraud_count > 0:
            alert_msg += "---\n"
            if fraud_count / total > 0.05:
                alert_msg += (
                    "⚠️ **High Alert**: Significant fraudulent activity detected."
                )

        summary = f"""
## ✅ Batch Prediction Complete

| Metric | Value |
|--- |--- |
| **Total Processed** | `{total:,}` |
| **Legitimate (Class 0)** | `{legit_count:,}` (`{legit_count / total:.1%}`) |
| **Fraudulent (Class 1)** | `{fraud_count:,}` (`{fraud_count / total:.1%}`) |
{alert_msg}

**💾 Download the complete results below. Session history has been updated.**
"""
        return df.head(100), output_path, summary

    except Exception as e:
        error_df = pd.DataFrame({"Error": [str(e)]})
        return error_df, None, f"**Batch Prediction Error**: {e}"


def get_prediction_stats() -> str:
    """Generate and format statistics from the current session's prediction history."""
    if not prediction_history:
        return (
            "## 📊 No predictions made in this session yet. "
            "Use the other tabs to get started!"
        )

    total = len(prediction_history)
    class_0 = sum(1 for p in prediction_history if p["prediction"] == 0)
    class_1 = total - class_0

    stats_header = f"""
## 📊 Session Analytics

| Metric | Value |
|--- |--- |
| **Total Predictions** | `{total}` |
| **Session Start Time** | `{prediction_history[0]['timestamp']}` |
| **Last Prediction Time** | `{prediction_history[-1]['timestamp']}` |

### 📈 Prediction Distribution
**Legitimate (Class 0):** `{class_0}` (`{class_0 / total:.1%}`)
<div style="background:#d1fae5; border-radius:5px; padding:2px 5px;
            width:{max(2, class_0 / total * 100)}%; color:#065f46;
            font-weight:bold;">{class_0}</div>

**Fraudulent (Class 1):** `{class_1}` (`{class_1 / total:.1%}`)
<div style="background:#fee2e2; border-radius:5px; padding:2px 5px;
            width:{max(2, class_1 / total * 100)}%; color:#991b1b;
            font-weight:bold;">{class_1}</div>

### 🕐 Recent Activity (Last 10)
"""
    recent_activity = "\n".join(
        [
            f"{i}. {'🔴' if p['prediction'] == 1 else '🟢'} "
            f"`{p['timestamp']}` → **Class {p['prediction']}**"
            for i, p in enumerate(reversed(prediction_history[-10:]), 1)
        ]
    )

    return stats_header + recent_activity


def load_sample_class_0() -> Tuple[float, ...]:
    """Return a tuple of feature values for a legitimate transaction."""
    return (
        62800.0,
        8.99,
        -1.508148,
        1.853128,
        0.220932,
        -0.146818,
        -0.528976,
        -0.626682,
        -0.166863,
        0.999947,
        -0.499523,
        -0.156841,
        0.847756,
        0.538299,
        -0.354516,
        0.352375,
        0.234339,
        0.906489,
        -0.010299,
        0.576838,
        0.256062,
        0.130144,
        -0.218249,
        -0.752300,
        0.059790,
        -0.071737,
        -0.017378,
        0.090617,
        0.204202,
        0.075687,
    )


def load_sample_class_1() -> Tuple[float, ...]:
    """Return a tuple of feature values for a fraudulent transaction."""
    return (
        35899.0,
        1.0,
        -2.857170,
        4.045601,
        -4.197299,
        5.487199,
        -3.070776,
        -1.422686,
        -5.651314,
        2.019657,
        -5.015491,
        -6.319708,
        3.779602,
        -8.077094,
        1.440889,
        -7.891909,
        0.530453,
        -7.954070,
        -14.265056,
        -5.771064,
        2.892170,
        0.981609,
        1.080323,
        -0.561384,
        0.102678,
        -0.067195,
        -0.476931,
        -0.103716,
        1.166961,
        0.663632,
    )


# --- CSS FOR SCROLLABLE V-FEATURES BOX ---
# This CSS targets the element with the ID 'v-features-box' and makes it a
# scrollable container.
CSS = """
#v-features-box {
    max-height: 400px;
    overflow-y: scroll;
    padding: 10px;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
}
"""

# ==================== GRADIO INTERFACE ====================
with gr.Blocks(
    title="Professional Fraud Detection System",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="gray"),  # type: ignore[attr-defined]
    css=CSS,
) as fraud_app:
    gr.Markdown(
        "# 🛡️ Credit Card Fraud Detection System\n"
        "### Professional ML-Powered Transaction Analysis"
    )

    with gr.Tabs():
        # ==================== TAB 1: SINGLE PREDICTION ====================
        with gr.Tab("🎯 Single Prediction", id=0):
            with gr.Row():
                with gr.Column(scale=1, min_width=350):
                    gr.Markdown("### 📝 Transaction Details")
                    with gr.Group():
                        time_input = gr.Number(
                            label="⏱️ Time (seconds)",
                            value=62800.0,
                            info="Time elapsed since first transaction.",
                        )
                        amount_input = gr.Number(
                            label="💵 Amount ($)",
                            value=8.99,
                            info="Transaction amount.",
                        )
                    with gr.Accordion("🔐 Advanced Features (V1-V28)", open=False):
                        with gr.Group(elem_id="v-features-box"):
                            v_inputs = [
                                gr.Number(label=f"V{i}", value=0.0, precision=6)
                                for i in range(1, 29)
                            ]

                    all_inputs = [time_input, amount_input] + v_inputs
                    with gr.Row():
                        predict_btn = gr.Button(
                            "🔍 Analyze Transaction", variant="primary", scale=2
                        )
                        clear_btn = gr.ClearButton(
                            components=all_inputs, value="🗑️ Clear", scale=1
                        )
                    with gr.Row():
                        sample_0_btn = gr.Button("Load Legitimate Example")
                        sample_1_btn = gr.Button("Load Fraudulent Example")

                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Analysis Results")
                    with gr.Row():
                        result_visualization = gr.HTML(
                            """
                            <div style="display: flex; align-items: center;
                                        justify-content: center; height: 100%;
                                        min-height: 450px; background: #f3f4f6;
                                        border-radius: 20px;">
                                <div style="text-align: center; color: #6b7280;">
                                    <div style="font-size: 48px; opacity: 0.5;">🎯</div>
                                    <h3 style="font-size: 18px;
                                               font-weight: 500;">Ready to Analyze</h3>
                                </div>
                            </div>
                            """
                        )
                        result_details = gr.Markdown(
                            "### Details will appear here after analysis."
                        )

            predict_btn.click(
                fn=make_prediction,
                inputs=all_inputs,
                outputs=[result_visualization, result_details],
            )
            sample_0_btn.click(
                fn=load_sample_class_0, inputs=None, outputs=all_inputs
            )
            sample_1_btn.click(
                fn=load_sample_class_1, inputs=None, outputs=all_inputs
            )

        # ==================== TAB 2: ANALYTICS DASHBOARD ====================
        with gr.Tab("📈 Analytics Dashboard", id=1):
            gr.Markdown(
                "### Session Statistics & History\n"
                "Track all predictions made during this session, including "
                "from batch analysis."
            )
            refresh_btn = gr.Button("🔄 Refresh Statistics", variant="secondary")
            stats_display = gr.Markdown("*Click refresh to view analytics.*")
            refresh_btn.click(
                fn=get_prediction_stats, inputs=None, outputs=[stats_display]
            )

        # ==================== TAB 3: BATCH ANALYSIS ====================
        with gr.Tab("📁 Batch Analysis", id=2):
            gr.Markdown(
                "### Upload a CSV for Bulk Transaction Analysis\n"
                "Your CSV must contain columns: `Time`, `Amount`, and "
                "`V1` through `V28`."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(
                        label="📂 Upload CSV File",
                        file_types=[".csv"],
                    )
                    batch_btn = gr.Button("🚀 Process Batch File", variant="primary")
                with gr.Column(scale=2):
                    batch_summary = gr.Markdown("###  Awaiting batch processing...")

            gr.Markdown("---")
            batch_output_df = gr.DataFrame(
                label="📊 Results Preview (First 100 rows)",
                wrap=True,
                interactive=False,
            )
            download_output_file = gr.File(label="💾 Download Full Results")

            batch_btn.click(
                fn=batch_predict,
                inputs=[file_input],
                outputs=[batch_output_df, download_output_file, batch_summary],
            )

        # ==================== TAB 4: ABOUT ====================
        with gr.Tab("ℹ️ Information", id=3):
            gr.Markdown(
                f"""
## 📦 Model & Application Information
| Property | Value |
|--- |--- |
| **Model File** | `{os.path.basename(MODEL_PATH)}` |
| **Model Type** | `{type(model).__name__}` |
| **Input Features** | 30 (Time, Amount, V1-V28) |
| **Output Classes** | 2 (0: Legitimate, 1: Fraudulent) |
---
## 🎨 Key Features
- **Real-time Predictions**: Instant fraud scoring with a side-by-side gauge.
- **Stable Layout**: Advanced features are in a scrollable window.
- **Batch Processing**: Analyze multiple transactions by uploading a CSV file.
- **Comprehensive Analytics**: A dashboard to track prediction statistics.
- **Export Capability**: Download results with detailed probability scores.
"""
            )

if __name__ == "__main__":
    fraud_app.launch(debug=True)
