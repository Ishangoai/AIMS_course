"""Merged Gradio fraud detection interface.

Combines interactive UI components with real model inference from
`gradioapp.utils.fraud_utils.predict_fraud`.  Provides:
- Single-transaction prediction with visualization.
- Batch CSV prediction.
- Example data fillers (random, zero, sample).
"""

from __future__ import annotations

import re
import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from gradioapp.utils.fraud_utils import predict_fraud


# =====================================================
# Helper functions
# =====================================================


def predict_transaction(time: float, amount: float, *v_features: float):
    """Predict a single transaction and render visualization.

    Args:
        time: Transaction time index.
        amount: Transaction amount.
        *v_features: The V1–V28 feature values.

    Returns:
        Tuple containing:
            - fraud_likelihood (str): Probability as percentage text.
            - gauge (go.Figure): Gauge plot of fraud probability.
            - feature_importance (go.Figure): Placeholder bar chart.
            - result_html (str): Formatted HTML summary.
    """
    features = [*v_features, amount]
    result_text = predict_fraud(features)

    match = re.search(r"\(([\d.]+)% probability\)", result_text)
    if match:
        prob = float(match.group(1)) / 100.0
        if "Legitimate" in result_text:
            prob = 1 - prob
    else:
        prob = 0.0

    fraud_probability = prob
    fraud_likelihood = f"{fraud_probability * 100:.2f}%"
    prediction_label = "Fraudulent" if fraud_probability >= 0.5 else "Legitimate"

    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=fraud_probability * 100,
            title={"text": "Fraud Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red" if fraud_probability >= 0.5 else "green"},
                "steps": [
                    {"range": [0, 50], "color": "lightgreen"},
                    {"range": [50, 100], "color": "salmon"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": fraud_probability * 100,
                },
            },
        )
    )

    importance = np.random.random(len(v_features))
    feature_importance = go.Figure(
        go.Bar(
            x=[f"V{i+1}" for i in range(len(v_features))],
            y=importance,
            marker_color="steelblue",
        )
    )
    feature_importance.update_layout(title="Feature Importance (simulated)")

    result_html = f"""
    <div style="font-size: 1.2em; font-family: Arial;">
        <b>Prediction:</b> {prediction_label}<br>
        <b>Fraud Probability:</b> {fraud_likelihood}<br>
        <b>Model Output:</b> {result_text}
    </div>
    """

    return fraud_likelihood, gauge, feature_importance, result_html


# =====================================================
# Batch CSV Prediction
# =====================================================


def batch_predict(csv_file: gr.File | None):
    """Perform batch predictions on uploaded CSV data.

    Args:
        csv_file: Uploaded CSV file object.

    Returns:
        Tuple[str, pd.DataFrame]: Message and dataframe of predictions.
    """
    if csv_file is None:
        return (
            "Please upload dataset",
            pd.DataFrame(columns=["TransactionID", "Prediction", "Fraud Probability"]),
        )

    df = pd.read_csv(csv_file.name)
    predictions = []
    for _, row in df.iterrows():
        inputs = [row.get(f"V{i}", 0.0) for i in range(1, 29)] + [
            row.get("Amount", 0.0)
        ]
        result = predict_fraud(inputs)
        match = re.search(r"\(([\d.]+)% probability\)", result)
        prob = 1 - (float(match.group(1)) / 100.0) if match else 0.0
        predictions.append(
            {
                "TransactionID": row.get("Time", None),
                "Prediction": result,
                "Fraud Probability": round(prob, 2),
            }
        )
    return "", pd.DataFrame(predictions)


def clear_predictions() -> pd.DataFrame:
    """Return an empty dataframe for clearing batch output."""
    return pd.DataFrame(columns=["TransactionID", "Prediction", "Fraud Probability"])


# =====================================================
# Randomizers / Fillers
# =====================================================


def fill_random() -> list[float]:
    """Generate random example feature values."""
    values = np.random.uniform(-5, 5, 30).tolist()
    return [int(values[0])] + [abs(values[1])] + values[2:]


def fill_zeros() -> list[int]:
    """Return zero-filled example values."""
    return [0 for _ in range(30)]


def fill_sample() -> list[float]:
    """Return a linearly spaced example feature vector."""
    sample = np.linspace(-1, 1, 30).tolist()
    return [int(sample[0])] + [abs(sample[1])] + sample[2:]


# =====================================================
# Gradio UI Layout
# =====================================================


with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# 💳 Fraud Detection System\nMonitor and classify financial transactions.")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### Input Transaction Data")

            time_input = gr.Number(label="Transaction Time")
            amount_input = gr.Number(label="Transaction Amount")
            v_inputs = [gr.Number(label=f"V{i}") for i in range(1, 29)]

            with gr.Row():
                random_btn = gr.Button("🎲 Random Values")
                zero_btn = gr.Button("🧊 Zero Fill")
                sample_btn = gr.Button("📊 Sample Values")

            predict_btn = gr.Button("🚀 Predict Transaction")
            clear_btn = gr.Button("🧹 Clear Inputs")

        with gr.Column(scale=3):
            gr.Markdown("### Prediction Output")

            fraud_likelihood = gr.Textbox(label="Fraud Probability")
            gauge_plot = gr.Plot(label="Fraud Probability Gauge")
            feature_importance_plot = gr.Plot(label="Feature Importance")
            result_html = gr.HTML(label="Prediction Summary")

            gr.Markdown("### 💾 Batch CSV Prediction")
            csv_input = gr.File(label="Upload CSV", file_types=[".csv"])
            batch_output = gr.Dataframe(
                headers=["TransactionID", "Prediction", "Fraud Probability"]
            )
            batch_btn = gr.Button("Run Batch Prediction")
            clear_batch_btn = gr.Button("Clear Results")

            gr.Markdown("### 🤖 System Info")
            gr.Markdown("*Model source:* gradioapp.utils.fraud_utils")

    predict_btn.click(
        predict_transaction,
        inputs=[time_input, amount_input] + v_inputs,
        outputs=[fraud_likelihood, gauge_plot, feature_importance_plot, result_html],
    )

    random_btn.click(fn=fill_random, inputs=None, outputs=[time_input, amount_input] + v_inputs)
    zero_btn.click(fn=fill_zeros, inputs=None, outputs=[time_input, amount_input] + v_inputs)
    sample_btn.click(fn=fill_sample, inputs=None, outputs=[time_input, amount_input] + v_inputs)

    batch_btn.click(fn=batch_predict, inputs=csv_input, outputs=[gr.Textbox(), batch_output])
    clear_batch_btn.click(fn=clear_predictions, inputs=None, outputs=batch_output)
