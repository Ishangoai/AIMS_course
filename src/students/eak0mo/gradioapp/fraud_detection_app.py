from __future__ import annotations

from typing import Any

import gradio as gr
import plotly.graph_objects as go
from gradioapp.utils.fraud_utils import predict_fraud

# -----------------------------
# Example transaction defaults
# -----------------------------
EXAMPLE_TRANSACTION: list[float] = [
    1.18081256, -0.151318004, 0.735193402, 0.394006458, -0.934756778,
    -0.943678475, -0.130194211, -0.148228409, 0.589976142, -0.308470408,
    -0.283332093, 0.614995531, -0.013868061, -0.142073926, -0.110624364,
    -0.241598526, 0.193491029, -0.982638787, 0.293116998, -0.042024044,
    -0.287134741, -0.721555218, 0.124741513, 0.779478755, 0.147347019,
    0.798826508, -0.068182379, 0.014936089, 31.4,
]


# -----------------------------
# Donut Meter Visualization
# -----------------------------
def wrapped_predict(*args: Any) -> tuple[go.Figure, str]:
    """Predicts fraud probability and visualizes it as a labeled donut meter."""

    # Safely parse all inputs
    inputs: list[float] = []
    for i, arg in enumerate(args):
        if arg is None:
            inputs.append(EXAMPLE_TRANSACTION[i])
        else:
            try:
                inputs.append(float(arg))
            except (ValueError, TypeError):
                inputs.append(EXAMPLE_TRANSACTION[i])

    result: str = predict_fraud(inputs)

    # Extract fraud probability (0–1)
    try:
        if "Fraudulent" in result:
            prob = float(result.split("(")[1].split("%")[0]) / 100
        elif "Legitimate" in result:
            prob = 1 - (float(result.split("(")[1].split("%")[0]) / 100)
        else:
            prob = 0.0
    except (IndexError, ValueError):
        prob = 0.0

    # Compute colors and status label dynamically
    if prob < 0.33:
        color = "#2ecc71"  # Green
        status = "Legitimate ✅"
    elif prob < 0.66:
        color = "#f1c40f"  # Yellow
        status = "Suspicious ⚠️"
    else:
        color = "#e74c3c"  # Red
        status = "Fraudulent 🚨"

    # --- Donut chart ---
    fig = go.Figure()

    # Add donut pie chart
    fig.add_trace(
        go.Pie(
            labels=["Fraudulent", "Legitimate"],
            values=[prob, 1 - prob],
            hole=0.65,
            direction="clockwise",
            textinfo="label+percent",
            textfont=dict(size=14, color="#333", family="Inter, sans-serif"),
            marker=dict(
                colors=["#e74c3c", "#2ecc71"],
                line=dict(color="#f9f9f9", width=3),
            ),
            hoverinfo="label+percent",
        )
    )

    # Add center label
    fig.add_annotation(
        text=f"<b>{prob * 100:.1f}%</b><br><span style='font-size:14px'>{status}</span>",
        showarrow=False,
        font=dict(size=22, color=color, family="Inter, sans-serif"),
        x=0.5,
        y=0.5,
        xanchor="center",
        yanchor="middle",
    )

    # --- Update layout (fixed syntax) ---
    fig.update_layout(
        title=dict(
            text="Fraud Probability Donut Meter",
            font=dict(size=22, color="#3b0d91", family="Inter, sans-serif"),
            x=0.5,
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            font=dict(size=14, family="Inter, sans-serif"),
        ),
        height=450,
        width=450,
        margin=dict(l=40, r=40, t=80, b=60),
        paper_bgcolor="#f9f9f9",
        plot_bgcolor="#f9f9f9",
    )

    # Styled HTML output
    fraud_text = (
        f"<div style='text-align:center; font-size:20px; margin-top:10px;'>"
        f"<b style='color:{color};'>{result}</b></div>"
    )

    return fig, fraud_text


# -----------------------------
# Example + reset handlers
# -----------------------------
def fill_example() -> list[float]:
    """Return default example transaction values."""
    return EXAMPLE_TRANSACTION


def reset_inputs() -> list[None]:
    """Reset all input sliders and fields to None."""
    return [None] * 29


# -----------------------------
# Gradio App UI
# -----------------------------
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo"),  # type: ignore
    css="""
    body { background-color: #f8fbff; font-family: 'Inter', sans-serif; }
    .feature-col { padding: 6px; }
    .centered-container {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        width: 100%;
        text-align: center;
        margin-top: 20px;
    }
    .gr-button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 8px 16px !important;
    }
    """,
) as fraud_app:
    # Header
    gr.Markdown(
        """
        <div style='text-align:center;'>
            <h1 style='color:#3b0d91;'>💳 Fraud Detection Predictor</h1>
            <p style='font-size:16px; color:#555; max-width:700px; margin:auto;'>
                Use the sliders below to simulate a transaction and check whether it's
                <strong>Fraudulent</strong> or <strong>Legitimate</strong>.
            </p>
            <p style='font-size:14px; color:#777;'>
                Built by: Elisha Komolafe 🇳🇬 & Lionel Cedric Gohouede 🇧🇯(2025)
            </p>
        </div>
        """,
    )

    gr.Markdown("### 🧮 Input Transaction Features")

    # Input features (4 columns × 7 sliders)
    with gr.Row():
        feature_inputs: list[gr.Slider] = []
        for col in range(4):
            with gr.Column(elem_classes=["feature-col"]):
                for i in range(col * 7, (col + 1) * 7):
                    feature_inputs.append(
                        gr.Slider(
                            label=f"V{i + 1}",
                            value=EXAMPLE_TRANSACTION[i],
                            minimum=-3.0,
                            maximum=3.0,
                            step=0.01,
                            interactive=True,
                        )
                    )

    # Amount input
    amount = gr.Textbox(
        label="💵 Transaction Amount",
        placeholder="Enter amount (e.g. 120.50)",
        value=str(EXAMPLE_TRANSACTION[28]),
    )

    # Action buttons
    with gr.Row():
        predict_btn = gr.Button("🚀 Predict", variant="primary")
        example_btn = gr.Button("💡 Example")
        reset_btn = gr.Button("🔄 Reset")

    # Visualization
    gr.Markdown("### 🔍 Prediction Visualization")
    with gr.Column(elem_classes=["centered-container"]):
        prob_chart = gr.Plot(label="Fraud Probability Donut Meter")
        result_html = gr.HTML()

    # Bind events
    inputs: list[Any] = feature_inputs + [amount]
    predict_btn.click(fn=wrapped_predict, inputs=inputs, outputs=[prob_chart, result_html])
    example_btn.click(fn=fill_example, outputs=inputs)
    reset_btn.click(fn=reset_inputs, outputs=inputs)
