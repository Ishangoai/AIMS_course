from __future__ import annotations

from typing import Any

import gradio as gr
from gradioapp.utils.fraud_utils import predict_fraud

# -----------------------------
# Example transaction defaults
# -----------------------------
EXAMPLE_TRANSACTION: list[float] = [
    1.18081256,
    -0.151318004,
    0.735193402,
    0.394006458,
    -0.934756778,
    -0.943678475,
    -0.130194211,
    -0.148228409,
    0.589976142,
    -0.308470408,
    -0.283332093,
    0.614995531,
    -0.013868061,
    -0.142073926,
    -0.110624364,
    -0.241598526,
    0.193491029,
    -0.982638787,
    0.293116998,
    -0.042024044,
    -0.287134741,
    -0.721555218,
    0.124741513,
    0.779478755,
    0.147347019,
    0.798826508,
    -0.068182379,
    0.014936089,
    31.4,
]


# -----------------------------
# Wrapped predict function
# -----------------------------
def wrapped_predict(*args: Any) -> str:
    """Predicts fraud probability and returns formatted HTML result."""
    inputs: list[float] = []
    for i, arg in enumerate(args):
        try:
            inputs.append(float(arg) if arg is not None else EXAMPLE_TRANSACTION[i])
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

    if prob < 0.33:
        color = "#2ecc71"  # Green
        status = "Legitimate ✅"
        bg_color = "#d5f4e6"
    elif prob < 0.66:
        color = "#f39c12"  # Orange
        status = "Suspicious ⚠️"
        bg_color = "#fef5e7"
    else:
        color = "#e74c3c"  # Red
        status = "Fraudulent 🚨"
        bg_color = "#fadbd8"

    fraud_html = f"""
    <div style="text-align:center; padding: 30px; background-color: {bg_color};
                border-radius: 15px; margin: 20px auto; max-width: 600px;
                border: 3px solid {color};">
        <div style="font-size: 48px; font-weight: bold; color: {color};
                    margin-bottom: 15px;">
            {prob * 100:.1f}%
        </div>
        <div style="font-size: 28px; font-weight: 600; color: {color};
                    margin-bottom: 20px;">
            {status}
        </div>
        <div style="font-size: 18px; color: #555; margin-top: 15px;">
            {result}
        </div>
        <div style="margin-top: 25px; padding-top: 20px;
                    border-top: 2px solid {color};">
            <div style="display: inline-block; margin: 0 20px;">
                <div style="width: 20px; height: 20px; background-color: #e74c3c;
                            display: inline-block; border-radius: 3px;
                            vertical-align: middle;"></div>
                <span style="margin-left: 8px; font-size: 14px; color: #555;">
                    Fraudulent ({prob * 100:.1f}%)
                </span>
            </div>
            <div style="display: inline-block; margin: 0 20px;">
                <div style="width: 20px; height: 20px; background-color: #2ecc71;
                            display: inline-block; border-radius: 3px;
                            vertical-align: middle;"></div>
                <span style="margin-left: 8px; font-size: 14px; color: #555;">
                    Legitimate ({(1 - prob) * 100:.1f}%)
                </span>
            </div>
        </div>
    </div>
    """

    return fraud_html


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
    gr.Markdown(
        """
        <div style="text-align:center;">
            <h1 style="color:#3b0d91;">💳 Fraud Detection Predictor</h1>
            <p style="font-size:16px; color:#555; max-width:700px; margin:auto;">
                Use the sliders below to simulate a transaction and check whether it's
                <strong>Fraudulent</strong> or <strong>Legitimate</strong>.
            </p>
            <p style="font-size:14px; color:#777;">
                Built by: Elisha Komolafe 🇳🇬 & Lionel Cedric Gohouede 🇧🇯 (2025)
            </p>
        </div>
        """,
    )

    gr.Markdown("### 🧮 Input Transaction Features")

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

    amount = gr.Textbox(
        label="💵 Transaction Amount",
        placeholder="Enter amount (e.g. 120.50)",
        value=str(EXAMPLE_TRANSACTION[28]),
    )

    with gr.Row():
        predict_btn = gr.Button("🚀 Predict", variant="primary")
        example_btn = gr.Button("💡 Example")
        reset_btn = gr.Button("🔄 Reset")

    gr.Markdown("### 🔍 Prediction Result")
    with gr.Column(elem_classes=["centered-container"]):
        result_html = gr.HTML()

    inputs: list[Any] = feature_inputs + [amount]
    predict_btn.click(fn=wrapped_predict, inputs=inputs, outputs=[result_html])
    example_btn.click(fn=fill_example, outputs=inputs)
    reset_btn.click(fn=reset_inputs, outputs=inputs)
