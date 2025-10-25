import re

import gradio as gr
import pandas as pd
from gradioapp.utils.fraud_utils import predict_fraud

# Example transaction values for testing
EXAMPLE_TRANSACTION = [
    -1.508147666,
    1.853128013,
    0.220932254,
    -0.146817811,
    -0.528975655,
    -0.626681837,
    -0.166862547,
    0.999946773,
    -0.49952308,
    -0.156840906,
    0.847755891,
    0.538299087,
    -0.354515904,
    0.35237478,
    0.23433853,
    0.906489408,
    -0.010299228,
    0.576837916,
    0.256061611,
    0.130144351,
    -0.218249367,
    -0.752299949,
    0.059790389,
    -0.071737144,
    -0.017378138,
    0.090616835,
    0.204201631,
    0.075687247,
    8.99,
]

COLUMN_NAMES = [f"V{i + 1}" for i in range(28)] + ["Amount"]


def wrapped_predict(*args):
    """Predicts fraud for a single transaction and returns result and HTML bar."""
    inputs = [arg if arg is not None else EXAMPLE_TRANSACTION[i] for i, arg in enumerate(args)]
    result = predict_fraud(inputs)

    try:
        if "Fraudulent" in result:
            prob = float(result.split("(")[1].split("%")[0]) / 100
        elif "Legitimate" in result:
            prob = 1 - (float(result.split("(")[1].split("%")[0]) / 100)
        else:
            prob = 0.0
    except Exception:
        prob = 0.0

    bar_html = f"""
    <div style='width:100%; background:#eee; border-radius:5px;'>
      <div style='width:{prob * 100:.1f}%; background:{"red" if prob > 0.5 else "green"}; padding:5px;
      border-radius:5px; color:white; text-align:center;'>
        {prob * 100:.1f}%
      </div>
    </div>
    """
    return result, bar_html


def fill_example():
    """Fills the input sliders with example transaction data."""
    return EXAMPLE_TRANSACTION


def reset_inputs():
    """Clears all input sliders."""
    return [None] * 29


def batch_predict(csv_file):
    """Processes a CSV file for batch fraud prediction."""
    if csv_file is None:
        return "Please upload dataset", pd.DataFrame(columns=["TransactionID", "Prediction", "Fraud Probability"])  # type: ignore

    df = pd.read_csv(csv_file.name)
    predictions = []

    for _, row in df.iterrows():
        inputs = [row.get(f"V{i}", EXAMPLE_TRANSACTION[i - 1]) for i in range(1, 29)]
        inputs.append(row.get("Amount", EXAMPLE_TRANSACTION[28]))
        result = predict_fraud(inputs)

        match = re.search(r"\(([\d.]+)% probability\)", result)
        prob = 1 - (float(match.group(1)) / 100.0) if match else 0.0

        predictions.append(
            {"TransactionID": row.get("Time", None), "Prediction": result, "Fraud Probability": round(prob, 4)}
        )

    return "Batch prediction complete.", pd.DataFrame(predictions)


def clear_predictions():
    """Clears the batch prediction dataframe."""
    return pd.DataFrame(columns=["TransactionID", "Prediction", "Fraud Probability"])  # type: ignore


def descriptive_analysis(csv_file):
    """Generates descriptive statistics for the uploaded CSV file."""
    if csv_file is None:
        return "Please upload a CSV file first.", pd.DataFrame()

    try:
        df = pd.read_csv(csv_file.name)
        desc_df = df.describe().reset_index()
        desc_df.rename(columns={"index": "Statistic"}, inplace=True)
        return f"Descriptive Statistics for {len(df)} transactions:", desc_df
    except Exception as e:
        return f"Error reading or processing CSV: {e}", pd.DataFrame()


# ---- Gradio UI ----
with gr.Blocks(css="body {background: #f2f7ff;}") as fraud_app:
    prediction_state = gr.State(pd.DataFrame(columns=["TransactionID", "Prediction", "Fraud Probability"]))  # type: ignore

    gr.Markdown("# AIMS Course: Fraud Detection Predictor")
    gr.Markdown("Fill in transaction features or upload CSV for batch prediction.")

    with gr.Row():
        with gr.Column():
            sliders_1 = [gr.Slider(-5, 5, value=EXAMPLE_TRANSACTION[i], label=f"V{i + 1}") for i in range(14)]
        with gr.Column():
            sliders_2 = [gr.Slider(-5, 5, value=EXAMPLE_TRANSACTION[i], label=f"V{i + 1}") for i in range(14, 28)]
            Amount = gr.Number(label="Transaction Amount", value=EXAMPLE_TRANSACTION[28], precision=2)

    predict_btn = gr.Button("Predict Single Transaction")
    example_btn = gr.Button("Fill Example Transaction")
    reset_btn = gr.Button("Reset Inputs")
    result = gr.Textbox(label="Prediction Result")
    prob_bar = gr.HTML(label="Fraud Probability")
    inputs = sliders_1 + sliders_2 + [Amount]
    predict_btn.click(fn=wrapped_predict, inputs=inputs, outputs=[result, prob_bar])
    example_btn.click(fn=fill_example, inputs=[], outputs=inputs)
    reset_btn.click(fn=reset_inputs, inputs=[], outputs=inputs)

    # --- 1. Data Analysis ---
    gr.Markdown("### 1. Data Analysis (Summary Statistics)")
    with gr.Row():
        csv_input = gr.File(label="Upload CSV", file_types=[".csv"], scale=3)
        analyze_btn = gr.Button("📊 View Data Statistics", scale=1, variant="secondary")

    with gr.Group(visible=False) as descriptive_group:
        gr.Markdown("## Descriptive Statistics of Uploaded Dataset")
        desc_text_output = gr.Textbox(label="Status", value="Statistics will appear here.")
        desc_dataframe_output = gr.Dataframe(
            headers=["Statistic"] + COLUMN_NAMES,
            wrap=True,
            interactive=False,
            label="Descriptive Statistics",
        )
        close_analysis_btn = gr.Button("Close Analysis")

    analyze_btn.click(
        fn=descriptive_analysis, inputs=[csv_input], outputs=[desc_text_output, desc_dataframe_output]
    ).then(fn=lambda: gr.update(visible=True), outputs=[descriptive_group])
    close_analysis_btn.click(fn=lambda: gr.update(visible=False), outputs=[descriptive_group])

    # --- 2. Batch Prediction ---
    gr.Markdown("### 2. Batch Prediction & Results")
    batch_output = gr.Dataframe(
        headers=["TransactionID", "Prediction", "Fraud Probability"],
        label="Batch Prediction Results",
        wrap=True,
        interactive=False,
    )

    with gr.Row():
        csv_btn = gr.Button("Run Batch Prediction", variant="primary")
        clear_btn = gr.Button("Clear Predictions")

    csv_btn.click(fn=batch_predict, inputs=csv_input, outputs=[gr.Textbox(visible=False), batch_output]).then(
        fn=lambda df: df, inputs=[batch_output], outputs=[prediction_state]
    )
    clear_btn.click(
        fn=lambda: (clear_predictions(), clear_predictions()), inputs=None, outputs=[batch_output, prediction_state]
    )
