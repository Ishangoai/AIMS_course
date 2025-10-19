import re

import gradio as gr
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Assuming this import works in your environment
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
    """Predicts fraud for a single transaction and returns result, HTML bar, and Plotly gauge."""
    # Handle missing inputs with example defaults
    inputs = [arg if arg is not None else EXAMPLE_TRANSACTION[i] for i, arg in enumerate(args)]

    result = predict_fraud(inputs)

    # Try to extract probability number from text
    try:
        if "Fraudulent" in result:
            prob = float(result.split("(")[1].split("%")[0]) / 100
        elif "Legitimate" in result:
            prob = 1 - (float(result.split("(")[1].split("%")[0]) / 100)
        else:
            prob = 0.0
    except Exception:
        prob = 0.0

    # Create a simple HTML bar to visualize probability
    bar_html = f"""
    <div style='width:100%; background:#eee; border-radius:5px;'>
      <div style='width:{prob * 100:.1f}%; background:{"red" if prob > 0.5 else "green"}; padding:5px;
      border-radius:5px; color:white; text-align:center;'>
        {prob * 100:.1f}%
      </div>
    </div>
    """

    # Create a Plotly gauge for visual appeal
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Fraud Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red" if prob > 0.5 else "green"},
                "steps": [{"range": [0, 50], "color": "lightgreen"}, {"range": [50, 100], "color": "pink"}],
            },
        )
    )

    return result, bar_html, gauge


def fill_example():
    """Fills the input sliders with example transaction data."""
    return EXAMPLE_TRANSACTION


def reset_inputs():
    """Clears all input sliders."""
    return [None] * 29


def batch_predict(csv_file):
    """Processes a CSV file for batch fraud prediction."""
    if csv_file is None:
        # Return empty data frame if no file is uploaded
        return "Please upload dataset", pd.DataFrame(columns=["TransactionID", "Prediction", "Fraud Probability"])  # type: ignore

    df = pd.read_csv(csv_file.name)
    predictions = []

    for _, row in df.iterrows():
        # Prepare inputs, using example data as fallback if columns are missing
        inputs = [row.get(f"V{i}", EXAMPLE_TRANSACTION[i - 1]) for i in range(1, 29)]
        inputs.append(row.get("Amount", EXAMPLE_TRANSACTION[28]))
        result = predict_fraud(inputs)

        # Extract probability (assuming result format is consistent)
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
        # Generate descriptive statistics for numerical columns
        desc_df = df.describe().reset_index()
        desc_df.rename(columns={"index": "Statistic"}, inplace=True)
        return f"Descriptive Statistics for {len(df)} transactions:", desc_df
    except Exception as e:
        return f"Error reading or processing CSV: {e}", pd.DataFrame()


def generate_batch_analysis_viz(predictions_df):
    """
    Generates a gauge for the average fraud probability and a histogram
    for the probability distribution across the predicted dataset.
    Both figures are set to the same height (350px).
    """
    if predictions_df.empty or "Fraud Probability" not in predictions_df.columns:
        return "No prediction data available for visualization.", go.Figure(), go.Figure()

    probabilities = predictions_df["Fraud Probability"]
    avg_prob = probabilities.mean() * 100

    # 1. Average Probability Gauge Chart (Plotly)
    gauge_fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=avg_prob,
            title={"text": "Average Fraud Probability (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red" if avg_prob > 5 else "green"},
                "steps": [{"range": [0, 5], "color": "lightgreen"}, {"range": [5, 100], "color": "pink"}],
                # Note: Setting a low threshold (5%) for "Fraud" in batch context
            },
        )
    )
    # Set explicit height for equal size
    gauge_fig.update_layout(height=350, margin=dict(t=50, b=0, l=0, r=0))

    # 2. Probability Distribution Histogram (Plotly Express)
    hist_fig = px.histogram(
        probabilities,
        x="Fraud Probability",
        nbins=20,
        title="Distribution of Fraud Probabilities",
        color_discrete_sequence=["#4B8BBE"],
    )
    # Set explicit height for equal size
    hist_fig.update_layout(
        xaxis_title="Fraud Probability (0.0 to 1.0)", yaxis_title="Count of Transactions", height=350
    )

    return f"Analysis complete for {len(predictions_df)} predictions.", gauge_fig, hist_fig


# --- Visibility Toggling Functions for gr.Group alternative ---
def show_group():
    """Returns an update object to make a group visible."""
    return gr.update(visible=True)


def hide_group():
    """Returns an update object to make a group invisible."""
    return gr.update(visible=False)


# ---- Gradio UI ----
with gr.Blocks(css="body {background: #f2f7ff;}") as fraud_app:
    # State component to hold the last predicted dataframe for visualization
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
    prob_chart = gr.Plot(label="Fraud Probability Gauge")
    inputs = sliders_1 + sliders_2 + [Amount]
    predict_btn.click(fn=wrapped_predict, inputs=inputs, outputs=[result, prob_bar, prob_chart])
    example_btn.click(fn=fill_example, inputs=[], outputs=inputs)
    reset_btn.click(fn=reset_inputs, inputs=[], outputs=inputs)

    # ==============================================================================
    # --- 1. Data Analysis (Summary Statistics) ---
    # ==============================================================================
    gr.Markdown("### 1. Data Analysis (Summary Statistics)")

    # Place the CSV input and analysis button in a Row for better layout
    with gr.Row():
        csv_input = gr.File(label="Upload CSV", file_types=[".csv"], scale=3)
        analyze_btn = gr.Button("📊 View Data Statistics", scale=1, variant="secondary")

    # --- Descriptive Analysis Group (Hidden) ---
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

    # Wire up the Descriptive Analysis buttons (Runs analysis AND shows group)
    analyze_btn.click(
        fn=descriptive_analysis, inputs=[csv_input], outputs=[desc_text_output, desc_dataframe_output]
    ).then(fn=show_group, outputs=[descriptive_group])

    close_analysis_btn.click(fn=hide_group, outputs=[descriptive_group])

    # ==============================================================================
    # --- 2. Batch Prediction & Results ---
    # ==============================================================================
    gr.Markdown("### 2. Batch Prediction & Results")

    # --- Prediction Visualization Group (Hidden and Moved Up) ---
    # This section is now defined before the batch prediction buttons
    with gr.Group(visible=False) as visualization_group:
        gr.Markdown("## Prediction Visualization and Distribution")
        viz_text_output = gr.Textbox(label="Status", value="Analysis results will appear here.")

        # Row for equal height charts
        with gr.Row():
            avg_prob_gauge = gr.Plot(label="Average Fraud Probability Gauge")
            prob_histogram = gr.Plot(label="Probability Distribution Histogram")

        close_viz_btn = gr.Button("Close Visualization")

    batch_output = gr.Dataframe(
        headers=["TransactionID", "Prediction", "Fraud Probability"],
        label="Batch Prediction Results",
        wrap=True,
        interactive=False,
    )

    with gr.Row():
        csv_btn = gr.Button("Run Batch Prediction", variant="primary")
        clear_btn = gr.Button("Clear Predictions")
        viz_btn = gr.Button("📈 View Prediction Analysis", variant="secondary")  # Button to trigger visualization

    # Keep track of sort toggle
    sort_toggle = gr.State(value=True)  # True = descending

    # Wire up Batch Prediction. It updates the DataFrame AND the State component.
    csv_btn.click(fn=batch_predict, inputs=csv_input, outputs=[gr.Textbox(visible=False), batch_output]).then(
        fn=lambda df: df,  # Pass the output dataframe to the state
        inputs=[batch_output],
        outputs=[prediction_state],
    )

    # Wire up Clear Predictions (clears dataframe and state)
    clear_btn.click(
        fn=lambda: (clear_predictions(), clear_predictions()), inputs=None, outputs=[batch_output, prediction_state]
    )

    # Wire up the Visualization button (Runs viz function AND shows group)
    viz_btn.click(
        fn=generate_batch_analysis_viz,
        inputs=[prediction_state],  # Use the state component containing the latest results
        outputs=[viz_text_output, avg_prob_gauge, prob_histogram],
    ).then(fn=show_group, outputs=[visualization_group])

    close_viz_btn.click(fn=hide_group, outputs=[visualization_group])
