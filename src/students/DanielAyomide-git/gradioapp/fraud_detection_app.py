import re

import gradio as gr
import matplotlib.pyplot as plt  # Matplotlib import
import numpy as np
import pandas as pd

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


def create_matplotlib_gauge(prob: float, title: str, height: int = 350):
    """
    Creates a simple Matplotlib approximation of a gauge chart.
    Returns a Matplotlib Figure object.
    """
    # Close any existing plot to prevent memory leaks in a loop
    plt.close("all")

    fig, ax = plt.subplots(figsize=(6, 6))  # Adjust size as needed, Gradio controls the final display size

    # Convert angle to radians for plotting

    # 1. Background Arc (The "Gauge")
    # Define colors for ranges (0-50% green, 50-100% pink)
    colors = ["lightgreen", "pink"]
    # Create the arcs
    ax.bar(0, 1.0, width=np.pi, bottom=0.0, linewidth=0, align="edge", color=colors[0], edgecolor=None, alpha=0.5)
    ax.bar(
        np.pi / 2,
        1.0,
        width=np.pi / 2,
        bottom=0.0,
        linewidth=0,
        align="edge",
        color=colors[1],
        edgecolor=None,
        alpha=0.5,
    )

    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 6))

    # Gauge arcs (0-50% and 50-100%)
    # Matplotlib polar angle is in radians, 0 is east, positive is counter-clockwise.
    # We want 0-100% to go from 180 deg (pi) down to 0 deg.

    # Range 0-50% (100% to 50% on the plot)
    # Start angle: pi, End angle: pi/2
    ax.barh(np.linspace(np.pi / 2, np.pi, 50), 1, color="lightgreen", align="edge", height=np.pi / 100)

    # Range 50-100% (50% to 0% on the plot)
    # Start angle: pi/2, End angle: 0
    ax.barh(np.linspace(0, np.pi / 2, 50), 1, color="pink", align="edge", height=np.pi / 100)

    # The actual pointer
    pointer_color = "red" if prob > 0.5 else "green"

    # Pointer angle: 180 deg (pi) for 0% to 0 deg (0) for 100%
    pointer_angle = np.pi * (1 - prob)

    # Draw the pointer
    ax.plot([pointer_angle, pointer_angle], [0, 1], color=pointer_color, linewidth=5, solid_capstyle="round")
    ax.plot([0], [0], marker="o", markersize=10, color="black")  # Center pivot

    # Set limits and style
    ax.set_theta_zero_location("W")  # Set 0 degree to the West (left)
    ax.set_theta_direction(-1)  # Clockwise direction
    ax.set_thetamin(0)  # Start at 0 degrees
    ax.set_thetamax(180)  # End at 180 degrees
    ax.set_rticks([])  # Remove radial ticks
    ax.set_xticks(np.linspace(0, np.pi, 5))  # Add tick marks for 0%, 25%, 50%, 75%, 100%
    ax.set_xticklabels(["100%", "75%", "50%", "25%", "0%"])  # Relabel ticks
    ax.tick_params(axis="x", pad=10)  # Add padding to labels

    ax.set_title(f"{title}\n{prob * 100:.1f}%", va="bottom")
    fig.subplots_adjust(top=0.85)

    return fig


def wrapped_predict(*args):
    """Predicts fraud for a single transaction and returns result, HTML bar, and Matplotlib gauge."""
    # Handle missing inputs with example defaults
    inputs = [arg if arg is not None else EXAMPLE_TRANSACTION[i] for i, arg in enumerate(args)]

    result = predict_fraud(inputs)

    # Try to extract probability number from text.
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

    # Create a Matplotlib gauge
    gauge_fig = create_matplotlib_gauge(prob, title="Fraud Probability (%)")

    return result, bar_html, gauge_fig


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
    for the probability distribution across the predicted dataset using Matplotlib.
    """
    if predictions_df.empty or "Fraud Probability" not in predictions_df.columns:
        # Return empty figures
        return "No prediction data available for visualization.", plt.Figure(), plt.Figure()

    probabilities = predictions_df["Fraud Probability"]
    avg_prob = probabilities.mean()

    # 1. Average Probability Gauge Chart (Matplotlib)
    # The Matplotlib gauge creation is now its own function for reusability
    gauge_fig = create_matplotlib_gauge(avg_prob, title="Average Fraud Probability (%)")

    # 2. Probability Distribution Histogram (Matplotlib)
    plt.close("all")  # Close previous figures
    hist_fig, ax = plt.subplots(figsize=(7, 4))  # Adjusted size for similar proportion

    # Use Matplotlib's hist function
    ax.hist(
        probabilities,
        bins=20,
        color="#4B8BBE",  # Use the Plotly Express default blue for consistency
        edgecolor="black",  # Add edge color for better bin separation
    )

    ax.set_title("Distribution of Fraud Probabilities")
    ax.set_xlabel("Fraud Probability (0.0 to 1.0)")
    ax.set_ylabel("Count of Transactions")
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping

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
    # Change gr.Plot to accept Matplotlib figure
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
            # Change gr.Plot to accept Matplotlib figures
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
