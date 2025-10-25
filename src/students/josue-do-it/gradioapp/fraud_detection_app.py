

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# IMPORTANT: This utility function is assumed to be the core prediction model logic.
from gradioapp.utils.fraud_utils import predict_fraud

# --- 1. MODEL FUNCTIONS (SIMULATION) ---
# NOTE: This function simulates the core model logic for local execution,
# and it is called via 'wrapped_predict'.


def predict_data(data):
    """
    Simulation function for fraud detection based on rules.
    It expects 29 features as input (V1-V28, Amount), excluding 'Time'.

    Args:
        data (list): List of 29 features (V1...V28, Amount).

    Returns:
        str: Formatted prediction message.
    """
    try:
        # Unpacking 29 expected features: V1 to V28, and the transaction amount
        (
            _V1, V2, _V3, V4, _V5, _V6, _V7, _V8, _V9, _V10, V11, _V12, _V13,
            V14, _V15, _V16, _V17, __V18, _V19, _V20, _V21, _V22, _V23, _V24, _V25,
            _V26, _V27, _V28, account
        ) = data
    except ValueError:
        # Error message reflecting the expectation of 29 features
        return "Error: 29 input features are required for prediction (V1-V28, Amount)."

    # Fraud detection conditions based on the provided rules
    if account > 5000 and V14 < -5 and V11 > 2:
        return "Fraud Detected! 🚨 (Very High Risk)"
    if account > 10000 and V4 > 5:
        return "High Probability of Fraud! ⚠️ (Abnormal Amount/Activity)"
    if V14 < -10 and V2 < -5:
        return "Potential Fraud! 🔍 (Suspicious Transaction Profile)"
    return "No Fraud Detected. ✅ (Normal Transaction)"


def wrapped_predict(*args):
    """
    Wrapper function to convert unpacked Gradio arguments
    into a single list for the predict_fraud model function.
    """
    return predict_fraud(list(args))


def get_model_accuracy():
    """Simulates retrieving the current model accuracy (MLflow)."""
    return np.random.uniform(88.0, 99.0)  # Simulation of high accuracy


# --- 2. VISUALIZATION FUNCTIONS ---


def show_head(file_path):
    """Displays the first few rows of the uploaded CSV file."""
    if file_path is None:
        return "No CSV file loaded."
    try:
        df = pd.read_csv(file_path.name)
        return df.head(5).to_markdown(index=False)
    except Exception as e:
        return f"Error reading file: {e}"


def update_column_choices(file_obj):
    """Reads CSV columns and updates dropdown choices."""
    if file_obj is None:
        return (gr.Dropdown(choices=[], interactive=True, value=None),) * 3 + (None,)

    file_path = file_obj.name
    try:
        df = pd.read_csv(file_path)
        # Clean-up: drop completely empty columns
        df.dropna(axis=1, how='all', inplace=True)
        columns = list(df.columns)
    except Exception as e:
        print(f"Error reading CSV for column update: {e}")
        return (gr.Dropdown(choices=[], interactive=True, value=None),) * 3 + (None,)

    return (
        gr.Dropdown(choices=columns, interactive=True, value=columns[0] if columns else None),
        gr.Dropdown(choices=columns, interactive=True, value=columns[0] if columns else None),
        gr.Dropdown(choices=columns, interactive=True, value=columns[0] if columns else None),
        file_path
    )


def generate_histogram(file_path, column):
    """Generates a histogram for a selected column."""
    if file_path is None or column is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, "Please load a CSV and select a column.", horizontalalignment='center',
                      verticalalignment='center', transform=ax.transAxes, color='gray')
        ax.axis('off')
        plt.close(fig)
        return fig

    try:
        df = pd.read_csv(file_path)
        data_to_plot = df[column].dropna()
    except Exception as e:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Error loading or column not found: {e}", horizontalalignment='center',
              verticalalignment='center', transform=ax.transAxes, color='red')
        ax.axis('off')
        plt.close(fig)
        return fig

    if not pd.api.types.is_numeric_dtype(data_to_plot):
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, f"Column '{column}' is not numeric for histogram.", horizontalalignment='center',
              verticalalignment='center', transform=ax.transAxes, color='orange')
        ax.axis('off')
        plt.close(fig)
        return fig

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(data_to_plot, bins=30, edgecolor='#333333', color='#4CAF50', alpha=0.7)
    ax.set_title(f"Variable Distribution: {column}", fontsize=14)
    ax.set_xlabel(column, fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.close(fig)
    return fig


def generate_scatter_plot(file_path, x_column, y_column):
    """Generates a scatter plot for two selected columns."""
    if file_path is None or x_column is None or y_column is None:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.text(0.5, 0.5, "Please load a CSV and select both X & Y columns.", horizontalalignment='center',
              verticalalignment='center', transform=ax.transAxes, color='gray')
        ax.axis('off')
        plt.close(fig)
        return fig

    try:
        df = pd.read_csv(file_path)
        data_to_plot = df.dropna(subset=[x_column, y_column])
    except Exception as e:
        fig, ax = plt.subplots(figsize=(10, 7))
        ax.text(0.5, 0.5, f"Error loading or columns not found: {e}", horizontalalignment='center',
              verticalalignment='center', transform=ax.transAxes, color='red')
        ax.axis('off')
        plt.close(fig)
        return fig

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(data_to_plot[x_column], data_to_plot[y_column], alpha=0.6, color='#2196F3', edgecolors='#1976D2',
              linewidths=0.5)
    ax.set_title(f"Relationship between {x_column} and {y_column}", fontsize=14)
    ax.set_xlabel(x_column, fontsize=12)
    ax.set_ylabel(y_column, fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.7)

    plt.tight_layout()
    plt.close(fig)
    return fig


# --- 3. GRADIO INTERFACE DEFINITION ---

custom_css = """
body {
    background: #f8f9fa; /* Light background */
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.gradio-container {
    max-width: 1200px;
    margin: 0 auto;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    background: white;
}
h1 {
    color: #007bff;
    text-align: center;
    padding-top: 20px;
    font-weight: 600;
}
h2, h3 {
    color: #333333;
    border-bottom: 2px solid #f0f0f0;
    padding-bottom: 5px;
    margin-top: 20px;
}
.gradio-button {
    background-color: #007bff !important;
    color: white !important;
    border-radius: 8px !important;
    transition: background-color 0.3s ease;
    font-weight: 500;
}
.gradio-button:hover {
    background-color: #0056b3 !important;
}
"""

with gr.Blocks(css="body {background: #f2f7ff;}", title="Fraud Detection") as fraud_app:
    # State variable to hold the path of the uploaded file
    csv_file_path_state = gr.State(None)

    gr.Markdown("# CREDIT CARD FRAUD DETECTION")
    gr.Markdown("---")

    with gr.Tabs():
        # --- TAB 1: MANUAL PREDICTION & ACCURACY ---
        with gr.TabItem("Prediction & Metrics"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### ➡️ Manual Transaction Feature Input")
                    account = gr.Slider(0, 25691, value=88.3, label="Amount", info="Transaction in USD", step=0.01)

                    # V1 to V28 features grouped in Accordions
                    V_features = {}
                    with gr.Accordion("Features V1 - V10 (PCA)", open=False):
                        with gr.Row():
                            for i in range(1, 11):
                                V_features[f'V{i}'] = gr.Slider(-25.0, 25.0, value=0.0, label=f"V{i}",
                      step=0.1, scale=1)
                    with gr.Accordion("Features V11 - V20 (PCA)", open=False):
                        with gr.Row():
                            for i in range(11, 21):
                                V_features[f'V{i}'] = gr.Slider(-25.0, 25.0, value=0.0, label=f"V{i}",
              step=0.1, scale=1)
                    with gr.Accordion("Features V21 - V28 (PCA)", open=False):
                        with gr.Row():
                            for i in range(21, 29):
                                V_features[f'V{i}'] = gr.Slider(-25.0, 25.0, value=0.0, label=f"V{i}",
              step=0.1, scale=1)

                    predict_btn = gr.Button("Analyze Transaction (Predict)", variant="primary")
                    result = gr.Textbox(label="Prediction Result", type="text", show_copy_button=True, lines=2)

                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Model Performance")
                    gauge_output = gr.Plot(label="Current Accuracy")

                    refresh_accuracy_btn = gr.Button("Refresh Accuracy", variant="secondary")

                    gr.Markdown("---")
                    gr.Markdown("### 📂 Load Data for Analysis")
                    file_input = gr.File(
                        file_types=[".csv"],
                        label="Upload a CSV File",
                        # placeholder="For Exploratory Data Analysis (EDA) in the 'Data Analysis' tab."
                    )

        with gr.TabItem("Data Analysis"):
            gr.Markdown("## Overview of Loaded Data")
            head_output = gr.Markdown(label="First Rows (Top 5)")
            gr.Markdown("---")

            with gr.Row():
                # Histogram Column
                with gr.Column():
                    gr.Markdown("### Distribution Histogram")
                    histogram_column_dropdown = gr.Dropdown(label="Select Column", choices=[], interactive=True)
                    histogram_button = gr.Button("Generate Histogram", variant="secondary")
                    histogram_output = gr.Plot(label="Column Histogram")

                # Scatter Plot Column
                with gr.Column():
                    gr.Markdown("### Scatter Plot (Relationship)")
                    scatter_x_column_dropdown = gr.Dropdown(label="X-Axis", choices=[], interactive=True)
                    scatter_y_column_dropdown = gr.Dropdown(label="Y-Axis", choices=[], interactive=True)
                    scatter_button = gr.Button("Generate Scatter Plot", variant="secondary")
                    scatter_output = gr.Plot(label="Scatter Plot X vs Y")

    # --- 4. EVENT HANDLERS ---

    # 4.1 Update after file upload
    file_input.change(
        fn=update_column_choices,
        inputs=file_input,
        outputs=[
            histogram_column_dropdown,
            scatter_x_column_dropdown,
            scatter_y_column_dropdown,
            csv_file_path_state
        ]
    ).success(
        fn=show_head,
        inputs=csv_file_path_state,
        outputs=head_output
    )

    # 4.2 Visualization Buttons
    histogram_button.click(fn=generate_histogram,
                           inputs=[csv_file_path_state, histogram_column_dropdown],
                           outputs=histogram_output)

    scatter_button.click(fn=generate_scatter_plot,
                         inputs=[csv_file_path_state, scatter_x_column_dropdown, scatter_y_column_dropdown],
                         outputs=scatter_output)

    all_inputs = [V_features[f'V{i}'] for i in range(1, 29)] + [account]
    predict_btn.click(
        fn=wrapped_predict,
        inputs=all_inputs,
        outputs=result
    )
