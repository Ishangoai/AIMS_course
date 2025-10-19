import os
import random
import tempfile

import gradio as gr
import joblib
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use('Agg')  # Use non-interactive backend

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)


# Prediction Function
# This function is what Gradio will call when a user interacts with the UI
def predict_fraud(
    time,
    v1,
    v2,
    v3,
    v4,
    v5,
    v6,
    v7,
    v8,
    v9,
    v10,
    v11,
    v12,
    v13,
    v14,
    v15,
    v16,
    v17,
    v18,
    v19,
    v20,
    v21,
    v22,
    v23,
    v24,
    v25,
    v26,
    v27,
    v28,
    amount,
):
    # Create a Pandas DataFrame that matches the model's expected input format
    input_df = pd.DataFrame(
        data={
            "Time": [time],
            "V1": [v1],
            "V2": [v2],
            "V3": [v3],
            "V4": [v4],
            "V5": [v5],
            "V6": [v6],
            "V7": [v7],
            "V8": [v8],
            "V9": [v9],
            "V10": [v10],
            "V11": [v11],
            "V12": [v12],
            "V13": [v13],
            "V14": [v14],
            "V15": [v15],
            "V16": [v16],
            "V17": [v17],
            "V18": [v18],
            "V19": [v19],
            "V20": [v20],
            "V21": [v21],
            "V22": [v22],
            "V23": [v23],
            "V24": [v24],
            "V25": [v25],
            "V26": [v26],
            "V27": [v27],
            "V28": [v28],
            "Amount": [amount],
        }
    )

    # Ensure all columns are float32 to match training format
    for col in input_df.columns:
        input_df[col] = input_df[col].astype("float32")

    prediction_result = model.predict(input_df)

    # The result is typically a NumPy array or list; get the first element
    predicted_value = prediction_result[0]

    # Format the output for display
    if predicted_value == 0:
        return "🟢 Legitimate Transaction"
    else:
        return "🔴 Fraudulent Transaction"


# Batch prediction function for CSV files
def predict_fraud_batch(csv_file):
    """
    Process a CSV file with multiple transactions and return predictions
    """
    if csv_file is None:
        return None, "Please upload a CSV file.", None

    try:
        # Read the CSV file
        df = pd.read_csv(csv_file.name)

        # Check if the required columns are present
        required_columns = [
            "Time",
            "V1",
            "V2",
            "V3",
            "V4",
            "V5",
            "V6",
            "V7",
            "V8",
            "V9",
            "V10",
            "V11",
            "V12",
            "V13",
            "V14",
            "V15",
            "V16",
            "V17",
            "V18",
            "V19",
            "V20",
            "V21",
            "V22",
            "V23",
            "V24",
            "V25",
            "V26",
            "V27",
            "V28",
            "Amount",
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return None, f"Missing required columns: {', '.join(missing_columns)}", None

        # Select only the required columns in the correct order
        input_df = df[required_columns].copy()

        # Ensure all columns are float32 to match training format
        for col in input_df.columns:
            input_df[col] = input_df[col].astype("float32")

        # Make predictions
        predictions = model.predict(input_df)

        # Add predictions to the dataframe
        result_df = df.copy()
        result_df["Prediction"] = predictions
        result_df["Prediction_Label"] = result_df["Prediction"].apply(
            lambda x: "Legitimate" if x == 0 else "Fraudulent"
        )

        # Create a temporary file to store results
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as temp_file:
            result_df.to_csv(temp_file.name, index=False)

        # Summary statistics
        total_transactions = len(result_df)
        fraudulent_count = (predictions == 1).sum()
        legitimate_count = (predictions == 0).sum()
        fraud_percentage = (fraudulent_count / total_transactions) * 100

        summary = f"""
        📊 **Batch Prediction Results:**
        - Total transactions processed: {total_transactions}
        - Fraudulent transactions: {fraudulent_count} ({fraud_percentage:.2f}%)
        - Legitimate transactions: {legitimate_count} ({100 - fraud_percentage:.2f}%)
        Results have been saved and are ready for download.
        """

        # Create pie chart
        plt.style.use('dark_background')
        _, ax = plt.subplots(figsize=(8, 6))

        # Data for pie chart
        labels = ['Legitimate', 'Fraudulent']
        sizes = [legitimate_count, fraudulent_count]
        colors = ['#2ecc71', '#e74c3c']  # Green for legitimate, red for fraudulent
        explode = (0, 0.1)  # Explode the fraudulent slice slightly

        # Create pie chart
        pie_result = ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                           startangle=90, explode=explode, shadow=True,
                           textprops={'fontsize': 12, 'color': 'white'})
        wedges = pie_result[0]
        autotexts = pie_result[1] if len(pie_result) > 1 else []
        if len(pie_result) > 2:
            autotexts = pie_result[2]

        # Customize the appearance
        ax.set_title('Fraud Detection Results\nDistribution of Transactions',
                    fontsize=16, fontweight='bold', color='white', pad=20)

        # Make percentages bold and larger
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(14)

        # Add legend
        ax.legend(wedges, [f'{label}\n{size:,} transactions' for label, size in zip(labels, sizes)],
                 title="Transaction Types", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1),
                 fontsize=10, title_fontsize=12)

        plt.tight_layout()

        # Save plot to temporary file
        plot_temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        plt.savefig(plot_temp_file.name, dpi=150, bbox_inches='tight',
                   facecolor='#2D3250', edgecolor='none')
        plt.close()

        return temp_file.name, summary, plot_temp_file.name

    except Exception as e:
        return None, f"Error processing file: {str(e)}", None


# Random data generation function
def generate_random_data():
    """Generate random transaction data for testing"""
    return (
        random.uniform(0, 172792),  # time
        random.uniform(-56.4, 56.4),  # V1
        random.uniform(-72.1, 22.1),  # V2
        random.uniform(-48.3, 9.4),   # V3
        random.uniform(-5.7, 16.9),   # V4
        random.uniform(-113.7, 34.8),  # V5
        random.uniform(-26.2, 73.3),  # V6
        random.uniform(-43.6, 120.6),  # V7
        random.uniform(-73.2, 20.0),  # V8
        random.uniform(-13.4, 15.6),  # V9
        random.uniform(-24.6, 23.7),  # V10
        random.uniform(-4.8, 12.0),   # V11
        random.uniform(-18.7, 7.8),   # V12
        random.uniform(-5.8, 7.1),    # V13
        random.uniform(-19.2, 10.5),  # V14
        random.uniform(-4.5, 8.9),    # V15
        random.uniform(-14.1, 17.3),  # V16
        random.uniform(-25.2, 9.2),   # V17
        random.uniform(-9.5, 5.0),    # V18
        random.uniform(-7.2, 5.0),    # V19
        random.uniform(-54.5, 39.4),  # V20
        random.uniform(-34.8, 27.2),  # V21
        random.uniform(-10.9, 10.5),  # V22
        random.uniform(-44.8, 22.5),  # V23
        random.uniform(-2.8, 4.6),    # V24
        random.uniform(-10.3, 7.5),   # V25
        random.uniform(-2.6, 3.5),    # V26
        random.uniform(-22.6, 31.6),  # V27
        random.uniform(-15.4, 33.8),  # V28
        random.uniform(0, 25691.16),  # amount
    )


# Custom CSS for the color scheme
css = """
:root {
    --primary-color: #2D3250;
    --secondary-color: #424769;
    --accent-color: #7077A1;
    --highlight-color: #F6B17A;
}

.gradio-container {
    background-color: var(--primary-color) !important;
}

.dark {
    background: var(--primary-color) !important;
}

/* Main app styling */
.app {
    background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%) !important;
}

/* Headers and text */
h1, h2, h3, h4, h5, h6 {
    color: var(--highlight-color) !important;
}

.prose {
    color: #ffffff !important;
}

/* Input fields styling */
.gr-textbox input, .gr-number input {
    background-color: var(--secondary-color) !important;
    border: 2px solid var(--accent-color) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}

.gr-textbox input:focus, .gr-number input:focus {
    border-color: var(--highlight-color) !important;
    box-shadow: 0 0 0 2px rgba(246, 177, 122, 0.3) !important;
}

/* Button styling */
.gr-button {
    background: linear-gradient(135deg, var(--accent-color) 0%, var(--highlight-color) 100%) !important;
    border: none !important;
    color: var(--primary-color) !important;
    font-weight: bold !important;
    border-radius: 8px !important;
    transition: all 0.3s ease !important;
}

.gr-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 12px rgba(112, 119, 161, 0.4) !important;
}

/* Secondary button styling */
.gr-button.secondary {
    background: var(--secondary-color) !important;
    color: var(--highlight-color) !important;
    border: 2px solid var(--accent-color) !important;
}

/* Tab styling */
.gr-tab-nav {
    background-color: var(--secondary-color) !important;
    border-bottom: 2px solid var(--accent-color) !important;
}

.gr-tab-nav button {
    background-color: transparent !important;
    color: var(--highlight-color) !important;
    border: none !important;
}

.gr-tab-nav button.selected {
    background-color: var(--accent-color) !important;
    color: white !important;
}

/* Block styling */
.gr-block {
    background-color: rgba(66, 71, 105, 0.3) !important;
    border: 1px solid var(--accent-color) !important;
    border-radius: 12px !important;
    padding: 20px !important;
    margin: 10px 0 !important;
}

/* File upload styling */
.gr-file {
    background-color: var(--secondary-color) !important;
    border: 2px dashed var(--accent-color) !important;
    border-radius: 8px !important;
}

/* Label styling */
label {
    color: var(--highlight-color) !important;
    font-weight: 500 !important;
}

/* Output text styling */
.gr-textbox textarea {
    background-color: var(--secondary-color) !important;
    border: 2px solid var(--accent-color) !important;
    color: #ffffff !important;
    border-radius: 8px !important;
}
"""


# Gradio Interface Definition
with gr.Blocks(title="Credit Card Fraud Detection", css=css, theme="base") as fraud_detection_app:
    gr.Markdown(
        """
        # 🏦 Credit Card Fraud Detection
        This model is loaded directly from the 'Production' stage in the MLflow Model Registry.
        **Note:** The V1-V28 features are PCA-transformed features from the original dataset for privacy protection.
        """
    )

    with gr.Tabs():
        with gr.TabItem("Single Transaction"):
            gr.Markdown(
                "### Enter transaction features to predict whether a credit card "
                "transaction is fraudulent or legitimate."
            )

            # All input fields organized in a structured block layout
            with gr.Group():
                gr.Markdown("### 📊 Transaction Input Fields")

                # Transaction basics
                with gr.Row():
                    time = gr.Number(label="Time", value=0.0, info="Time elapsed since first transaction")
                    amount = gr.Number(label="Amount", value=100.0, info="Transaction amount")

                # PCA Features organized in a grid
                gr.Markdown("#### PCA Features (V1-V28)")
                with gr.Row():
                    with gr.Column(scale=1):
                        with gr.Row():
                            v1 = gr.Number(label="V1", value=0.0)
                            v2 = gr.Number(label="V2", value=0.0)
                            v3 = gr.Number(label="V3", value=0.0)
                            v4 = gr.Number(label="V4", value=0.0)
                        with gr.Row():
                            v5 = gr.Number(label="V5", value=0.0)
                            v6 = gr.Number(label="V6", value=0.0)
                            v7 = gr.Number(label="V7", value=0.0)
                            v8 = gr.Number(label="V8", value=0.0)
                        with gr.Row():
                            v9 = gr.Number(label="V9", value=0.0)
                            v10 = gr.Number(label="V10", value=0.0)
                            v11 = gr.Number(label="V11", value=0.0)
                            v12 = gr.Number(label="V12", value=0.0)
                        with gr.Row():
                            v13 = gr.Number(label="V13", value=0.0)
                            v14 = gr.Number(label="V14", value=0.0)

                    with gr.Column(scale=1):
                        with gr.Row():
                            v15 = gr.Number(label="V15", value=0.0)
                            v16 = gr.Number(label="V16", value=0.0)
                            v17 = gr.Number(label="V17", value=0.0)
                            v18 = gr.Number(label="V18", value=0.0)
                        with gr.Row():
                            v19 = gr.Number(label="V19", value=0.0)
                            v20 = gr.Number(label="V20", value=0.0)
                            v21 = gr.Number(label="V21", value=0.0)
                            v22 = gr.Number(label="V22", value=0.0)
                        with gr.Row():
                            v23 = gr.Number(label="V23", value=0.0)
                            v24 = gr.Number(label="V24", value=0.0)
                            v25 = gr.Number(label="V25", value=0.0)
                            v26 = gr.Number(label="V26", value=0.0)
                        with gr.Row():
                            v27 = gr.Number(label="V27", value=0.0)
                            v28 = gr.Number(label="V28", value=0.0)

            # Action buttons in a single row
            with gr.Group():
                gr.Markdown("### 🎯 Actions")
                with gr.Row():
                    predict_btn = gr.Button("🔍 Analyze Transaction", variant="primary", size="lg", scale=2)
                    random_btn = gr.Button("🎲 Generate Random Data", variant="secondary", size="lg", scale=1)
                    example_legitimate_btn = gr.Button("📝 Load Legitimate Example",
                    variant="secondary", size="sm", scale=1)
                    example_fraud_btn = gr.Button("⚠️ Load Fraud Example", variant="secondary", size="sm", scale=1)

            # Output section
            with gr.Group():
                gr.Markdown("### 📋 Prediction Result")
                output_prediction = gr.Textbox(label="Result", interactive=False, lines=2)

        with gr.TabItem("Batch Predictions"):
            gr.Markdown(
                """
                ### Upload a CSV file for batch fraud detection
                **Required CSV Format:**
                Your CSV file must contain the following columns:
                - `Time`, `V1`, `V2`, ..., `V28`, `Amount`
                The CSV can contain additional columns, but the above columns are required for predictions.
                Results will include all original columns plus prediction results.
                """
            )

            with gr.Group():
                gr.Markdown("### 📁 File Upload & Processing")
                with gr.Row():
                    with gr.Column(scale=2):
                        csv_input = gr.File(label="Upload CSV File", file_types=[".csv"], file_count="single")
                        batch_predict_btn = gr.Button("🔍 Predict Batch", variant="primary", size="lg")

                    with gr.Column(scale=3):
                        batch_results = gr.Textbox(label="Batch Prediction Summary", interactive=False, lines=8)

                with gr.Row():
                    with gr.Column(scale=1):
                        download_file = gr.File(label="Download Results", interactive=False)
                    with gr.Column(scale=2):
                        pie_chart = gr.Image(label="Fraud Distribution", type="filepath", interactive=False)

    # Random data button handler
    random_btn.click(
        fn=generate_random_data,
        outputs=[
            time,
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            v8,
            v9,
            v10,
            v11,
            v12,
            v13,
            v14,
            v15,
            v16,
            v17,
            v18,
            v19,
            v20,
            v21,
            v22,
            v23,
            v24,
            v25,
            v26,
            v27,
            v28,
            amount,
        ],
    )

    # Button click handlers
    predict_btn.click(
        fn=predict_fraud,
        inputs=[
            time,
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            v8,
            v9,
            v10,
            v11,
            v12,
            v13,
            v14,
            v15,
            v16,
            v17,
            v18,
            v19,
            v20,
            v21,
            v22,
            v23,
            v24,
            v25,
            v26,
            v27,
            v28,
            amount,
        ],
        outputs=output_prediction,
    )

    # Batch prediction handler
    batch_predict_btn.click(
        fn=predict_fraud_batch,
        inputs=[csv_input],
        outputs=[download_file, batch_results, pie_chart],
    )

    # Example data functions
    def load_legitimate_example():
        # Example of a legitimate transaction (standardized values)
        return (
            -1.677,  # time
            1.024,
            2.001,
            -4.770,
            3.819,
            -1.272,
            -1.735,
            -3.059,
            0.890,
            0.415,
            -3.956,  # V1-V10
            3.572,
            -7.186,
            0.147,
            -5.249,
            1.678,
            -2.641,
            -1.312,
            -0.392,
            1.118,  # V11-V20
            0.204,
            0.343,
            -0.054,
            0.710,
            -0.372,
            -2.032,
            0.367,
            0.395,
            0.020,  # V21-V28
            -0.453,  # amount (standardized)
        )

    def load_fraud_example():
        # Example of a fraudulent transaction (standardized values)
        return (
            0.243,  # time
            -25.826,
            19.167,
            -25.390,
            11.125,
            -16.683,
            3.934,
            -37.060,
            -28.760,
            -11.127,
            -23.228,  # V1-V10
            3.786,
            -10.522,
            -2.657,
            -3.793,
            -4.499,
            -6.558,
            -12.867,
            -5.805,
            -1.254,  # V11-V20
            7.907,
            -16.922,
            5.704,
            3.510,
            0.054,
            -0.672,
            -0.209,
            -4.950,
            -0.448,  # V21-V28
            -0.447,  # amount (standardized)
        )

    example_legitimate_btn.click(
        fn=load_legitimate_example,
        outputs=[
            time,
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            v8,
            v9,
            v10,
            v11,
            v12,
            v13,
            v14,
            v15,
            v16,
            v17,
            v18,
            v19,
            v20,
            v21,
            v22,
            v23,
            v24,
            v25,
            v26,
            v27,
            v28,
            amount,
        ],
    )

    example_fraud_btn.click(
        fn=load_fraud_example,
        outputs=[
            time,
            v1,
            v2,
            v3,
            v4,
            v5,
            v6,
            v7,
            v8,
            v9,
            v10,
            v11,
            v12,
            v13,
            v14,
            v15,
            v16,
            v17,
            v18,
            v19,
            v20,
            v21,
            v22,
            v23,
            v24,
            v25,
            v26,
            v27,
            v28,
            amount,
        ],
    )

# Launch the App
if __name__ == "__main__":
    print("Launching Gradio app to interact with the fraud detection model")
    fraud_detection_app.launch(server_name="0.0.0.0", server_port=7862)
