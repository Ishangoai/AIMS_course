"""
Main Gradio application for Credit Card Fraud Detection.
Interactive interface for single and batch predictions.
"""

import gradio as gr
import gradio.themes
import numpy as np
from gradioapp.gradio_utils import (
    FraudDetectionModel,
    load_default_dataset,
    predict_batch_transactions,
    predict_single_transaction,
)

# =========================
# Configuration
# =========================

MODEL_NAME = "fraud_detection_rf"
MODEL_STAGE = "latest"
DB_PATH = "/home/mirandraf/Desktop/AIMS_course/src/students/efandresena/dagster/ml_fraud/mlflow_artifacts/mlflow_fraud_tracking.db"  # noqa: E501
DATA_URL = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"

# Feature names (excluding 'Class' label), reduced to 10 features
FEATURE_NAMES = ['V14', 'V17', 'V12', 'V10', 'V16', 'V11', 'V4', 'V3', 'V9', 'V18']


# =========================
# Initialize Model and Data
# =========================

print("🔄 Loading fraud detection model...")
fraud_model = FraudDetectionModel(
    model_name=MODEL_NAME,
    model_stage=MODEL_STAGE,
    db_path=DB_PATH
)
print("✅ Model loaded successfully!")

print("🔄 Loading default dataset...")
default_df, feature_ranges = load_default_dataset(DATA_URL)
print(f"✅ Default dataset loaded: {len(default_df)} transactions")


# =========================
# Wrapper Functions for Gradio
# =========================

def single_prediction_wrapper(*slider_values):
    """Wrapper for single transaction prediction."""
    try:
        fig, label = predict_single_transaction(
            model=fraud_model,
            slider_values=list(slider_values),
            feature_names=FEATURE_NAMES
        )
        return fig, gr.Label(value=label, label="Prediction Result")
    except Exception as e:
        print(f"Error in single prediction: {e}")
        return None, gr.Label(value=f"Error: {str(e)}", label="Error")


def batch_prediction_wrapper(file, use_default):
    """Wrapper for batch transaction prediction."""
    try:
        results = predict_batch_transactions(
            model=fraud_model,
            file=file,
            use_default=use_default,
            default_df=default_df,
            feature_names=FEATURE_NAMES
        )
        return results
    except Exception as e:
        print(f"Error in batch prediction: {e}")
        return None, None, None, None, None, None


# =========================
# Gradio Interface
# =========================

with gr.Blocks(theme=gradio.themes.Soft(), title="Fraud Detection - Mirindra & TIAO") as demo:

    # Header
    gr.Markdown("""
    # 💳 Credit Card Fraud Detection
    ### Interactive ML Model by Mirindra and TIAO

    This application uses a Random Forest classifier trained on credit card transaction data
    to detect potentially fraudulent transactions in real-time.

    Choose between **Single Prediction** for individual transactions or **Batch Prediction**
    for analyzing multiple transactions at once.
    """)

    with gr.Tabs():

        # =======================
        # Single Prediction Tab
        # =======================
        with gr.TabItem("🔍 Single Transaction Prediction"):
            gr.Markdown("""
            ### Predict fraud risk for a single transaction
            Adjust the feature sliders below to input transaction details, then click **Run Prediction**.
            """)

            with gr.Row():
                # Sliders organized in columns
                all_sliders = []
                num_features_per_col = 10  # all 10 fit in one column here

                for col_idx in range(0, len(FEATURE_NAMES), num_features_per_col):
                    with gr.Column(scale=1):
                        col_features = FEATURE_NAMES[col_idx:col_idx + num_features_per_col]
                        gr.Markdown(f"#### Features {col_idx + 1} to {col_idx + len(col_features)}")

                        for feature in col_features:
                            min_val, max_val = feature_ranges[feature]
                            step = abs(max_val - min_val) / 1000 if max_val != min_val else 0.01
                            default_val = float(np.mean([min_val, max_val]))

                            slider = gr.Slider(
                                minimum=min_val,
                                maximum=max_val,
                                value=default_val,
                                label=feature,
                                interactive=True,
                                step=step
                            )
                            all_sliders.append(slider)

                # Results column
                with gr.Column(scale=1):
                    gr.Markdown("#### 🎯 Prediction Result")
                    predict_single_button = gr.Button("🚀 Run Prediction", variant="primary", size="lg")
                    output_gauge = gr.Plot(label="Fraud Probability Gauge")
                    output_label = gr.Label(label="Result")

                    gr.Markdown("""
                    **Interpretation:**
                    - 🟢 **Not Fraud**: Low probability of fraudulent activity
                    - 🔴 **FRAUD**: High probability of fraudulent activity
                    """)

            predict_single_button.click(
                fn=single_prediction_wrapper,
                inputs=all_sliders,
                outputs=[output_gauge, output_label]
            )

        # =======================
        # Batch Prediction Tab
        # =======================
        with gr.TabItem("📊 Batch Transaction Prediction"):
            gr.Markdown(f"""
            ### Analyze multiple transactions at once
            Upload your own CSV file or use the default dataset for batch predictions.

            **CSV Format Requirements:**
            - Must contain the following {len(FEATURE_NAMES)} features: {', '.join(FEATURE_NAMES)}
            - No 'Class' column needed (this is what we predict)
            """)

            with gr.Row():
                # Input column
                with gr.Column(scale=1):
                    gr.Markdown("#### 📥 Input Options")
                    upload_file = gr.File(
                        label="Upload CSV File",
                        file_types=[".csv"],
                        type="filepath"
                    )
                    use_default_checkbox = gr.Checkbox(
                        label="Use default sample dataset",
                        value=True
                    )
                    predict_batch_button = gr.Button(
                        "🚀 Run Batch Predictions",
                        variant="primary",
                        size="lg"
                    )

                    gr.Markdown("---")
                    gr.Markdown("#### 💾 Download Results")
                    download_file = gr.File(label="Download CSV with predictions")

                    gr.Markdown("""
                    **Output CSV includes:**
                    - All original features
                    - Fraud_Prediction (0 or 1)
                    - Fraud_Probability (0.0 to 1.0)
                    - Fraud_Prediction_Label
                    """)

                # Visualizations column
                with gr.Column(scale=2):
                    gr.Markdown("#### 📈 Results Visualizations")

                    with gr.Tabs():
                        with gr.TabItem("💰 Transaction Amounts"):
                            output_amount_hist = gr.Plot(label="Amount Distribution")

                        with gr.TabItem("🥧 Fraud Proportion"):
                            output_fraud_pie = gr.Plot(label="Fraud vs Non-Fraud")

                        # Show plots for first 3 features in FEATURE_NAMES (excluding Amount)
                        for i in range(min(3, len(FEATURE_NAMES))):
                            feature = FEATURE_NAMES[i]
                            with gr.TabItem(f"📊 {feature}"):
                                locals()[f"output_feature{i + 1}"] = gr.Plot(label=f"{feature} Distribution")

            # Build output list for batch_prediction_wrapper outputs
            batch_outputs = [
                output_amount_hist,
                output_fraud_pie,
            ]
            for i in range(min(3, len(FEATURE_NAMES))):
                batch_outputs.append(locals()[f"output_feature{i + 1}"])
            batch_outputs.append(download_file)  # pyright: ignore [reportArgumentType]

            predict_batch_button.click(
                fn=batch_prediction_wrapper,
                inputs=[upload_file, use_default_checkbox],
                outputs=batch_outputs
            )

        # =======================
        # About Tab
        # =======================
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ## About This Application

            ### 🎯 Purpose
            This fraud detection system helps identify potentially fraudulent credit card transactions
            using machine learning. It's built using:
            - **Model**: Random Forest Classifier
            - **Framework**: Dagster for orchestration, MLflow for tracking
            - **Interface**: Gradio for interactive predictions

            ### 📊 Dataset Information
            - **Source**: Credit Card Fraud Detection Dataset
            - **Features**: 30 numerical features (Time, V1-V28, Amount)
            - **Target**: Binary classification (Fraud vs Non-Fraud)
            - **Note**: Features V1-V28 are PCA-transformed for privacy

            ### 👥 Developers
            **Mirindra & TIAO** - AIMS Course October 2025

            ### 🔧 Technical Details
            - Model training pipeline built with Dagster
            - Hyperparameter tuning with GridSearchCV
            - MLflow for experiment tracking and model registry
            - Interactive deployment with Gradio

            ### 📝 Usage Tips
            - **Single Prediction**: Best for analyzing individual suspicious transactions
            - **Batch Prediction**: Process multiple transactions efficiently
            - **Interpretation**: Focus on transactions with high fraud probability (>0.5)

            ### ⚠️ Disclaimer
            This is a demonstration model for educational purposes. For production use,
            ensure proper validation, monitoring, and regulatory compliance.
            """)

    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #666;">
        <p>💳 Fraud Detection System v1.0 | Built with Dagster, MLflow & Gradio</p>
        <p>© 2025 Mirindra & TIAO - AIMS Course Project</p>
    </div>
    """)


# =========================
# Launch Application
# =========================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Launching Fraud Detection Gradio App...")
    print("=" * 60 + "\n")

    demo.launch(
        debug=True,
        share=True,
        server_name="127.0.0.1",
        server_port=7870
    )
