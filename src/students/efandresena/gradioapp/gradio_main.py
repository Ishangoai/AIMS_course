"""
Main Gradio application for Credit Card Fraud Detection.
Interactive interface for single and batch predictions.
Updated to load from .pkl file with full Pyright type safety.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import gradio.themes
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .gradio_utils import (
    FraudDetectionModel,
    load_default_dataset,
    predict_batch_transactions,
    predict_single_transaction,
)

# =========================
# Configuration
# =========================

# Get model path - look in the same directory as this file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(CURRENT_DIR, "model.pkl"))

DATA_URL = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"

# Feature names (excluding 'Class' label), reduced to 10 features
FEATURE_NAMES: List[str] = ['V14', 'V17', 'V12', 'V10', 'V16', 'V11', 'V4', 'V3', 'V9', 'V18']

# Set temp directory for GCP
os.environ.setdefault('TMPDIR', '/tmp')

# =========================
# Lazy Model Loading
# =========================

_fraud_model: Optional[FraudDetectionModel] = None
_default_df: Optional[pd.DataFrame] = None
_feature_ranges: Optional[Dict[str, Tuple[float, float]]] = None
_model_load_error: Optional[str] = None


def get_model_and_data() -> Tuple[FraudDetectionModel, pd.DataFrame, Dict[str, Tuple[float, float]]]:
    """Lazy load model and data on first use."""
    global _fraud_model, _default_df, _feature_ranges, _model_load_error

    if _fraud_model is None and _model_load_error is None:
        try:
            print("🔄 Loading fraud detection model...")
            print(f"Looking for model at: {MODEL_PATH}")
            _fraud_model = FraudDetectionModel(model_path=MODEL_PATH)
            print("✅ Model loaded successfully!")

            print("🔄 Loading default dataset...")
            _default_df, _feature_ranges = load_default_dataset(DATA_URL)
            print(f"✅ Default dataset loaded: {len(_default_df)} transactions")
        except Exception as e:
            _model_load_error = str(e)
            print(f"❌ Error loading model: {e}")
            raise

    if _model_load_error:
        raise RuntimeError(f"Model failed to load: {_model_load_error}")

    # Type assertions to satisfy type checker
    assert _fraud_model is not None, "Model should be loaded"
    assert _default_df is not None, "DataFrame should be loaded"
    assert _feature_ranges is not None, "Feature ranges should be loaded"

    return _fraud_model, _default_df, _feature_ranges


# =========================
# Wrapper Functions for Gradio
# =========================

def single_prediction_wrapper(*slider_values: float) -> Tuple[Optional[Figure], gr.Label]:
    """Wrapper for single transaction prediction."""
    try:
        fraud_model, _, _ = get_model_and_data()

        fig, label = predict_single_transaction(
            model=fraud_model,
            slider_values=list(slider_values),
            feature_names=FEATURE_NAMES
        )
        return fig, gr.Label(value=label, label="Prediction Result")
    except Exception as e:
        print(f"Error in single prediction: {e}")
        return None, gr.Label(value="error_msg", label="Error")


def batch_prediction_wrapper(
    file: Optional[Any],
    use_default: bool
) -> Tuple[Optional[Figure], Optional[Figure], Optional[Figure], Optional[Figure], Optional[Figure], Optional[str]]:
    """Wrapper for batch transaction prediction."""
    try:
        fraud_model, default_df, _ = get_model_and_data()

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

def build_interface() -> gr.Blocks:
    """Build the Gradio interface."""

    # Try to load model and data for slider ranges
    feature_ranges: Dict[str, Tuple[float, float]]
    try:
        _, _, feature_ranges = get_model_and_data()
    except Exception as e:
        print(f"Warning: Could not load model for initial setup: {e}")
        # Use default ranges if model can't load
        feature_ranges = {name: (-5.0, 5.0) for name in FEATURE_NAMES}

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
                    all_sliders: List[gr.Slider] = []
                    num_features_per_col = 10

                    for col_idx in range(0, len(FEATURE_NAMES), num_features_per_col):
                        with gr.Column(scale=1):
                            col_features = FEATURE_NAMES[col_idx:col_idx + num_features_per_col]
                            gr.Markdown(f"#### Features {col_idx + 1} to {col_idx + len(col_features)}")

                            for feature in col_features:
                                min_val, max_val = feature_ranges.get(feature, (-5.0, 5.0))
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

                            with gr.TabItem(f"📊 {FEATURE_NAMES[0]}"):
                                output_feature1 = gr.Plot(label=f"{FEATURE_NAMES[0]} Distribution")

                            with gr.TabItem(f"📊 {FEATURE_NAMES[1]}"):
                                output_feature2 = gr.Plot(label=f"{FEATURE_NAMES[1]} Distribution")

                            with gr.TabItem(f"📊 {FEATURE_NAMES[2]}"):
                                output_feature3 = gr.Plot(label=f"{FEATURE_NAMES[2]} Distribution")

                # Build output list
                batch_outputs: List[Any] = [
                    output_amount_hist,
                    output_fraud_pie,
                    output_feature1,
                    output_feature2,
                    output_feature3,
                    download_file
                ]

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
                - **Framework**: Scikit-learn
                - **Interface**: Gradio for interactive predictions

                ### 📊 Dataset Information
                - **Source**: Credit Card Fraud Detection Dataset
                - **Features**: 30 numerical features (Time, V1-V28, Amount)
                - **Target**: Binary classification (Fraud vs Non-Fraud)
                - **Note**: Features V1-V28 are PCA-transformed for privacy

                ### 👥 Developers
                **Mirindra & TIAO** - AIMS Course October 2025

                ### 🔧 Technical Details
                - Model training with scikit-learn
                - Interactive deployment with Gradio
                - Real-time predictions with probability scores

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
            <p>💳 Fraud Detection System v1.0 | Built with Scikit-learn & Gradio</p>
            <p>© 2025 Mirindra & TIAO - AIMS Course Project</p>
        </div>
        """)

    return demo


# Build interface
demo: gr.Blocks = build_interface()


# =========================
# Launch Application
# =========================

if __name__ == "__main__":
    demo.launch(share=True)
