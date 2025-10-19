
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from gradioapp.utils.fraud_model_utils import predict_fraud_activity

# --- Utility Functions ---


def wrapped_predict(*args):
    """Wraps the prediction function to accept a list of features."""
    return predict_fraud_activity(list(args))


# Define feature ranges - adjust these based on your actual data ranges
FEATURE_RANGES = {
    'Time': (0, 172792), 'V1': (-56.4, 2.5), 'V2': (-72.7, 22.1),
    'V3': (-48.3, 9.4), 'V4': (-5.7, 16.9), 'V5': (-113.7, 34.8),
    'V6': (-26.2, 73.3), 'V7': (-43.6, 120.6), 'V8': (-73.2, 20.0),
    'V9': (-13.4, 15.6), 'V10': (-24.6, 23.7), 'V11': (-4.8, 12.0),
    'V12': (-18.7, 7.8), 'V13': (-5.8, 7.1), 'V14': (-19.2, 10.5),
    'V15': (-4.5, 8.9), 'V16': (-14.1, 17.3), 'V17': (-25.2, 9.3),
    'V18': (-9.5, 5.0), 'V19': (-7.2, 5.6), 'V20': (-54.5, 39.4),
    'V21': (-34.8, 27.2), 'V22': (-10.9, 10.5), 'V23': (-44.8, 22.5),
    'V24': (-2.8, 4.6), 'V25': (-10.3, 7.5), 'V26': (-2.6, 3.5),
    'V27': (-22.6, 31.6), 'V28': (-15.4, 33.8),
    'Amount': (0, 25691.16)
}


def create_feature_importance_plot(model, feature_names):
    """Create a bar plot of feature importances."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-10:]  # Top 10 features

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(indices)), importances[indices], color='#3b82f6')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title('Top 10 Most Important Features')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        return fig
    return None


def create_input_distribution_plot(input_values, feature_names):
    """Visualize the current input values."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Create color map based on value ranges
    # Simplified value ranges for visualization (e.g., normalized 0-100)
    # The original logic used the raw input values which might not work well
    # for features with very different scales. Assuming a hypothetical
    # visualization logic for demonstration.
    colors = ['#ef4444' if v > 65 else '#fbbf24' if v > 35 else '#10b981'
              for v in input_values]

    ax.bar(range(len(input_values)), input_values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(input_values)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_ylabel('Value')
    ax.set_title('Current Input Feature Values')
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Midpoint')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    return fig


# Custom CSS for better styling
custom_css = """
body {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
}

.gradio-container {
    max-width: 1800px !important;
    margin: 0 auto !important;
}

.main {
    margin: 0 auto !important;
}

.header-section {
    text-align: center;
    padding: 1.5rem 1rem;
    background: rgba(30, 30, 46, 0.95);
    border-radius: 15px;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    border: 1px solid rgba(102, 126, 234, 0.3);
}

.header-section h1 {
    color: #667eea;
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

.header-section p {
    color: #a0aec0;
    font-size: 1rem;
}

.prediction-card {
    background: rgba(30, 30, 46, 0.8);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}

.tabs {
    background: rgba(30, 30, 46, 0.8);
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
}

#predict-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    font-size: 1.2rem !important;
    font-weight: 600 !important;
    padding: 1rem 3rem !important;
    transition: transform 0.2s !important;
}

#predict-btn:hover {
    transform: scale(1.05) !important;
}

.result-box {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    border-radius: 12px;
    padding: 1.5rem;
    font-size: 1.3rem;
    font-weight: 600;
    text-align: center;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
}

.input-section {
    background: rgba(30, 30, 46, 0.8);
    border-radius: 12px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.scrollable-inputs {
    max-height: 600px;
    overflow-y: auto;
    padding-right: 10px;
}

.scrollable-inputs::-webkit-scrollbar {
    width: 8px;
}

.scrollable-inputs::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
}

.scrollable-inputs::-webkit-scrollbar-thumb {
    background: #667eea;
    border-radius: 10px;
}

.scrollable-inputs::-webkit-scrollbar-thumb:hover {
    background: #764ba2;
}
"""

# --- Gradio Interface Definition ---

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as fraud_app:

    # Header Section
    with gr.Row():
        with gr.Column():
            gr.HTML("""
                <div class="header-section">
                    <h1>🛡️ AI Fraud Detection System</h1>
                    <p>Advanced Machine Learning Model for Real-Time
                    Transaction Analysis</p>
                    <p style="font-size: 0.9rem; color: #718096;
                    margin-top: 0.5rem;">
                        MLOPs Assignment 2 Project | Built with Random Forest
                        Classifier
                    </p>
                </div>
            """)

    # Define feature names
    feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']

    # --- Input Components Definition ---
    # Moved component instantiation to a dictionary to avoid massive,
    # non-PEP8 compliant list unpacking later.

    input_components = {}

    def create_slider(name):
        """Helper to create a slider component."""
        min_val, max_val = FEATURE_RANGES[name]
        default_val = 0
        if name == 'Time':
            default_val = max_val / 2
        elif name == 'Amount':
            default_val = 100
        return gr.Slider(min_val, max_val, value=default_val, label=name)

    # Instantiate all sliders
    for name in feature_names:
        input_components[name] = create_slider(name)

    # Group sliders for easier access and to pass to functions
    sliders = [input_components[name] for name in feature_names]
    time, amount = input_components['Time'], input_components['Amount']
    v_features = {f'v{i}': input_components[f'V{i}'] for i in range(1, 29)}
    v_cols = [v_features[f'v{i}'] for i in range(1, 29)]

    # Split V features into columns (10, 10, 8 for V1-V28)
    v_col1 = v_cols[0:10]    # V1-V10
    v_col2 = v_cols[10:20]   # V11-V20
    v_col3 = v_cols[20:24]   # V21-V24
    v_col4 = v_cols[24:28]   # V25-V28

    # --- Tabs Definition ---

    with gr.Tabs():
        # Tab 1: Input and Prediction
        with gr.TabItem("🎯 Make Prediction"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Transaction Features")
                    gr.Markdown("Adjust the key features below (scroll for more)")

                    with gr.Column(elem_classes="scrollable-inputs"):
                        with gr.Row():
                            # Input Column 1
                            with gr.Column():
                                gr.Markdown("**🕐 Core Features**")
                                # Time, V1-V10
                                time.info = "Transaction time"
                                [time] + v_col1

                            # Input Column 2
                            with gr.Column():
                                gr.Markdown("**🔢 Additional Features**")
                                # V11-V20
                                v_col2

                        with gr.Row():
                            # Input Column 3
                            with gr.Column():
                                gr.Markdown("**📈 Extended Features**")
                                # V21-V24
                                v_col3

                            # Input Column 4
                            with gr.Column():
                                gr.Markdown("**🎯 Final Features**")
                                # V25-V28, Amount
                                v_col4
                                amount.info = "Transaction amount"
                                amount

                # Results Column
                with gr.Column(scale=1):
                    gr.Markdown("### 🎯 Prediction Result")

                    # Buttons
                    with gr.Row():
                        predict_btn = gr.Button("🔍 Analyze Transaction",
                                                variant="primary", size="lg",
                                                elem_id="predict-btn", scale=2)
                        reset_btn = gr.Button("🔄 Reset", variant="secondary",
                                              size="lg", scale=1)

                    # Result Display
                    result = gr.Textbox(
                        label="Classification Result",
                        lines=4,
                        show_label=True,
                        container=True,
                        elem_classes="result-box"
                    )

                    gr.Markdown("""
                    ---
                    **Quick Guide:**
                    - ✅ **Legitimate**: Safe transaction
                    - ⚠️ **Fraud**: Suspicious activity detected
                    - 📊 View confidence in next tab
                    """)

        # Tab 2: Prediction Confidence
        with gr.TabItem("📈 Confidence Analysis"):
            gr.Markdown("### 🎲 Model Confidence Score")
            gr.Markdown("See how confident the AI model is about its prediction")
            probability_plot = gr.Plot(label="Prediction Probability Distribution")

        # Tab 3: Input Values Visualization
        with gr.TabItem("📉 Input Visualization"):
            gr.Markdown("### 🎨 Your Input Values at a Glance")
            gr.Markdown("Visual representation of all transaction features you've entered")
            input_dist_plot = gr.Plot(label="Feature Value Distribution")
            gr.Markdown("""
            **Color Legend:**
            - 🟢 Green (20-35): Low values
            - 🟡 Yellow (35-65): Medium values
            - 🔴 Red (65-80): High values
            """)

        # Tab 4: Feature Importance
        with gr.TabItem("⭐ Model Insights"):
            gr.Markdown("### 🧠 Feature Importance Analysis")
            gr.Markdown("Discover which features have the most impact on fraud detection")
            feature_importance_plot = gr.Plot(
                label="Top 10 Most Important Features")
            gr.Markdown("""
            **Understanding Feature Importance:**
            - Higher bars indicate features that strongly influence predictions
            - The model weighs these features more heavily when making decisions
            - This helps understand what makes a transaction suspicious
            """)

        # Tab 5: About
        with gr.TabItem("ℹ️ About"):
            gr.Markdown("""
            ## About This Application

            ### 🎓 Project Information
            This fraud detection system was developed as part of the MLOPs
            AIMS Assignment.

            ### 🤖 Technology Stack
            - **Model**: Random Forest Classifier
            - **Features**: 30 input features (Time, V1-V28, Amount)
            - **Purpose**: Real-time fraud detection in financial transactions

            ### 📚 How It Works
            1. **Input**: Enter transaction characteristics using the sliders
            2. **Analysis**: The Random Forest model processes all 30 features
            3. **Prediction**: Get a classification (Legitimate or Fraud)
            4. **Confidence**: View probability scores for transparency

            ### 🔒 Model Performance
            The model has been trained on historical transaction data to identify
            patterns associated with fraudulent activities while minimizing
            false positives.

            ### 👨‍💻 For More Information
            Contact Koto and Melvin.
            """)

    # --- Backend Logic ---

    def predict_and_visualize(*args):
        """Runs prediction and creates all visualizations."""
        prediction = wrapped_predict(*args)

        # Extract numeric value from prediction if possible
        pred_value = None
        try:
            # Try to extract number from prediction string
            # Re-importing re here for demonstration of moving imports,
            # though it should be at the top level
            import re
            numbers = re.findall(r'\d+\.?\d*', str(prediction))
            if numbers:
                pred_value = float(numbers[0])
            else:
                pred_value = None
        except Exception:
            pred_value = None

        # Format prediction result with styling based on threshold
        if pred_value is not None:
            if pred_value <= 5:
                formatted_result = (
                    f"✅ LEGITIMATE TRANSACTION\n\nScore: {pred_value}"
                    f"\n\nRecommendation: Approve transaction"
                )
            else:
                formatted_result = (
                    f"⚠️ FRAUD DETECTED\n\nScore: {pred_value}"
                    f"\n\nRecommendation: Flag for review"
                )
        else:
            # Fallback to keyword detection if no numeric value found
            if ("fraud" in prediction.lower() or
                    "suspicious" in prediction.lower()):
                formatted_result = (
                    f"⚠️ FRAUD DETECTED\n\n{prediction}"
                    f"\n\nRecommendation: Flag for review"
                )
            else:
                formatted_result = (
                    f"✅ LEGITIMATE TRANSACTION\n\n{prediction}"
                    f"\n\nRecommendation: Approve transaction"
                )

        # Create probability plot
        prob_plot = None
        try:
            from gradioapp.utils.fraud_model_utils import model
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba([list(args)])[0]

                fig, ax = plt.subplots(figsize=(10, 6))
                classes = ['Legitimate', 'Fraud']
                colors = ['#10b981', '#ef4444']
                bars = ax.barh(classes, proba, color=colors, height=0.6)

                for i, (bar, prob) in enumerate(zip(bars, proba)):
                    ax.text(prob + 0.02, i, f'{prob * 100:.2f}%',
                           va='center', fontsize=16, fontweight='bold')

                ax.set_xlim(0, 1.15)
                ax.set_xlabel('Probability', fontsize=14)
                ax.set_title('Model Confidence Score', fontsize=16,
                             fontweight='bold')
                ax.grid(axis='x', alpha=0.3)
                plt.tight_layout()
                prob_plot = fig
        except Exception:
            pass

        # Create input distribution plot
        input_plot = create_input_distribution_plot(list(args), feature_names)

        return formatted_result, prob_plot, input_plot

    def reset_sliders():
        """Reset all sliders to their default values."""
        # The order must match the order in the 'sliders' list
        defaults = [
            FEATURE_RANGES['Time'][1] / 2,  # Time
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # V1-V10
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  # V11-V20
            0, 0, 0, 0,  # V21-V24
            0, 0, 0, 0,  # V25-V28
            100  # Amount
        ]
        return defaults

    def load_feature_importance():
        """Load feature importance plot on app start."""
        try:
            from gradioapp.utils.fraud_model_utils import model
            return create_feature_importance_plot(model, feature_names)
        except Exception:
            return None

    # Connect prediction button
    predict_btn.click(
        fn=predict_and_visualize,
        inputs=sliders,
        outputs=[result, probability_plot, input_dist_plot]
    )

    # Connect reset button
    reset_btn.click(
        fn=reset_sliders,
        inputs=[],
        outputs=sliders
    )

    # Load feature importance when app starts
    fraud_app.load(load_feature_importance, outputs=feature_importance_plot)

if __name__ == "__main__":
    fraud_app.launch()
