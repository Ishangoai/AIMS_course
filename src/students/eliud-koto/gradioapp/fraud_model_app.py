import gradio as gr
import matplotlib.pyplot as plt
import numpy as np

# Note: Ensure this utility path is correct and accessible.
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

    # Normalize values for simple color mapping (0-100 range)
    normalized_values = []
    for name, value in zip(feature_names, input_values):
        min_val, max_val = FEATURE_RANGES[name]
        # Avoid division by zero if min and max are the same
        if max_val == min_val:
            normalized_values.append(50)
        else:
            normalized_val = 100 * (value - min_val) / (max_val - min_val)
            normalized_values.append(normalized_val)

    # Create color map based on normalized values
    colors = ['#ef4444' if v > 65 else '#fbbf24' if v > 35 else '#10b981'
              for v in normalized_values]

    # Use input_values for bar height
    ax.bar(range(len(input_values)), input_values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(input_values)))
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_ylabel('Raw Value')
    ax.set_title('Current Input Feature Values')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    return fig


# Custom CSS for better styling
custom_css = """
.gradio-container {
    max-width: 1800px !important;
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

with gr.Blocks(css=custom_css, theme="black") as fraud_app:

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
                                # Create Time slider
                                time_min, time_max = FEATURE_RANGES['Time']
                                time_slider = gr.Slider(
                                    minimum=time_min,
                                    maximum=time_max,
                                    value=time_max / 2,
                                    label='Time'
                                )

                                # Create V1-V10 sliders
                                v_sliders_1 = []
                                for i in range(1, 11):
                                    min_val, max_val = FEATURE_RANGES[f'V{i}']
                                    slider = gr.Slider(
                                        minimum=min_val,
                                        maximum=max_val,
                                        value=0,
                                        label=f'V{i}'
                                    )
                                    v_sliders_1.append(slider)

                            # Input Column 2
                            with gr.Column():
                                gr.Markdown("**🔢 Additional Features**")
                                # Create V11-V20 sliders
                                v_sliders_2 = []
                                for i in range(11, 21):
                                    min_val, max_val = FEATURE_RANGES[f'V{i}']
                                    slider = gr.Slider(
                                        minimum=min_val,
                                        maximum=max_val,
                                        value=0,
                                        label=f'V{i}'
                                    )
                                    v_sliders_2.append(slider)

                        with gr.Row():
                            # Input Column 3
                            with gr.Column():
                                gr.Markdown("**📈 Extended Features**")
                                # Create V21-V24 sliders
                                v_sliders_3 = []
                                for i in range(21, 25):
                                    min_val, max_val = FEATURE_RANGES[f'V{i}']
                                    slider = gr.Slider(
                                        minimum=min_val,
                                        maximum=max_val,
                                        value=0,
                                        label=f'V{i}'
                                    )
                                    v_sliders_3.append(slider)

                            # Input Column 4
                            with gr.Column():
                                gr.Markdown("**🎯 Final Features**")
                                # Create V25-V28 sliders
                                v_sliders_4 = []
                                for i in range(25, 29):
                                    min_val, max_val = FEATURE_RANGES[f'V{i}']
                                    slider = gr.Slider(
                                        minimum=min_val,
                                        maximum=max_val,
                                        value=0,
                                        label=f'V{i}'
                                    )
                                    v_sliders_4.append(slider)

                                # Create Amount slider
                                amount_min, amount_max = FEATURE_RANGES['Amount']
                                amount_slider = gr.Slider(
                                    minimum=amount_min,
                                    maximum=amount_max,
                                    value=100,
                                    label='Amount'
                                )

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
                    result = gr.HTML(
                        label="Classification Result",
                        elem_classes="prediction-card",
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
            **Color Legend (based on normalized value within feature range):**
            - 🟢 Green: Low values (0-35%)
            - 🟡 Yellow: Medium values (35-65%)
            - 🔴 Red: High values (65-100%)
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

    # Collect all sliders in order
    all_sliders = [time_slider] + v_sliders_1 + v_sliders_2 + v_sliders_3 + v_sliders_4 + [amount_slider]

    # --- Backend Logic ---

    def predict_and_visualize(*args):
        """Runs prediction and creates all visualizations."""
        import re

        prediction = wrapped_predict(*args)

        # Extract numeric value from prediction
        pred_value = None
        try:
            numbers = re.findall(r'\d+\.\d+|\d+', str(prediction))
            if numbers:
                pred_value = float(numbers[0])
        except Exception:
            pred_value = None

        # --- HTML Formatting for Result Box ---

        if pred_value is not None and pred_value > 0.5:
            result_html = (
                f'<div class="result-box" style="background: linear-gradient('
                f'135deg, #ef4444 0%, #dc2626 100%);">'
                f'⚠️ <strong>FRAUD DETECTED</strong>'
                f'<br>Score: {pred_value:.4f}'
                f'<br><br>Recommendation: Flag for review'
                f'</div>'
            )
        elif pred_value is not None:
            result_html = (
                f'<div class="result-box" style="background: linear-gradient('
                f'135deg, #10b981 0%, #059669 100%);">'
                f'✅ <strong>LEGITIMATE TRANSACTION</strong>'
                f'<br>Score: {pred_value:.4f}'
                f'<br><br>Recommendation: Approve transaction'
                f'</div>'
            )
        else:
            # Fallback for text-only prediction
            if "fraud" in prediction.lower() or "suspicious" in prediction.lower():
                result_html = (
                    f'<div class="result-box" style="background: linear-gradient('
                    f'135deg, #ef4444 0%, #dc2626 100%);">'
                    f'⚠️ <strong>FRAUD DETECTED</strong>'
                    f'<br><br>{prediction}'
                    f'<br><br>Recommendation: Flag for review'
                    f'</div>'
                )
            else:
                result_html = (
                    f'<div class="result-box" style="background: linear-gradient('
                    f'135deg, #10b981 0%, #059669 100%);">'
                    f'✅ <strong>LEGITIMATE TRANSACTION</strong>'
                    f'<br><br>{prediction}'
                    f'<br><br>Recommendation: Approve transaction'
                    f'</div>'
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
        except Exception as e:
            print(f"Error creating probability plot: {e}")

        # Create input distribution plot
        input_plot = create_input_distribution_plot(list(args), feature_names)

        return result_html, prob_plot, input_plot

    def reset_sliders():
        """Reset all sliders to their default values."""
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
        except Exception as e:
            print(f"Error loading feature importance: {e}")
            return None

    # Connect prediction button
    predict_btn.click(
        fn=predict_and_visualize,
        inputs=all_sliders,
        outputs=[result, probability_plot, input_dist_plot]
    )

    # Connect reset button
    reset_btn.click(
        fn=reset_sliders,
        inputs=[],
        outputs=all_sliders
    )

    # Load feature importance when app starts
    fraud_app.load(load_feature_importance, outputs=feature_importance_plot)

if __name__ == "__main__":
    fraud_app.launch()
