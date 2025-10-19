import io
import os
from datetime import datetime

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from gradioapp.styles.fraud_app import custom_css
from gradioapp.utils.fraud_detection_utils import predict_credit_card_fraud


def wrapped_predict(*args):
    """Wrap model predictor"""
    return predict_credit_card_fraud(list(args))


ABOUT_FILE = os.path.join(
    os.path.dirname(__file__),
    "docs/about_fraud_app.md"
)


def load_about_markdown():
    """Load about markdown file"""
    if os.path.exists(ABOUT_FILE):
        with open(ABOUT_FILE, "r", encoding="utf-8") as f:
            return f.read()
    else:
        return (
            "⚠️ *About file not found. "
            "Please ensure `about.md` exists in the same directory.*"
        )


def wrapped_predict_ui(*args):
    """UI wrapper to format Gradio callouts"""
    result = wrapped_predict(*args)

    error_style = (
        "padding: 20px; border-radius: 12px; "
        "background: linear-gradient(135deg, #fee 0%, #fdd 100%); "
        "border-left: 5px solid #c33; margin: 10px 0; "
        "box-shadow: 0 4px 6px rgba(0,0,0,0.1);"
    )

    warning_style = (
        "padding: 20px; border-radius: 12px; "
        "background: linear-gradient(135deg, #fff3cd 0%, #ffe69c 100%); "
        "border-left: 5px solid #ff9800; margin: 10px 0; "
        "box-shadow: 0 4px 6px rgba(0,0,0,0.1);"
    )

    success_style = (
        "padding: 20px; border-radius: 12px; "
        "background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); "
        "border-left: 5px solid #28a745; margin: 10px 0; "
        "box-shadow: 0 4px 6px rgba(0,0,0,0.1);"
    )

    if not result.get("status"):
        error_msg = result.get('error', 'Unknown error')
        return f"""
        <div style="{error_style}">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 32px;">❌</span>
                <div>
                    <div style="font-size: 18px; font-weight: bold;
                                color: #c33; margin-bottom: 5px;">
                        Prediction Failed
                    </div>
                    <div style="color: #733; font-size: 14px;">
                        {error_msg}
                    </div>
                </div>
            </div>
        </div>
        """

    pred = result.get("prediction")
    msg = result.get("message", "")

    if pred == 1:
        return f"""
        <div style="{warning_style}">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 32px;">⚠️</span>
                <div>
                    <div style="font-size: 20px; font-weight: bold;
                                color: #e65100; margin-bottom: 5px;">
                        High Risk Transaction
                    </div>
                    <div style="color: #7d4f00; font-size: 15px;
                                line-height: 1.5;">
                        {msg}
                    </div>
                </div>
            </div>
        </div>
        """
    else:
        return f"""
        <div style="{success_style}">
            <div style="display: flex; align-items: center; gap: 12px;">
                <span style="font-size: 32px;">✅</span>
                <div>
                    <div style="font-size: 20px; font-weight: bold;
                                color: #155724; margin-bottom: 5px;">
                        Low Risk Transaction
                    </div>
                    <div style="color: #155724; font-size: 15px;
                                line-height: 1.5;">
                        {msg}
                    </div>
                </div>
            </div>
        </div>
        """


def process_batch_file(file):
    """Process uploaded CSV or JSON file for batch predictions"""
    try:
        if file is None:
            return None, "❌ Please upload a file", None, None

        file_path = file.name

        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            msg = "❌ Unsupported format. Please upload CSV or JSON."
            return None, msg, None, None

        required_cols = [
            'V14', 'V17', 'V10', 'V12', 'V11', 'V16',
            'V4', 'V9', 'V18', 'V7', 'V3', 'Amount'
        ]

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            msg = f"❌ Missing columns: {', '.join(missing_cols)}"
            return None, msg, None, None

        predictions = []
        for _, row in df.iterrows():
            features = [row[col] for col in required_cols]
            result = predict_credit_card_fraud(features)  # type: ignore
            if result.get("status"):
                predictions.append(result.get("prediction", 0))
            else:
                predictions.append(-1)

        df['Prediction'] = predictions
        df['Risk_Label'] = df['Prediction'].apply(
            lambda x: 'Fraud' if x == 1 else ('Normal' if x == 0 else 'Error')
        )

        total = len(df)
        fraud_count = (df['Prediction'] == 1).sum()
        normal_count = (df['Prediction'] == 0).sum()
        error_count = (df['Prediction'] == -1).sum()

        card_style = (
            "padding: 15px; background: white; border-radius: 8px; "
            "border-left: 4px solid {color};"
        )

        summary = f"""
        <div style="padding: 20px; border-radius: 12px;
                    background: #f8f9fa; margin: 10px 0;">
            <h3 style="margin-top: 0; color: #333;">
                📊 Batch Prediction Summary
            </h3>
            <div style="display: grid; grid-template-columns: repeat(2, 1fr);
                        gap: 15px; margin-top: 15px;">
                <div style="{card_style.format(color='#007bff')}">
                    <div style="font-size: 14px; color: #666;">
                        Total Transactions
                    </div>
                    <div style="font-size: 28px; font-weight: bold;
                                color: #007bff;">
                        {total}
                    </div>
                </div>
                <div style="{card_style.format(color='#28a745')}">
                    <div style="font-size: 14px; color: #666;">
                        Normal Transactions
                    </div>
                    <div style="font-size: 28px; font-weight: bold;
                                color: #28a745;">
                        {normal_count}
                    </div>
                </div>
                <div style="{card_style.format(color='#ff9800')}">
                    <div style="font-size: 14px; color: #666;">
                        Fraudulent Transactions
                    </div>
                    <div style="font-size: 28px; font-weight: bold;
                                color: #ff9800;">
                        {fraud_count}
                    </div>
                </div>
                <div style="{card_style.format(color='#dc3545')}">
                    <div style="font-size: 14px; color: #666;">
                        Fraud Rate
                    </div>
                    <div style="font-size: 28px; font-weight: bold;
                                color: #dc3545;">
                        {fraud_count / total * 100:.1f}%
                    </div>
                </div>
            </div>
            {f'<div style="margin-top: 10px; padding: 10px; '
             f'background: #fff3cd; border-radius: 6px; color: #856404;">'
             f'⚠️ {error_count} predictions failed</div>'
             if error_count > 0 else ''}
        </div>
        """

        plot = create_fraud_plot(normal_count, fraud_count)

        return df, summary, df, plot

    except Exception as e:
        return None, f"❌ Error processing file: {str(e)}", None, None


def create_fraud_plot(normal_count, fraud_count):
    """Create a bar chart showing fraud vs non-fraud distribution"""
    fig, ax = plt.subplots(figsize=(8, 5))

    categories = ['Normal', 'Fraud']
    counts = [normal_count, fraud_count]
    colors = ['#28a745', '#ff9800']

    bars = ax.bar(categories, counts, color=colors, alpha=0.8, edgecolor='black')

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontsize=14,
            fontweight='bold'
        )

    ax.set_ylabel('Number of Transactions', fontsize=12, fontweight='bold')
    ax.set_title(
        'Fraud Detection Results Distribution',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)

    return buf


def prepare_plot_download():
    """Prepare plot for download"""
    return f"fraud_distribution_chart_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}.png"


def process_and_save(file):
    """Process file and prepare results for download"""
    table, summary, df, plot = process_batch_file(file)
    results_file = save_results(df)
    plot_file = save_plot_as_file(plot) if plot else None
    return table, summary, results_file, plot_file


def get_sample_data():
    """Get sample batch data"""
    return {
        'V14': [0.35, -10.5, 1.2, -8.3, 0.45],
        'V17': [-0.01, -15.0, 0.5, -12.1, 0.2],
        'V10': [-0.16, -5.0, 0.8, -4.2, -0.3],
        'V12': [0.54, -7.5, 1.1, -6.8, 0.6],
        'V11': [0.85, 5.0, 0.9, 4.2, 0.7],
        'V16': [0.91, -8.0, 0.4, -7.1, 0.8],
        'V4': [-0.15, 5.0, -0.2, 4.5, -0.1],
        'V9': [-0.50, -3.0, -0.6, -2.8, -0.4],
        'V18': [0.58, -4.0, 0.3, -3.5, 0.5],
        'V7': [-0.17, -1.0, -0.1, -0.9, -0.2],
        'V3': [0.22, -2.0, 0.3, -1.8, 0.25],
        'Amount': [8.99, 500.0, 25.50, 450.0, 15.75]
    }


def create_sample_batch_file():
    """Create a sample batch file for download"""
    df = pd.DataFrame(get_sample_data())
    sample_path = "sample_batch.csv"
    df.to_csv(sample_path, index=False)
    return sample_path


def load_sample_to_upload():
    """Load sample file and return for file upload component"""
    sample_path = create_sample_batch_file()
    return sample_path


def save_results(df):
    """Save batch prediction results to CSV"""
    if df is None or df.empty:
        return None
    output_path = "batch_predictions_results.csv"
    df.to_csv(output_path, index=False)
    return output_path


def save_plot_as_file(plot_buffer):
    """Save plot buffer to file for download"""
    if plot_buffer is None:
        return None
    plot_path = "fraud_distribution_chart.png"
    with open(plot_path, 'wb') as f:
        f.write(plot_buffer.getvalue())
    return plot_path


SAMPLE_NORMAL = {
    "V14": 0.35237478, "V17": -0.010299228, "V10": -0.156840906,
    "V12": 0.538299087, "V11": 0.847755891, "V16": 0.906489408,
    "V4": -0.146817811, "V9": -0.49952308, "V18": 0.576837916,
    "V7": -0.166862547, "V3": 0.220932254, "Amount": 8.99
}

SAMPLE_FRAUD = {
    "V14": -10.5, "V17": -15.0, "V10": -5.0, "V12": -7.5,
    "V11": 5.0, "V16": -8.0, "V4": 5.0, "V9": -3.0,
    "V18": -4.0, "V7": -1.0, "V3": -2.0, "Amount": 500.0
}

with gr.Blocks(css=custom_css) as fraud_app:
    gr.Markdown("# Credit Card Fraud Detector")

    with gr.Tabs():
        with gr.Tab("🔍 Fraud Detection"):
            gr.Markdown(
                "### Enter transaction features to predict fraud risk"
            )

            result = gr.HTML(label="Prediction Result")

            with gr.Row():
                with gr.Column():
                    v14 = gr.Slider(
                        minimum=-20, maximum=20,
                        value=SAMPLE_NORMAL["V14"],
                        label="V14", elem_classes="orange-slider"
                    )
                    v17 = gr.Slider(
                        minimum=-20, maximum=20,
                        value=SAMPLE_NORMAL["V17"],
                        label="V17", elem_classes="orange-slider"
                    )
                    v10 = gr.Slider(
                        minimum=-20, maximum=20,
                        value=SAMPLE_NORMAL["V10"],
                        label="V10", elem_classes="orange-slider"
                    )
                    v12 = gr.Slider(
                        minimum=-20, maximum=20,
                        value=SAMPLE_NORMAL["V12"],
                        label="V12", elem_classes="orange-slider"
                    )
                    v11 = gr.Slider(
                        minimum=-20, maximum=20,
                        value=SAMPLE_NORMAL["V11"],
                        label="V11", elem_classes="orange-slider"
                    )
                    v16 = gr.Slider(
                        minimum=-20, maximum=20,
                        value=SAMPLE_NORMAL["V16"],
                        label="V16", elem_classes="orange-slider"
                    )
                with gr.Column():
                    v4 = gr.Slider(
                        minimum=-20, maximum=20,
                        value=SAMPLE_NORMAL["V4"],
                        label="V4", elem_classes="orange-slider"
                    )
                    v9 = gr.Slider(
                        minimum=-20, maximum=20,
                        value=SAMPLE_NORMAL["V9"],
                        label="V9", elem_classes="orange-slider"
                    )
                    v18 = gr.Slider(
                        minimum=-20, maximum=20,
                        value=SAMPLE_NORMAL["V18"],
                        label="V18", elem_classes="orange-slider"
                    )
                    v7 = gr.Slider(
                        minimum=-20, maximum=20,
                        value=SAMPLE_NORMAL["V7"],
                        label="V7", elem_classes="orange-slider"
                    )
                    v3 = gr.Slider(
                        minimum=-20, maximum=20,
                        value=SAMPLE_NORMAL["V3"],
                        label="V3", elem_classes="orange-slider"
                    )
                    amount = gr.Slider(
                        minimum=0, maximum=1000,
                        value=SAMPLE_NORMAL["Amount"],
                        label="Amount", elem_classes="orange-slider"
                    )

            with gr.Row():
                predict_btn = gr.Button("🔍 Predict", variant="primary")
                load_normal_btn = gr.Button("Load Sample Normal Case")
                load_fraud_btn = gr.Button("Load Sample Fraud Case")
                clear_btn = gr.Button("Clear")

            gr.Markdown(
                "### Group Members: Khadija Edarzi & Similoluwa Okunowo"
            )

        with gr.Tab("📊 Batch Predictions"):
            gr.Markdown(
                "### Upload CSV or JSON for batch fraud detection"
            )
            gr.Markdown(
                "**Required columns:** V14, V17, V10, V12, V11, V16, "
                "V4, V9, V18, V7, V3, Amount"
            )

            file_upload = gr.File(
                label="Upload File (CSV or JSON)",
                file_types=[".csv", ".json"]
            )

            with gr.Row():
                process_btn = gr.Button(
                    "🔍 Process Batch",
                    variant="primary",
                    scale=2
                )
                load_sample_btn = gr.Button(
                    "📥 Load Sample File",
                    scale=1
                )

            batch_summary = gr.HTML(label="Results Summary")

            with gr.Accordion("📋 Detailed Results", open=False):
                results_table = gr.Dataframe(
                    label="Transaction Details",
                    wrap=True,
                    interactive=False
                )

            with gr.Accordion("📈 Visualization", open=False):
                plot_output = gr.Image(
                    label="Fraud Distribution Chart",
                    type="filepath"
                )
                download_plot_btn = gr.Button("💾 Download Chart")
                plot_download = gr.File(
                    label="Chart File",
                    interactive=False,
                    visible=False
                )

            download_results = gr.File(
                label="Download Results",
                interactive=False,
                visible=True
            )

            gr.Markdown(
                "### Group Members: Khadija Edarzi & Similoluwa Okunowo"
            )

        with gr.Tab("📖 About"):
            gr.Markdown(load_about_markdown())

    predict_btn.click(
        fn=wrapped_predict_ui,
        inputs=[v14, v17, v10, v12, v11, v16, v4, v9, v18, v7, v3, amount],
        outputs=result
    )

    def _load_sample_normal():
        return tuple(SAMPLE_NORMAL.values())

    def _load_sample_fraud():
        return tuple(SAMPLE_FRAUD.values())

    def _clear_all():
        return (0.0,) * 12

    load_normal_btn.click(
        fn=_load_sample_normal,
        inputs=[],
        outputs=[v14, v17, v10, v12, v11, v16, v4, v9, v18, v7, v3, amount]
    )

    load_fraud_btn.click(
        fn=_load_sample_fraud,
        inputs=[],
        outputs=[v14, v17, v10, v12, v11, v16, v4, v9, v18, v7, v3, amount]
    )

    clear_btn.click(
        fn=_clear_all,
        inputs=[],
        outputs=[v14, v17, v10, v12, v11, v16, v4, v9, v18, v7, v3, amount]
    )

    process_btn.click(
        fn=process_and_save,
        inputs=[file_upload],
        outputs=[results_table, batch_summary, download_results, plot_output]
    )

    load_sample_btn.click(
        fn=load_sample_to_upload,
        inputs=[],
        outputs=[file_upload]
    )

if __name__ == "__main__":
    fraud_app.launch(server_name="0.0.0.0", share=False, debug=True)
