import tempfile

import gradio as gr
import gradio.themes as themes
import pandas as pd

from .ui_utils import upload_file_and_get_predictions


def build_interface():
    theme = themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=["" "Segoe UI", "Helvetica Neue", "Arial", "sans-serif"]
    )

    with gr.Blocks(theme=theme) as demo:

        # Inject CSS to style prediction panel and tabs for dark theme and readability
        gr.Markdown("""
        <style>
        /* Prediction panel container - dark background */
        #prediction-panel {
            background-color: #121212 !important;
            color: #e0e0ff !important;
            padding: 10px;
            border-radius: 8px;
        }

        /* Tabs container inside prediction panel */
        #prediction-panel .gradio-tabs,
        #prediction-panel .gradio-tabs > div {
            background-color: #121212 !important;
            color: #e0e0ff !important;
        }

        /* Tab buttons */
        #prediction-panel .gradio-tabs .tab-item {
            background-color: #1e1e1e !important;
            color: #a0c0ff !important;
            border-color: #2a2a2a !important;
        }

        /* Active tab button */
        #prediction-panel .gradio-tabs .tab-item.selected {
            background-color: #3a3a6a !important;
            color: #ffffff !important;
            font-weight: bold !important;
        }

        /* Tab panel content background */
        #prediction-panel .tab-content {
            background-color: #121212 !important;
            color: #d0d0ff !important;
        }

        /* DataFrame table header and cells */
        #prediction-panel .gradio-dataframe th,
        #prediction-panel .gradio-dataframe td {
            background-color: #1e1e1e !important;
            color: #c0d0ff !important;
            font-weight: 600;
        }

        /* DataFrame general text */
        #prediction-panel .gradio-dataframe {
            color: #c0d0ff !important;
        }

        /* Download button */
        #prediction-panel button {
            background-color: #2a2a4a !important;
            color: #c0d0ff !important;
        }

        /* Error message */
        #error-msg {
            color: #ff6666 !important;
        }
        </style>
        """)

        gr.Markdown("## Fraud Detection Model Interface")

        # -----------------------------
        # CSV Upload + Prediction Components
        # -----------------------------
        with gr.Row():
            with gr.Column(scale=1):
                csv_file = gr.File(label="Upload CSV file or drag and drop", file_types=['.csv'])
                upload_button = gr.Button("Predict")
                error_message = gr.Markdown("", visible=False, elem_id="error-msg")

            with gr.Column(scale=2):
                with gr.Group(visible=True, elem_id="prediction-panel") as prediction_panel:
                    gr.Markdown("### Predictions")
                    prediction_output = gr.DataFrame(headers=["sample_index", "prediction"], interactive=False)
                    download_button = gr.DownloadButton("Download Predictions CSV", visible=False)

        # -------------------
        # Loading Spinner
        # -------------------
        with gr.Group(visible=False) as spinner_box:
            gr.HTML("""
                <div class="loading-container">
                    <div class="custom-spinner"></div>
                    <p>Waiting for predictions...</p>
                </div>
            """)

        # -------------------
        # Prediction Function
        # -------------------
        def do_predict(file):
            if file is None:
                return (
                    gr.update(visible=True, value="Please upload a CSV file."),
                    None,
                    gr.update(visible=False),
                    None,
                    gr.update(visible=False)
                )

            error, pred_text, pred_filename = upload_file_and_get_predictions(file)
            if error:
                return (
                    gr.update(visible=True, value=error),
                    None,
                    gr.update(visible=False),
                    None,
                    gr.update(visible=False)
                )

            from io import StringIO
            pred_df = pd.read_csv(StringIO(pred_text))

            return (
                gr.update(visible=False),
                pred_df,
                gr.update(visible=True),
                pred_filename,
                gr.update(visible=True)
            )

        upload_button.click(
            lambda: gr.update(visible=True),
            outputs=spinner_box
        ).then(
            do_predict,
            inputs=csv_file,
            outputs=[
                error_message,
                prediction_output,
                download_button,
                download_button,
                prediction_panel
            ]
        ).then(
            lambda: gr.update(visible=False),
            outputs=spinner_box
        )

        def get_file_path(file_path):
            return file_path

        download_button.click(get_file_path, inputs=download_button, outputs=download_button)

        # -----------------------------
        # Feature Sliders Input Section
        # -----------------------------
        feature_names = [
            'Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
        ]

        gr.Markdown("### Or enter a single sample manually")

        sliders = []
        with gr.Row():
            with gr.Column(scale=3):
                for fname in feature_names:
                    sliders.append(gr.Slider(minimum=0, maximum=200000, value=0, step=0.01, label=fname))

        predict_sliders_btn = gr.Button("Predict from sliders")
        error_message_sliders = gr.Markdown("", visible=False, elem_id="error-msg")

        def predict_from_sliders(*vals):

            # Create CSV from slider values
            df = pd.DataFrame([vals], columns=feature_names)  # pyright: ignore
            csv_string = df.to_csv(index=False)

            # Write to temp file
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".csv", delete=False) as tmp_file:
                tmp_file.write(csv_string)
                tmp_file.flush()

            # Create a mock file object with .name (just like Gradio upload)
            class UploadedFile:
                def __init__(self, name):
                    self.name = name

            fake_file = UploadedFile(tmp_file.name)

            # Call your existing backend prediction function
            error, pred_text, pred_filename = upload_file_and_get_predictions(fake_file)

            if error:
                return (
                    gr.update(visible=True, value=error),
                    None,
                    gr.update(visible=False),
                    None,
                    gr.update(visible=False)
                )

            from io import StringIO
            pred_df = pd.read_csv(StringIO(pred_text))

            return (
                gr.update(visible=False),
                pred_df,
                gr.update(visible=True),
                pred_filename,
                gr.update(visible=True)
            )

        predict_sliders_btn.click(
            predict_from_sliders,
            inputs=sliders,
            outputs=[
                error_message_sliders,
                prediction_output,
                download_button,
                download_button,
                prediction_panel
            ]
        )

    return demo


if __name__ == "__main__":
    fraud_detector = build_interface()
    fraud_detector.launch()
