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

        with gr.Row():
            with gr.Column(scale=1):
                csv_file = gr.File(label="Upload CSV file or drag and drop", file_types=['.csv'])
                upload_button = gr.Button("Predict")
                error_message = gr.Markdown("", visible=False, elem_id="error-msg")

            with gr.Column(scale=2):
                # Predictions panel (initially visible but empty)
                with gr.Group(visible=True, elem_id="prediction-panel") as prediction_panel:
                    gr.Markdown("### Predictions")
                    prediction_output = gr.DataFrame(headers=["sample_index", "prediction"], interactive=False)
                    download_button = gr.DownloadButton("Download Predictions CSV", visible=False)

        # Custom loading spinner (hidden by default)
        with gr.Group(visible=False) as spinner_box:
            gr.HTML("""
                <div class="loading-container">
                    <div class="custom-spinner"></div>
                    <p>Waiting for predictions...</p>
                </div>
            """)

        # CSV preview removed as per your previous request

        # Prediction function
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

        # Button click event chain:
        # 1. Show spinner
        # 2. Run prediction
        # 3. Hide spinner
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

        # Download prediction file handler
        def get_file_path(file_path):
            return file_path

        download_button.click(get_file_path, inputs=download_button, outputs=download_button)

    return demo
