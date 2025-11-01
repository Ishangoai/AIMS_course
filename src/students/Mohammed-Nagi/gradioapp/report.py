"""
Enhanced Gradio interface for Agentic Report Generation System.
"""

import contextlib
import io
import json
import os
import queue
import sys
import tempfile
import threading
import time
from datetime import datetime

import gradio as gr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reporter.agent import run_agent, visualize_graph


class QueueIO(io.TextIOBase):
    """Custom IO class to capture stdout to a queue."""
    def __init__(self, q):
        self.queue = q

    def write(self, s):
        self.queue.put(s)
        return len(s)


def parse_log_for_progress(log_text: str) -> dict:
    """Extract progress information from logs."""
    stages = {
        "PLANNER": 0,
        "RESEARCHER": 20,
        "WRITER": 40,
        "VALIDATOR": 60,
        "REVIEWER": 80,
        "FINALIZING": 90
    }

    current_stage = "Starting..."
    progress = 0

    for stage, pct in stages.items():
        if stage in log_text:
            current_stage = stage.title()
            progress = pct

    if "Report generation complete!" in log_text:
        current_stage = "Complete"
        progress = 100

    return {
        "stage": current_stage,
        "progress": progress
    }


def format_metadata_display(metadata: dict) -> str:
    """Format metadata as HTML for display."""
    if not metadata:
        return "<p>No metadata available</p>"

    validation_icon = "✅" if metadata.get('validation_passed') else "⚠️"

    html = f"""
    <div style='padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px; color: white; font-family: Arial;'>
        <h2 style='margin-top: 0; font-size: 24px;'>📊 Report Statistics</h2>
        <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin-top: 20px;'>
            <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;'>
                <div style='font-size: 14px; opacity: 0.9;'>Word Count</div>
                <div style='font-size: 32px; font-weight: bold;'>{metadata.get('final_word_count', 0)}</div>
            </div>
            <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;'>
                <div style='font-size: 14px; opacity: 0.9;'>Sections</div>
                <div style='font-size: 32px; font-weight: bold;'>{metadata.get('total_sections', 0)}</div>
            </div>
            <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;'>
                <div style='font-size: 14px; opacity: 0.9;'>Iterations</div>
                <div style='font-size: 32px; font-weight: bold;'>{metadata.get('iterations_needed', 0)}</div>
            </div>
            <div style='background: rgba(255,255,255,0.1); padding: 15px; border-radius: 8px;'>
                <div style='font-size: 14px; opacity: 0.9;'>Validation</div>
                <div style='font-size: 32px; font-weight: bold;'>{validation_icon}</div>
            </div>
        </div>
        <div style='margin-top: 20px; font-size: 14px; opacity: 0.8;'>
            <strong>Title:</strong> {metadata.get('title', 'N/A')}
        </div>
    </div>
    """
    return html


def generate_report_stream(topic: str, temperature: float, max_iterations: int):
    """
    Generates report with real-time streaming of progress.

    Args:
        topic: Report topic
        temperature: Controls Writer agent creativity (0.0-1.0)
        max_iterations: Maximum revision attempts before stopping

    Yields: (log_text, progress_html, metadata_html, report_text, download_data)
    """
    if not topic.strip():
        yield ("", "<p>⚠️ Please enter a topic</p>", "", "", None)
        return

    log_queue = queue.Queue()
    result_queue = queue.Queue()

    def agent_thread():
        try:
            with contextlib.redirect_stdout(QueueIO(log_queue)):
                result = run_agent(topic, temperature, max_iterations)
                result_queue.put(result)
        except Exception as e:
            result_queue.put(e)

    thread = threading.Thread(target=agent_thread, daemon=True)
    thread.start()

    full_log = ""
    start_time = time.time()

    progress_html = """
    <div style='text-align: center; padding: 20px;'>
        <div style='font-size: 18px; margin-bottom: 10px;'>🚀 Starting generation...</div>
        <div style='width: 100%; background: #e0e0e0; height: 30px; border-radius: 15px; overflow: hidden;'>
            <div style='width: 0%; background: linear-gradient(90deg, #667eea, #764ba2);
                        height: 100%; transition: width 0.3s;'></div>
        </div>
    </div>
    """

    yield (full_log, progress_html, "", "", None)

    while thread.is_alive() or not log_queue.empty():
        try:
            log_line = log_queue.get(timeout=0.1)
            full_log += log_line

            progress_info = parse_log_for_progress(full_log)
            progress_pct = progress_info['progress']
            current_stage = progress_info['stage']

            progress_html = f"""
            <div style='text-align: center; padding: 20px;'>
                <div style='font-size: 18px; margin-bottom: 10px; font-weight: bold;'>
                    {current_stage}
                </div>
                <div style='width: 100%; background: #e0e0e0; height: 30px; border-radius: 15px;
                            overflow: hidden; position: relative;'>
                    <div style='width: {progress_pct}%; background: linear-gradient(90deg, #667eea, #764ba2);
                                height: 100%; transition: width 0.5s;'></div>
                    <div style='position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                                color: white; font-weight: bold; font-size: 14px; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>
                        {progress_pct}%
                    </div>
                </div>
                <div style='font-size: 12px; margin-top: 10px; color: #666;'>
                    Elapsed: {int(time.time() - start_time)}s
                </div>
            </div>
            """  # noqa: E501

            yield (full_log, progress_html, "", "", None)

        except queue.Empty:
            time.sleep(0.1)

    thread.join()

    final_result = result_queue.get()

    if isinstance(final_result, Exception):
        error_html = f"""
        <div style='padding: 20px; background: #ffebee; border-left: 4px solid #f44336; border-radius: 5px;'>
            <h3 style='color: #c62828; margin-top: 0;'>❌ Error Occurred</h3>
            <p style='color: #d32f2f;'>{str(final_result)}</p>
        </div>
        """
        full_log += f"\n\n❌ Error: {final_result}"
        yield (full_log, error_html, "", "", None)
        return

    elapsed_time = int(time.time() - start_time)

    final_report = final_result.get("final_report", "")
    metadata = final_result.get("metadata", {})
    review = final_result.get("review_feedback", "")

    metadata_html = format_metadata_display(metadata)

    report_display = f"""
# {metadata.get('title', 'Generated Report')}

{final_report}

---

## 📝 Editorial Review

{review}

---

*Generated in {elapsed_time} seconds*
"""

    download_data = {
        "report": final_report,
        "metadata": metadata,
        "review": review,
        "generated_at": datetime.now().isoformat(),
        "topic": topic,
        "generation_time_seconds": elapsed_time
    }

    success_html = """
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                border-radius: 10px; color: white;'>
        <div style='font-size: 32px; margin-bottom: 10px;'>✅</div>
        <div style='font-size: 20px; font-weight: bold;'>Generation Complete!</div>
        <div style='font-size: 14px; margin-top: 5px; opacity: 0.9;'>
            Report ready for download
        </div>
    </div>
    """

    yield (full_log, success_html, metadata_html, report_display, download_data)


def save_report_files(download_data):
    """Save report to TXT and JSON format."""
    if not download_data:
        return None, None

    txt_content = f"""{download_data['metadata'].get('title', 'Report')}

Topic: {download_data['topic']}
Generated: {download_data['generated_at']}
Word Count: {download_data['metadata'].get('final_word_count', 0)}
Validation: {'Passed' if download_data['metadata'].get('validation_passed') else 'Failed'}

{'=' * 70}

{download_data['report']}

{'=' * 70}

Editorial Review

{download_data['review']}

{'=' * 70}

Generated by Agentic Report Generation System in {download_data['generation_time_seconds']}s
"""

    json_content = json.dumps(download_data, indent=2)

    return txt_content, json_content


def generate_workflow_diagram():
    """Generate and return the workflow diagram."""
    try:
        diagram = visualize_graph()
        if isinstance(diagram, str):
            # It's Mermaid text
            return diagram, "text"
        else:
            # It's PNG bytes
            return diagram, "image"
    except Exception as e:
        return f"Error generating diagram: {str(e)}", "error"


def handle_diagram_display():
    """Handle workflow diagram generation and display."""
    try:
        diagram = visualize_graph()
        if isinstance(diagram, bytes):
            # Save bytes to a temporary file and return the path
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                tmp.write(diagram)
                return tmp.name
        else:
            # If it's a string (Mermaid), return error message
            return None
    except Exception as e:
        print(f"Error generating diagram: {e}")
        return None


# Custom CSS
custom_css = """
.gradio-container {
    font-family: 'Inter', sans-serif;
}

.custom-button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: bold !important;
    transition: transform 0.2s !important;
}

.custom-button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 5px 15px rgba(0,0,0,0.2) !important;
}

.example-box {
    background: #f5f5f5;
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
    cursor: pointer;
    transition: background 0.2s;
}

.example-box:hover {
    background: #e0e0e0;
}
"""

# Build Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="AI Report Generator") as report_app:

    # Header
    gr.Markdown("""
    <div style='text-align: center; padding: 30px 0;'>
        <h1 style='font-size: 3em; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                   -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            🤖 Agentic Report Generator
        </h1>
        <p style='font-size: 1.2em; color: #666; margin-top: 10px;'>
            AI-powered research and writing system for technical reports
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            # Input Section
            gr.Markdown("### 📝 Report Configuration")

            topic_input = gr.Textbox(
                label="Topic",
                placeholder="E.g., 'MLOps Best Practices' or 'CI/CD for Machine Learning'",
                lines=2,
                elem_classes=["custom-input"]
            )

            with gr.Row():
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.7,
                    step=0.1,
                    label="🌡️ Writer Creativity (Temperature)",
                    info="Controls writing style: Higher = more creative, Lower = more factual"
                )

                iterations_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="🔄 Max Revisions",
                    info="Maximum revision attempts"
                )

            # Example topics
            gr.Markdown("**💡 Example Topics:**")
            examples = gr.Examples(
                examples=[
                    ["MLOps Best Practices"],
                    ["CI/CD Pipelines for Machine Learning"],
                    ["API Design Patterns for Microservices"],
                    ["Gradio for Building ML Interfaces"],
                    ["Model Monitoring and Observability"]
                ],
                inputs=[topic_input],
                label=None
            )

            generate_btn = gr.Button(
                "🚀 Generate Report",
                variant="primary",
                size="lg",
                elem_classes=["custom-button"]
            )

            view_diagram_btn = gr.Button(
                "View Workflow Diagram",
                variant="primary",
                size="lg",
                elem_classes=["custom-button"]
            )

            # Progress Section
            gr.Markdown("### 📊 Generation Progress")
            progress_display = gr.HTML()

            # Download Section
            gr.Markdown("### 💾 Download Options")
            with gr.Row():
                download_txt = gr.DownloadButton(
                    "📄 Download TXT",
                    visible=False
                )
                download_json = gr.DownloadButton(
                    "📦 Download JSON",
                    visible=False
                )

        with gr.Column(scale=3):
            # Output Section
            gr.Markdown("### 📄 Generated Report")
            report_output = gr.Markdown(
                value="*Report will appear here after generation...*",
                show_label=False
            )

            # Metadata Display
            metadata_display = gr.HTML()

            # Workflow Diagram Display
            with gr.Accordion("Workflow Diagram", open=False):
                diagram_output = gr.Image(
                    label="System Architecture",
                    show_label=False
                )

            # Logs (Collapsible)
            with gr.Accordion("🔍 View Agent Logs", open=False):
                logs_display = gr.Textbox(
                    lines=15,
                    show_copy_button=True,
                    interactive=False,
                    show_label=False
                )

    # Hidden state for download data
    download_state = gr.State()

    # Event handlers
    def handle_generate(topic, temp, max_iter):
        """Handle generation and enable downloads."""
        for log, progress, metadata, report, data in generate_report_stream(topic, temp, max_iter):
            # Update download buttons visibility
            show_downloads = data is not None
            yield (
                log,
                progress,
                metadata,
                report,
                data,
                gr.update(visible=show_downloads),
                gr.update(visible=show_downloads)
            )

    def prepare_txt_download(data):
        if not data:
            return None
        txt_content, _ = save_report_files(data)

        # Create a safe filename from the topic
        topic = data.get('topic', 'report').replace(' ', '_').lower()
        safe_topic = ''.join(c for c in topic if c.isalnum() or c == '_')[:50]

        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, f"report_{safe_topic}.txt")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(txt_content)

        return filepath

    def prepare_json_download(data):
        if not data:
            return None
        _, json_content = save_report_files(data)

        # Create a safe filename from the topic
        topic = data.get('topic', 'report').replace(' ', '_').lower()
        safe_topic = ''.join(c for c in topic if c.isalnum() or c == '_')[:50]

        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, f"report_{safe_topic}.json")

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(json_content)

        return filepath

    generate_btn.click(
        fn=handle_generate,
        inputs=[topic_input, temperature_slider, iterations_slider],
        outputs=[
            logs_display,
            progress_display,
            metadata_display,
            report_output,
            download_state,
            download_txt,
            download_json
        ]
    )

    download_txt.click(
        fn=prepare_txt_download,
        inputs=[download_state],
        outputs=[download_txt]
    ).then(
        lambda: None,  # Clear after download
        None,
        None
    )

    download_json.click(
        fn=prepare_json_download,
        inputs=[download_state],
        outputs=[download_json]
    ).then(
        lambda: None,
        None,
        None
    )

    view_diagram_btn.click(
        fn=handle_diagram_display,
        inputs=[],
        outputs=[diagram_output]
    )

    # Footer
    gr.Markdown("""
    <div style='text-align: center; padding: 20px; color: #999; font-size: 0.9em;'>
        <p>Powered by LangGraph • Google Gemini • Wikipedia API</p>
        <p>Built with ❤️ for ML-Ops Assignment 3</p>
    </div>
    """)
