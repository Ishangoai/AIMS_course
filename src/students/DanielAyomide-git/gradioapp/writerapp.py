import contextlib
import io
import os
import queue
import sys
import tempfile
import threading
import time

import gradio as gr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from essay_writer.agent import run_agent


# ========== STREAMING UTILITY ==========
class QueueIO(io.TextIOBase):
    def __init__(self, q):
        self.queue = q

    def write(self, s):
        self.queue.put(s)
        return len(s)


# ========== REPORT GENERATION ==========
def generate_report(topic: str):
    if not topic.strip():
        yield "⚠ Please enter a topic.", "", None
        return

    log_queue = queue.Queue()
    result_queue = queue.Queue()

    def agent_task():
        try:
            with contextlib.redirect_stdout(QueueIO(log_queue)):
                result = run_agent(topic)
                result_queue.put(result)
        except Exception as e:
            result_queue.put(e)

    thread = threading.Thread(target=agent_task)
    thread.start()

    logs = ""
    while thread.is_alive() or not log_queue.empty():
        try:
            line = log_queue.get(timeout=0.1)
            logs += line
            yield logs, "", None
        except queue.Empty:
            time.sleep(0.1)

    thread.join()
    result = result_queue.get()

    if isinstance(result, Exception):
        logs += f"\n❌ Error: {result}"
        report_text = ""
        txt_file = None
    else:
        report_text = result.get("final_report", "No report generated.")

        # Save TXT
        if report_text:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", prefix="Report_")
            tmp_file.write(report_text.encode("utf-8"))
            tmp_file.close()
            txt_file = tmp_file.name
        else:
            txt_file = None

    yield logs, report_text, txt_file


# ========== UI ==========
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Agentic Research Report Writer",
    css="""
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #1b1d26;
            color: #e0e0e0;
        }
        .main-card {
            max-width: 900px;
            margin: auto;
            padding: 2em;
            background-color: rgba(255,255,255,0.05);
            border-radius: 16px;
            box-shadow: 0 0 20px rgba(0,0,0,0.4);
        }
        h1.title-text {
            font-size: 3em;
            font-weight: 700;
            color: #00d4ff;
            letter-spacing: 1px;
            margin-bottom: 0.3em;
        }
        p.subtitle {
            font-size: 1.2em;
            color: #c9c9c9;
        }
        .gr-button {
            background-color: #00d4ff !important;
            color: #0f1116 !important;
            font-weight: 600;
        }
        .final-output-markdown {
            max-height: 600px;
            overflow-y: auto;
            background-color: rgba(255,255,255,0.05);
            border-radius: 10px;
            padding: 1em;
        }
        .gr-textbox textarea {
            font-size: 0.95em;
        }
    """
) as essay:

    with gr.Column(elem_classes=["main-card"]):
        # Header
        gr.Markdown(
            """
            <div style='text-align:center'>
                <h1 class='title-text'>Welcome to Deep Research</h1>
            </div>
            """
        )

        gr.Markdown("<div style='height: 2em'></div>")

        # Input + Button
        topic_input = gr.Textbox(
            label="Topic:",
            placeholder="e.g., The Impact of AI on Renewable Energy",
            scale=4,
            autofocus=True,
        )
        generate_button = gr.Button("🚀 Generate Report", variant="primary", scale=0)

        gr.Markdown("<div style='height: 1em'></div>")

        # Agent logs
        with gr.Accordion("Click to view logs", open=True):
            thinking_box = gr.Textbox(
                label="Thinking...",
                lines=15,
                show_copy_button=True,
                interactive=False,
                autoscroll=True,
            )

        # Final report
        final_report_display = gr.Markdown(
            label="Final Report",
            elem_classes=["final-output-markdown"]
        )

        # TXT download
        txt_download = gr.File(
            label="Download TXT",
            file_count="single",
            type="filepath"
        )

        # Button click
        generate_button.click(
            fn=generate_report,
            inputs=topic_input,
            outputs=[thinking_box, final_report_display, txt_download]
        )
