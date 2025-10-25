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
        yield "⚠ Please enter a topic.", "", None, "", ""
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
            yield logs, "", None, "", ""
        except queue.Empty:
            time.sleep(0.1)

    thread.join()
    result = result_queue.get()

    if isinstance(result, Exception):
        logs += f"\n❌ Error: {result}"
        report_text = ""
        txt_file = None
        word_count = ""
        editable_text = ""
    else:
        report_text = result.get("final_report", "No report generated.")
        report_text = report_text.replace("*", "\\*")
        editable_text = report_text
        word_count = f"📝 Word Count: {len(report_text.split())}" if report_text else ""

        # Save TXT
        if report_text:
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", prefix="Report_")
            tmp_file.write(report_text.encode("utf-8"))
            tmp_file.close()
            txt_file = tmp_file.name
        else:
            txt_file = None

    yield logs, report_text, txt_file, word_count, editable_text


# ========== SAVE EDITED REPORT ==========
def save_edited_report(edited_text: str):
    if not edited_text.strip():
        return None, "⚠ Nothing to save."
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", prefix="Edited_Report_")
    tmp_file.write(edited_text.encode("utf-8"))
    tmp_file.close()
    word_count = f"📝 Word Count: {len(edited_text.split())}"
    return tmp_file.name, word_count


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
    """
) as essay:

    with gr.Column(elem_classes=["main-card"]):
        # Header
        gr.Markdown(
            """
            <div style='text-align:center'>
                <h1 class='title-text'>Welcome to Deep Research</h1>
                <p class='subtitle'>Generate, review, and edit structured research reports with live word count.</p>
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
        generate_button = gr.Button("🚀 Generate Report", variant="primary")

        gr.Markdown("<div style='height: 1em'></div>")

        # Logs
        with gr.Accordion("Click to view logs", open=True):
            thinking_box = gr.Textbox(
                label="Thinking...",
                lines=15,
                show_copy_button=True,
                interactive=False,
                autoscroll=True,
            )

        # Final report (read-only preview - plain text)
        final_report_display = gr.Textbox(
            label="Final Report (Preview)",
            lines=25,
            interactive=False,
            show_copy_button=True,
            autoscroll=True,
            elem_classes=["final-output-markdown"]
        )

        # Editable text area
        edited_textbox = gr.Textbox(
            label="✏️ Edit Your Report (optional)",
            lines=20,
            interactive=True,
            visible=True,
            show_copy_button=True
        )

        # Word count
        word_count_display = gr.Markdown(label="Word Count", value="")

        # TXT download buttons
        txt_download = gr.File(label="Download Original TXT", file_count="single", type="filepath")
        edited_download = gr.File(label="Download Edited TXT", file_count="single", type="filepath")

        # Save button
        save_button = gr.Button("💾 Save Edited Report")

        # Bind functions
        generate_button.click(
            fn=generate_report,
            inputs=topic_input,
            outputs=[
                thinking_box,
                final_report_display,
                txt_download,
                word_count_display,
                edited_textbox,
            ],
        )

        save_button.click(
            fn=save_edited_report,
            inputs=edited_textbox,
            outputs=[edited_download, word_count_display],
        )


if __name__ == "__main__":
    essay.launch()
