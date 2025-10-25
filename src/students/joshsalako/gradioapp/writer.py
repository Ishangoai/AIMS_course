import contextlib
import io
import os
import queue
import sys
import threading
import time

import gradio as gr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from essay_writer.agent import run_agent


class QueueIO(io.TextIOBase):
    def __init__(self, q):
        self.queue = q

    def write(self, s):
        self.queue.put(s)
        return len(s)


def generate_report(topic: str):
    """
    Runs the agent in a separate thread, streams its stdout logs,
    and provides the final report for download.
    """
    if not topic.strip():
        yield "Please enter a topic to generate a report.", "", gr.File(visible=False)
        return

    log_queue = queue.Queue()
    result_queue = queue.Queue()

    def agent_thread_target():
        try:
            with contextlib.redirect_stdout(QueueIO(log_queue)):
                result = run_agent(topic)
                result_queue.put(result)
        except Exception as e:
            result_queue.put(e)

    thread = threading.Thread(target=agent_thread_target)
    thread.start()

    full_log = ""
    while thread.is_alive() or not log_queue.empty():
        try:
            log_line = log_queue.get(timeout=0.1)
            full_log += log_line
            yield full_log, "", gr.File(visible=False)
        except queue.Empty:
            time.sleep(0.1)

    thread.join()

    final_result = result_queue.get()
    if isinstance(final_result, Exception):
        full_log += f"\n\nAn error occurred: {final_result}"
        final_report = "Report generation failed due to an error."
        yield full_log, final_report, gr.File(visible=False)
        return

    final_report = final_result.get("final_report", "")
    if not final_report:
        full_log += "\n\nFailed to generate the report."
        final_report = "Report generation failed to produce a result."
        yield full_log, final_report, gr.File(visible=False)
        return

    # Create and save the report file
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)

    sanitized_topic = "".join(c for c in topic.strip() if c.isalnum() or c in (' ', '_')).rstrip()
    file_name = f"{sanitized_topic.replace(' ', '_').lower()}_report.txt"
    file_path = os.path.join(reports_dir, file_name)

    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(final_report)
        yield full_log, final_report, gr.File(value=file_path, visible=True)
    except IOError as e:
        full_log += f"\n\nError saving report file: {e}"
        final_report += "\n\n(Could not save report to a file)"
        yield full_log, final_report, gr.File(visible=False)


with gr.Blocks(theme=gr.themes.Soft(), title="Agentic Report Writer") as essay:
    gr.Markdown(
        "<h1 style='text-align: center; margin-bottom: 0.5em; font-size: 3em;'>Agentic Report Writer</h1>"
    )
    gr.Markdown(
        "<p style='text-align: center; margin-bottom: 2em;'>"
        "This tool uses a team of AI agents to research and write a comprehensive, "
        "1000-word report on any topic you provide."
        "</p>"
    )

    gr.Markdown("<div style='height: 4.5em'></div>")

    topic_input = gr.Textbox(
        label="Topic:",
        placeholder="e.g., The Impact of AI on Renewable Energy",
        scale=4,
        autofocus=True,
    )
    gr.Markdown("<div style='height: 0.7em'></div>")
    generate_button = gr.Button("Generate Report", variant="primary", scale=0)

    with gr.Accordion("Click to view agent's live workstream...", open=False):
        thinking_box = gr.Textbox(
            label="Working...",
            lines=10,
            show_copy_button=True,
            interactive=False,
            autoscroll=True,
        )

    final_report_display = gr.Markdown(label="Final Report")
    download_file = gr.File(label="Download Report", visible=False)

    generate_button.click(
        fn=generate_report,
        inputs=topic_input,
        outputs=[thinking_box, final_report_display, download_file]
    )

if __name__ == "__main__":
    essay.launch()
