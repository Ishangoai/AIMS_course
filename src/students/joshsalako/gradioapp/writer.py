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
    Runs the agent in a separate thread and streams its stdout logs
    to the Gradio interface in real-time.
    """
    if not topic.strip():
        yield "Please enter a topic to generate a report.", ""
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
            yield full_log, ""
        except queue.Empty:
            time.sleep(0.1)

    thread.join()

    final_result = result_queue.get()
    if isinstance(final_result, Exception):
        full_log += f"\n\nAn error occurred: {final_result}"
        final_report = ""
    else:
        final_report = final_result.get("final_report", "Failed to generate the report.")

    yield full_log, final_report


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

    gr.Markdown("<div style='height: 1.0em'></div>")

    with gr.Accordion("Agent's Live Workstream", open=False):
        thinking_box = gr.Textbox(
            label="Thinking...",
            lines=15,
            show_copy_button=True,
            interactive=False,
            autoscroll=True,
        )

    final_report_display = gr.Markdown(label="Final Report")

    generate_button.click(
        fn=generate_report,
        inputs=topic_input,
        outputs=[thinking_box, final_report_display]
    )

if __name__ == "__main__":
    essay.launch()
