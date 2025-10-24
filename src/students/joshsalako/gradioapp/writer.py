import contextlib
import io
import os
import sys

import gradio as gr

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from essay_writer.agent import run_agent


def generate_report(topic: str):
    """
    Runs the report generation agent and captures its verbose output.
    """
    if not topic.strip():
        return "Please enter a topic to generate a report.", ""

    # Use a string stream to capture the standard output from the agent
    log_stream = io.StringIO()

    thinking_output = "The agent is thinking... Please wait."
    yield thinking_output, ""

    try:
        with contextlib.redirect_stdout(log_stream):
            result = run_agent(topic)

        logs = log_stream.getvalue()
        final_report = result.get("final_report", "Failed to generate the report.")

        yield logs, final_report

    except Exception as e:
        error_message = f"An unexpected error occurred: {e}"
        yield error_message, ""


# --- Gradio UI Definition ---
with gr.Blocks(theme=gr.themes.Soft(), title="Agentic Report Writer") as demo:
    gr.Markdown("# Agentic Report Writer")
    gr.Markdown("Enter a topic below and the agent will write a 1000 word report.")

    with gr.Row():
        topic_input = gr.Textbox(
            label="Topic",
            placeholder="e.g., The Impact of AI on Renewable Energy",
            scale=4
        )
        generate_button = gr.Button("Generate Report", variant="primary", scale=1)

    with gr.Accordion("Agent's Thought Process", open=False):
        thinking_box = gr.Textbox(
            label="Logs",
            lines=15,
            show_copy_button=True,
            interactive=False,
        )

    final_report_display = gr.Markdown(label="Final Report")

    generate_button.click(
        fn=generate_report,
        inputs=topic_input,
        outputs=[thinking_box, final_report_display]
    )

if __name__ == "__main__":
    demo.launch()
