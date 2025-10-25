import contextlib
import io
import os
import queue
import sys
import tempfile
import threading
import time

import gradio as gr

# Ensure text_agent is discoverable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from text_agent.agent import run_agent

# ========== STREAMING UTILITIES ==========


class QueueIO(io.TextIOBase):
    def __init__(self, q):
        self.queue = q

    def write(self, s):
        self.queue.put(s)
        return len(s)


# --- Save content to a temporary file ---
def save_to_temp_file(content, topic, is_warhammer):
    """Saves the final report content to a temporary .txt file."""
    topic_slug = "".join(c if c.isalnum() else "_" for c in topic[:30]).strip("_")

    if is_warhammer:
        file_prefix = f"WH40K_List_{topic_slug}_"
    else:
        file_prefix = f"Report_{topic_slug}_"

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt", prefix=file_prefix) as tmp_file:
        tmp_file.write(content.replace("***\n\n", "").replace("##", "").replace("#", ""))
        file_path = tmp_file.name
    return file_path
# -----------------------------------------------------


def generate_report(topic: str):  # noqa: C901
    """
    Runs the agent in a background thread and streams stdout logs
    to Gradio in real time.
    """
    if not topic.strip():
        # Yield 4 outputs now: log, report text, file path (None), and word count ("")
        yield "Please enter a topic to generate a report.", "", None, ""
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
    # Update the yield to include a temporary placeholder for the file component (None) and word count ("")
    while thread.is_alive() or not log_queue.empty():
        try:
            log_line = log_queue.get(timeout=0.1)
            full_log += log_line
            yield full_log, "", None, "Processing..."  # Yield log, empty report, None file, status text
        except queue.Empty:
            time.sleep(0.1)

    thread.join()

    final_result = result_queue.get()

    final_file_path = None  # Default value for the file output
    word_count_output = "N/A"  # Default word count output

    if isinstance(final_result, Exception):
        full_log += f"\n\n⚠️ An error occurred: {final_result}"
        final_report = ""
    else:
        final_report = final_result.get("final_report", "Failed to generate the report.")
        is_warhammer = final_result.get("is_warhammer", False)

        # Preserve the original markdown content for the file save and word count
        raw_report_content = final_report

        # --- WORD COUNT CALCULATION ---
        # Simple word count: split by whitespace
        if raw_report_content:
            word_count = len(raw_report_content.split())
            if is_warhammer:
                # For Warhammer lists, report points instead of words
                # Try to extract the total points from the list format
                try:
                    import re
                    match = re.search(r'\+ TOTAL ARMY POINTS:\s*(\d+)pts', raw_report_content, re.IGNORECASE)
                    if match:
                        word_count_output = f"Points: {match.group(1)}pts"
                    else:
                        word_count_output = f"Words: {word_count} (List)"
                except ImportError:
                    word_count_output = f"Words: {word_count} (List)"
            else:
                word_count_output = f"Words: {word_count}"
        # ------------------------------

        if is_warhammer:
            # Format for Markdown display
            final_report = (
                f"# ⚔️ Warhammer 40K Tactical Dossier ⚔️\n\n"
                f"***\n\n"
                f"{final_report}"
            )
        else:
            # Format for Markdown display
            final_report = (
                f"# 📘 Analytical Research Report 📘\n\n"
                f"***\n\n"
                f"{final_report}"
            )

        # Create and save the file
        final_file_path = save_to_temp_file(raw_report_content, topic, is_warhammer)

    # Yield final outputs: log, report text, file path, and word count
    yield full_log, final_report, final_file_path, word_count_output


# ========== UI ==========

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="Agentic Report writer with list creation",
    css="""
        /* ... existing CSS ... */
        .word-count-box {
            background-color: rgba(255, 255, 255, 0.1);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 8px;
            padding: 10px;
            text-align: center;
            font-weight: bold;
            color: #d4af37; /* Gold/Warhammer color */
            font-size: 1.1em;
        }
    """
) as essay:

    with gr.Column(elem_classes=["main-card"]):
        gr.Markdown(
            """
            <div style='text-align:center; padding-top: 0.5em;'>
                <h1 style='font-size:2.6em; margin-bottom:0;'>Intelligent Report Writer & Warhammer 40k list Writer</h1>
                <p style='font-size:1.1em; color:#ccc; margin-top:0.3em;'>
                    Where <span style="color:#4a90e2;">React powered AI </span>
                     meets <span style="color:#d4af37;">grimdark strategy </span>.<br>
                    Write 1000 word essays or Warhammer 40K tactical lists — fully autonomous. <br>
                    By Elisha Komolafe and Muhammed Said 🇳🇬.
                </p>
            </div>
            """,
        )

        gr.Markdown("<hr style='margin: 1em 0; border-color: rgba(255,255,255,0.2);'>")

        # Input and button
        with gr.Row():
            topic_input = gr.Textbox(
                label="Enter your topic or army concept:",
                placeholder="Write a report on The Ethics of AI in Healthcare or Ultramarines 2000pt Army List",
                scale=4,
                autofocus=True,
            )
            generate_button = gr.Button("⚙️ Generate", variant="primary", scale=1)

        gr.Markdown("<div style='height: 1em'></div>")

        # Live log area (agent thoughts)
        with gr.Accordion("React Agents Live operation", open=False):
            thinking_box = gr.Textbox(
                label="Agent Console Output",
                lines=18,
                show_copy_button=True,
                interactive=False,
                autoscroll=True,
                value="Initializing agent sequence...",
            )

        gr.Markdown("<div style='height: 1em'></div>")

        # Final report display setup
        gr.Markdown(
            "<h2 style='text-align:center; color:#eee;'>📜 Final Report</h2>"
        )

        # New Row for word count, report, and download
        with gr.Row():
            # New Textbox for Word/Point Count
            word_count_display = gr.Textbox(
                label="Word/Point Count",
                interactive=False,
                scale=1,
                elem_classes=["word-count-box"]
            )

            # Add the gr.File component for download
            file_output = gr.File(
                label="Download Report",
                file_count="single",
                type="filepath",
                scale=1,
                visible=True
            )

        # The main report output area (placed after the controls row)
        with gr.Row():
            final_report_display = gr.Markdown(
                label="Generated Output",
                show_label=False,
                elem_classes=["final-output-markdown"],
            )

        # Click handler - UPDATED to include word_count_display
        generate_button.click(
            fn=generate_report,
            inputs=topic_input,
            outputs=[thinking_box, final_report_display, file_output, word_count_display],
        )

        gr.Markdown(
            "<div class='footer-note'>© 2025 Elisha Komolafe, "
            "Muhammed Said with Ishango.Ai — Built for AIMS Assignment 3</div>"
        )


# Launch
if __name__ == "__main__":
    essay.launch()
