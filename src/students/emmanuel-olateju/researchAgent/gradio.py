"""
Gradio interface for Agentic Report Generation System.
Simplified structure with topic selection and dual modes.
"""

import contextlib
import io
import os
import queue
import sys
import threading
import time

import gradio as gr

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print(f"\n🔍 DEBUG: App directory: {current_dir}")
print(f"🔍 DEBUG: Python files found: {[f for f in os.listdir(current_dir) if f.endswith('.py')]}")

# Import the actual agent from same directory
MOCK_MODE = False
try:
    print("🔍 DEBUG: Attempting to import 'agent' module...")
    import agent as agent_module
    run_agent = agent_module.run_agent
    visualize_graph = getattr(agent_module, 'visualize_graph', lambda: None)
    print("✅ SUCCESS: Real agent imported!")
    print(f"   Module file: {agent_module.__file__}")
except Exception as e:
    print(f"❌ FAILED to import agent: {type(e).__name__}: {e}")
    print("\n⚠️  Falling back to MOCK MODE")
    MOCK_MODE = True

    # Mock fallback
    def mock_run_agent(topic, temperature, max_iterations):
        print(f"⚠️  MOCK: Generating fake report for '{topic}'")
        time.sleep(2)
        return {
            "final_report": f"# MOCK REPORT: {topic}\n\n**ERROR: Real agent.py not loaded**\n\nThis is a mock report.",
            "metadata": {"title": f"MOCK: {topic}", "topic": topic},
            "quality_metrics": {
                "word_count": 100,
                "word_count_accuracy": False,
                "structure_valid": False,
                "sections_count": 1,
                "citations_count": 0,
                "fact_check_rate": 0.0,
                "claims_verified": 0,
                "total_claims": 0,
                "overall_quality": "MOCK",
                "iterations_used": 0,
                "research_sources": 0
            },
            "review_feedback": "MOCK MODE ACTIVE"
        }

    run_agent = mock_run_agent
    def visualize_graph():
        return None

# Print status
if MOCK_MODE:
    print("\n" + "=" * 70)
    print("🔴 MOCK MODE ACTIVE - Fake reports will be generated!")
    print("=" * 70 + "\n")
else:
    print("\n" + "=" * 70)
    print("🟢 PRODUCTION MODE - Real agent ready!")
    print("=" * 70 + "\n")


# Try to import markdown and PDF libraries
try:
    from markdown import markdown  # type: ignore  # noqa: I001
    from weasyprint import HTML
    PDF_AVAILABLE = True
except ImportError:
    print("⚠️  PDF export requires: pip install markdown weasyprint")
    PDF_AVAILABLE = False


class QueueIO(io.TextIOBase):
    """Custom IO class to capture stdout to a queue."""
    def __init__(self, q):
        self.queue = q

    def write(self, s):
        self.queue.put(s)
        return len(s)


def get_actual_topic(dropdown_choice, custom_text):
    """Get the actual topic to use."""
    if dropdown_choice == "Custom":
        return custom_text.strip() if custom_text.strip() else "Custom Topic"
    return dropdown_choice


def simple_mode_generate(dropdown_topic, custom_topic, temperature, max_iterations):
    """Generate report in simple mode (fully automatic) with streaming logs."""
    topic = get_actual_topic(dropdown_topic, custom_topic)

    if not topic or topic == "Custom Topic":
        error_msg = "⚠️ **Error:** Please select a topic or enter a custom topic."
        return (error_msg, "", "", "")

    # Create queues for logging
    log_queue = queue.Queue()
    result_queue = queue.Queue()

    def agent_thread():
        """Run agent in separate thread with stdout capture."""
        try:
            with contextlib.redirect_stdout(QueueIO(log_queue)):
                result = run_agent(topic, temperature, max_iterations)
                result_queue.put(result)
        except Exception as e:
            result_queue.put(e)

    # Start agent thread
    thread = threading.Thread(target=agent_thread, daemon=True)
    thread.start()

    full_log = ""
    start_time = time.time()

    # Initial status
    status = f"🚀 **Generating report on:** {topic}\n\nPlease wait...\n\n"
    yield (status, "", "", "")

    # Stream logs while agent is running
    while thread.is_alive() or not log_queue.empty():
        try:
            log_line = log_queue.get(timeout=0.1)
            full_log += log_line

            # Update status with recent log
            elapsed = int(time.time() - start_time)
            status = (
                f"🚀 **Generating report on:** {topic}\n\n"
                f"⏱️ Elapsed: {elapsed}s\n\n"
                f"**Recent Activity:**\n```\n{full_log[-500:]}\n```"
            )
            yield (status, "", "", full_log)
        except queue.Empty:
            time.sleep(0.1)

    thread.join()

    # Get result
    try:
        result = result_queue.get(timeout=1)
    except queue.Empty:
        error_msg = "❌ **Error:** Timeout - no result received from agent"
        return (error_msg, "", "", full_log)

    # Check if result is an exception
    if isinstance(result, Exception):
        error_msg = f"❌ **Error occurred:**\n\n```\n{str(result)}\n```"
        return (error_msg, "", "", full_log)

    # Extract data
    report = result.get("final_report", "")
    metadata = result.get("metadata", {})
    metrics = result.get("quality_metrics", {})

    # Format success status
    elapsed_total = int(time.time() - start_time)
    status = (
        f"✅ **Generation Complete!** ({elapsed_total}s)\n\n"
        f"**Topic:** {metadata.get('title', topic)}\n"
        f"**Word Count:** {metrics.get('word_count', 0)} words\n"
        f"**Quality:** {metrics.get('overall_quality', 'Unknown')}\n"
        f"**Sections:** {metrics.get('sections_count', 0)}\n"
        f"**Citations:** {metrics.get('citations_count', 0)}\n"
        f"**Fact Check:** {int(metrics.get('fact_check_rate', 0) * 100)}%\n"
        f"**Iterations:** {metrics.get('iterations_used', 0)}\n"
        f"**Word Count Adjustments:** {metrics.get('word_count_adjustments', 0)}"
    )

    yield (status, report, report, full_log)


def hitl_generate_draft(dropdown_topic, custom_topic, temperature, max_iterations):
    """Generate initial draft for human review."""
    topic = get_actual_topic(dropdown_topic, custom_topic)

    if not topic or topic == "Custom Topic":
        return (
            "⚠️ **Error:** Please select a topic or enter a custom topic.",
            "",
            "",
            gr.update(interactive=False),
            gr.update(interactive=False)
        )

    try:
        # Generate draft (single iteration)
        result = run_agent(topic, temperature, 1)

        report = result.get("final_report", "")
        metadata = result.get("metadata", {})
        metrics = result.get("quality_metrics", {})

        status = (
            f"📋 **Draft Ready for Review**\n\n"
            f"**Topic:** {metadata.get('title', topic)}\n"
            f"**Word Count:** {metrics.get('word_count', 0)} words\n"
            f"**Quality:** {metrics.get('overall_quality', 'Unknown')}\n\n"
            f"👉 **Review the draft below.**\n"
            f"- If satisfied, click '✅ Approve & Finalize'\n"
            f"- To request changes, provide feedback and click '🔄 Revise Draft'"
        )

        return (
            status,
            report,
            report,
            gr.update(interactive=True),   # Enable feedback box
            gr.update(interactive=True)    # Enable approve button
        )

    except Exception as e:
        error_msg = f"❌ **Error:** {str(e)}"
        return (
            error_msg,
            "",
            "",
            gr.update(interactive=False),
            gr.update(interactive=False)
        )


def hitl_finalize(dropdown_topic, custom_topic, temperature, max_iterations, feedback, current_draft):
    """Finalize report with optional feedback."""
    topic = get_actual_topic(dropdown_topic, custom_topic)

    try:
        # If feedback provided, mention it
        if feedback and feedback.strip():
            status_msg = "🔄 **Revising with your feedback...**\n\n"
        else:
            status_msg = "✅ **Finalizing report...**\n\n"  # type:ignore  # noqa: F841

        # Generate final version
        # TODO: In real implementation, pass feedback to agent for revision
        result = run_agent(topic, temperature, max_iterations)

        report = result.get("final_report", "")
        metadata = result.get("metadata", {})
        metrics = result.get("quality_metrics", {})

        status = (
            f"✅ **Report Finalized!**\n\n"
            f"**Topic:** {metadata.get('title', topic)}\n"
            f"**Word Count:** {metrics.get('word_count', 0)} words\n"
            f"**Quality:** {metrics.get('overall_quality', 'Unknown')}\n"
            f"**Iterations:** {metrics.get('iterations_used', 0)}\n\n"
            f"📥 Your report is ready for download."
        )

        return (status, report, report)

    except Exception as e:
        error_msg = f"❌ **Error:** {str(e)}"
        return (error_msg, "", "")


def prepare_download_text(report_text, dropdown_topic, custom_topic):
    """Prepare report for download as text file."""
    if not report_text or not report_text.strip():
        filename = "no_report.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("No report has been generated yet. Please generate a report first.")
        return filename

    topic = get_actual_topic(dropdown_topic, custom_topic)
    safe_topic = topic.replace(" ", "_").replace("/", "_").lower()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"report_{safe_topic}_{timestamp}.txt"

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report_text)
        return filename
    except Exception as e:
        error_filename = "download_error.txt"
        with open(error_filename, "w", encoding="utf-8") as f:
            f.write(f"Error preparing download: {str(e)}")
        return error_filename


def prepare_download_pdf(report_text, dropdown_topic, custom_topic):
    """Prepare report for download as PDF file."""
    if not PDF_AVAILABLE:
        error_filename = "pdf_not_available.txt"
        with open(error_filename, "w", encoding="utf-8") as f:
            f.write("PDF export requires: pip install markdown weasyprint\n\n")
            f.write("Please install these libraries to enable PDF export.")
        return error_filename

    if not report_text or not report_text.strip():
        filename = "no_report.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("No report has been generated yet. Please generate a report first.")
        return filename

    topic = get_actual_topic(dropdown_topic, custom_topic)
    safe_topic = topic.replace(" ", "_").replace("/", "_").lower()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"report_{safe_topic}_{timestamp}.pdf"

    try:
        # Convert markdown to HTML
        html_content = markdown(report_text, extensions=['extra', 'codehilite']) # type: ignore  # noqa: E261

        # Add styling
        styled_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{
                    font-family: 'Georgia', serif;
                    line-height: 1.6;
                    max-width: 800px;
                    margin: 40px auto;
                    padding: 20px;
                    color: #333;
                }}
                h1 {{
                    color: #2c3e50;
                    border-bottom: 3px solid #3498db;
                    padding-bottom: 10px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                }}
                h3 {{
                    color: #7f8c8d;
                }}
                code {{
                    background-color: #f4f4f4;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                }}
                pre {{
                    background-color: #f4f4f4;
                    padding: 15px;
                    border-radius: 5px;
                    overflow-x: auto;
                }}
            </style>
        </head>
        <body>
            {html_content}
        </body>
        </html>
        """

        # Convert HTML to PDF
        HTML(string=styled_html).write_pdf(filename)  # type: ignore
        return filename

    except Exception as e:
        error_filename = "pdf_error.txt"
        with open(error_filename, "w", encoding="utf-8") as f:
            f.write(f"Error generating PDF: {str(e)}\n\n")
            f.write("Try downloading as text file instead.")
        return error_filename


# Build Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Agentic Report Generator") as app:

    # Header
    gr.Markdown(
        """
        # 🤖 Agentic Report Generator

        **Multi-Agent AI System with Analytical Rewriting**

        Generate comprehensive technical reports using a 7-agent system with advanced features:
        - 📊 Low-temperature content analysis (Phase 1)
        - ✏️ Strategic rewriting at moderate temperature (Phase 2)
        - 🔍 Automated fact-checking and validation
        - 📚 Wikipedia-powered research

        ---
        """
    )

    with gr.Tabs() as tabs:
        # ============ SIMPLE MODE TAB ============
        with gr.Tab("🚀 Simple Mode"):
            gr.Markdown("### Fully automatic report generation with real-time progress")

            with gr.Row():
                with gr.Column(scale=1):
                    topic_simple = gr.Radio(
                        choices=["CI/CD", "MLOps", "APIs", "Gradio", "Kubernetes", "Docker", "Custom"],
                        value="CI/CD",
                        label="📚 Select Topic"
                    )

                    custom_simple = gr.Textbox(
                        label="Custom Topic (if selected above)",
                        placeholder="Enter your custom topic here...",
                        lines=2,
                        visible=False
                    )

                    temp_simple = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="🌡️ Temperature",
                        info="Lower = factual, Higher = creative"
                    )

                    iter_simple = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="🔄 Max Iterations",
                        info="Maximum refinement attempts"
                    )

                    generate_simple_btn = gr.Button(
                        "🚀 Generate Report",
                        variant="primary",
                        size="lg"
                    )

                with gr.Column(scale=2):
                    status_simple = gr.Markdown("**Status:** Ready to generate")

                    with gr.Tabs():
                        with gr.Tab("📄 Report"):
                            output_simple_html = gr.Markdown(
                                value="*Your report will appear here...*",
                                height=500
                            )

                        with gr.Tab("📝 Plain Text"):
                            output_simple_text = gr.Textbox(
                                value="",
                                lines=25,
                                show_copy_button=True,
                                label="Raw Text"
                            )

                        with gr.Tab("📋 Agent Logs"):
                            logs_simple = gr.Textbox(
                                value="",
                                lines=25,
                                show_copy_button=True,
                                label="System Logs",
                                interactive=False
                            )

                    with gr.Row():
                        download_simple_txt_btn = gr.Button("💾 Download TXT", size="sm")
                        download_simple_pdf_btn = gr.Button("📄 Download PDF", size="sm")

                    download_simple_file = gr.File(label="Download", interactive=False)

        # ============ HITL MODE TAB ============
        with gr.Tab("👤 Human-in-the-Loop Mode"):
            gr.Markdown("### Review draft and provide feedback before finalization")

            with gr.Row():
                with gr.Column(scale=1):
                    topic_hitl = gr.Radio(
                        choices=["CI/CD", "MLOps", "APIs", "Gradio", "Kubernetes", "Docker", "Custom"],
                        value="MLOps",
                        label="📚 Select Topic"
                    )

                    custom_hitl = gr.Textbox(
                        label="Custom Topic (if selected above)",
                        placeholder="Enter your custom topic here...",
                        lines=2,
                        visible=False
                    )

                    temp_hitl = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="🌡️ Temperature"
                    )

                    iter_hitl = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="🔄 Max Iterations"
                    )

                    draft_btn = gr.Button(
                        "📝 Generate Draft",
                        variant="secondary",
                        size="lg"
                    )

                with gr.Column(scale=2):
                    status_hitl = gr.Markdown("**Status:** Ready to generate draft")

                    gr.Markdown("#### 📋 Draft Report")
                    with gr.Tabs():
                        with gr.Tab("📄 HTML"):
                            draft_html = gr.Markdown(
                                value="*Draft will appear here...*",
                                height=350
                            )
                        with gr.Tab("📝 Text"):
                            draft_text = gr.Textbox(
                                value="",
                                lines=15,
                                show_copy_button=True
                            )

                    feedback_box = gr.Textbox(
                        label="💬 Your Feedback (Optional)",
                        placeholder="Leave empty to approve, or provide feedback for revision...",
                        lines=3,
                        interactive=False
                    )

                    with gr.Row():
                        approve_btn = gr.Button(
                            "✅ Approve & Finalize",
                            variant="primary",
                            interactive=False
                        )
                        revise_btn = gr.Button(
                            "🔄 Revise Draft",
                            variant="secondary",
                            interactive=False
                        )

                    gr.Markdown("#### 📄 Final Report")
                    with gr.Tabs():
                        with gr.Tab("📄 HTML"):
                            final_html = gr.Markdown(
                                value="*Final report will appear here...*",
                                height=350
                            )
                        with gr.Tab("📝 Text"):
                            final_text = gr.Textbox(
                                value="",
                                lines=15,
                                show_copy_button=True
                            )

                    with gr.Row():
                        download_hitl_txt_btn = gr.Button("💾 Download TXT", size="sm")
                        download_hitl_pdf_btn = gr.Button("📄 Download PDF", size="sm")

                    download_hitl_file = gr.File(label="Download", interactive=False)

    # Footer
    mode_status = "🟢 PRODUCTION" if not MOCK_MODE else "🔴 MOCK"
    gr.Markdown(
        f"""
        ---
        **Status:** {mode_status} | **Powered by:** LangGraph • Google Gemini • Wikipedia API
        **Architecture:** 7-Agent System (Planner → Research → Writer → Fact Checker → Validator → Reviewer)
        **Key Feature:** Two-Phase Analytical Rewriting (Low-temp Analysis + Strategic Rewrite)
        """
    )

    # ============ EVENT HANDLERS ============

    # Toggle custom topic visibility
    def toggle_custom(choice):
        return gr.update(visible=(choice == "Custom"))

    topic_simple.change(toggle_custom, topic_simple, custom_simple)
    topic_hitl.change(toggle_custom, topic_hitl, custom_hitl)

    # Simple Mode - Generation (with streaming)
    generate_simple_btn.click(
        fn=simple_mode_generate,
        inputs=[topic_simple, custom_simple, temp_simple, iter_simple],
        outputs=[status_simple, output_simple_html, output_simple_text, logs_simple]
    )

    # Simple Mode - Downloads
    download_simple_txt_btn.click(
        fn=prepare_download_text,
        inputs=[output_simple_text, topic_simple, custom_simple],
        outputs=[download_simple_file]
    )

    download_simple_pdf_btn.click(
        fn=prepare_download_pdf,
        inputs=[output_simple_text, topic_simple, custom_simple],
        outputs=[download_simple_file]
    )

    # HITL Mode - Draft
    draft_btn.click(
        fn=hitl_generate_draft,
        inputs=[topic_hitl, custom_hitl, temp_hitl, iter_hitl],
        outputs=[status_hitl, draft_html, draft_text, feedback_box, approve_btn]
    )

    # HITL Mode - Approve (no feedback)
    approve_btn.click(
        fn=hitl_finalize,
        inputs=[topic_hitl, custom_hitl, temp_hitl, iter_hitl, gr.Textbox(value=""), draft_text],
        outputs=[status_hitl, final_html, final_text]
    )

    # HITL Mode - Revise (with feedback)
    revise_btn.click(
        fn=hitl_finalize,
        inputs=[topic_hitl, custom_hitl, temp_hitl, iter_hitl, feedback_box, draft_text],
        outputs=[status_hitl, final_html, final_text]
    )

    # HITL Mode - Downloads
    download_hitl_txt_btn.click(
        fn=prepare_download_text,
        inputs=[final_text, topic_hitl, custom_hitl],
        outputs=[download_hitl_file]
    )

    download_hitl_pdf_btn.click(
        fn=prepare_download_pdf,
        inputs=[final_text, topic_hitl, custom_hitl],
        outputs=[download_hitl_file]
    )


if __name__ == "__main__":
    app.launch(share=False, server_name="0.0.0.0", server_port=7860)
