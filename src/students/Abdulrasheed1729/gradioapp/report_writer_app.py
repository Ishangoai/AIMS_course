"""Gradio interface for the agentic report writing system."""

import gradio as gr
from report_writer.word_counter import count_words
from report_writer.workflow import generate_report

# Load environment variables
# load_dotenv()
#
# # Check for API key
# if not os.getenv("GOOGLE_API_KEY"):
#     print("Warning: GOOGLE_API_KEY not found in environment variables")

# Global state for human-in-the-loop
current_draft = None
current_state = None


def generate_initial_draft(topic, temperature, max_iterations):
    """Generate initial draft for human review."""
    global current_draft, current_state

    try:
        # Generate report without human feedback first
        result = generate_report(
            topic=topic,
            temperature=temperature,
            max_iterations=1,  # Just one iteration for initial draft
        )

        current_draft = result.get("report", "")
        current_state = result

        # Format progress messages
        messages = "\n".join(result.get("messages", []))

        word_info = count_words(current_draft) if current_draft else {"word_count": 0, "status": "N/A"}

        return (
            current_draft or "No draft generated",
            f"**Draft Status:** {result.get('status', 'UNKNOWN')}\n"
            f"**Word Count:** {word_info['word_count']} ({word_info['status']})\n\n"
            f"**Agent Activity Log:**\n{messages}",
            gr.update(interactive=True),  # Enable feedback box
            gr.update(interactive=True),  # Enable finalize button
        )

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        return (
            f"Error generating draft: {str(e)}",
            f"**Status:** Error occurred\n\n```\n{error_details}\n```",
            gr.update(interactive=False),
            gr.update(interactive=False),
        )


def finalize_report(topic, temperature, max_iterations, human_feedback):
    """Finalize report with optional human feedback."""
    global current_draft

    try:
        # Generate final report with human feedback
        result = generate_report(
            topic=topic,
            temperature=temperature,
            max_iterations=max_iterations,
            human_feedback=human_feedback if human_feedback and human_feedback.strip() else None,
        )

        final_report = result.get("report", "")

        # Format progress messages
        messages = "\n".join(result.get("messages", []))

        word_info = count_words(final_report) if final_report else {"word_count": 0, "status": "N/A"}

        status_text = (
            f"**Final Status:** {result.get('status', 'UNKNOWN')}\n"
            f"**Word Count:** {word_info['word_count']} ({word_info['status']})\n"
            f"**Iterations:** {result.get('iterations', 0)}\n\n"
            f"**Agent Activity Log:**\n{messages}"
        )

        return (final_report or "No report generated", status_text)

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        return (
            f"Error finalizing report: {str(e)}",
            f"**Status:** Error occurred\n\n```\n{error_details}\n```",
        )


def prepare_download(report, topic):
    """Prepare report for download."""
    print(f"prepare_download called with topic: {topic}")
    print(f"Report length: {len(report) if report else 0}")

    if not report or not report.strip():
        print("No report content to download")
        # Create empty file with message
        filename = "no_report.txt"
        with open(filename, "w", encoding="utf-8") as f:
            f.write("No report has been generated yet. Please generate a report first.")
        return filename

    # Create filename
    safe_topic = topic.replace(" ", "_").replace("/", "_").lower()
    filename = f"report_{safe_topic}.txt"

    # Write to file for download
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"✓ File prepared for download: {filename}")
        print(f"✓ File size: {len(report)} characters")
        return filename
    except Exception as e:
        print(f"✗ Error preparing download: {e}")
        import traceback

        traceback.print_exc()
        # Return error file
        error_filename = "download_error.txt"
        with open(error_filename, "w", encoding="utf-8") as f:
            f.write(f"Error preparing download: {str(e)}")
        return error_filename


def simple_generate(topic, temperature, max_iterations):
    """Simple generation without human-in-the-loop."""
    try:
        result = generate_report(topic=topic, temperature=temperature, max_iterations=max_iterations)

        final_report = result.get("report", "")
        messages = "\n".join(result.get("messages", []))
        word_info = count_words(final_report) if final_report else {"word_count": 0, "status": "N/A"}

        status_text = (
            f"**Status:** {result.get('status', 'UNKNOWN')}\n"
            f"**Word Count:** {word_info['word_count']} ({word_info['status']})\n"
            f"**Iterations:** {result.get('iterations', 0)}\n\n"
            f"**Agent Activity Log:**\n{messages}"
        )

        return (final_report or "No report generated", status_text)

    except Exception as e:
        import traceback

        error_details = traceback.format_exc()
        return (
            f"Error: {str(e)}",
            f"**Status:** Error occurred\n\n```\n{error_details}\n```",
        )


# Create Gradio interface
with gr.Blocks(title="Agentic Report Writer", theme=gr.themes.Monochrome()) as report_writer:
    gr.Markdown(
        """
        # 📝 Agentic Report Writing System

        Generate comprehensive technical reports using a multi-agent AI system.
        Choose from CI/CD, MLOps, APIs, Gradio, or enter a custom topic.

        **Note:** Generation takes 2-4 minutes.
        """
    )

    with gr.Tabs() as tabs:
        # Simple Mode Tab
        with gr.Tab("Simple Mode"):
            gr.Markdown("### Quick report generation without human review")

            with gr.Row():
                with gr.Column():
                    topic_simple = gr.Dropdown(
                        choices=["CI/CD", "MLOps", "APIs", "Gradio", "Custom"],
                        value="MLOps",
                        label="Topic",
                        info="Select a topic or choose Custom",
                    )
                    custom_topic_simple = gr.Textbox(
                        label="Custom Topic",
                        placeholder="Enter custom topic if selected above",
                        visible=False,
                    )
                    temperature_simple = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                        info="Higher = more creative, Lower = more focused",
                    )
                    max_iter_simple = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Max Iterations",
                        info="Maximum revision cycles",
                    )
                    generate_btn_simple = gr.Button("Generate Report", variant="primary", size="lg")

                with gr.Column():
                    status_simple = gr.Markdown("**Status:** Ready to generate")
                    with gr.Tabs():
                        with gr.Tab("📄 Rendered (HTML)"):
                            report_simple_rendered = gr.Markdown(
                                value="*Generated report will appear here...*",
                                height=600,
                            )
                        with gr.Tab("📝 Markdown Source"):
                            report_simple_raw = gr.Textbox(
                                value="",
                                lines=25,
                                max_lines=30,
                                show_copy_button=True,
                                label="Raw Markdown (for copying/editing)",
                            )
                    download_btn_simple = gr.Button("📥 Download Markdown", size="sm", variant="secondary")
                    download_file_simple = gr.File(
                        label="📥 Click the file below to download",
                        interactive=False,
                        type="filepath",
                    )

        # Human-in-the-Loop Mode Tab
        with gr.Tab("Human-in-the-Loop Mode"):
            gr.Markdown("### Review and provide feedback before final generation")

            with gr.Row():
                with gr.Column():
                    topic_hitl = gr.Dropdown(
                        choices=["CI/CD", "MLOps", "APIs", "Gradio", "Custom"],
                        value="MLOps",
                        label="Topic",
                    )
                    custom_topic_hitl = gr.Textbox(
                        label="Custom Topic",
                        placeholder="Enter custom topic if selected above",
                        visible=False,
                    )
                    temperature_hitl = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )
                    max_iter_hitl = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Max Iterations",
                    )
                    draft_btn = gr.Button("Generate Draft", variant="secondary", size="lg")

                with gr.Column():
                    status_hitl = gr.Markdown("**Status:** Ready to generate draft")
                    with gr.Tabs():
                        with gr.Tab("📄 Rendered (HTML)"):
                            draft_output_rendered = gr.Markdown(
                                value="*Draft report will appear here...*",
                                height=350,
                            )
                        with gr.Tab("📝 Markdown Source"):
                            draft_output_raw = gr.Textbox(
                                value="",
                                lines=15,
                                show_copy_button=True,
                            )
                    feedback_input = gr.Textbox(
                        label="Your Feedback (Optional)",
                        placeholder="Provide feedback to improve the report...",
                        lines=3,
                        interactive=False,
                    )
                    finalize_btn = gr.Button("Finalize Report", variant="primary", interactive=False)
                    with gr.Tabs():
                        with gr.Tab("📄 Rendered (HTML)"):
                            final_output_rendered = gr.Markdown(
                                value="*Final report will appear here...*",
                                height=350,
                            )
                        with gr.Tab("📝 Markdown Source"):
                            final_output_raw = gr.Textbox(
                                value="",
                                lines=15,
                                show_copy_button=True,
                            )
                    download_btn_hitl = gr.Button("📥 Download Markdown", size="sm", variant="secondary")
                    download_file_hitl = gr.File(
                        label="📥 Click the file below to download",
                        interactive=False,
                        type="filepath",
                    )

        # About Tab
        # with gr.Tab("About"):
        #     gr.Markdown(
        #         """
        #         ## System Architecture
        #
        #         This agentic system uses **LangGraph** to orchestrate multiple specialized agents:
        #
        #         1. **Research Agent** - Gathers information using LLM knowledge base
        #         2. **Writer Agent** - Creates structured report drafts
        #         3. **Fact Checker Agent** - Verifies accuracy and identifies issues
        #         4. **Editor Agent** - Refines content and ensures word count compliance
        #         5. **Quality Control Agent** - Final review and approval
        #
        #         ### Features
        #         - ✅ Automated word counting (1000 ± 50 words)
        #         - ✅ Structure validation (intro, main, conclusion)
        #         - ✅ Iterative refinement with feedback loops
        #         - ✅ Human-in-the-loop review option
        #         - ✅ Powered by Google Gemini Pro
        #
        #         ### Workflow
        #         ```
        #         Research → Writer → Fact Checker → Editor → Quality Control
        #                         ↑                                    ↓
        #                         └────────── (if revision needed) ────┘
        #         ```
        #
        #         ### Requirements
        #         - Google API key (required)
        #
        #         Set this in a `.env` file or environment variables.
        #
        #         ### Generation Time
        #         - Simple mode: 2-4 minutes
        #         - Draft generation: 1-2 minutes
        #         - Finalization: 2-4 minutes
        #         """
        #     )

    # Event handlers for custom topic visibility
    def toggle_custom(choice):
        return gr.update(visible=(choice == "Custom"))

    topic_simple.change(toggle_custom, topic_simple, custom_topic_simple)
    topic_hitl.change(toggle_custom, topic_hitl, custom_topic_hitl)

    # Get actual topic (handle custom)
    def get_topic(dropdown, custom):
        return custom if dropdown == "Custom" else dropdown

    # Simple mode handlers
    def simple_generate_wrapper(t, c, temp, mi):
        report, status = simple_generate(get_topic(t, c), temp, mi)
        return report, report, status  # rendered, raw, status

    generate_btn_simple.click(
        fn=simple_generate_wrapper,
        inputs=[topic_simple, custom_topic_simple, temperature_simple, max_iter_simple],
        outputs=[report_simple_rendered, report_simple_raw, status_simple],
    )

    def download_simple_handler(r, t, c):
        result = prepare_download(r, get_topic(t, c))
        print(f"Download handler returning: {result}")
        return result

    download_btn_simple.click(
        fn=download_simple_handler,
        inputs=[report_simple_raw, topic_simple, custom_topic_simple],
        outputs=[download_file_simple],
    )

    # HITL mode handlers
    def draft_wrapper(t, c, temp, mi):
        draft, status, feedback_update, btn_update = generate_initial_draft(get_topic(t, c), temp, mi)
        return (
            draft,
            draft,
            status,
            feedback_update,
            btn_update,
        )  # rendered, raw, status, feedback, btn

    draft_btn.click(
        fn=draft_wrapper,
        inputs=[topic_hitl, custom_topic_hitl, temperature_hitl, max_iter_hitl],
        outputs=[
            draft_output_rendered,
            draft_output_raw,
            status_hitl,
            feedback_input,
            finalize_btn,
        ],
    )

    def finalize_wrapper(t, c, temp, mi, fb):
        final, status = finalize_report(get_topic(t, c), temp, mi, fb)
        return final, final, status  # rendered, raw, status

    finalize_btn.click(
        fn=finalize_wrapper,
        inputs=[
            topic_hitl,
            custom_topic_hitl,
            temperature_hitl,
            max_iter_hitl,
            feedback_input,
        ],
        outputs=[final_output_rendered, final_output_raw, status_hitl],
    )

    def download_hitl_handler(r, t, c):
        result = prepare_download(r, get_topic(t, c))
        print(f"Download handler returning: {result}")
        return result

    download_btn_hitl.click(
        fn=download_hitl_handler,
        inputs=[final_output_raw, topic_hitl, custom_topic_hitl],
        outputs=[download_file_hitl],
    )
