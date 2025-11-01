import os

import gradio as gr
from agents.report.report_agent import (
    MLOpsReportAgent,
    MLOpsReportOrchestrator,
    create_quality_gauge,
    create_stats_pie,
    create_word_count_gauge,
    create_workflow_timeline,
    export_report,
)


def markdown_to_html(text):  # noqa: C901
    """
    Convert markdown text to styled HTML

    Args:
        text: Markdown formatted text

    Returns:
        HTML formatted text
    """
    lines = text.split('\n')
    html_content = []
    in_paragraph = False

    for line in lines:
        stripped = line.strip()

        if not stripped:
            if in_paragraph:
                html_content.append('</p>')
                in_paragraph = False
            html_content.append('<br>')
            continue

        # Handle headers
        if stripped.startswith('###'):
            if in_paragraph:
                html_content.append('</p>')
                in_paragraph = False
            content = stripped.replace('###', '').strip()
            html_content.append(f'<h3 style="color: #34495e; margin-top: 10px; margin-bottom: 10px; font-weight: 600;">{content}</h3>')  # noqa: E501
        elif stripped.startswith('##'):
            if in_paragraph:
                html_content.append('</p>')
                in_paragraph = False
            content = stripped.replace('##', '').strip()
            html_content.append(f'<h2 style="color: #34495e; margin-top: 10px; margin-bottom: 10px; border-left: 4px solid #667eea; padding-left: 15px; font-weight: 700;">{content}</h2>')  # noqa: E501
        elif stripped.startswith('#'):
            if in_paragraph:
                html_content.append('</p>')
                in_paragraph = False
            content = stripped.replace('#', '').strip()
            html_content.append(f'<h1 style="color: #34495e; border-bottom: 3px solid #667eea; padding-bottom: 10px; margin-top: 0; margin-bottom: 10px; font-weight: 800;">{content}</h1>')  # noqa: E501
        else:
            # Handle bold text
            processed_line = line
            # Replace **text** with <strong>text</strong>
            import re
            processed_line = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', processed_line)
            # Replace *text* with <em>text</em>
            processed_line = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'<em>\1</em>', processed_line)

            if not in_paragraph:
                html_content.append(
                '<p style="color: #34495e; line-height: 1.8; '
                'margin: 12px 0; text-align: justify;">'
            )
                in_paragraph = True
            html_content.append(processed_line)

    if in_paragraph:
        html_content.append('</p>')

    full_html = f"""
    <div style="font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 100%;
                padding: 25px;
                background: white;
                border-radius: 10px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.05);">
        {''.join(html_content)}
    </div>
    """

    return full_html


def generate_report_ui(topic, custom_topic, temperature):
    """
    Main UI function for report generation

    Args:
        topic: Selected topic from dropdown
        custom_topic: Custom topic input
        temperature: Model temperature

    Returns:
        Tuple of UI outputs
    """
    final_topic = custom_topic.strip() if custom_topic and custom_topic.strip() else topic

    orchestrator = MLOpsReportOrchestrator()
    result = orchestrator.generate_report(topic=final_topic, temperature=temperature)

    report_text = result['report']
    word_count = result['word_count']
    quality_score = result['quality_score']

    # Convert to HTML for display
    report_html = markdown_to_html(report_text)

    # Create charts
    fig_words = create_word_count_gauge(word_count)
    fig_quality = create_quality_gauge(quality_score)
    fig_timeline = create_workflow_timeline(result['workflow_log'])
    fig_pie = create_stats_pie(word_count)

    # Workflow text
    workflow_text = "WORKFLOW LOG\n" + "=" * 50 + "\n"
    for log in result['workflow_log']:
        workflow_text += f"{log['timestamp']} | {log['step']}: {log['status']}\n"
        if log['details']:
            workflow_text += f"   Details: {log['details']}\n"

    # Stats text
    stats_text = f"""### STATISTICS SUMMARY

| Metric | Value | Status |
|--------|-------|--------|
| **Word Count** | {word_count} | {"Valid" if result['is_valid'] else "Adjust"} |
| **Target** | 950-1050 words | {"OK" if result['is_valid'] else "Review"} |
| **Quality Score** | {quality_score}/10 | {"Excellent" if quality_score >= 8 else "Good"} |
| **Temperature** | {temperature} | OK |

---

### QUALITY ANALYSIS

**Strengths:**
- Clear structure with headings
- Accurate technical content
- Appropriate length

**Status:** Ready for review"""

    return (
        report_html,
        report_text,
        fig_words,
        fig_quality,
        fig_pie,
        fig_timeline,
        stats_text,
        workflow_text,
        final_topic,
        result['is_valid']
    )


# Build Gradio Interface
with gr.Blocks(theme=gr.themes.Ocean()) as llm_report:  # type: ignore

    gr.HTML("""
    <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white; padding: 30px; border-radius: 15px; margin-bottom: 20px;">
        <h1 style="margin:0; font-size: 3em;">MLOps Report Generator</h1>
        <h2 style="margin:10px 0 0 0;">Multi-Agent System with LangGraph & Web Search</h2>
    </div>
    """)

    with gr.Tabs():

        # TAB 1: Information
        with gr.Tab("Information"):
            gr.Markdown("""
            # Welcome to MLOps Report Generator

            ## The 4 Intelligent Agents

            1. **Research Agent** - Uses web search to gather current information
            2. **Outline Agent** - Creates document structure
            3. **Writer Agent** - Writes the complete report (auto-adjusts to 950-1050 words)
            4. **Fact-Checker** - Validates quality and accuracy

            ## Features

            - LangGraph Workflow Orchestration
            - Interactive Visualizations (Matplotlib charts)
            - Export Formats: TXT and HTML (with styling)
            - Human-in-the-Loop Editing
            - Multi-language Translation (English, French, German, Spanish)
            - Fast Generation (30-45 seconds)

            ## Workflow

            The system uses LangGraph to orchestrate a multi-agent workflow with automatic iteration
            for word count adjustment (950-1050 words target).
            """)

        # TAB 2: Report Generation
        with gr.Tab("Report Generation"):

            with gr.Row():
                # Left Column - Controls
                with gr.Column(scale=1):
                    gr.Markdown("## Configuration")

                    topic_input = gr.Dropdown(
                        choices=[
                            "MLOps Overview and Best Practices",
                            "CI/CD in Machine Learning",
                            "Model Monitoring and Observability",
                            "Feature Stores in MLOps",
                            "ML Model Deployment Strategies"
                        ],
                        value="MLOps Overview and Best Practices",
                        label="Topic"
                    )

                    custom_topic = gr.Textbox(
                        label="Or Custom Topic",
                        placeholder="e.g., 'Model Drift Detection'"
                    )

                    temperature = gr.Slider(0.0, 1.0, 0.7, step=0.1, label="Temperature")

                    generate_btn = gr.Button("Generate Draft", variant="primary", size="lg")

                # Right Column - Results
                with gr.Column(scale=2):
                    gr.Markdown("## Generated Draft")

                    draft_output_html = gr.HTML(label="Draft Report")

                    draft_output_raw = gr.Textbox(
                        label="Raw Text (for editing/export)",
                        lines=20,
                        show_copy_button=True,
                        visible=True
                    )

                    gr.Markdown("### Human-in-the-Loop")

                    with gr.Row():
                        edit_btn = gr.Button("Edit Draft", variant="secondary")
                        validate_btn = gr.Button("Validate & Finalize", variant="primary", visible=False)

                    edited_output = gr.Textbox(
                        label="Edit Report Here",
                        lines=20,
                        visible=False
                    )

                    final_output_html = gr.HTML(
                        label="Final Report",
                        visible=False
                    )

                    final_output_raw = gr.Textbox(
                        label="Final Report (Raw)",
                        lines=20,
                        visible=False,
                        show_copy_button=True
                    )

            gr.Markdown("---")

            # Export Section
            with gr.Row():
                gr.Markdown("## Export & Translation")

            with gr.Row():
                format_selector = gr.Radio(
                    choices=["txt", "html"],
                    value="txt",
                    label="Export Format"
                )

                language_selector = gr.Radio(
                    choices=["English", "French", "German", "Spanish"],
                    value="English",
                    label="Translation Language"
                )

            with gr.Row():
                export_btn = gr.Button("Export Report", variant="secondary")
                translate_btn = gr.Button("Translate Report", variant="secondary")

            export_file = gr.File(label="Download File")
            translated_output = gr.Textbox(label="Translated Report", lines=10, visible=False)

        # TAB 3: Statistics
        with gr.Tab("Statistics"):
            stats_text_output = gr.Markdown()

            with gr.Row():
                gauge_words_output = gr.Plot(label="Word Count")
                gauge_quality_output = gr.Plot(label="Quality Score")

            pie_output = gr.Plot(label="Section Distribution")

        # TAB 4: Workflow
        with gr.Tab("Workflow"):
            timeline_output = gr.Plot(label="Timeline")
            workflow_text_output = gr.Textbox(label="Detailed Log", lines=15)

    # State variables
    current_draft = gr.State("")
    current_final = gr.State("")
    current_topic = gr.State("")
    is_valid = gr.State(False)

    # Event Handlers
    def on_generate(topic, custom, temp):
        """Handle generate button click"""
        report_html, report_raw, fig_w, fig_q, fig_p, fig_t, stats, workflow, final_topic, valid = generate_report_ui(topic, custom, temp)  # noqa: E501
        return {
            draft_output_html: report_html,
            draft_output_raw: report_raw,
            gauge_words_output: fig_w,
            gauge_quality_output: fig_q,
            pie_output: fig_p,
            timeline_output: fig_t,
            stats_text_output: stats,
            workflow_text_output: workflow,
            current_draft: report_raw,
            current_topic: final_topic,
            is_valid: valid,
            edited_output: gr.update(visible=False),
            validate_btn: gr.update(visible=False),
            final_output_html: gr.update(visible=False),
            final_output_raw: gr.update(visible=False)
        }

    def on_edit(draft):
        """Handle edit button click"""
        return {
            edited_output: gr.update(visible=True, value=draft),
            validate_btn: gr.update(visible=True),
            draft_output_raw: gr.update(visible=True)
        }

    def on_validate(edited):
        """Handle validate button click"""
        edited_html = markdown_to_html(edited)
        return {
            final_output_html: gr.update(visible=True, value=edited_html),
            final_output_raw: gr.update(visible=True, value=edited),
            current_final: edited,
            edited_output: gr.update(visible=False),
            validate_btn: gr.update(visible=False)
        }

    def on_export(report, format_type, topic):
        """Handle export button click"""
        if not report or not report.strip():
            return None

        if not topic or not topic.strip():
            topic = "mlops_report"

        try:
            filepath = export_report(report, format_type, topic)
            if filepath and os.path.exists(filepath):
                return filepath
            else:
                print(f"Error: File not created at {filepath}")
                return None
        except Exception as e:
            print(f"Export error: {e}")
            return None

    def on_translate(report, language):
        """Handle translate button click"""
        if not report:
            return gr.update(visible=False)

        agent = MLOpsReportAgent()
        translated = agent.translation_agent(report, language)

        return gr.update(visible=True, value=translated)

    # Connect events
    generate_btn.click(
        on_generate,
        [topic_input, custom_topic, temperature],
        [draft_output_html, draft_output_raw, gauge_words_output, gauge_quality_output, pie_output,
         timeline_output, stats_text_output, workflow_text_output,
         current_draft, current_topic, is_valid, edited_output, validate_btn,
         final_output_html, final_output_raw]
    )

    edit_btn.click(
        on_edit,
        [current_draft],
        [edited_output, validate_btn, draft_output_raw]
    )

    validate_btn.click(
        on_validate,
        [edited_output],
        [final_output_html, final_output_raw, current_final, edited_output, validate_btn]
    )

    export_btn.click(
        lambda report, final, fmt, topic: on_export(final if final else report, fmt, topic),
        [current_draft, current_final, format_selector, current_topic],
        [export_file]
    )

    translate_btn.click(
        lambda report, final, lang: on_translate(final if final else report, lang),
        [current_draft, current_final, language_selector],
        [translated_output]
    )

    gr.HTML("""
    <div style="text-align: center; padding: 20px; margin-top: 30px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 10px; color: white;">
        <p style="margin: 0; font-size: 1.1em;">Powered by Gemini 2.0 Flash with LangGraph & Web Search</p>
        <p style="margin: 5px 0 0 0; font-size: 0.9em;">Structured Multi-Agent Report Generation System</p>
    </div>
    """)
