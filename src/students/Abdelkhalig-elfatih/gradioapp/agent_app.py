"""
Gradio Interface for Agentic Report Writer (Google Gemini)
Simple functional implementation without classes
"""

import os
from datetime import datetime
from io import BytesIO

import gradio as gr
from agent import AgenticReportWriter
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

# Global variables
writer = None
current_draft = ""


def initialize_writer(api_key: str, temperature: float):
    """Initialize the writer with user settings"""
    global writer
    try:
        writer = AgenticReportWriter(
            google_api_key=api_key,
            model="gemini-2.0-flash-exp",
            temperature=temperature
        )
        # Generate the diagram
        try:
            graph_image = writer.graph.get_graph().draw_mermaid_png()
            # Save to file
            with open("agentic_workflow_diagram.png", "wb") as f:
                f.write(graph_image)
        except Exception as e:
            print(f"Error: {e}")

        return "System initialized successfully"
    except Exception as e:
        return f"Error initializing system: {str(e)}"


def generate_report(topic: str, api_key: str, temperature: float, progress=gr.Progress()):
    """Generate report with progress tracking"""
    global writer, current_draft

    if not topic or not topic.strip():
        return "Please enter a topic", "", 0, ""

    # Initialize if needed
    if not writer:
        init_msg = initialize_writer(api_key, temperature)
        if "Error" in init_msg:
            return init_msg, "", 0, ""

    try:
        progress(0, desc="Researching topic...")
        progress(0.2, desc="Creating outline...")
        progress(0.4, desc="Writing sections...")
        progress(0.7, desc="Quality checking...")
        progress(0.9, desc="Editing and formatting...")

        # Generate report
        result = writer.generate_report(topic)

        try:
            # Generate diagram as bytes
            graph_bytes = writer.graph.get_graph().draw_mermaid_png()

        # Keep it in memory as a BytesIO object
            image_buffer = BytesIO(graph_bytes)

            img = Image.open(image_buffer)
            img.save("agentic_workflow_diagram.png", format="PNG")

        except Exception as e:
            print(f"Error generating diagram: {e}")

        progress(1.0, desc="Complete")

        # Prepare metadata
        metadata = f"""
**Report Metadata**
- **Topic:** {result['topic']}
- **Word Count:** {result['word_count']}
- **Target Range:** 950-1050 words
- **Iterations:** {result['iterations']}
- **Model:** gemini-2.0-flash-exp
- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        current_draft = result['report']

        return (
            "Report generated successfully",
            result['report'],
            result['word_count'],
            metadata
        )

    except Exception as e:
        return f"Error generating report: {str(e)}", "", 0, ""


def save_report(report_text: str, topic: str):
    """Save report to text file"""
    if not report_text:
        return "No report to save"

    try:
        # Create reports directory if it doesn't exist
        os.makedirs("reports", exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        safe_topic = "".join(c for c in topic if c.isalnum() or c in (' ', '-', '_')).rstrip()
        safe_topic = safe_topic.replace(' ', '_')[:50]
        filename = f"reports/report_{safe_topic}_{timestamp}.txt"

        # Save file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Topic: {topic}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            f.write(report_text)

        return f"Report saved to: {filename}"

    except Exception as e:
        return f"Error saving report: {str(e)}"


def edit_draft(edited_text: str, additional_instructions: str, api_key: str, temperature: float):
    """Human-in-the-loop: Edit and refine the draft"""
    global writer, current_draft

    if not edited_text:
        return "No draft to refine", edited_text, 0

    if not additional_instructions:
        # Just update word count
        word_count = len(edited_text.split())
        return "Draft updated (no AI refinement)", edited_text, word_count

    try:
        # Use LLM to apply user's instructions
        if not writer:
            initialize_writer(api_key, temperature)

        from langchain.prompts import ChatPromptTemplate

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert editor. The user has provided a draft report
            and specific instructions for refinement. Apply these instructions while
            maintaining the overall structure and quality of the report.
            Keep the word count as close to the original as possible unless instructed otherwise."""),
            ("human", """Draft report:
            {draft}
            User instructions:
            {instructions}

            Apply the instructions and return the refined report.""")
        ])

        response = writer.llm.invoke(
            prompt.format_messages(
                draft=edited_text,
                instructions=additional_instructions
            )
        )

        refined_text = response.content
        word_count = len(refined_text.split())

        current_draft = refined_text

        return (
            "Draft refined successfully",
            refined_text,
            word_count
        )

    except Exception as e:
        return f"Error refining draft: {str(e)}", edited_text, len(edited_text.split())


# Custom CSS
custom_css = """
.gradio-container {
    font-family: 'Arial', sans-serif;
}
.header {
    text-align: center;
    padding: 20px;
    background: linear-gradient(135deg, #4285F4 0%, #34A853 50%, #FBBC05 100%);
    color: white;
    border-radius: 10px;
    margin-bottom: 20px;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as app:

    # Header
    gr.HTML("""
    <div class="header">
        <h1>Agentic Report Writer</h1>
        <p>Multi-Agent System powered by Google Gemini 2.0 Flash</p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Configuration")

            api_key = gr.Textbox(
                label="Google API Key",
                type="password",
                value=os.getenv("GOOGLE_API_KEY", ""),
                placeholder="AIza..."
            )

            gr.Markdown("**Model:** gemini-2.0-flash-exp (fixed)")

            temperature = gr.Slider(
                label="Temperature",
                minimum=0.0,
                maximum=1.0,
                value=0.7,
                step=0.1,
                info="Higher = more creative"
            )

            gr.Markdown("## Report Topic")

            topic_input = gr.Textbox(
                label="Enter Topic",
                placeholder="e.g., MLOps Best Practices, CI/CD Pipeline Design, API Development with FastAPI",
                lines=2
            )

            example_topics = gr.Dropdown(
                label="Or select an example topic:",
                choices=[
                    "MLOps: Best Practices and Implementation Strategies",
                    "CI/CD Pipelines: Modern Approaches and Tools",
                    "RESTful API Design: Principles and Best Practices",
                    "Gradio: Building Interactive ML Applications",
                    "Container Orchestration with Kubernetes in ML",
                    "Model Monitoring and Observability in Production"
                ]
            )

            generate_btn = gr.Button("Generate Report", variant="primary", size="lg")
            status_output = gr.Textbox(label="Status", lines=2)

        with gr.Column(scale=2):
            gr.Markdown("## Generated Report")

            with gr.Tab("Final Report"):
                report_output = gr.Textbox(
                    label="Report",
                    lines=20,
                    max_lines=30,
                    show_copy_button=True
                )

                with gr.Row():
                    word_count_output = gr.Number(label="Word Count", precision=0)

                save_btn = gr.Button("Save as TXT", variant="secondary")
                save_status = gr.Textbox(label="Save Status")

            with gr.Tab("Edit Draft (Human-in-the-Loop)"):
                gr.Markdown("""
                **Human-in-the-Loop Editing**

                Edit the draft directly or provide instructions for AI-assisted refinement.
                """)

                draft_editor = gr.Textbox(
                    label="Editable Draft",
                    lines=15,
                    max_lines=25,
                    interactive=True
                )

                edit_instructions = gr.Textbox(
                    label="Refinement Instructions (optional)",
                    placeholder="""e.g., 'Add more examples in the second section'
                    or 'Make the conclusion more impactful'""",
                    lines=3
                )

                refine_btn = gr.Button("Refine Draft", variant="secondary")

                edit_status = gr.Textbox(label="Edit Status")
                edited_word_count = gr.Number(label="Word Count", precision=0)

            with gr.Tab("Metadata and Outline"):
                metadata_output = gr.Markdown()

    # Example topic selection
    example_topics.change(
        fn=lambda x: x,
        inputs=[example_topics],
        outputs=[topic_input]
    )

    # Generate report
    generate_btn.click(
        fn=generate_report,
        inputs=[topic_input, api_key, temperature],
        outputs=[status_output, report_output, word_count_output, metadata_output]
    ).then(
        fn=lambda x: x,
        inputs=[report_output],
        outputs=[draft_editor]
    )

    # Save report as TXT
    save_btn.click(
        fn=save_report,
        inputs=[report_output, topic_input],
        outputs=[save_status]
    )

    # Edit/refine draft
    refine_btn.click(
        fn=edit_draft,
        inputs=[draft_editor, edit_instructions, api_key, temperature],
        outputs=[edit_status, draft_editor, edited_word_count]
    )

    # Instructions
    with gr.Accordion("Instructions", open=False):
        gr.Markdown("""
        ### How to Use
        1. **Configure Settings**: Enter your Google API key and adjust temperature
        2. **Choose Topic**: Enter a custom topic or select from examples
        3. **Generate**: Click "Generate Report" and wait for the multi-agent system to work
        4. **Review**: Check the generated report and metadata
        5. **Edit (Optional)**: Use the "Edit Draft" tab for human-in-the-loop refinement
        6. **Save**: Save the final report as TXT file
        ### Agent Workflow
        1. **Research Agent**: Identifies key research areas
        2. **Outline Agent**: Creates structured outline
        3. **Writing Agent**: Writes each section
        4. **Quality Agent**: Reviews for accuracy and completeness
        5. **Editor Agent**: Ensures proper length (950-1050 words) and formatting
        ### Features
        - **Multi-Agent System**: Specialized agents for each task
        - **Powered by Gemini 2.0 Flash**: Fast and efficient
        - **Quality Control**: Automatic fact-checking and editing
        - **Human-in-the-Loop**: Edit drafts directly or with AI assistance
        - **Flexible Topics**: Any course-related topic (MLOps, CI/CD, APIs, etc.)
        - **Export**: Save reports as TXT files
        ### Getting Google API Key
        1. Go to https://makersuite.google.com/app/apikey
        2. Create a new API key
        3. Copy and paste it above
        """)

# Run the application
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )
