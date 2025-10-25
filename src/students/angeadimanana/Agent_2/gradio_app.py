"""
Gradio Interface for the Agentic Report System
"""

import re
from datetime import datetime

import gradio as gr
from config import GOOGLE_API_KEY, MODEL_NAME, TEMPERATURE
from langchain_google_genai import ChatGoogleGenerativeAI
from workflow.report_workflow import ReportWorkflow

# Initialize LLM globally
llm = ChatGoogleGenerativeAI(
    model=MODEL_NAME,
    temperature=TEMPERATURE,
    google_api_key=GOOGLE_API_KEY
)


def extract_topic_from_prompt(user_input):
    """
    Extract the topic from user input
    Supports topics: APIs, Gradio, MLflow, Dagster, Agentic AI,
    Github flow for app deployment, CI/CD with Github Actions,
    Data Engineering
    """
    user_input_lower = user_input.lower()

    # Define topic keywords
    topics = {
        "apis": ["api", "apis", "restful", "graphql", "rest api", "web api", "rest", "endpoint"],
        "gradio": ["gradio", "interface", "ui", "user interface", "web interface"],
        "mlflow": ["mlflow", "ml flow", "mlops", "machine learning operations", "experiment tracking"],
        "github": ["github", "git", "git hub", "git flow", "CI/CD", "github actions"],
        "Agentic AI": ["agentic", "agentics", "agents AI", "LangChain", "LangGraph"]
    }

    # Check for each topic
    for topic_name, keywords in topics.items():
        for keyword in keywords:
            if keyword in user_input_lower:
                return topic_name.upper() if topic_name != "mlflow" else "MLflow"

    # If no specific topic found, return the input as is
    return user_input


def calculate_report_metrics(report_text):
    """
    Calculate various metrics for the report
    """
    # Word count
    words = re.findall(r'\b[\w\'-]+\b', report_text, flags=re.UNICODE)
    word_count = len([w for w in words if w.strip()])

    # Character count
    char_count = len(report_text)

    # Paragraph count
    paragraphs = [p for p in report_text.split('\n\n') if p.strip()]
    paragraph_count = len(paragraphs)

    # Section count (headers with ##)
    sections = re.findall(r'^#{1,3}\s+.+$', report_text, re.MULTILINE)
    section_count = len(sections)

    # Estimated reading time (average 200 words per minute)
    reading_time = max(1, round(word_count / 200))

    # Sentence count (approximate)
    sentences = re.split(r'[.!?]+', report_text)
    sentence_count = len([s for s in sentences if s.strip()])

    return {
        'word_count': word_count,
        'char_count': char_count,
        'paragraph_count': paragraph_count,
        'section_count': section_count,
        'reading_time': reading_time,
        'sentence_count': sentence_count
    }


def generate_report(user_prompt, progress=gr.Progress()):
    """
    Generate report based on user prompt
    """
    if not user_prompt or user_prompt.strip() == "":
        return (
            "❌ **Error:** Please enter a topic or prompt.",
            "",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            None
        )

    try:
        # Extract topic
        progress(0.1, desc="Extracting topic...")
        topic = extract_topic_from_prompt(user_prompt)

        # Create workflow
        progress(0.2, desc="Initializing workflow...")
        workflow = ReportWorkflow(llm)

        # Generate report
        progress(0.3, desc="Generating report...")
        results = workflow.run(topic, verbose=False)

        progress(0.9, desc="Calculating metrics...")

        # Extract report
        final_report = results['final_report']
        qa_score = results['qa_score']
        status = results['status']
        iterations = results['iterations']

        # Calculate metrics
        metrics = calculate_report_metrics(final_report)

        # Format status message
        status_emoji = "✅" if status == "APPROVED" else "⚠️"
        status_message = f"{status_emoji} **Status:** {status}\n\n**Topic Detected:** {topic}"

        # Format QA info
        qa_info = f"**QA Score:** {qa_score:.2f}/1.0 | **Iterations:** {iterations}"

        progress(1.0, desc="Complete!")

        # Store in session state for download
        return (
            final_report,
            status_message,
            qa_info,
            str(metrics['word_count']),
            str(metrics['char_count']),
            str(metrics['paragraph_count']),
            str(metrics['section_count']),
            f"{metrics['reading_time']} min",
            str(metrics['sentence_count']),
            final_report  # For download
        )

    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return (
            f"❌ **Error:** {str(e)}\n\n**Details:**\n```\n{error_details}\n```",
            "",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            "N/A",
            None
        )


def save_as_txt(report_content):
    """
    Save report as TXT file and return the content for download
    """
    if not report_content:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.txt"

    # Write to file
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report_content)

    return filename


def save_as_md(report_content):
    """
    Save report as Markdown file
    """
    if not report_content:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.md"

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(report_content)

    return filename


def save_as_html(report_content):
    """
    Save report as HTML file with basic styling
    """
    if not report_content:
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.html"

    # Convert markdown-like syntax to HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Report - {timestamp}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 800px;
            margin: 40px auto;
            padding: 20px;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-bottom: 1px solid #ecf0f1;
            padding-bottom: 8px;
        }}
        h3 {{
            color: #7f8c8d;
        }}
        p {{
            text-align: justify;
            margin: 15px 0;
        }}
        .timestamp {{
            color: #95a5a6;
            font-size: 0.9em;
            text-align: right;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="timestamp">Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
"""

    # Simple conversion of markdown to HTML
    lines = report_content.split('\n')
    for line in lines:
        if line.strip():
            if line.startswith('# '):
                html_content += f"        <h1>{line[2:]}</h1>\n"
            elif line.startswith('## '):
                html_content += f"        <h2>{line[3:]}</h2>\n"
            elif line.startswith('### '):
                html_content += f"        <h3>{line[4:]}</h3>\n"
            else:
                html_content += f"        <p>{line}</p>\n"
        else:
            html_content += "        <br>\n"

    html_content += """    </div>
</body>
</html>"""

    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return filename


def create_interface():
    """
    Create the Gradio interface
    """
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .header-text {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #667eea;
    }
    """

    with gr.Blocks(theme=gr.themes.Soft(), css=custom_css, title="Agentic Report Generator") as app:
        # Header
        gr.HTML("""
        <div class="header-text">
            <h1>📝 Agentic Report Generator</h1>
            <h3>Generate Professional Technical Reports with AI</h3>
        </div>
        """)

        gr.Markdown("""
        This application uses a multi-agent system to automatically generate high-quality technical reports
        on one of the following topics: **APIs**, **Gradio**, **MLflow**, **Dagster**, **Agentic AI**,
        **Github flows and actions (CI/CD)**
        Simply enter a prompt or question about any of these topics, and the system will:
        1. 🔍 Research the topic
        2. ✍️ Write a comprehensive report (~1000 words)
        3. ✅ Fact-check the content
        4. 📝 Edit for clarity and structure
        5. 🎯 Perform quality assurance
        ---
        """)

        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                gr.Markdown("## 🎯 Input")
                user_input = gr.Textbox(
                    label="Enter your topic or prompt",
                    placeholder="Example: Tell me about RESTful APIs and their best practices...",
                    lines=3,
                    info="💡 Supported topics: APIs, Gradio, MLflow, Dagster, Github flow & actions"
                )

                with gr.Row():
                    generate_btn = gr.Button("🚀 Generate Report", variant="primary", size="lg")
                    clear_btn = gr.Button("🗑️ Clear", size="lg")

                # Examples
                gr.Examples(
                    examples=[
                        ["What are RESTful APIs and how do they work?"],
                        ["Explain Gradio and its use cases for machine learning"],
                        ["Tell me about MLflow for machine learning operations"],
                        ["How do GraphQL APIs differ from REST APIs?"],
                        ["What are the best practices for building user interfaces with Gradio?"],
                        ["How can MLflow help with experiment tracking and model versioning?"],
                        ["What is the role of Dagster?"],
                        ["Agentic AI or AI agents?"],
                        ["Explain the github flows, github actions"]
                    ],
                    inputs=user_input,
                    label="📚 Example Prompts"
                )

        # Status section
        with gr.Row():
            with gr.Column(scale=1):
                status_output = gr.Markdown(label="📊 Status")
            with gr.Column(scale=1):
                qa_output = gr.Markdown(label="⚡ Quality Metrics")

        # Report output section
        gr.Markdown("## 📄 Generated Report")
        report_output = gr.Textbox(
            label="Report Content",
            lines=25,
            show_copy_button=True,
            interactive=False
        )

        # Metrics section
        gr.Markdown("## 📊 Report Metrics")
        with gr.Row():
            with gr.Column():
                word_count = gr.Textbox(label="📝 Word Count", interactive=False)
            with gr.Column():
                char_count = gr.Textbox(label="🔤 Characters", interactive=False)
            with gr.Column():
                para_count = gr.Textbox(label="📑 Paragraphs", interactive=False)

        with gr.Row():
            with gr.Column():
                section_count = gr.Textbox(label="📂 Sections", interactive=False)
            with gr.Column():
                reading_time = gr.Textbox(label="⏱️ Reading Time", interactive=False)
            with gr.Column():
                sentence_count = gr.Textbox(label="💬 Sentences", interactive=False)

        # Download section
        gr.Markdown("## 💾 Download Report")
        gr.Markdown("Save your generated report in different formats:")

        with gr.Row():
            download_txt = gr.Button("📄 Download as TXT", size="sm", variant="secondary")
            download_md = gr.Button("📋 Download as Markdown", size="sm", variant="secondary")
            download_html = gr.Button("🌐 Download as HTML", size="sm", variant="secondary")

        download_file = gr.File(label="📥 Downloaded File", interactive=False)

        # Hidden state to store report content
        report_state = gr.State()

        # Event handlers
        generate_btn.click(
            fn=generate_report,
            inputs=[user_input],
            outputs=[
                report_output,
                status_output,
                qa_output,
                word_count,
                char_count,
                para_count,
                section_count,
                reading_time,
                sentence_count,
                report_state
            ]
        )

        clear_btn.click(
            fn=lambda: ("", "", "", "", "", "", "", "", "", None),
            outputs=[
                user_input,
                report_output,
                status_output,
                qa_output,
                word_count,
                char_count,
                para_count,
                section_count,
                reading_time,
                sentence_count
            ]
        )

        download_txt.click(
            fn=save_as_txt,
            inputs=[report_state],
            outputs=[download_file]
        )

        download_md.click(
            fn=save_as_md,
            inputs=[report_state],
            outputs=[download_file]
        )

        download_html.click(
            fn=save_as_html,
            inputs=[report_state],
            outputs=[download_file]
        )

        # Footer
        gr.Markdown("""
        ---
        ### 💡 Tips for Better Results:
        - **Be specific** in your prompts for more focused reports
        - The system automatically detects topics related to **APIs**, **Gradio**, or **MLflow**,
        **Dagster**, **Github**
        - Reports are generated with **~1000 words** and undergo multiple quality checks
        - Download your report in **TXT**, **Markdown**, or **HTML** format
        """)

    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    )
