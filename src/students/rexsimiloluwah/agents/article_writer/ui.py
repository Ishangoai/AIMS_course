# ruff: noqa
import sys
import time
from io import StringIO

import gradio as gr
from agents.article_writer.main import run_article_writer
from agents.article_writer.searcher import WikipediaSearcher

# Global variable to capture logs
log_buffer = StringIO()
logs_list = []


class LogCapture:
    def __init__(self, original_stdout):
        self.original_stdout = original_stdout
        self.buffer = StringIO()

    def write(self, text):
        self.original_stdout.write(text)
        self.original_stdout.flush()
        global logs_list
        if text.strip():
            logs_list.append(text)

    def flush(self):
        self.original_stdout.flush()


def run_article_generation(topic, model_name, writer_temp, rewriter_temp, validator_temp,
                           target_words, tolerance, max_revisions,
                           writer_max_tokens, rewriter_max_tokens, validator_max_tokens,
                           progress=gr.Progress()):
    """
    Runs the article generation and returns outputs for the UI
    """
    start_time = time.time()

    global logs_list
    logs_list = []  # Reset logs

    # Capture stdout
    original_stdout = sys.stdout
    sys.stdout = LogCapture(original_stdout)

    try:
        progress(0, desc="Initializing...")

        # Initialize searcher
        searcher = WikipediaSearcher()

        progress(0.1, desc="Starting article generation...")

        # Run the generator
        response = run_article_writer(
            topic=topic,
            target_words=int(target_words),
            tolerance=int(tolerance),
            max_revisions=int(max_revisions),
            model_name=model_name,
            searcher=searcher,
            writer_temperature=writer_temp,
            rewriter_temperature=rewriter_temp,
            validator_temperature=validator_temp,
            writer_max_tokens=int(writer_max_tokens) if writer_max_tokens else None,
            rewriter_max_tokens=int(rewriter_max_tokens) if rewriter_max_tokens else None,
            validator_max_tokens=int(validator_max_tokens) if validator_max_tokens else None,
            save_files=False
        )

        progress(0.9, desc="Finalizing results...")

        # Calculate generation time
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        if minutes > 0:
            time_str = f"{minutes} min {seconds} sec"
        else:
            time_str = f"{seconds} sec"

        # Extract results
        result = response.get("result", {})
        metadata = response.get("metadata", {})
        details = result.get("details", {})

        # Safeguarded article (final output)
        safeguarded_article = result.get("safeguarded_article", "⚠️ No safeguarded article generated.")

        # Original LLM article
        # article_draft = result.get("final_article_llm", "⚠️ No article draft generated.")

        # Outline
        outline = details.get("outline", "⚠️ No outline generated.")

        # Sources/Research insights
        sources = details.get("sources", [])
        if sources:
            sources_text = "## Research Sources\n\n"
            for idx, source in enumerate(sources, 1):
                title = source.get("title", "Unknown")
                link = source.get("link", "")
                content_preview = source.get("content", "")[:200] + "..."
                sources_text += f"### {idx}. {title}\n"
                sources_text += f"**Link:** [{link}]({link})\n\n"
                sources_text += f"{content_preview}\n\n---\n\n"
        else:
            sources_text = "No research sources available."

        # Metadata statistics with tables
        limits = details.get("limits", {})
        temperatures = details.get("temperatures", {})
        max_tokens_config = details.get("max_tokens", {})

        metadata_text = f"""## Generation Statistics

### ⏱️ Generation Time
**Total Time:** {time_str}

### Article Metrics

| Metric | Value |
|--------|-------|
| **Topic** | {details.get('topic', 'N/A')} |
| **Word Count** | {details.get('word_count', 0)} words |
| **Safeguarded Word Count** | {result.get('safeguarded_word_count', 0)} words |
| **Revisions Used** | {details.get('revisions_used', 0)} / {metadata.get('max_revisions', 0)} |
| **Validation Status** | {'✅ Valid' if result.get('was_valid', False) else '⚠️ Invalid'} |
| **Met Limits Without Safeguard** | {'✅ Yes' if metadata.get('met_limits_without_safeguard', False) else '❌ No'} |

### Target Limits

| Parameter | Value |
|-----------|-------|
| **Target Words** | {limits.get('target_words', 0)} |
| **Tolerance** | ± {limits.get('tolerance', 0)} |
| **Min Words** | {limits.get('min_words', 0)} |
| **Max Words** | {limits.get('max_words', 0)} |

### Model Configuration

| Setting | Value |
|---------|-------|
| **Model** | {details.get('model', 'N/A')} |
| **Writer Temperature** | {temperatures.get('writer', 0)} |
| **Rewriter Temperature** | {temperatures.get('rewriter', 0)} |
| **Validator Temperature** | {temperatures.get('validator', 0)} |

### Token Limits

| Component | Max Tokens |
|-----------|------------|
| **Writer** | {max_tokens_config.get('writer') or '2048 (Default)'} |
| **Rewriter** | {max_tokens_config.get('rewriter') or '2048 (Default)'} |
| **Validator** | {max_tokens_config.get('validator') or '2048 (Default)'} |

### Validation Feedback
{details.get('validation_feedback') or 'No feedback provided.'}
"""

        # Get logs
        logs_output = "\n".join(logs_list) if logs_list else "No logs captured."

        progress(1.0, desc="Complete!")
        sys.stdout = original_stdout

        # Success message with time
        success_msg = f"✅ Article generated successfully in {time_str}!"

        return (
            safeguarded_article,
            outline,
            sources_text,
            metadata_text,
            logs_output,
            gr.update(visible=True),  # Show action buttons row
            gr.update(visible=False),  # Hide loading indicator
            success_msg  # Success notification
        )

    except Exception as e:
        sys.stdout = original_stdout

        # Calculate time even on error
        end_time = time.time()
        elapsed_time = end_time - start_time
        minutes = int(elapsed_time // 60)
        seconds = int(elapsed_time % 60)

        if minutes > 0:
            time_str = f"{minutes} min {seconds} sec"
        else:
            time_str = f"{seconds} sec"

        error_msg = f"⚠️ **Error:** {str(e)}"
        logs_output = "\n".join(logs_list) if logs_list else f"Error occurred: {str(e)}"
        return (
            error_msg,
            error_msg,
            error_msg,
            error_msg,
            logs_output,
            gr.update(visible=False),  # Hide action buttons on error
            gr.update(visible=False),  # Hide loading indicator
            f"❌ Generation failed after {time_str}"  # Error notification
        )


def check_topic_input(topic):
    """Check if topic is entered to enable/disable generate button"""
    if topic and topic.strip():
        return gr.update(interactive=True)
    else:
        return gr.update(interactive=False)

custom_css = """
/* Import modern fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global font settings */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* Body text size */
body, .gradio-container {
    font-size: 16px !important;
    line-height: 1.6 !important;
}

/* Input labels and descriptions */
label, .label-wrap {
    font-size: 15px !important;
}

/* Markdown content (articles, results) */
.markdown-text, .prose {
    font-size: 17px !important;
    line-height: 1.7 !important;
}

/* Buttons */
button {
    font-size: 15px !important;
}

/* Code and logs use monospace */
code, pre, .logs-container {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
    font-size: 14px !important;
}

.main-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    border-radius: 12px;
    margin-bottom: 2rem;
    color: white;
}

.main-header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
}

.main-header p {
    margin: 0.5rem 0 0 0;
    font-size: 1.1rem;
    opacity: 0.95;
}

.example-topics {
    margin-top: 0.5rem;
    margin-bottom: 1rem;
    padding: 1rem;
    background: rgba(245, 87, 108, 0.05);
    border-radius: 8px;
}

.example-topics .gallery {
    gap: 0.75rem !important;
}

.example-topics .gallery button {
    border: 2px solid #f5576c !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 4px rgba(245, 87, 108, 0.1) !important;
}

.example-topics .gallery button:hover {
    border-color: #f093fb !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(245, 87, 108, 0.2) !important;
}

label {
    background: transparent !important;
    color: #333 !important;
    font-weight: 600 !important;
}

.label-wrap {
    background: transparent !important;
}

span.label-wrap span {
    background: transparent !important;
    color: #333 !important;
}

button.primary {
    background: linear-gradient(135deg, #2d5016 0%, #3d6b1f 100%) !important;
    border: none !important;
    color: white !important;
}

button.primary:hover:not(:disabled) {
    background: linear-gradient(135deg, #3d6b1f 0%, #4d7c2f 100%) !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(45, 80, 22, 0.3) !important;
}

button.primary:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

button.secondary {
    background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%) !important;
    border: none !important;
    color: white !important;
}

button.secondary:hover {
    background: linear-gradient(135deg, #f7931e 0%, #ffaa4d 100%) !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(255, 107, 53, 0.3) !important;
}

button[size="sm"] {
    background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%) !important;
    border: none !important;
    color: white !important;
    padding: 0.5rem 1rem !important;
}

button[size="sm"]:hover {
    background: linear-gradient(135deg, #4b5563 0%, #374151 100%) !important;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(75, 85, 99, 0.3) !important;
}

.generate-section {
    margin: 1.5rem 0;
}

.main-container {
    display: flex;
    gap: 2rem;
    align-items: flex-start;
}

.input-column {
    flex: 1;
    min-width: 400px;
}

.output-column {
    flex: 1.5;
    min-width: 500px;
}

#component-0 {
    max-width: 1600px;
    margin: 0 auto;
}

.logs-container {
    background: #1e1e1e;
    color: #d4d4d4;
    padding: 1rem;
    border-radius: 8px;
    font-family: 'Courier New', monospace;
    font-size: 0.9rem;
    max-height: 600px;
    overflow-y: auto;
}

.action-buttons-container {
    background: rgba(45, 80, 22, 0.05);
    padding: 1rem;
    border-radius: 8px;
    margin-top: 1rem;
}

.loading-message {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    margin: 1rem 0;
    animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.8; }
}

.success-notification {
    background: linear-gradient(135deg, #2d5016 0%, #3d6b1f 100%);
    color: white;
    padding: 1rem 2rem;
    border-radius: 8px;
    text-align: center;
    margin: 1rem 0;
    font-size: 1.1rem;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(45, 80, 22, 0.3);
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green"), css=custom_css) as demo:  # type: ignore
    gr.HTML("""
        <div class="main-header">
            <h1>Article Writer</h1>
            <p>A simple agentic system for researching and writing articles on any topic.</p>
        </div>
    """)

    with gr.Row(elem_classes="main-container"):
        # Left Column - Inputs
        with gr.Column(elem_classes="input-column"):
            gr.Markdown("## 📝 Input Configuration")

            with gr.Group():
                topic = gr.Textbox(
                    label="📝 Article Topic",
                    placeholder="Enter your article topic here...",
                    lines=2,
                    scale=1
                )

                with gr.Row(elem_classes="example-topics"):
                    gr.Examples(
                        examples=[
                            ["CI/CD Pipelines"],
                            ["MLOps"],
                            ["Microservices Architecture"],
                            ["Kubernetes Orchestration"],
                            ["DevSecOps Best Practices"]
                        ],
                        inputs=topic,
                        label="💡 Example Topics"
                    )

            with gr.Accordion("⚙️ Advanced Configuration", open=False):
                with gr.Row():
                    model_name = gr.Dropdown(
                        choices=["gemini-2.0-flash-exp", "gemini-2.5-pro"],
                        value="gemini-2.0-flash-exp",
                        label="🤖 Model",
                        info="Select the AI model to use"
                    )

                gr.Markdown("### Temperature Settings")
                with gr.Row():
                    writer_temp = gr.Slider(0.0, 1.0, value=0.2,
                                            step=0.05,
                                            label="Writer",
                                            info="Creativity level")
                with gr.Row():
                    rewriter_temp = gr.Slider(0.0, 1.0, value=0.1,
                                              step=0.05,
                                              label="Rewriter",
                                              info="Refinement focus")
                with gr.Row():
                    validator_temp = gr.Slider(0.0, 1.0, value=0.0,
                                               step=0.05,
                                               label="Validator",
                                               info="Strictness level")

                gr.Markdown("### Content Parameters")
                with gr.Row():
                    target_words = gr.Number(value=1000, label="Target Words", info="Desired article length")
                with gr.Row():
                    tolerance = gr.Number(value=50, label="Tolerance (±)", info="Acceptable word variance")
                with gr.Row():
                    max_revisions = gr.Number(value=3, label="Max Revisions", info="Quality iterations")

                gr.Markdown("### Token Limits (Leave at 2048 for defaults)")
                with gr.Row():
                    writer_max_tokens = gr.Number(value=2048, label="Writer Max Tokens", precision=0)
                with gr.Row():
                    rewriter_max_tokens = gr.Number(value=2048, label="Rewriter Max Tokens", precision=0)
                with gr.Row():
                    validator_max_tokens = gr.Number(value=2048, label="Validator Max Tokens", precision=0)

            with gr.Row(elem_classes="generate-section"):
                generate_btn = gr.Button("🚀 Generate Article", variant="primary", size="lg", scale=2, interactive=False)
                clear_btn = gr.Button("🔄 Clear", variant="secondary", size="lg", scale=1)

        # Right Column - Outputs
        with gr.Column(elem_classes="output-column"):
            gr.Markdown("## 📤 Generated Results")

            with gr.Tabs():
                with gr.Tab("🏁 Final Result"):
                    # Loading indicator
                    loading_indicator = gr.Markdown("", visible=False)
                    # Success notification area
                    success_notification = gr.Markdown("", visible=False, elem_classes="success-notification")
                    safeguarded_box = gr.Markdown(label="Safeguarded Final Result", value="")

                    # Action buttons container
                    with gr.Group(visible=False, elem_classes="action-buttons-container") as action_buttons_row:
                        gr.Markdown("### 📥 Export Options")
                        with gr.Row():
                            with gr.Column(scale=2):
                                download_format = gr.Dropdown(
                                    choices=["Markdown (.md)", "Text (.txt)"],
                                    value="Markdown (.md)",
                                    label="Download Format",
                                )
                            with gr.Column(scale=1):
                                download_btn = gr.Button("⬇️ Download", size="lg", variant="primary")

                        with gr.Row():
                            copy_btn = gr.Button("📋 Copy to Clipboard", size="lg", variant="secondary", scale=1)

                        # File output for download (now visible to show download link)
                        download_file = gr.File(label="📥 Your Download", visible=True, interactive=False)

                with gr.Tab("📜 Logs"):
                    logs_box = gr.Code(label="Generation Logs", language="shell", value="", elem_classes="logs-container")
                    refresh_logs_btn = gr.Button("🔄 Refresh Logs", size="sm")

                with gr.Tab("📋 Outline"):
                    outline_box = gr.Markdown(label="Generated Outline", value="")

                with gr.Tab("🔍 Sources"):
                    sources_box = gr.Markdown(label="Research Sources", value="")

                with gr.Tab("📊 Metadata"):
                    metadata_box = gr.Markdown(label="Generation Statistics", value="")

                with gr.Tab("ℹ️ About"):
                    gr.Markdown("""
                    ## How It Works
                    
                    This AI Article Generator uses a **multi-agent system** to create high-quality, research-backed articles. Here's the workflow:
                    
                    ### 🔄 Generation Pipeline
                    
                    1. **Planning Phase** 🎯
                       - Analyzes your topic
                       - Creates a structured outline
                       - Defines article scope and sections
                    
                    2. **Research Phase** 🔍
                       - Searches Wikipedia for relevant information
                       - Extracts key facts and context
                       - Gathers authoritative sources
                    
                    3. **Writing Phase** ✍️
                       - Generates initial article draft
                       - Follows the planned outline
                       - Integrates research findings
                       - Targets specified word count
                    
                    4. **Validation Phase** ✅
                       - Checks word count compliance
                       - Validates structure and flow
                       - Ensures quality standards
                       - Provides detailed feedback
                    
                    5. **Revision Phase** 🔧
                       - Refines based on validation feedback
                       - Adjusts word count if needed
                       - Improves clarity and coherence
                       - Iterates up to max revisions
                    
                    6. **Safeguard Phase** 🛡️
                       - Applies final word count enforcement
                       - Ensures hard limits are met
                       - Preserves article integrity
                    
                    ### 🤖 Multi-Agent Architecture
                    
                    - **Planner Agent**: Creates article structure
                    - **Writer Agent**: Generates content (higher temperature for creativity)
                    - **Validator Agent**: Checks quality (zero temperature for consistency)
                    - **Rewriter Agent**: Refines content (lower temperature for precision)
                    
                    ### ⚙️ Configuration Options
                    
                    - **Temperature**: Controls creativity vs. consistency
                      - Higher (0.5-1.0) = More creative
                      - Lower (0.0-0.3) = More focused
                    
                    - **Target Words & Tolerance**: Sets desired length
                      - Target: Your ideal article length
                      - Tolerance: Acceptable variance (±)
                    
                    - **Max Revisions**: Quality vs. speed tradeoff
                      - More revisions = Higher quality
                      - Fewer revisions = Faster generation
                    
                    - **Token Limits**: Controls response length per agent
                      - Default: 2048 tokens
                      - Increase for longer outputs
                    
                    ### 📊 Output Tabs
                    
                    - **Final Result**: Safeguarded, production-ready article
                    - **Article Draft**: Original LLM output before safeguarding
                    - **Outline**: Planned structure and sections
                    - **Sources**: Research materials with links
                    - **Metadata**: Detailed generation statistics
                    - **Logs**: Real-time generation process logs
                    
                    ### 🚀 Tips for Best Results
                    
                    1. Be specific with topics (e.g., "CI/CD Pipelines" vs. "Technology")
                    2. Use appropriate word counts (800-2000 for most topics)
                    3. Start with default temperatures and adjust if needed
                    4. Allow 3-5 revisions for quality articles
                    5. Check metadata to see if safeguarding was needed
                                
                    ### Limitations
                    
                    - Relies on Wikipedia; may miss niche topics
                    - Generated content should be reviewed for accuracy
                    - LLMs think in tokens, not words. Might be difficult to hit exact word counts. Might be a skill issue too, haha.
                    
                    ### 🎓 Powered By
                    
                    - **LangGraph**: Multi-agent orchestration
                    - **Google Gemini**: State-of-the-art language model
                    - **Wikipedia API**: Authoritative research source
                    - **Gradio**: Interactive web interface
                    
                    ---
                    
                    **Ready to generate?** Enter a topic above and click Generate! 🚀
                    """)

    # Enable/disable generate button based on topic input
    topic.change(
        fn=check_topic_input,
        inputs=[topic],
        outputs=[generate_btn]
    )

    # Show loading indicator when generation starts
    def show_loading():
        return (
            gr.update(value="## 🔄 Generating your article...\n\nPlease wait while our AI agents research, write, and validate your content. This may take 1-2 minutes.", visible=True),
            gr.update(visible=False),  # Hide action buttons during generation
            gr.update(visible=False),  # Hide success notification
        )

    generate_btn.click(
        fn=show_loading,
        inputs=None,
        outputs=[loading_indicator, action_buttons_row, success_notification],
        queue=False
    ).then(
        fn=run_article_generation,
        inputs=[
            topic, model_name,
            writer_temp, rewriter_temp, validator_temp,
            target_words, tolerance, max_revisions,
            writer_max_tokens, rewriter_max_tokens, validator_max_tokens
        ],
        outputs=[
            safeguarded_box,
            outline_box,
            sources_box,
            metadata_box,
            logs_box,
            action_buttons_row,
            loading_indicator,
            success_notification
        ],
        show_progress=True  # type: ignore
    ).then(
        fn=lambda msg: gr.update(value=msg, visible=True),
        inputs=[success_notification],
        outputs=[success_notification]
    )

    # Clear button action
    def clear_all():
        return (
            "",  # topic
            "",  # safeguarded_box
            "",  # outline_box
            "",  # sources_box
            "",  # metadata_box
            "",  # logs_box
            gr.update(visible=False),  # Hide action buttons
            gr.update(visible=False),  # Hide loading indicator
            gr.update(visible=False),  # Hide success notification
            gr.update(interactive=False)  # Disable generate button
        )

    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[
            topic,
            safeguarded_box,
            outline_box,
            sources_box,
            metadata_box,
            logs_box,
            action_buttons_row,
            loading_indicator,
            success_notification,
            generate_btn
        ]
    )

    # Copy functionality - copies the actual markdown content
    def copy_to_clipboard(content):
        # Return the content itself so Gradio can handle clipboard
        if not content or content.startswith("⚠️"):
            gr.Warning("No content to copy!")
            return None
        gr.Info("✅ Content copied to clipboard! Use Ctrl+V to paste.")
        return content

    copy_btn.click(
        fn=copy_to_clipboard,
        inputs=[safeguarded_box],
        outputs=None,
        js="(content) => {navigator.clipboard.writeText(content); return content;}"
    )

    # Download functionality - creates actual file download
    def create_download_file(content, format_choice):
        if not content or content.startswith("⚠️"):
            gr.Warning("No valid content to download!")
            return None

        import os
        import tempfile
        from datetime import datetime

        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if "Markdown" in format_choice:
            extension = ".md"
            filename = f"article_{timestamp}.md"
        else:
            extension = ".txt"
            filename = f"article_{timestamp}.txt"

        # Create a temporary file that persists
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        gr.Info(f"✅ Download ready: {filename}")
        return filepath

    download_btn.click(
        fn=create_download_file,
        inputs=[safeguarded_box, download_format],
        outputs=download_file
    ).then(
        fn=lambda x: gr.update(visible=True, value=x) if x else gr.update(visible=False),
        inputs=[download_file],
        outputs=[download_file]
    )

    # Refresh logs
    def get_current_logs():
        return "\n".join(logs_list) if logs_list else "No logs available yet."

    refresh_logs_btn.click(
        fn=get_current_logs,
        inputs=None,
        outputs=logs_box
    )

    gr.Markdown("""
    ---
    <div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">
        <p>Research → Write → Validate → Safeguard</p>
        <p>Made with ❤️ using LangGraph + Google Gemini + Gradio</p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(share=False)
