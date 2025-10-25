# report_agent_app/gradioapp.py
"""
Gradio Interface - Enhanced Version
- 3-writer system with progress tracking
- Temperature control
- Plain text display, markdown download
- Original light CSS design
- Entertaining status updates
"""

import re
import time
from datetime import datetime
from pathlib import Path

import gradio as gr

# Import your modules
from .fandresena_agent import count_words, generate_essay_with_context
from .web_search import get_search_context

# -------------------------
# Configuration
# -------------------------
MIN_WORDS = 950
MAX_WORDS = 1050

# -------------------------
# Entertaining Messages
# -------------------------
ENTERTAINING_MESSAGES = [
    "🧠 AI neurons firing up...",
    "📚 Scanning MLOps knowledge base...",
    "✨ Channeling technical excellence...",
    "🔥 Cooking up some quality content...",
    "🎯 Optimizing word choice...",
    "🚀 Deploying linguistic algorithms...",
    "💡 Illuminating complex concepts...",
    "⚡ Processing at lightning speed...",
    "🎨 Painting with words...",
    "🔬 Conducting quality analysis...",
]


# -------------------------
# Helper Functions
# -------------------------
def convert_to_plain_text(markdown_text):
    """Convert markdown to plain text for display"""
    # Remove markdown symbols but keep structure
    text = markdown_text

    # Replace headers with plain text equivalents
    text = re.sub(r'^# (.*?)$', r'\1', text, flags=re.MULTILINE)  # Main title
    text = re.sub(r'^## (.*?)$', r'\n\1\n' + '=' * 40, text, flags=re.MULTILINE)  # Sections
    text = re.sub(r'^### (.*?)$', r'\n\1\n' + '-' * 30, text, flags=re.MULTILINE)  # Subsections

    # Remove bold/italic
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)

    # Remove code blocks
    text = re.sub(r'`(.*?)`', r'\1', text)

    # Clean up multiple newlines
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


# -------------------------
# Generation Function
# -------------------------
def generate_with_streaming(topic, use_web_search, temperature):
    """
    Generate essay with progress updates and entertaining messages
    ALL OUTPUTS MUST BE STRINGS for Gradio Markdown components
    """
    if not topic.strip():
        yield (
            "⚠️ Please enter a topic",
            "",
            "📝 Words: 0",
            "📄 Paragraphs: 0",
            "❌ No Input",
            False,
            False,
            False
        )
        return

    # Reset state - ALL STRINGS
    yield (
        "🚀 Initializing...",
        "",
        "📝 Words: 0",
        "📄 Paragraphs: 0",
        "⏳ Starting",
        False,
        False,
        False
    )
    time.sleep(0.3)

    progress_log = ""

    def add_progress(msg):
        """Add progress message"""
        nonlocal progress_log
        progress_log += msg + "\n"
        return progress_log

    try:
        # Step 1: Web search (optional)
        search_context = ""
        if use_web_search:
            add_progress("🔍 Performing web search...")
            yield (
                progress_log,
                "",
                "📝 Words: 0",
                "📄 Paragraphs: 0",
                "🌐 Searching",
                False,
                False,
                False
            )
            time.sleep(0.3)

            search_context = get_search_context(topic)

            if "error" in search_context.lower() or not search_context.strip():
                search_context = ""
                add_progress("⚠️ Web search unavailable, continuing without it")
            else:
                add_progress("✅ Web search completed - latest info retrieved")

            yield (
                progress_log,
                "",
                "📝 Words: 0",
                "📄 Paragraphs: 0",
                "✅ Search Done",
                True,
                False,
                False
            )
            time.sleep(0.3)

        # Step 2: Generate essay with progress callbacks
        add_progress(f"\n🎯 Generating report (Temperature: {temperature})")
        add_progress("⏱️ This will take 2-3 minutes...\n")
        yield (
            progress_log,
            "",
            "📝 Words: 0",
            "📄 Paragraphs: 0",
            "✍️ Writing",
            True,
            False,
            False
        )
        time.sleep(0.5)

        essay = ""
        current_step = 0

        def progress_callback(message):
            """Handle progress updates from agent"""
            nonlocal progress_log, current_step

            # Add entertaining message occasionally
            if current_step % 2 == 0 and current_step < len(ENTERTAINING_MESSAGES):
                add_progress(ENTERTAINING_MESSAGES[current_step])

            add_progress(message)
            current_step += 1

        # Generate essay
        essay = generate_essay_with_context(
            topic=topic,
            web_search_context=search_context,
            temperature=temperature,
            progress_callback=progress_callback
        )

        # Calculate final metrics
        word_count = count_words(essay)
        para_count = len([p for p in essay.split('\n\n') if p.strip()])

        # Determine status
        if MIN_WORDS <= word_count <= MAX_WORDS:
            status = "✅ Valid (Perfect!)"
        elif word_count < MIN_WORDS:
            status = f"⚠️ Too Short (-{MIN_WORDS - word_count} words)"
        else:
            status = f"⚠️ Too Long (+{word_count - MAX_WORDS} words)"

        add_progress(f"\n{'=' * 50}")
        add_progress("🎉 GENERATION COMPLETE!")
        add_progress(f"📊 Final word count: {word_count}")
        add_progress(f"📄 Paragraphs: {para_count}")
        add_progress(f"✅ Status: {status}")
        add_progress(f"{'=' * 50}")

        # Convert markdown to plain text for display
        plain_text = convert_to_plain_text(essay)

        # Final output - ALL STRINGS for Markdown components
        yield (
            progress_log,
            plain_text,
            f"📝 Words: {word_count}",
            f"📄 Paragraphs: {para_count}",
            status,
            True,
            True,
            True
        )

    except Exception as e:
        error_msg = f"❌ Error occurred: {str(e)}"
        add_progress(error_msg)
        yield (
            progress_log,
            "",
            "📝 Words: 0",
            "📄 Paragraphs: 0",
            "❌ Error",
            False,
            False,
            False
        )


# -------------------------
# Download Functions
# -------------------------
def save_markdown(essay, topic):
    """Save essay as markdown file"""
    if not essay:
        return None

    # Clean filename
    clean_topic = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in topic)
    clean_topic = clean_topic.replace(' ', '_')[:30]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"report_{clean_topic}_{timestamp}.md"
    filepath = Path.cwd() / filename

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"# {topic}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Word Count:** {count_words(essay)}\n")
        f.write("**Course:** MLOps Technical Report\n\n")
        f.write("---\n\n")
        f.write(essay)

    return str(filepath)


def save_txt(essay, topic):
    """Save essay as plain text file"""
    if not essay:
        return None

    clean_topic = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in topic)
    clean_topic = clean_topic.replace(' ', '_')[:30]

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"report_{clean_topic}_{timestamp}.txt"
    filepath = Path.cwd() / filename

    plain_text = convert_to_plain_text(essay)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(f"{topic}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Word Count: {count_words(essay)}\n")
        f.write("Course: MLOps Technical Report\n\n")
        f.write("-" * 60 + "\n\n")
        f.write(plain_text)

    return str(filepath)


# -------------------------
# Gradio Interface
# -------------------------

# Load custom CSS
custom_css = ""
try:
    with open('report_agent_app/app.css') as f:
        custom_css = f.read()
except:  # noqa: E722
    print("Warning: Could not load app.css")

with gr.Blocks(css=custom_css, title="MLOps ReportGen Pro") as report_app:

    # State variables
    essay_state = gr.State("")
    topic_state = gr.State("")

    # Main container
    with gr.Row(elem_classes='main_row'):
        with gr.Column(elem_classes='main_column'):

            # Title
            gr.Markdown('ML Ops ReportGen Pro', elem_classes='app_title')

            # Input row
            with gr.Row(elem_classes='top_row'):
                prompt = gr.Textbox(
                    container=False,
                    elem_classes='prompt_area',
                    placeholder='E.g., CI/CD Pipelines with GitHub Actions for ML Projects'
                )
                action_button = gr.Button('Generate', elem_classes='action-button')

            # Temperature control (new)
            with gr.Row(elem_classes='top_row'):
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.6,
                    step=0.1,
                    label="🌡️ Temperature (Creativity)",
                    info="Lower = More focused, Higher = More creative"
                )
                use_search = gr.Checkbox(
                    label='🔍 Enable Web Search',
                    container=False,
                    elem_classes='pcheck_box',
                    value=False
                )

            # Main content area
            with gr.Row():
                # LEFT: Preview (70%)
                with gr.Column(elem_classes='column-shadow prcolumn'):
                    gr.Markdown('Preview', elem_classes='pheading')
                    preview = gr.TextArea(
                        container=False,
                        label='',
                        elem_classes='preview',
                        lines=30,
                        max_lines=50
                    )
                    with gr.Row():
                        word_count_display = gr.Markdown('📝 Words: 0', elem_classes='props')
                        para_count_display = gr.Markdown('📄 Paragraphs: 0', elem_classes='props')
                    gr.Markdown('Created by Fandresena & Vicent', elem_classes='pcredits')

                # RIGHT: Controls (30%)
                with gr.Column(elem_classes='column-shadow pcolumn-shadow'):
                    gr.Markdown('Process Status', elem_classes='cheading')

                    draft_check = gr.Checkbox(
                        label='Draft',
                        container=False,
                        elem_classes='pcheck_box',
                        interactive=False
                    )
                    plagarism_check = gr.Checkbox(
                        label='Quality Review',
                        container=False,
                        elem_classes='pcheck_box',
                        interactive=False
                    )
                    final_check = gr.Checkbox(
                        label='Final',
                        container=False,
                        elem_classes='pcheck_box',
                        interactive=False
                    )

                    status_info = gr.Markdown(
                        '<div class="stat_info">Ready to generate</div>'
                    )

                    # Progress log
                    progress_log = gr.TextArea(
                        label="📋 Generation Log",
                        lines=10,
                        max_lines=15,
                        interactive=False
                    )

                    # Download buttons
                    gr.Markdown("### 💾 Download Options")
                    download_md_btn = gr.Button('Download Markdown', elem_classes='paction-button')
                    download_txt_btn = gr.Button('Download Plain Text', elem_classes='paction-button-green')

                    download_file = gr.File(label="", visible=False)

    # -------------------------
    # Event Handlers
    # -------------------------

    # Generate button
    generate_click = action_button.click(
        fn=generate_with_streaming,
        inputs=[prompt, use_search, temperature_slider],
        outputs=[
            progress_log,
            preview,
            word_count_display,
            para_count_display,
            status_info,
            draft_check,
            plagarism_check,
            final_check
        ],
        show_progress=True
    )

    # Store topic and essay after generation
    generate_click.then(
        fn=lambda p, prev: (p, prev),
        inputs=[prompt, preview],
        outputs=[topic_state, essay_state]
    )

    # Download Markdown
    download_md_btn.click(
        fn=save_markdown,
        inputs=[essay_state, topic_state],
        outputs=[download_file]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[download_file]
    )

    # Download Plain Text
    download_txt_btn.click(
        fn=save_txt,
        inputs=[essay_state, topic_state],
        outputs=[download_file]
    ).then(
        fn=lambda: gr.update(visible=True),
        outputs=[download_file]
    )

# -------------------------
# Launch
# -------------------------


def launch():
    """Launch the app"""
    report_app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )


if __name__ == "__main__":
    launch()
