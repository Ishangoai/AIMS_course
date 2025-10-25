import os
import re
import sys

import gradio as gr

# Add parent directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.writer.agents import approve_report, request_changes, save_report, start_generation


def calculate_text_stats(text):
    """Calculate text statistics"""
    if not text or not text.strip():
        return "0", "0", "0", "0"

    # Word count
    words = len(text.split())

    # Character count (including spaces)
    chars = len(text)

    # Sentence count (simple approximation)
    sentences = len(re.findall(r'[.!?]+', text))

    # Paragraph count
    paragraphs = len([p for p in text.split('\n\n') if p.strip()])
    if paragraphs == 0:
        paragraphs = len([p for p in text.split('\n') if p.strip()])

    return str(words), str(chars), str(sentences), str(paragraphs)


with gr.Blocks(title='LangGraph Report Writer - 1000 Words', theme=gr.themes.Soft()) as writer_app:
    gr.Markdown('''
    # 🧠 LangGraph Agentic Report Writer
    **Multi-Agent Pipeline**: Research → Write → Fact-Check → **You Review** → Refine
    ''')

    session_id_state = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=2):
            topic_input = gr.Textbox(
                label='📝 Topic',
                placeholder='e.g., Quantum Computing, Blockchain, AGI',
                lines=2
            )
        with gr.Column(scale=1):
            generate_btn = gr.Button('🚀 Generate Report', variant='primary', size='lg')

    # Agent Progress Tracking
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🔄 Agent Pipeline Progress")
            progress_html = gr.HTML(value="""
                <div style="display: flex; gap: 10px; padding: 15px; background: #f5f5f5; border-radius: 8px;">
                    <div id="agent-research" style="flex: 1; padding: 10px; background: #e0e0e0;
                         border-radius: 5px; text-align: center;">
                        🔍 Research<br><small>Waiting...</small>
                    </div>
                    <div id="agent-write" style="flex: 1; padding: 10px; background: #e0e0e0;
                         border-radius: 5px; text-align: center;">
                        ✍️ Write<br><small>Waiting...</small>
                    </div>
                    <div id="agent-factcheck" style="flex: 1; padding: 10px; background: #e0e0e0;
                         border-radius: 5px; text-align: center;">
                        🔎 Fact-Check<br><small>Waiting...</small>
                    </div>
                    <div id="agent-review" style="flex: 1; padding: 10px; background: #e0e0e0;
                         border-radius: 5px; text-align: center;">
                        👤 Review<br><small>Waiting...</small>
                    </div>
                    <div id="agent-refine" style="flex: 1; padding: 10px; background: #e0e0e0;
                         border-radius: 5px; text-align: center;">
                        ✨ Refine<br><small>Waiting...</small>
                    </div>
                </div>
            """)

    status_box = gr.Textbox(label='📊 Status', lines=2, interactive=False)

    # Text Statistics Panel
    with gr.Row():
        with gr.Column(scale=1):
            word_count_display = gr.Textbox(
                label='📏 Word Count',
                value='0',
                interactive=False
            )
        with gr.Column(scale=1):
            char_count_display = gr.Textbox(
                label='📝 Character Count',
                value='0',
                interactive=False
            )
        with gr.Column(scale=1):
            sentence_count_display = gr.Textbox(
                label='📋 Sentence Count',
                value='0',
                interactive=False
            )
        with gr.Column(scale=1):
            paragraph_count_display = gr.Textbox(
                label='📄 Paragraph Count',
                value='0',
                interactive=False
            )

    report_editor = gr.Textbox(
        label='📄 Report (Editable)',
        lines=25,
        placeholder='Generated report appears here...'
    )

    feedback_box = gr.Textbox(
        label='💬 Your Feedback',
        placeholder='What would you like changed?',
        lines=4,
        visible=False
    )

    with gr.Row():
        approve_btn = gr.Button('✅ Approve', variant='primary', visible=False)
        changes_btn = gr.Button('🔄 Request Changes', variant='secondary', visible=False)

    download_btn = gr.Button('💾 Download', variant='primary', visible=False)
    download_file = gr.File(label='📥 File')

    gr.Markdown("""
    ---
    ### 📖 Instructions:
    1. Enter topic → Click "Generate Report"
    2. System generates report (950-1000 words)
    3. Review the report → Edit if needed
    4. **Approve** or provide **Feedback** for changes
    5. System refines while maintaining word limit
    6. Repeat until satisfied → Download
    """)

    # JavaScript for real-time text statistics
    report_editor.change(
        fn=lambda text: calculate_text_stats(text),
        inputs=[report_editor],
        outputs=[word_count_display, char_count_display, sentence_count_display, paragraph_count_display]
    )

    # Events

    generate_btn.click(
        start_generation,
        inputs=[topic_input],
        outputs=[
            status_box, report_editor, approve_btn, feedback_box, changes_btn,
            session_id_state, progress_html, word_count_display, char_count_display,
            sentence_count_display, paragraph_count_display
        ]
    )

    approve_btn.click(
        approve_report,
        inputs=[report_editor, session_id_state],
        outputs=[status_box, report_editor, download_btn, progress_html]
    )

    changes_btn.click(
        request_changes,
        inputs=[report_editor, feedback_box, session_id_state],
        outputs=[
            status_box, report_editor, download_btn, approve_btn, changes_btn,
            progress_html, word_count_display, char_count_display,
            sentence_count_display, paragraph_count_display
        ]
    )

    download_btn.click(save_report, inputs=[report_editor], outputs=[download_file])
