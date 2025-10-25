import re

import gradio as gr

from .agent import citation_generator, drafting_process

custom_css = ""
with open('report_agent_app/app.css') as f:
    custom_css = f.read()
    f.close()


def markdown_to_plaintext(markdown_text: str) -> str:
    """Convert Markdown to readable plain text using only built-ins."""
    if not markdown_text or not markdown_text.strip():
        return ""  # return empty string instead of None

    text = markdown_text.replace('\r\n', '\n').replace('\r', '\n')
    lines = text.split('\n')
    output_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            output_lines.append("")
            continue

        # Headings
        if stripped.startswith("### "):
            heading = stripped[4:].strip()
            output_lines.append(f"{heading}\n{'-' * len(heading)}")
            continue
        elif stripped.startswith("## "):
            heading = stripped[3:].strip()
            output_lines.append(f"{heading}\n{'=' * len(heading)}")
            continue
        elif stripped.startswith("# "):
            heading = stripped[2:].strip().upper()
            output_lines.append(f"{heading}\n{'=' * len(heading)}")
            continue

        # Lists
        if stripped.startswith(("- ", "* ")):
            stripped = "• " + stripped[2:].strip()

        # Inline formatting
        stripped = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped)  # bold
        stripped = re.sub(r"\*(.*?)\*", r"\1", stripped)      # italic
        stripped = re.sub(r"__(.*?)__", r"\1", stripped)
        stripped = re.sub(r"_(.*?)_", r"\1", stripped)
        stripped = re.sub(r"`(.*?)`", r"'\1'", stripped)

        output_lines.append(stripped)

    return "\n".join(output_lines).strip()


def convert_and_save_report(markdown_text):
    """Convert markdown → plain text and save to .txt for Gradio download."""
    plain_text = markdown_to_plaintext(markdown_text)
    file_path = "report.txt"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(plain_text or "No content provided.")
    return file_path


def convert_and_save_references(markdown_text):
    """Convert markdown → plain text and save to .txt for Gradio download."""
    citation_text = citation_generator(markdown_text)
    plain_text = markdown_to_plaintext(citation_text)
    file_path = "references.txt"

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(plain_text or "No content provided.")
    return file_path


with gr.Blocks(css=custom_css) as report_app:
    with gr.Row(elem_classes='main_row'):
        with gr.Column(elem_classes='main_column'):
            gr.Markdown('ML Ops ReportGen Pro', elem_classes='app_title')
            with gr.Row(elem_classes='top_row'):
                prompt = gr.Textbox(container=False, elem_classes='prompt_area',
                placeholder='''Eg. report about agentic systems for future.''')
                action_button = gr.Button('Generate', elem_classes='action-button')

            with gr.Row():
                with gr.Column(elem_classes='column-shadow prcolumn'):
                    heading = gr.Markdown('Preview', elem_classes='pheading')
                    preview = gr.TextArea(container=False, label='', elem_classes='preview')
                    with gr.Row():
                        word_count = gr.Markdown('', elem_classes='props')

                    gr.Markdown('Created by Frandresena & Vicent', elem_classes='pcredits')

                with gr.Column(elem_classes='column-shadow pcolumn-shadow'):
                    status_info = gr.Markdown('Click generate to try!', elem_classes='stat_info')

                    reset = gr.Button('Reset', elem_classes='paction-button-green')

                    download = gr.Button('Download Txt', elem_classes='paction-button')
                    file_output_report = gr.File(label="Generate Plain Text")

                    download_refs = gr.Button('Generate References', elem_classes='paction-button')
                    file_output_refs = gr.File(label="Download References")

                    download.click(fn=convert_and_save_report, inputs=preview, outputs=file_output_report)
                    download_refs.click(fn=convert_and_save_references, inputs=preview, outputs=file_output_refs)
                    def reset_app():
                        return '', 'Generate', 'Click generate to again!', '', ''
                    reset.click(fn=reset_app, outputs=[prompt, action_button, status_info, word_count, preview])

    def start_draft(prompt, action, preview):
        words = ''
        content = ''
        if (action == "Revise Draft"):
            if (preview == ''):
                return '', '', 'Please type comment or prompt also in prompt area!', "Generate", ''
            content, words = drafting_process(preview)
        elif (action == "Generate"):
            content, words = drafting_process(prompt)
        return content, words, "Report ready! Please review and comment in prompt area if revision needed...",
        "Revise Draft", ''

    action_button.click(fn=start_draft, inputs=[prompt, action_button, preview], outputs=[preview, word_count,
    status_info, action_button, prompt])
