# File: app.py
"""
Gradio interface for Text Processing App
Location: khadijaedarzi9/gradio_app/app.py
This file contains only UI code and imports functions from operations.py
"""

import datetime

import gradio as gr
from gradioapp.utils.operations import (
    FileExporter,
    analyze_text,
    convert_text,
    correct_text,
    generate_word_cloud,
    reverse_text,
    save_as_txt,
)


class TextStyler:
    """Apply styling to text based on font, size, and type."""

    FONT_FAMILIES = {
        "Arial": "Arial, sans-serif",
        "Times New Roman": "'Times New Roman', serif",
        "Courier": "'Courier New', monospace"
    }

    @staticmethod
    def apply_style(text, font, size, text_type):
        """Apply HTML styling to text."""
        if not text:
            return ""

        font_family = TextStyler.FONT_FAMILIES.get(font, "Arial, sans-serif")
        font_size = f"{size}px"

        # Apply text type styling
        if text_type == "Title":
            weight = "bold"
            transform = "uppercase"
        elif text_type == "Subtitle":
            weight = "600"
            transform = "capitalize"
        else:  # Body
            weight = "normal"
            transform = "none"

        style = (
            f"font-family: {font_family}; "
            f"font-size: {font_size}; "
            f"font-weight: {weight}; "
            f"text-transform: {transform};"
        )

        return f'<div style="{style}">{text}</div>'


def build_interface():  # noqa: C901
    """Construct and return the Gradio Blocks interface."""
    # theme = gr.themes.Soft(
    #     primary_hue="indigo",
    #     neutral_hue="slate",
    #     font=["Inter", "sans-serif"],
    #     radius_size="lg",
    # )

    css = """
    .word-cloud-html { min-height: 200px; }
    .styled-text { padding: 15px; border-radius: 8px;
                   background-color: #f8f9fa; }
    """

    with gr.Blocks(css=css) as interface:
        gr.Markdown("# Text Processing App")
        gr.Markdown(
            "Refine, convert, analyze and export your text."
        )

        with gr.Tabs():
            # ============== CORRECTION TAB ==============
            with gr.Tab("Correction"):
                with gr.Row():
                    font_dropdown = gr.Dropdown(
                        label="Font",
                        choices=["Arial", "Times New Roman", "Courier"],
                        value="Arial",
                        interactive=True,
                    )
                    size_dropdown = gr.Dropdown(
                        label="Character size",
                        choices=["10", "12", "14", "16", "18"],
                        value="12",
                        interactive=True,
                    )
                    text_type = gr.Dropdown(
                        label="Text type",
                        choices=["Title", "Subtitle", "Body"],
                        value="Body",
                        interactive=True,
                    )

                with gr.Row():
                    with gr.Column():
                        input_text = gr.Textbox(
                            label="Enter your text below",
                            lines=10,
                            placeholder="Start typing... text will be corrected and styled automatically",
                        )
                        reset_btn = gr.Button("↺ Reset", size="lg")

                    with gr.Column():
                        corrected = gr.HTML(
                            value="<div style='padding: 15px; min-height: 300px; "
                                  "border: 1px solid #ddd; border-radius: 8px; "
                                  "background-color: #f8f9fa;'>"
                                  "<p style='text-align: center; color: #666;'>"
                                  "Your corrected and styled text will appear here as you type</p>"
                                  "</div>",
                            label="Corrected & Styled Text (Live Preview)"
                        )

                        # Hidden textbox to store plain corrected text
                        corrected_plain = gr.Textbox(
                            visible=False,
                            value=""
                        )

                def handle_live_correction(text, font, size, text_type):
                    """Correct text and apply styling in real-time."""
                    if not text or not text.strip():
                        empty_html = (
                            "<div style='padding: 15px; min-height: 300px; "
                            "border: 1px solid #ddd; border-radius: 8px; "
                            "background-color: #f8f9fa;'>"
                            "<p style='text-align: center; color: #666;'>"
                            "Your corrected and styled text will appear here as you type</p>"
                            "</div>"
                        )
                        return empty_html, ""

                    corrected_text = correct_text(text)
                    styled = TextStyler.apply_style(
                        corrected_text,
                        font,
                        size,
                        text_type
                    )
                    # Wrap in a nice container
                    styled_html = (
                        f"<div style='padding: 15px; min-height: 300px; "
                        f"border: 1px solid #ddd; border-radius: 8px; "
                        f"background-color: #f8f9fa;'>{styled}</div>"
                    )
                    return styled_html, corrected_text

                # Live update as user types
                input_text.change(
                    handle_live_correction,
                    inputs=[input_text, font_dropdown, size_dropdown, text_type],
                    outputs=[corrected, corrected_plain]
                )

                # Update style when dropdowns change
                for dropdown in [font_dropdown, size_dropdown, text_type]:
                    dropdown.change(
                        handle_live_correction,
                        inputs=[input_text, font_dropdown, size_dropdown, text_type],
                        outputs=[corrected, corrected_plain]
                    )

                reset_btn.click(
                    lambda: (
                        "",
                        "<div style='padding: 15px; min-height: 300px; "
                        "border: 1px solid #ddd; border-radius: 8px; "
                        "background-color: #f8f9fa;'>"
                        "<p style='text-align: center; color: #666;'>"
                        "Your corrected and styled text will appear here as you type</p></div>",
                        ""
                    ),
                    inputs=None,
                    outputs=[input_text, corrected, corrected_plain]
                )

            # ============== CONVERTER TAB ==============
            with gr.Tab("Converter"):
                with gr.Row():
                    with gr.Column():
                        conv_choice = gr.Dropdown(
                            label="Conversion Type",
                            choices=["Uppercase", "Lowercase", "Title Case"],
                            value="Uppercase",
                        )
                        conv_btn = gr.Button(
                            "Convert",
                            variant="primary",
                            size="lg"
                        )
                    with gr.Column():
                        conv_output = gr.Textbox(
                            label="Converted text",
                            lines=10
                        )

                def handle_conversion(corrected_text, mode):
                    """Handle text conversion."""
                    if not corrected_text:
                        return "Please correct text in the Correction tab first."
                    return convert_text(corrected_text, mode)

                conv_btn.click(
                    handle_conversion,
                    inputs=[corrected_plain, conv_choice],
                    outputs=conv_output
                )

            # ============== REVERSER TAB ==============
            with gr.Tab("Reverser"):
                with gr.Row():
                    with gr.Column():
                        rev_choice = gr.Radio(
                            label="Reverse Type",
                            choices=["words", "characters"],
                            value="words",
                            info="Choose how to reverse the text"
                        )
                        rev_btn = gr.Button(
                            "Reverse",
                            variant="primary",
                            size="lg"
                        )
                    with gr.Column():
                        rev_output = gr.Textbox(
                            label="Reversed text",
                            lines=10
                        )

                def handle_reversal(corrected_text, mode):
                    """Handle text reversal."""
                    if not corrected_text:
                        return "Please correct text in the Correction tab first."
                    return reverse_text(corrected_text, mode)

                rev_btn.click(
                    handle_reversal,
                    inputs=[corrected_plain, rev_choice],
                    outputs=rev_output
                )

            # ============== ANALYZER TAB ==============
            with gr.Tab("Analyzer & Word Cloud"):
                with gr.Row():
                    with gr.Column():
                        ana_btn = gr.Button(
                            "Analyze Text",
                            variant="primary",
                            size="lg"
                        )
                    with gr.Column():
                        ana_output = gr.Textbox(
                            label="Analysis Results",
                            lines=8,
                            interactive=False
                        )

                gr.Markdown("### Word Cloud Visualization")
                wc_output = gr.HTML(
                    value="<p style='text-align: center; color: #666;'>"
                          "Click 'Analyze Text' to generate word cloud</p>",
                    elem_classes="word-cloud-html"
                )

                def handle_analysis(corrected_text):
                    """Handle text analysis and word cloud generation."""
                    if not corrected_text:
                        no_text_msg = "Please correct text in the " \
                                     "Correction tab first."
                        no_text_html = "<p style='text-align: center; " \
                                      "color: #666;'>No text to analyze</p>"
                        return no_text_msg, no_text_html

                    analysis = analyze_text(corrected_text)
                    wc = generate_word_cloud(corrected_text)

                    ana_str = (
                        f"Total Words: {analysis['word_count']}\n\n"
                        f"Total Characters: {analysis['char_count']}\n\n"
                        f"Sentences: {analysis['sentence_count']}\n\n"
                        f"Average Word Length: "
                        f"{analysis['avg_word_length']} characters\n\n"
                        f"Most Frequent Word: '{analysis['most_common']}'"
                    )

                    return ana_str, wc

                ana_btn.click(
                    handle_analysis,
                    inputs=corrected_plain,
                    outputs=[ana_output, wc_output],
                )

            # ============== DOWNLOAD TAB ==============
            with gr.Tab("Download"):
                gr.Markdown("### Export all your processed text")

                with gr.Row():
                    format_dropdown = gr.Dropdown(
                        label="Choose Export Format",
                        choices=["txt"],
                        value="txt"
                    )
                    download_btn = gr.Button(
                        "Download File",
                        variant="primary",
                        size="lg"
                    )

                def handle_download(corrected_text, conv_text, rev_text,
                                   analysis_text, fmt):
                    """Handle file download and return the file directly."""
                    if not corrected_text:
                        # Return a dummy file with error message
                        error_path = FileExporter._get_filename("txt")
                        with open(error_path, "w", encoding="utf-8") as f:
                            f.write("ERROR: No content to export.\n")
                            f.write("Please correct text first in the Correction tab.")
                        return error_path

                    corrected_val = corrected_text or ""
                    converted_val = conv_text or "Not converted yet"
                    reversed_val = rev_text or "Not reversed yet"
                    analysis_val = analysis_text or "Not analyzed yet"

                    timestamp = datetime.datetime.now().strftime(
                        '%Y-%m-%d %H:%M:%S'
                    )

                    aggregated = (
                        f"╔{'═' * 68}╗\n"
                        f"║{' ' * 68}║\n"
                        f"║{'Text Processing App - COMPLETE EXPORT':^68}║\n"
                        f"║{' ' * 68}║\n"
                        f"╚{'═' * 68}╝\n\n"
                        f"Export Date: {timestamp}\n"
                        f"Format: {fmt.upper()}\n"
                        f"{'=' * 70}\n\n\n"
                        f"{'CORRECTED TEXT':^70}\n"
                        f"{'-' * 70}\n\n{corrected_val}\n\n\n"
                        f"{'=' * 70}\n\n\n"
                        f"{'CONVERTED TEXT':^70}\n"
                        f"{'-' * 70}\n\n{converted_val}\n\n\n"
                        f"{'=' * 70}\n\n\n"
                        f"{'REVERSED TEXT':^70}\n"
                        f"{'-' * 70}\n\n{reversed_val}\n\n\n"
                        f"{'=' * 70}\n\n\n"
                        f"{'TEXT ANALYSIS':^70}\n"
                        f"{'-' * 70}\n\n{analysis_val}\n\n\n"
                        f"{'=' * 70}\n"
                        f"End of document - Generated by Text Processing App\n"
                        f"{'=' * 70}\n"
                    )

                    try:
                        if fmt == "txt":
                            path = save_as_txt(aggregated)
                        return path
                    except Exception as e:
                        error_path = FileExporter._get_filename("txt")
                        with open(error_path, "w", encoding="utf-8") as f:
                            f.write("ERROR: Export failed\n")
                            f.write(f"Details: {str(e)}")
                        return error_path

                download_btn.click(
                    handle_download,
                    inputs=[
                        corrected_plain,
                        conv_output,
                        rev_output,
                        ana_output,
                        format_dropdown
                    ],
                    outputs=gr.File(label="Download will start automatically"),
                )

    return interface


if __name__ == "__main__":
    demo = build_interface()
    demo.launch()
