import os

import gradio as gr

# 👉 Fix: Rediriger le dossier temporaire de Gradio vers ton répertoire personnel
os.environ["GRADIO_TEMP_DIR"] = os.path.expanduser("~/.gradio_tmp")
os.makedirs(os.environ["GRADIO_TEMP_DIR"], exist_ok=True)


# --- Utility Functions ---


def process_text(text, case_option, reverse_words, reverse_chars):
    if not text:
        return ""

    processed = text

    # Case conversion
    if case_option == "Uppercase":
        processed = processed.upper()
    elif case_option == "Lowercase":
        processed = processed.lower()
    elif case_option == "Title Case":
        processed = processed.title()

    # Text reversing
    if reverse_words:
        processed = " ".join(processed.split()[::-1])
    if reverse_chars:
        processed = processed[::-1]

    return processed


def analyze_text(text):
    if not text.strip():
        return 0, 0, 0.0

    words = text.split()
    word_count = len(words)
    char_count = len(text)
    avg_word_len = sum(len(w) for w in words) / word_count if word_count else 0

    return word_count, char_count, round(avg_word_len, 2)


# --- Main Function ---


def text_tool(text, case_option, reverse_words, reverse_chars):
    processed = process_text(text, case_option, reverse_words, reverse_chars)
    word_count, char_count, avg_word_len = analyze_text(text)

    stats = (
        f"**Word Count:** {word_count}\n"
        f"**Character Count:** {char_count}\n"
        f"**Average Word Length:** {avg_word_len}"
    )

    return processed, stats


# --- UI Layout ---
with gr.Blocks(css="body {background: #f2f7ff;}") as demo:
    gr.Markdown("## 🧰 Text Utility App\nPerform various text operations easily!")

    with gr.Row():
        text_input = gr.Textbox(label="Enter text here", lines=5, placeholder="Type or paste text...")

    with gr.Accordion("Text Operations", open=True):
        with gr.Tabs():
            with gr.Tab("Case Converter"):
                case_option = gr.Radio(
                    ["Uppercase", "Lowercase", "Title Case"],
                    label="Choose Case Conversion",
                    value="Uppercase"
                )

            with gr.Tab("Text Reverser"):
                reverse_words = gr.Checkbox(label="Reverse Word Order", value=False)
                reverse_chars = gr.Checkbox(label="Reverse All Characters", value=False)

            with gr.Tab("Text Analyzer"):
                stats_output = gr.Markdown("**Word Count:** 0\n**Character Count:** 0\n**Average Word Length:** 0.0")

    with gr.Row():
        result_output = gr.Textbox(label="Output", lines=5, interactive=False)

    with gr.Row():
        clear_btn = gr.Button("🧹 Clear")

    # --- Interactivity ---
    text_input.change(
        text_tool,
        inputs=[text_input, case_option, reverse_words, reverse_chars],
        outputs=[result_output, stats_output]
    )

    case_option.change(
        text_tool,
        inputs=[text_input, case_option, reverse_words, reverse_chars],
        outputs=[result_output, stats_output]
    )

    reverse_words.change(
        text_tool,
        inputs=[text_input, case_option, reverse_words, reverse_chars],
        outputs=[result_output, stats_output]
    )

    reverse_chars.change(
        text_tool,
        inputs=[text_input, case_option, reverse_words, reverse_chars],
        outputs=[result_output, stats_output]
    )

    clear_btn.click(
        fn=lambda: ("", "", "**Word Count:** 0\n**Character Count:** 0\n**Average Word Length:** 0.0"),
        outputs=[text_input, result_output, stats_output]
    )

# --- Launch ---
demo.launch()
