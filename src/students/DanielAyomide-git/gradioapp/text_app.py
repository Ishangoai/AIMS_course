# text_app.py
import gradio as gr  # Import the Scrabble module
from gradioapp.utils import text_app_utils


def process_text(text, case_option, reverse_words, reverse_chars):
    if not text:
        return ""

    # Case conversion
    if case_option == "Uppercase":
        text = text.upper()
    elif case_option == "Lowercase":
        text = text.lower()
    elif case_option == "Title Case":
        text = text.title()

    # Reversal options
    if reverse_words:
        text = " ".join(text.split()[::-1])
    if reverse_chars:
        text = text[::-1]

    return text


def text_stats(text):
    if not text:
        return {"Word Count": 0, "Character Count": 0, "Average Word Length": 0}
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    avg_word_len = sum(len(w) for w in words) / word_count if word_count else 0
    return {
        "Word Count": word_count,
        "Character Count": char_count,
        "Average Word Length": round(avg_word_len, 2)
    }


def clear_all():
    return "", "", {"Word Count": 0, "Character Count": 0, "Average Word Length": 0}


# ------------------- BUILD MAIN APP -------------------
with gr.Blocks(theme=gr.themes.Soft()) as text_app:
    gr.Markdown("# 🨁 Text Formatting Application and Scrabble-ish gamemode 🨃 \n"
                "### 🨂 Perform text operations and test your might against the computer 😎 🨀")

    # --- SIDE-BY-SIDE Input + Analyzer ---
    with gr.Row(equal_height=True):
        input_box = gr.Textbox(label="Enter your text here", lines=6, placeholder="Type or paste text...")

        # Live statistics next to input
        with gr.Column():
            gr.Markdown("### 📊 Live Text Analyzer")
            stats_output = gr.JSON(label="Live Statistics", value={
                "Word Count": 0, "Character Count": 0, "Average Word Length": 0
            })

    # --- Operations Accordion ---
    with gr.Accordion("Text Operations", open=True):
        with gr.Tab("Case Converter"):
            case_option = gr.Radio(
                ["Uppercase", "Lowercase", "Title Case"],
                label="Choose Case",
                value="Lowercase"
            )

        with gr.Tab("Text Reverser"):
            reverse_words = gr.Checkbox(label="Reverse Word Order")
            reverse_chars = gr.Checkbox(label="Reverse All Characters")

    # --- Output and Buttons ---
    output_box = gr.Textbox(label="Output", lines=5)
    with gr.Row():
        clear_btn = gr.Button("Clear", variant="stop")

    gr.Markdown("---")

    # --- Scrabble Section ---
    gr.Markdown("## 🮖 Mini Scrabble Game")
    gr.Markdown("Test your vocabulary by forming complex words from the given chracters!!")

    # 🧩 This is the correct way to include the Scrabble subapp
    scrabble_game = text_app_utils.scrabble_demo  # export name in text_app_utils.py
    scrabble_game.render()  # Render the scrabble subapp inside this main app

    # --- Event connections ---
    input_box.change(process_text, inputs=[input_box, case_option, reverse_words, reverse_chars]
                        , outputs=output_box)
    case_option.change(process_text, inputs=[input_box, case_option, reverse_words, reverse_chars]
                        , outputs=output_box)
    reverse_words.change(process_text, inputs=[input_box, case_option, reverse_words, reverse_chars]
                        , outputs=output_box)
    reverse_chars.change(process_text, inputs=[input_box, case_option, reverse_words, reverse_chars]
                        , outputs=output_box)

    input_box.change(text_stats, inputs=input_box, outputs=stats_output)
    clear_btn.click(clear_all, outputs=[input_box, output_box, stats_output])
