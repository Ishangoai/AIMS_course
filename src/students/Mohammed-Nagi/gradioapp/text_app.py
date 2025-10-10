import gradio as gr


def case_converter(text, case_option):
    if not text:
        return ""
    
    if case_option == "Uppercase":
        text = text.upper()
    elif case_option == "Lowercase":
        text = text.lower()
    elif case_option == "Titlecase":
        text = text.title()
    
    return text


def text_reverser(text, reverse_word, reverse_char):
    if not text:
        return ""

    if reverse_word and not reverse_char:
        words = text.split()
        text = " ".join(reversed(words))
    elif reverse_char and not reverse_word:
        text = text[::-1]
    elif reverse_word and reverse_char:
        words = text.split()
        text = " ".join(reversed(words))[::-1]

    return text


def text_analyser(text):
    if not text:
        return "Words: 0\nCharacters: 0\nAverage Word Length: 0"
    
    char_count = len(text.replace(" ", ""))
    words = text.split()
    word_count = len(words)
    avg_word_len = round(sum(len(w) for w in words) / word_count, 2)

    return f"Words: {word_count}\nCharacters: {char_count}\nAverage Word Length: {avg_word_len}"


def process_text(text, case_option, reverse_word, reverse_char):
    text = case_converter(text, case_option)
    text = text_reverser(text, reverse_word, reverse_char)

    return text


def clear():
        clear = ""
        return [clear, clear]


with gr.Blocks() as text_app:
    gr.Markdown("# Text Editor App")

    input_text = gr.Textbox(label="Enter Your Text", lines=5, placeholder="type or paste text here...")
    
    with gr.Tabs():
        with gr.Tab("Case Converter"):
            case_option = gr.Radio(["Uppercase", "Lowercase", "Titlecase"], label="Choose Case", value="Uppercase")

        with gr.Tab("Text Reverser"):
            reverse_word = gr.Checkbox(label="Reverse Word Order")
            reverse_char = gr.Checkbox(label="Reverse All Characters")

    analysis_box = gr.Textbox(label="Analysis", interactive=False, lines=3)

    input_text.input(fn=text_analyser, inputs=input_text, outputs=analysis_box)

    output_text = gr.Textbox(label="Output Text", lines=5)

    run_button = gr.Button("Apply Changes")

    run_button.click(
        fn=process_text,
        inputs=[input_text, case_option, reverse_word, reverse_char],
        outputs=output_text
    )

    clear_button = gr.Button("Clear Output")

    clear_button.click(
        fn=clear,
        inputs=None,
        outputs=[output_text, analysis_box]
    )
