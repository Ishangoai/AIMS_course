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
        return [0, 0, 0]

    char_count = len(text.replace(" ", ""))
    words = text.split()
    word_count = len(words)
    avg_len = round(sum(len(w) for w in words) / word_count, 2)

    return word_count, char_count, avg_len


def process_and_analyse(text, case_option, reverse_word, reverse_char):
    text = case_converter(text, case_option)
    text = text_reverser(text, reverse_word, reverse_char)
    words, chars, length = text_analyser(text)

    return text, words, chars, length


def clear_all():
    return ["", "", 0, 0, 0]


# Text Editor App
with gr.Blocks() as text_app:
    gr.Markdown("# Text Editor App")

    # Input Box
    input_text = gr.Textbox(label="Enter Your Text", lines=5, placeholder="type or paste text here...")

    # Control Tabs
    with gr.Tabs():
        with gr.Tab("Case Converter"):
            case_option = gr.Radio(["Uppercase", "Lowercase", "Titlecase"], label="Choose Case", value="Uppercase")

        with gr.Tab("Text Reverser"):
            reverse_word = gr.Checkbox(label="Reverse Word Order")
            reverse_char = gr.Checkbox(label="Reverse Character Order")

    # Analysis Display
    with gr.Group():
        gr.Markdown(" **Text Analysis:**")

        with gr.Row():
            word_count = gr.Label(label="Word Count", value=0)
            char_count = gr.Label(label="Character Count", value=0)
            avg_len = gr.Label(label="Average Word Length", value=0)

    # Dynamic Analysis
    input_text.input(fn=text_analyser, inputs=input_text, outputs=[word_count, char_count, avg_len])

    # Output Box
    output_text = gr.Textbox(label="Output Text", lines=5)

    # Run Button
    run_button = gr.Button("Apply Changes")

    run_button.click(
        fn=process_and_analyse,
        inputs=[input_text, case_option, reverse_word, reverse_char],
        outputs=[output_text, word_count, char_count, avg_len]
    )

    # Clear Button
    clear_button = gr.Button("Clear")

    clear_button.click(
        fn=clear_all,
        inputs=None,
        outputs=[input_text, output_text, word_count, char_count, avg_len]
    )
