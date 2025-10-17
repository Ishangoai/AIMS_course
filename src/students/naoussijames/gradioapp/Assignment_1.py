import gradio as gr


# Case conversion functions
def to_upper(text):

    return text.upper()


def to_lower(text):

    return text.lower()


def to_title(text):

    return text.title()


# Text reversal functions
def reverse_words_order(text):

    reverse_words = text.split()[::-1]
    return " ".join(reverse_words)


def reverse_characters(text):

    reverse = [word[::-1] for word in text.split()]
    return " ".join(reverse)


def text_reverser(text, reverse_order, reverse_chars):

    result = text
    if reverse_order:
        result = reverse_words_order(result)
    if reverse_chars:
        result = reverse_characters(result)
    return result


# Text analyser function
def text_analyser(text):

    words = text.split()
    if not words:
        return "No text entered."
    char_count = sum(len(word) for word in words)
    word_count = len(words)
    avg_length = char_count / word_count
    return (
        f"Text Analysis:\n"
        f"- Character count: {char_count}\n"
        f"- Word count: {word_count}\n"
        f"- Average word length: {avg_length:.2f}"
    )


# Gradio Interface with Accordions
with gr.Blocks(title="Text Tools") as my_app:

    gr.Markdown("## Text Tools Suite\nUse the accordions below to transform or analyze your text.")
    text_input = gr.Textbox(label="Enter text", placeholder="Type something here...")

    # Accordion 1: Case Conversion
    with gr.Accordion("Case Conversion", open=False):
        with gr.Row():
            upper_btn = gr.Button("UPPERCASE")
            lower_btn = gr.Button("lowercase")
            title_btn = gr.Button("Title Case")

    # Accordion 2: Text Reversal
    with gr.Accordion("Text Reversal", open=False):
        reverse_order_checkbox = gr.Checkbox(label="Reverse word order")
        reverse_chars_checkbox = gr.Checkbox(label="Reverse characters in words")
        apply_reverse_btn = gr.Button("Apply Reversal")

    # Accordion 3: Text Analysis
    with gr.Accordion("Text Analysis", open=False):
        analyser_btn = gr.Button("Analyse Text")

    # Output box
    output_box = gr.Textbox(label="Results of your operations")

    # Connect button clicks to functions
    upper_btn.click(fn=to_upper, inputs=text_input, outputs=output_box)
    lower_btn.click(fn=to_lower, inputs=text_input, outputs=output_box)
    title_btn.click(fn=to_title, inputs=text_input, outputs=output_box)

    apply_reverse_btn.click(
        fn=text_reverser,
        inputs=[text_input, reverse_order_checkbox, reverse_chars_checkbox],
        outputs=output_box
    )

    analyser_btn.click(fn=text_analyser, inputs=text_input, outputs=output_box)
