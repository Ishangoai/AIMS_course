import gradio as gr


def analyze(text: str) -> str:
    char_count = len(text)
    word_count = len(text.split(" "))
    avg_word_length = char_count / word_count if word_count > 0 else 0
    display = f"""
    Character Count: {char_count}
    Word Count: {word_count}
    Average Word Length: {avg_word_length:.2f}
    """
    return display


def convert(text, case) -> str:
    if case == "upper":
        return text.upper()
    else:
        return text.lower()


def reverse(text: str, reverse_order: bool, reverse_chars: bool) -> str:
    if reverse_order:
        text = " ".join(reversed(text.split(" ")))
    if reverse_chars:
        text = text[::-1]
    return text


with gr.Blocks() as demo:

    textbox = gr.Textbox(label="Enter your name")
    output_box = gr.Textbox(label="Output", interactive=False)

    with gr.Accordion("Case converter", open=False):
        converter = gr.Radio(choices=["upper", "lower"], label="Case")
        converter.change(fn=convert, inputs=[textbox, converter], outputs=output_box)
    with gr.Accordion("Reverse Text", open=False):
        reverse_order = gr.Checkbox(label="Reverse Word Order")
        reverser_chars = gr.Checkbox(label="Reverse All Characters")
        textbox.change(fn=reverse, inputs=[textbox, reverse_order, reverser_chars], outputs=output_box)
    with gr.Accordion("Text analyzer"):
        analyzer = gr.Textbox(label="Word Count, Character Count, Average word length", interactive=True)
        textbox.change(fn=analyze, inputs=textbox, outputs=analyzer)


if __name__ == "__main__":
    demo.launch()
