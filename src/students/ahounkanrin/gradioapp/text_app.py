import gradio as gr


def case_converter(text, case):
    if case == "Uppercase":
        return text.upper()
    elif case == "Lowercase":
        return text.lower()
    elif case == "Titlecase":
        return text.title()
    return text


def reverse_words_order(text, checkbox_checked):
    if checkbox_checked:
        words = text.split(" ")
        reversed_words = ""
        for x in words[::-1]:
            reversed_words = reversed_words + " " + x
        return reversed_words
    return text


def reverse_all_characters(text, checkbox_checked):
    if checkbox_checked:
        return text[::-1]
    return text


def word_counter(text):
    words = text.split(" ")
    return len(words)


def charater_counter(text):
    return len(text)


def average_word_length_counter(text):
    return charater_counter(text) // word_counter(text)


def reset(text):
    return text, None, False, False


with gr.Blocks() as text_app:
    gr.Markdown("# Text Manipulation App")

    with gr.Row():
        input_text = gr.Textbox(label="Enter your text here")

    with gr.Row():

        with gr.Tab("Case Converter"):

            with gr.Row():
                with gr.Column():
                    case_buttons = gr.Radio(choices=["Uppercase", "Lowercase", "Titlecase"],
                                            label="Chose an option", value=None)

        with gr.Tab("Text Reverser"):

            with gr.Row():

                with gr.Column():
                    reversed_words_checkbox = gr.Checkbox(label="Reverse Word Order")
                with gr.Column():
                    reversed_characters_checkbox = gr.Checkbox(label="Reverse All Characters")

        with gr.Tab("Text Analyzer"):

            with gr.Row():

                with gr.Column():
                    words_count = gr.Textbox(value="0", label="Word Count")
                with gr.Column():
                    character_count = gr.Textbox(value="0", label="Character Count")
                with gr.Column():
                    average_word_length = gr.Textbox(value="0", label="Average Word Length")

    with gr.Row():
        output_text = gr.Textbox(label="Output Text")

    with gr.Row():
        with gr.Column():
            clear_button = gr.Button("Clear")
        with gr.Column():
            reset_button = gr.Button("Reset")

    case_buttons.change(fn=case_converter, inputs=[input_text, case_buttons], outputs=output_text)
    clear_button.click(fn=lambda: None, inputs=None, outputs=output_text)
    reset_button.click(fn=reset, inputs=input_text,
                       outputs=[output_text, case_buttons, reversed_words_checkbox, reversed_characters_checkbox])

    reversed_words_checkbox.change(fn=reverse_words_order,
                                   inputs=[input_text, reversed_words_checkbox],
                                   outputs=output_text)
    reversed_characters_checkbox.change(fn=reverse_all_characters,
                                        inputs=[input_text, reversed_characters_checkbox],
                                        outputs=output_text)

    input_text.change(fn=word_counter, inputs=input_text, outputs=words_count)
    input_text.change(fn=charater_counter, inputs=input_text, outputs=character_count)
    input_text.change(fn=average_word_length_counter, inputs=input_text, outputs=average_word_length)
    input_text.change(fn=lambda x: x, inputs=input_text, outputs=output_text)
