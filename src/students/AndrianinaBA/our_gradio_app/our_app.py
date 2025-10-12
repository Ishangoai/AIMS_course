
# ============================================================
#  Project      : Text Analyzer Web App
#  Framework    : Gradio
#  Language     : Python
#
#  Description  : A simple interactive application that process text  and also analyzes
#                 user-inputted text and returns metrics such as word count, character count, and more.
#                 As a bonus, we added a wordart option to generate wordart images from the text.
#
#  Authors      : Vicent & Andrianina
#  Assignment   : ML Ops Assignment 1
#
# ============================================================

from io import BytesIO

import gradio as gr
import requests
from PIL import Image

from .utils.text_analyzer import case_converter, text_analyzer, text_reverser_all_character, text_reverser_word

style = ''
with open("our_gradio_app/utils/style.css") as f:
    style = f.read()


def update_display(input_text, case_option, reverse_all, reverse_word):
    output_text = input_text
    if reverse_all:
        output_text = text_reverser_all_character(output_text, reverse_all)
    elif reverse_word:
        output_text = text_reverser_word(output_text, reverse_word)

    return case_converter(output_text, case_option)


def apply_events(input_widgets, input_options, output_options):
    for widget in input_widgets:
        widget.change(fn=update_display, inputs=input_options, outputs=output_options)


def update_statistics(input_text):
    return text_analyzer(input_text)


def update_image(input_text):
    url = make_url(input_text) if input_text != '' else make_url()
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    return img


def make_url(text='Click button to generate word art ....'):
    url = f"https://quickchart.io/wordcloud?format=png&backgroundColor=white&maxNumWords=200&fontScale=25&width=500&height=500&text={text}"

    return url


def delete_text():
    url = make_url()
    return '', url


with gr.Blocks(css=style) as our_gradio_instance:
    """
    We thought it would be nice to have a title and small description for our app here.
    """
    gr.Markdown("# AIMS Course: Text Analyzer by Vicent and Andrianina", elem_classes='app-title')
    gr.Markdown("This app helps you understand your text better. Try out 😎", elem_classes="app-intro")

    input_text = gr.Textbox(label="", lines=3, placeholder="Go head, type something here...", elem_classes="text_input")

    with gr.Tab("Case Converter"):
        gr.Markdown("Select the text case that you would like and see the change below.")

        case_radio_button = gr.Radio(choices=["Upper Case", "Lower Case", "Title Case"]
                                , value="Upper Case"
                                , label=""
                               )

    with gr.Tab("Text Reverser"):
        gr.Markdown("Reverse the input text.")
        reverse_word = gr.Checkbox(label="Reverse Word Order", value=False)
        reverse_all = gr.Checkbox(label="Reverse all characters", value=False)

    with gr.Tab("Text Analyzer"):
        gr.Markdown("Text statistics")
        with gr.Row():
            word_count_output = gr.Textbox(label="Word Count")
            char_count_output = gr.Textbox(label="Character Count")
            average_count_output = gr.Textbox(label="Average Characters per Word")

    with gr.Tab("WordArt"):
        generate_button = gr.Button(value="Generate", elem_classes='custom-button', size="sm")
        with gr.Row():
            default_text = "Click button to generate word art."
            placeholder_url = make_url()
            image = gr.Image(height=300)
            try:
                image.value = placeholder_url
            except Exception:
                print("Image loading failed!")

    with gr.Blocks():

        output_text_box = gr.Textbox(label="Result",
                                     placeholder="Nothing to display...",
                                     lines=3, elem_classes='result'
                                     )

        delete_button = gr.Button("Clear Text", elem_classes='custom-button-green')

        # Here we are updating the result based on the input typed
        input_options = [input_text, case_radio_button, reverse_all, reverse_word]
        input_widgets = [
             input_text,
             case_radio_button,
             reverse_all,
             reverse_word
        ]

        generate_button.click(fn=update_image, inputs=output_text_box, outputs=image)

        apply_events(input_widgets, input_options, output_text_box)
        input_text.change(fn=update_statistics, inputs=input_text, outputs=[word_count_output,
                                                                           char_count_output,
                                                                           average_count_output]
                                                                            )

        delete_button.click(fn=delete_text, inputs=None, outputs=[input_text, image])

if __name__ == "__main__":
    our_gradio_instance.launch()
