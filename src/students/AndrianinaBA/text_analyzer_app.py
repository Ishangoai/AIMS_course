

import gradio as gr
from gradioapp.utils.text_analyzer import case_converter, text_analyzer, text_reverser_all_character, text_reverser_word

with gr.Blocks(css="body {background: #f2f7ff;}") as text_analyzer_app_instance:
    gr.Markdown("# AIMS Course: Text Analyzer by Vicent and Andrianina")
    gr.Markdown("This app provides various text analysis functionalities.")
    input_text = gr.Textbox(label="Input Text", lines=3, placeholder="Enter text here...")

    with gr.Tab("Case Converter"):
        """
        For the first tab, where we select the case conversion option.
        """
        gr.Markdown("Convert text to different cases.")
        case_option = gr.Radio(choices=["Upper Case", "Lower Case", "Title Case"]
                                , value="Upper Case"
                                , label="Select Case"
                               )

    with gr.Tab("Text Reverser"):
        """
        For the second tab, where we select the text reversal options.
        """
        gr.Markdown("Reverse the input text.")
        reverse_word = gr.Checkbox(label="Reverse Word Order", value=False)
        reverse_all = gr.Checkbox(label="Reverse all characters", value=False)

    with gr.Tab("Text Analyzer"):
        """
        For the third tab, where we analyze the text."""
        gr.Markdown("Analyze the input text for word and character count.")
        analyze_button = gr.Button("Analyze")
        word_count_output = gr.Textbox(label="Word Count")
        char_count_output = gr.Textbox(label="Character Count")
        average_count_output = gr.Textbox(label="Average Characters per Word")

        analyze_button.click(fn=text_analyzer, inputs=input_text, outputs=[word_count_output,
                                                                             char_count_output,
                                                                             average_count_output])

    with gr.Blocks():
        gr.Markdown("## Your actions here")
        def delete_text():
            print("Deleting the text...")
            return

        def finally_process_text(input_text_no_mod, case_option, reverse_word, reverse_all):
            if reverse_all:
                output_text = text_reverser_all_character(input_text_no_mod, reverse_all)
            elif reverse_word:
                output_text = text_reverser_word(input_text_no_mod, reverse_word)
            else:
                output_text = input_text_no_mod
            return case_converter(output_text, case_option)

        result_button = gr.Button("Result")
        output_text_box = gr.Textbox(value="Your processed text here", lines=3)
        result_button.click(fn=finally_process_text, inputs=[input_text,
                                                              case_option,
                                                              reverse_word,
                                                              reverse_all],
                                                              outputs=output_text_box)
        delete_button = gr.Button("Clear Text")
        delete_button.click(fn=delete_text, inputs=None, outputs=input_text)

if __name__ == "__main__":
    text_analyzer_app_instance.launch()
