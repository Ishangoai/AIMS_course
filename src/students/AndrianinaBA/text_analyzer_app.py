

import gradio as gr
from gradioapp.utils.text_analyzer import case_converter, text_analyzer, text_reverser

# text = "Beams of fire sweep thorough my hearst, thrust of pain increasingly engaged, i am no one now only agony"
# res = case_converter(text=text, option="Title Case")
# print(res)

with gr.Blocks(css="body {background: #f2f7ff;}") as text_analyzer_app_instance:
    gr.Markdown("# AIMS Course: Text Analyzer")
    gr.Markdown("This app provides various text analysis functionalities.")

    with gr.Tab("Case Converter"):
        gr.Markdown("Convert text to different cases.")
        input_text = gr.Textbox(label="Input Text", lines=4, placeholder="Enter text here...")
        case_option = gr.Radio(choices=["Upper Case", "Lower Case", "Title Case"]
                                , value="Upper Case"
                                , label="Select Case"
                               )
        convert_button = gr.Button("Convert")
        output_text = gr.Textbox(label="Converted Text", lines=4)

        convert_button.click(fn=case_converter, inputs=[input_text, case_option], outputs=output_text)

    with gr.Tab("Text Reverser"):
        gr.Markdown("Reverse the input text.")
        reverse_input = gr.Textbox(label="Input Text", lines=4, placeholder="Enter text here...")
        reverse_button = gr.Button("Reverse")
        reversed_output = gr.Textbox(label="Reversed Text", lines=4)

        reverse_button.click(fn=text_reverser, inputs=reverse_input, outputs=reversed_output)

    with gr.Tab("Text Analyzer"):
        gr.Markdown("Analyze the input text for word and character count.")
        analyze_input = gr.Textbox(label="Input Text", lines=4, placeholder="Enter text here...")
        analyze_button = gr.Button("Analyze")
        word_count_output = gr.Textbox(label="Word Count")
        char_count_output = gr.Textbox(label="Character Count")
        average_count_output = gr.Textbox(label="Average Characters per Word")

        analyze_button.click(fn=text_analyzer, inputs=analyze_input, outputs=[word_count_output,
                                                                             char_count_output,
                                                                             average_count_output])
if __name__ == "__main__":
    text_analyzer_app_instance.launch()
