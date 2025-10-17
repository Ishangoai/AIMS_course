import gradio as gr

# case_convertor_interface=gr.Interface("s")
# text_reversor_interface=gr.Interface("s")
# text_analyzer_interface=gr.Interface("s")

with gr.Blocks(css="body {background: #f00;}") as our_app:
    with gr.Column():
        title = gr.Label("Adrianina & Vicent's GradioApp")
        with gr.Row():
            input_box = gr.Textbox(placeholder='Type here ...', label='')

        # tabs=gr.TabbedInteface([],['Case Convertor', 'Text Reversor', 'Text Analyzer'])

if __name__ == '__main__':
      our_app.launch()
