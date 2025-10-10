import gradio as gr

# Define a simple function that takes a name and returns a greeting
def greet(name):
    return f"Hello, {name}!"

# Create a Gradio interface
# inputs: gr.Textbox() creates a text input field for the user
# outputs: gr.Textbox() displays the output as text
image_app = gr.Interface(fn=greet, inputs=gr.Textbox(label="Enter your name"), outputs="text")
