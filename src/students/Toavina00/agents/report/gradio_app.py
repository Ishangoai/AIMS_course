import gradio as gr
from graph import generate_report


def llm_call(topic: str):
    return generate_report(topic)


with gr.Blocks(css=".gradio-container {width: 70%; margin: auto}") as iface:
    gr.Markdown("# Report writter")
    gr.Markdown("## Write a report about a MLOps related topic")

    with gr.Row():
        topic = gr.Text(label="Choose the topic", max_length=25, scale=3)
        generate = gr.Button("Generate Report", scale=1)

    result = gr.Textbox(lines=50, label="Report")

    generate.click(llm_call, inputs=[topic], outputs=[result])

if __name__ == "__main__":
    iface.launch()
