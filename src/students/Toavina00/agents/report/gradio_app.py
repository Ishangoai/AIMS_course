import gradio as gr
from agents.report.graph import generate_report


def llm_call(topic: str, temperature: float):
    return generate_report(topic, temperature)


with gr.Blocks(css=".gradio-container {width: 70%; margin: auto}") as iface:
    gr.Markdown("# Report writter")
    gr.Markdown("## Write a report about a MLOps related topic")

    with gr.Row():
        topic = gr.Text(label="Choose the topic", max_length=25, scale=3)
        generate = gr.Button("Generate Report", scale=1)

    slider = gr.Slider(0.0, 2.0, 1.0, label="Model Temperature")

    result = gr.Textbox(lines=50, label="Report")

    generate.click(llm_call, inputs=[topic, slider], outputs=[result])

if __name__ == "__main__":
    iface.launch()
