import gradio as gr

from graph import agent

def llm_call(topic: str):
    messages = {"role": "user", "content": topic}
    response = agent.invoke({"messages": messages})
    return response["messages"][-1].content


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
