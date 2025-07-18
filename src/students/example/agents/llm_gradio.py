
import gradio as gr

from graph import graph


def chat_with_graph(message, history):
    """Pass the full chat history to the graph and return the assistant's reply."""
    messages = []
    if history:
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
    messages.append({"role": "user", "content": message})
    inputs = {"messages": messages}
    result = graph.invoke(inputs)
    reply = result["messages"][-1].content
    return reply


def main():
    """Launch the Gradio UI for the currency assistant chat."""
    with gr.Blocks() as demo:
        gr.Markdown("# Global Currency Assistant.\nYou can tell me which currency you prefer to convert")
        gr.ChatInterface(
            fn=chat_with_graph,
            title="Currency Assistant",
            description="Ask about currency conversion and get the latest exchange rates."
        )
    demo.launch()

if __name__ == "__main__":
    main()
