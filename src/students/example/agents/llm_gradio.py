
import gradio as gr
from agents.graph import graph


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


with gr.Blocks() as llm_chat:
    gr.Markdown("# AI general assistant.\nYou can tell me latest info by using search tool")
    gr.ChatInterface(
        fn=chat_with_graph,
        title="Currency Assistant",
        description="Ask about any current information in natural language."
    )


if __name__ == "__main__":
    llm_chat.launch()
