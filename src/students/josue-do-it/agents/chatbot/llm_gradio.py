import gradio as gr
from agents.chatbot.graph import graph


def chat_with_graph(message, history):
    """Pass the full chat history to the graph and return the assistant's reply."""
    # With type="messages", history is already in the correct format
    messages = history.copy() if history else []
    messages.append({"role": "user", "content": message})

    # Invoke the graph with the messages and get the response
    response = graph.invoke({"messages": messages})

    # Extract the assistant's reply from the response
    return response["messages"][-1].content


with gr.Blocks() as llm_chat:
    gr.Markdown("# AI general assistant.\nThis bot uses search tools to find current information.")
    gr.ChatInterface(fn=chat_with_graph, title="Simple AI Assistant", type="messages")


if __name__ == "__main__":
    llm_chat.launch()
