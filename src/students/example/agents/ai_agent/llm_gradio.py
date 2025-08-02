import io

import gradio as gr
from agents.ai_agent.langchain_orchestrator import create_essay_orchestrator
from PIL import Image

# Create the multi-agent orchestrator
orchestrator = create_essay_orchestrator()


def chat_with_graph(message, history):
    """Process essay writing requests through the multi-agent system."""
    initial_state = orchestrator.create_initial_state(message)
    response = ""
    final_state = None

    for chunk in orchestrator.app.stream(initial_state, stream_mode="updates"):
        agent, state = chunk.popitem()
        final_state = state

        # Stream messages from the state
        messages = state.get("messages", [])
        if messages:
            latest_message = messages[-1]
            if isinstance(latest_message, dict):
                content = latest_message.get("content", "")
                response += f"**{agent.upper()}:** {content}\n\n"
                yield response

    # Add final essay when complete
    if final_state:
        final_essay = final_state.get("final_essay")
        if final_essay:
            response += f"**Final Essay:**\n\n{final_essay}\n"
            yield response


with gr.Blocks() as llm_chat:
    gr.Markdown(
        "# AI Essay Writing Assistant\n\n"
        "This multi-agent system helps you write comprehensive essays. "
        "Provide your topic and let the AI agents collaborate to create well-structured content."
    )

    with gr.Tabs():
        with gr.Tab("Chat"):
            gr.ChatInterface(
                fn=chat_with_graph,
                title="Essay Writing Assistant",
                description="Provide your essay topic for detailed, well-structured content.",
                type="messages",
            )

        with gr.Tab("Agent Graph"):
            gr.Markdown("## Multi-Agent Workflow Visualization")
            try:
                graph_bytes = orchestrator.app.get_graph().draw_mermaid_png()
                graph_image = Image.open(io.BytesIO(graph_bytes))
                gr.Image(value=graph_image, label="Agent Workflow Graph")
            except Exception as e:
                gr.Markdown(f"**Error generating graph:** {str(e)}")
                gr.Markdown("Graph visualization is not available.")

llm_chat.queue(max_size=10)
if __name__ == "__main__":
    llm_chat.launch()
