import gradio as gr
from agents.ai_agent.agents.planner import PlannerAgent

agent = PlannerAgent()


def chat_with_graph(message, history):
    """Process essay writing requests through the multi-agent system for long-form content creation."""
    # With type="messages", history is already in the correct format
    messages = history.copy() if history else []
    messages.append({"role": "user", "content": message})

    # Use the latest user message for essay research and planning
    result = agent.conduct_research(message)  # TODO: change agentic planner to invoke graph
    return result


with gr.Blocks() as llm_chat:
    gr.Markdown(
        "# 📝 AI Essay Writing Assistant\n\n"
        "This specialized multi-agent system is designed to help you write long, cohesive essays "
        "(typically 2000+ words). The AI agents work collaboratively to research your topic, "
        "develop structured arguments, gather supporting evidence, and craft well-organized, "
        "comprehensive essays with proper flow and academic rigor."
    )
    gr.ChatInterface(
        fn=chat_with_graph,
        title="Long-Form Essay Writing Assistant",
        description=(
            "Provide your essay topic or research question, and let our AI agents help you "
            "create detailed, well-structured essays of 2000+ words with thorough research "
            "and compelling arguments."
        ),
        type="messages"
    )


if __name__ == "__main__":
    llm_chat.launch()
