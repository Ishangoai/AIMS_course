import gradio as gr
from agents.ai_agent.agents.planner import PlannerAgent
from agents.ai_agent.state import create_initial_state, get_state_summary

agent = PlannerAgent()


def chat_with_graph(message, history):
    """Process essay writing requests through the multi-agent system for long-form content creation."""
    # With type="messages", history is already in the correct format
    messages = history.copy() if history else []
    messages.append({"role": "user", "content": message})

    # Create initial state for the essay system
    initial_state = create_initial_state(
        topic=message,
        target_word_count=2000
    )

    # Add the conversation history to the state
    initial_state["messages"] = messages

    # Stream the processing steps
    yield "🔍 **Starting essay planning process...**\n\n"

    yield "📚 **Step 1: Conducting research on your topic...**\n\n"

    # Conduct research first to show progress
    try:
        research_result = agent.conduct_research(message)
        yield f"✅ **Research completed!**\n\n**Research Summary:**\n{research_result[:500]}...\n\n"
    except Exception as e:
        yield f"⚠️ **Research encountered issues:** {str(e)}\n\n"

    yield "📋 **Step 2: Creating detailed essay outline...**\n\n"

    # Use the full agent capabilities - plan the essay with research and outline
    try:
        result_state = agent.plan_essay(initial_state)

        # Show the outline
        if result_state.get("outline"):
            outline_text = "\n".join([f"{i + 1}. {section}" for i, section in enumerate(result_state["outline"])])
            yield f"✅ **Essay outline created!**\n\n**Proposed Essay Structure:**\n{outline_text}\n\n"

        # Get comprehensive summary
        yield "📊 **Step 3: Generating comprehensive analysis...**\n\n"
        summary = get_state_summary(result_state)
        yield f"🎯 **Planning Complete!**\n\n{summary}"

    except Exception as e:
        yield f"❌ **Planning failed:** {str(e)}\n\nFalling back to basic research results."


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
