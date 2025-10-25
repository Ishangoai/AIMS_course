import re
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

# Load environment variables
load_dotenv()
API_KEY = "AIzaSyAnkAcME4ZQrwujP2u-NJ0vsmuwTHBVmA4"

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
print("✅ API Key loaded")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=API_KEY,
    temperature=0.1
)


# ---------------- State Definition ---------------- #
class ArticleState(TypedDict):
    topic: str
    outline: str
    article: str
    word_count: int
    iteration: int
    max_iterations: int


# ---------------- Prompts ---------------- #
outline_prompt = PromptTemplate.from_template("""
Write an outline for an article on {topic}. The outline should be bullet points
starting with Introduction and ending with Conclusion. There should be 7 points in total.
""")

content_prompt = PromptTemplate.from_template("""
You are a specialist in writing scientific articles. Write an article
that is of professional standard based on the outline below.
Outline:
{outline}

Your article MUST be exactly around 1000 words (between 950-1050 words).
""")

decrease_editor_prompt = PromptTemplate.from_template("""
You are a professional text editor. Edit the article below by reducing the
number of words to **exactly between 950-1050 words** while preserving meaning and flow.

Current word count: {word_count}
Target: 950-1050 words

-----------------------------------------------------------------
{article}
-----------------------------------------------------------------

Return ONLY the edited article, no explanations.
""")

increase_editor_prompt = PromptTemplate.from_template("""
You are a professional text editor. Edit the article below by increasing the
number of words to **exactly between 950-1050 words** while maintaining clarity and coherence.

Current word count: {word_count}
Target: 950-1050 words

-----------------------------------------------------------------
{article}
-----------------------------------------------------------------

Return ONLY the edited article, no explanations.
""")

# ---------------- Chains ---------------- #
outline_chain = outline_prompt | llm | StrOutputParser()
content_chain = content_prompt | llm | StrOutputParser()
decrease_editor_chain = decrease_editor_prompt | llm | StrOutputParser()
increase_editor_chain = increase_editor_prompt | llm | StrOutputParser()


# ---------------- Utility Functions ---------------- #
def count_words(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text))


# ---------------- Node Functions ---------------- #
def generate_outline(state: ArticleState) -> ArticleState:
    """Generate article outline"""
    print(f"📝 Generating outline for: {state['topic']}")
    outline = outline_chain.invoke({"topic": state["topic"]})
    state["outline"] = outline
    return state


def generate_article(state: ArticleState) -> ArticleState:
    """Generate article from outline"""
    print("✍️ Generating article...")
    article = content_chain.invoke({"outline": state["outline"]})
    state["article"] = article
    state["word_count"] = count_words(article)
    state["iteration"] = 0
    print(f"📊 Initial word count: {state['word_count']}")
    return state


def check_word_count(state: ArticleState) -> Literal["valid", "too_long", "too_short", "max_iterations"]:
    """Check if word count is within acceptable range"""
    wc = state["word_count"]

    # Check if max iterations reached
    if state["iteration"] >= state["max_iterations"]:
        print(f"⚠️ Max iterations ({state['max_iterations']}) reached")
        return "max_iterations"

    # Check word count range
    if 950 <= wc <= 1050:
        print(f"✅ Word count valid: {wc}")
        return "valid"
    elif wc > 1050:
        print(f"🔻 Article too long: {wc} words")
        return "too_long"
    else:
        print(f"🔺 Article too short: {wc} words")
        return "too_short"


def decrease_length(state: ArticleState) -> ArticleState:
    """Reduce article length"""
    print(f"📉 Reducing article length (iteration {state['iteration'] + 1})...")
    edited = decrease_editor_chain.invoke({
        "article": state["article"],
        "word_count": state["word_count"]
    })
    state["article"] = edited
    state["word_count"] = count_words(edited)
    state["iteration"] += 1
    print(f"📊 New word count: {state['word_count']}")
    return state


def increase_length(state: ArticleState) -> ArticleState:
    """Increase article length"""
    print(f"📈 Increasing article length (iteration {state['iteration'] + 1})...")
    edited = increase_editor_chain.invoke({
        "article": state["article"],
        "word_count": state["word_count"]
    })
    state["article"] = edited
    state["word_count"] = count_words(edited)
    state["iteration"] += 1
    print(f"📊 New word count: {state['word_count']}")
    return state


def finalize_article(state: ArticleState) -> ArticleState:
    """Final step - article is ready"""
    print(f"🎉 Article finalized with {state['word_count']} words")
    return state


# ---------------- Build Graph ---------------- #
def create_article_graph():
    workflow = StateGraph(ArticleState)

    # Add nodes
    workflow.add_node("generate_outline", generate_outline)
    workflow.add_node("generate_article", generate_article)
    workflow.add_node("decrease_length", decrease_length)
    workflow.add_node("increase_length", increase_length)
    workflow.add_node("finalize", finalize_article)

    # Set entry point
    workflow.set_entry_point("generate_outline")

    # Add edges
    workflow.add_edge("generate_outline", "generate_article")

    # Conditional edges based on word count
    workflow.add_conditional_edges(
        "generate_article",
        check_word_count,
        {
            "valid": "finalize",
            "too_long": "decrease_length",
            "too_short": "increase_length",
            "max_iterations": "finalize"
        }
    )

    workflow.add_conditional_edges(
        "decrease_length",
        check_word_count,
        {
            "valid": "finalize",
            "too_long": "decrease_length",
            "too_short": "increase_length",
            "max_iterations": "finalize"
        }
    )

    workflow.add_conditional_edges(
        "increase_length",
        check_word_count,
        {
            "valid": "finalize",
            "too_long": "decrease_length",
            "too_short": "increase_length",
            "max_iterations": "finalize"
        }
    )

    workflow.add_edge("finalize", END)

    return workflow.compile()


# ---------------- Main Function ---------------- #
def get_article(topic: str, max_iterations: int = 3):
    """Generate article using LangGraph"""
    graph = create_article_graph()

    # Initialize state
    initial_state = ArticleState(
        topic=topic,
        outline="",
        article="",
        word_count=0,
        iteration=0,
        max_iterations=max_iterations
    )

    # Run the graph
    result = graph.invoke(initial_state)

    return result["article"], result["word_count"]


# ---------------- Visualize Graph ---------------- #
def visualize_graph():
    """Generate and save the graph visualization"""
    graph = create_article_graph()

    try:
        # Generate PNG visualization
        png_data = graph.get_graph().draw_mermaid_png()
        with open("article_generation_graph.png", "wb") as f:
            f.write(png_data)
        print("✅ Graph visualization saved as 'article_generation_graph.png'")
    except Exception as e:
        print(f"⚠️ Could not generate PNG: {e}")
        print("Generating Mermaid diagram instead...")

        # Generate Mermaid diagram
        mermaid = graph.get_graph().draw_mermaid()
        with open("article_generation_graph.mmd", "w") as f:
            f.write(mermaid)
        print("✅ Mermaid diagram saved as 'article_generation_graph.mmd'")
        print("   View it at: https://mermaid.live/")


# ---------------- Run Test ---------------- #
if __name__ == "__main__":
    # Generate and visualize the graph
    print("🎨 Generating graph visualization...")
    visualize_graph()

    print("\n" + "=" * 60)
    print("📰 Generating Article...")
    print("=" * 60 + "\n")

    # Generate article
    final_article, word_count = get_article("MLOps")

    print("\n" + "=" * 60)
    print(f"📄 Final Article ({word_count} words):\n")
    print("=" * 60)
    print(final_article)
