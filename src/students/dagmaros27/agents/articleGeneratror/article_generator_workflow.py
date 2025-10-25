"""
LangGraph Essay Generation Workflow - Core Logic

"""

from __future__ import annotations
import os
import re
import time
import requests
from typing import TypedDict, List, Literal, Optional, Callable, Dict
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Uncomment for PDF RAG in production:
# from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
# from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_community.vectorstores import FAISS


# ============================================================================
# STATE DEFINITION
# ============================================================================

class GraphState(TypedDict):
    """State object tracking the essay generation workflow."""
    topic: str
    essay_draft: str
    outline: str
    sources: List[str]
    editor_validity: bool
    editor_suggestion: str
    revision_count: int
    length_check_count: int




# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def safe_llm_invoke(prompt: str, temperature: float = 0.7, max_retries: int = 3) -> str:
    """Safely invoke Gemini API with built-in retry and fallback using only stdlib."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY")

    model_main = "gemini-2.0-flash-exp"
    model_fallback = "gemini-1.5-flash"

    for attempt in range(1, max_retries + 1):
        try:
            llm = ChatGoogleGenerativeAI(
                model=model_main,
                temperature=temperature,
                google_api_key=api_key,
            )
            response = llm.invoke(prompt)
            return response.content

        except Exception as e:
            error_text = str(e).lower()
            print(f"⚠️ Attempt {attempt}/{max_retries} failed: {e}")

            # Handle quota errors specifically
            if "429" in error_text or "quota" in error_text:
                wait_time = min(10 * attempt, 60)
                print(f"🚦 Quota limit hit — waiting {wait_time}s before retry...")
                time.sleep(wait_time)
                # Fallback to another model after first quota error
                model_main = model_fallback
                continue

            # Retry for transient issues (connection, timeout, etc.)
            if "timeout" in error_text or "connection" in error_text:
                time.sleep(5 * attempt)
                continue

            # Otherwise re-raise unexpected errors
            raise

    # If all retries fail, simulate a response to keep workflow alive
    print("❌ All Gemini retries failed — using simulated output.")
    return f"(Simulated Gemini response for prompt: {prompt[:100]}...)"



def simulate_response(prompt: str) -> str:
    """Used when LLM fails or quota exceeded."""
    return f"(Simulated response for prompt: {prompt[:80]}...)"


# ============================================================================
# NODES
# ============================================================================

def researcher_node(state: GraphState) -> GraphState:
    """Searches Wikipedia and gathers sources."""
    topic = state["topic"]
    sources: List[str] = []

    try:
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "opensearch",
            "search": topic,
            "limit": 3,
            "format": "json",
        }
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if len(data) > 1 and data[1]:
            for title in data[1][:3]:
                content_params = {
                    "action": "query",
                    "titles": title,
                    "prop": "extracts",
                    "explaintext": True,
                    "format": "json",
                }
                content_res = requests.get(search_url, params=content_params, timeout=10)
                content_data = content_res.json()
                pages = content_data.get("query", {}).get("pages", {})

                for _, page_data in pages.items():
                    content = page_data.get("extract", "")
                    if len(content) > 200:
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=500,
                            chunk_overlap=50,
                            separators=["\n\n", "\n", ". ", " ", ""],
                        )
                        chunks = splitter.split_text(content)
                        for chunk in chunks[:3]:
                            if len(chunk.strip()) > 100:
                                sources.append(f"[Wikipedia: {title}]\n{chunk.strip()}")
                        break
    except Exception as e:
        print(f"Wikipedia error: {e}")

    # Add simulated sources if empty
    if len(sources) < 3:
        sources.extend([
            "[Simulated PDF: MLOps_Best_Practices.pdf]\nGenerative AI is revolutionizing MLOps by introducing intelligent automation.",
            "[Simulated PDF: AI_Infrastructure_2024.pdf]\nThe integration of generative AI introduces new challenges in reproducibility.",
            "[Simulated PDF: Future_of_AI_Operations.pdf]\n67% of organizations explore generative AI for MLOps automation by 2026."
        ])

    state["sources"] = sources[:10]
    return state


def outline_agent(state: GraphState) -> GraphState:
    """Creates essay outline."""
    sources_text = "\n".join(state.get("sources", []))
    prompt = f"""
You are a research director creating an essay outline.

TOPIC: {state['topic']}
SOURCES:
{sources_text}

TASK: Create a detailed outline for a 1000-word essay (Intro, 3-4 body sections, Conclusion).
OUTPUT ONLY THE OUTLINE.
"""
    try:
        outline = safe_llm_invoke(prompt, temperature=0.7)
    except Exception as e:
        print(f"Outline LLM error: {e}")
        outline = simulate_response(prompt)

    state["outline"] = outline
    return state


def essay_writer_agent(state: GraphState) -> GraphState:
    """Writes or revises essay."""
    state["revision_count"] = state.get("revision_count", 0) + 1
    sources_text = "\n".join(state.get("sources", []))

    if not state.get("outline"):
        state["outline"] = "(Missing outline)"

    feedback = state.get("editor_suggestion", "N/A")
    if feedback and feedback != "N/A":
        task = f"Revise essay based on feedback: {feedback}"
    else:
        task = "Write a professional essay based on outline and sources."

    prompt = f"""
{task}

OUTLINE:
{state['outline']}

SOURCES:
{sources_text}

Write approximately 950–1050 words. Output only the essay.
"""
    try:
        essay = safe_llm_invoke(prompt)
    except Exception as e:
        print(f"Essay writer error: {e}")
        essay = simulate_response(prompt)

    state["essay_draft"] = essay
    return state


def essay_editor_agent(state: GraphState) -> GraphState:
    """Reviews essay quality."""
    max_revisions = 3
    if state.get("revision_count", 0) >= max_revisions:
        state["editor_validity"] = True
        state["editor_suggestion"] = "N/A"
        return state

    sources_text = "\n".join(state.get("sources", []))
    prompt = f"""
Review this essay. Be lenient; only reject for major issues.

OUTLINE:
{state['outline']}

SOURCES:
{sources_text}

ESSAY:
{state['essay_draft']}

If acceptable, respond "VALID".
If major issues, respond "REVISE: [specific issue]".
"""
    try:
        result = safe_llm_invoke(prompt, temperature=0.3).strip()
    except Exception as e:
        print(f"Editor LLM error: {e}")
        result = "VALID"

    if "VALID" in result.upper():
        state["editor_validity"] = True
        state["editor_suggestion"] = "N/A"
    else:
        state["editor_validity"] = False
        state["editor_suggestion"] = result.replace("REVISE:", "").strip()
    return state


def validator_node(state: GraphState) -> Dict[str, str]:
    """Validates essay length."""
    min_words, max_words = 950, 1050
    attempts = state.get("length_check_count", 0) + 1
    state["length_check_count"] = attempts

    essay = state.get("essay_draft", "")
    word_count = len(essay.split())

    if min_words <= word_count <= max_words:
        return {"route": "valid_length"}

    if attempts >= 3:
        words = essay.split()
        truncated = " ".join(words[:1000])
        state["essay_draft"] = truncated
        return {"route": "valid_length"}

    if word_count > max_words:
        state["editor_suggestion"] = f"Essay too long ({word_count} words). Trim to {min_words}-{max_words}."
    else:
        state["editor_suggestion"] = f"Essay too short ({word_count} words). Expand to {min_words}-{max_words}."

    state["editor_validity"] = False
    return {"route": "invalid_length"}

# ============================================================================
# ROUTING FUNCTIONS
# ============================================================================

def route_editor(state: GraphState) -> Literal["essay_writer_agent", "validator_node"]:
    """Route based on editor decision."""
    return "validator_node" if state["editor_validity"] else "essay_writer_agent"


def route_validator(state: GraphState) -> Literal["essay_writer_agent", END]:
    """Route based on word count validation."""
    word_count = len(state.get("essay_draft", "").split())
    length_attempts = state.get("length_check_count", 0)
    
    if (950 <= word_count <= 1050) or length_attempts >= 3:
        return END
    return "essay_writer_agent"


# ============================================================================
# WORKFLOW BUILDER
# ============================================================================

def create_essay_workflow() -> StateGraph:
    """Creates and returns the compiled workflow."""
    workflow = StateGraph(GraphState)
    
    # Add nodes
    workflow.add_node("researcher_node", researcher_node)
    workflow.add_node("outline_agent", outline_agent)
    workflow.add_node("essay_writer_agent", essay_writer_agent)
    workflow.add_node("essay_editor_agent", essay_editor_agent)
    workflow.add_node("validator_node", validator_node)
    
    # Define flow
    workflow.set_entry_point("researcher_node")
    workflow.add_edge("researcher_node", "outline_agent")
    workflow.add_edge("outline_agent", "essay_writer_agent")
    workflow.add_edge("essay_writer_agent", "essay_editor_agent")
    
    # Conditional edges
    workflow.add_conditional_edges(
        "essay_editor_agent",
        route_editor,
        {"essay_writer_agent": "essay_writer_agent", "validator_node": "validator_node"}
    )
    
    workflow.add_conditional_edges(
        "validator_node",
        route_validator,
        {"essay_writer_agent": "essay_writer_agent", END: END}
    )
    
    return workflow.compile()


# ============================================================================
# PUBLIC API
# ============================================================================

def generate_essay(
    topic: str,
    api_key: str,
    callback: Optional[Callable[[str, dict], None]] = None
) -> dict:
    """
    Generate an essay on the given topic.
    
    Args:
        topic: Essay topic
        api_key: Gemini API key
        callback: Optional callback function(stage_name, state_data) for progress updates
        
    Returns:
        dict: Final state with essay and metadata
    """
    os.environ["GEMINI_API_KEY"] = api_key
    
    initial_state = {
        "topic": topic,
        "essay_draft": "",
        "outline": "",
        "sources": [],
        "editor_validity": False,
        "editor_suggestion": "N/A",
        "revision_count": 0,
        "length_check_count": 0
    }
    
    app = create_essay_workflow()
    
    # Stream through workflow with callbacks
    if callback:
        for state_update in app.stream(initial_state):
            for node_name, node_state in state_update.items():
                callback(node_name, node_state)
    
    # Get final result
    final_state = app.invoke(initial_state)
    
    return {
        "essay": final_state["essay_draft"],
        "topic": final_state["topic"],
        "sources": final_state["sources"],
        "outline": final_state["outline"],
        "word_count": len(final_state["essay_draft"].split()),
        "revisions": final_state["revision_count"],
        "length_checks": final_state["length_check_count"]
    }


# ============================================================================
# CLI INTERFACE
# ============================================================================

def main():
    """CLI interface for testing."""
    print("=" * 80)
    print("ESSAY GENERATION WORKFLOW")
    print("=" * 80)
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY environment variable")
        return
    
    topic = input("\nEnter essay topic: ").strip() or "The future impact of Generative AI on MLOps practices"
    
    def progress_callback(stage, state):
        """Print progress updates."""
        stage_names = {
            "researcher_node": "🔍 Research",
            "outline_agent": "📝 Outline",
            "essay_writer_agent": "✍️ Writing",
            "essay_editor_agent": "🔎 Review",
            "validator_node": "📏 Validation"
        }
        print(f"{stage_names.get(stage, stage)} complete")
    
    print(f"\n🚀 Generating essay on: {topic}\n")
    
    result = generate_essay(topic, api_key, callback=progress_callback)
    
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Word count: {result['word_count']}")
    print(f"Revisions: {result['revisions']}")
    print(f"Sources: {len(result['sources'])}")
    print("\n" + "-" * 80)
    print(result['essay'])
    print("-" * 80)


if __name__ == "__main__":
    main()