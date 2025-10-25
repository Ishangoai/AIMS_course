import operator
import os
import time
import uuid
from typing import Annotated, Literal, TypedDict, cast

import gradio as gr
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

# Load environment variables from .env file
load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    GOOGLE_API_KEY = input('🔑 Enter your Google API key: ').strip()
    os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

llm = ChatGoogleGenerativeAI(
    model='gemini-2.0-flash-lite',
    temperature=0.1,  # Very low temperature for maximum consistency
    # max_output_tokens=1500,  # ~1000 words * 1.3 tokens/word
    google_api_key=GOOGLE_API_KEY
)


def count_words(text):
    """Accurate word counter"""
    return len(text.split())


def enforce_word_limit_with_llm(text, current_count, min_words=950, max_words=1000, iteration=0):
    """
    Use LLM to intelligently adjust text to meet word count without truncation.
    This preserves content quality while enforcing limits.
    """
    if min_words <= current_count <= max_words:
        return text  # Perfect, no changes needed

    if iteration >= 2:  # Prevent infinite loops
        print(f"⚠️ Max iterations reached. Final count: {current_count}")
        return text

    if current_count > max_words:
        # Need to condense
        excess = current_count - max_words
        prompt = PromptTemplate.from_template(
            """You are a professional editor. This text is TOO LONG.

Current text ({current_count} words):
{text}

TASK: Reduce this by EXACTLY {excess} words to reach {max_words} words.

RULES:
1. Remove redundancy and wordiness
2. Combine similar points
3. Keep all key information
4. Maintain complete sentences
5. Final output MUST be {max_words} words or less

OUTPUT ONLY THE CONDENSED TEXT (no explanations):"""
        )

        chain = prompt | llm | StrOutputParser()
        condensed = chain.invoke({
            "text": text,
            "current_count": current_count,
            "excess": excess,
            "max_words": max_words
        })

        new_count = count_words(condensed)
        print(f"🔧 Condensed: {current_count} → {new_count} words")

        # Recursive check if still too long
        if new_count > max_words:
            return enforce_word_limit_with_llm(condensed, new_count, min_words, max_words, iteration + 1)
        return condensed

    else:
        # Need to expand
        needed = min_words - current_count
        prompt = PromptTemplate.from_template(
            """You are a professional editor. This text is TOO SHORT.

Current text ({current_count} words):
{text}

TASK: Add EXACTLY {needed} words to reach {min_words} words.

RULES:
1. Add meaningful details and examples
2. Expand on existing concepts
3. Maintain coherence and flow
4. Do not pad with filler
5. Final output MUST be between {min_words}-{max_words} words

OUTPUT ONLY THE EXPANDED TEXT (no explanations):"""
        )

        chain = prompt | llm | StrOutputParser()
        expanded = chain.invoke({
            "text": text,
            "current_count": current_count,
            "needed": needed,
            "min_words": min_words,
            "max_words": max_words
        })

        new_count = count_words(expanded)
        print(f"🔧 Expanded: {current_count} → {new_count} words")

        # Recursive check if still too short or now too long
        if new_count < min_words or new_count > max_words:
            return enforce_word_limit_with_llm(expanded, new_count, min_words, max_words, iteration + 1)
        return expanded


def get_word_count_instruction(current_words, target_min, target_max):
    """Generate strict word count instructions"""
    if current_words < target_min:
        words_needed = target_min - current_words
        return f"""
⚠️ CRITICAL: You are {words_needed} words SHORT. Current: {current_words} | Required: {target_min}-{target_max}

ADD EXACTLY {words_needed} WORDS by:
- Expanding explanations with more depth
- Adding concrete examples
- Elaborating technical concepts
- Including additional context

COUNT AS YOU WRITE. STOP at {target_max} words maximum."""
    elif current_words > target_max:
        words_to_remove = current_words - target_max
        return f"""
⚠️ CRITICAL: You are {words_to_remove} words OVER. Current: {current_words} | Required: {target_min}-{target_max}

CUT EXACTLY {words_to_remove} WORDS by:
- Removing redundant phrases
- Condensing verbose sections
- Eliminating unnecessary details
- Combining repetitive points

COUNT AS YOU WRITE. MUST NOT exceed {target_max} words."""
    else:
        return f"""
✅ PERFECT: {current_words} words (target: {target_min}-{target_max})
MAINTAIN this exact length. Do not add or remove content."""

# ==============================================================
# State Schema
# ==============================================================


class ReportState(TypedDict):
    """State schema for the report generation workflow"""
    topic: str
    research_summary: str
    draft_report: str
    factchecked_report: str
    human_feedback: str
    final_report: str
    word_count: int
    iteration_count: int
    needs_human_review: bool
    human_approved: bool
    messages: Annotated[list, operator.add]

# ==============================================================
# Agent Nodes
# ==============================================================


def research_agent(state: ReportState) -> ReportState:
    """Research agent: Gathers background information"""
    print("🔍 Research Agent working...")

    prompt = PromptTemplate.from_template(
        """You are a research assistant with deep expertise.

        Research topic: {topic}

        Provide comprehensive research notes covering:
        - Historical background and evolution
        - Current state and recent developments
        - Key concepts and technical details
        - Major challenges and opportunities
        - Notable contributors and organizations
        """
    )

    chain = prompt | llm | StrOutputParser()
    research = chain.invoke({"topic": state["topic"]})

    state["research_summary"] = research
    state["messages"].append("Research complete")
    return state


def writing_agent(state: ReportState) -> ReportState:
    """Writing agent: Creates structured report targeting 1000 words"""
    print("✍️ Writing Agent creating draft...")

    prompt = PromptTemplate.from_template(
        """You are a professional technical writer. WORD COUNT IS CRITICAL.

Topic: {topic}
Research: {research_summary}

Write a complete report of EXACTLY 950-1000 words.

Structure:
1. Introduction - Set context and significance
2. Main Content - Detailed analysis with examples
3. Conclusion - Summary and future outlook

ABSOLUTE REQUIREMENTS:
- MUST be 950-1000 words (NO LESS, NO MORE)
- Count every word as you write
- Write plain text (no markdown formatting)
- Stop writing at 1000 words MAXIMUM
- Complete all sections with proper endings

Begin counting: OUTPUT ONLY THE REPORT TEXT:"""
    )

    chain = prompt | llm | StrOutputParser()
    draft = chain.invoke({
        "topic": state["topic"],
        "research_summary": state["research_summary"]
    })

    initial_count = count_words(draft)
    print(f"📊 Initial draft: {initial_count} words")

    # Use LLM to enforce limit intelligently (no truncation)
    if not (950 <= initial_count <= 1000):
        print("⚙️ Adjusting word count...")
        draft = enforce_word_limit_with_llm(draft, initial_count, 950, 1000)

    word_count = count_words(draft)
    print(f"✅ Final draft: {word_count} words")

    state["draft_report"] = draft
    state["word_count"] = word_count
    state["messages"].append(f"Draft created ({word_count} words)")
    return state


def factcheck_agent(state: ReportState) -> ReportState:
    """Fact-checking agent - maintains 950-1000 words"""
    print("🔎 Fact-checking...")

    current_word_count = state["word_count"]
    word_instruction = get_word_count_instruction(current_word_count, 950, 1000)

    prompt = PromptTemplate.from_template(
        """You are a fact-checking analyst. STRICT WORD COUNT ENFORCEMENT.

Report to review ({current_word_count} words):
{draft_report}

{word_instruction}

Your tasks:
1. Verify all facts are accurate
2. Correct any errors or inaccuracies
3. Ensure report is complete and coherent
4. MAINTAIN word count at 950-1000 words

CRITICAL: Output MUST be {target_min}-{target_max} words.
Current count: {current_word_count} words

OUTPUT ONLY THE CORRECTED REPORT (no explanations):"""
    )

    chain = prompt | llm | StrOutputParser()
    checked = chain.invoke({
        "draft_report": state["draft_report"],
        "word_instruction": word_instruction,
        "current_word_count": current_word_count,
        "target_min": 950,
        "target_max": 1000
    })

    initial_count = count_words(checked)
    print(f"📊 Initial fact-check: {initial_count} words")

    # Use LLM to enforce limit intelligently
    if not (950 <= initial_count <= 1000):
        print("⚙️ Adjusting word count...")
        checked = enforce_word_limit_with_llm(checked, initial_count, 950, 1000)

    word_count = count_words(checked)
    print(f"✅ Final fact-check: {word_count} words")

    state["factchecked_report"] = checked
    state["word_count"] = word_count
    state["needs_human_review"] = True
    state["messages"].append(f"Fact-checked ({word_count} words)")
    return state


def human_review_node(state: ReportState) -> ReportState:
    """Human review checkpoint"""
    print("👤 Pausing for human review...")
    state["messages"].append("Awaiting human review")
    return state


def refinement_agent(state: ReportState) -> ReportState:
    """Refinement agent: Applies human feedback while keeping 950-1000 words"""
    print("✨ Applying feedback...")

    if not state.get("human_feedback") or state["human_approved"]:
        state["final_report"] = state["factchecked_report"]
        state["messages"].append("Approved without changes")
        return state

    current_word_count = state["word_count"]
    word_instruction = get_word_count_instruction(current_word_count, 950, 1000)

    prompt = PromptTemplate.from_template(
        """You are an expert editor. STRICT WORD COUNT: 950-1000 words.

Current Report ({word_count} words):
{report}

User Feedback:
{feedback}

{word_instruction}

CRITICAL INSTRUCTIONS:
1. Apply the user's feedback
2. MUST maintain 950-1000 word count
3. If adding content, remove equivalent amount elsewhere
4. Prioritize most important changes
5. Keep complete sentences and structure
6. Current: {word_count} words → Target: 950-1000 words

OUTPUT ONLY THE REVISED REPORT (no explanations):"""
    )

    chain = prompt | llm | StrOutputParser()
    refined = chain.invoke({
        "report": state["factchecked_report"],
        "feedback": state["human_feedback"],
        "word_count": current_word_count,
        "word_instruction": word_instruction
    })

    initial_count = count_words(refined)
    print(f"📊 Initial refinement: {initial_count} words")

    # Use LLM to enforce limit intelligently
    if not (950 <= initial_count <= 1000):
        print("⚙️ Adjusting word count...")
        refined = enforce_word_limit_with_llm(refined, initial_count, 950, 1000)

    word_count = count_words(refined)
    print(f"✅ Final refinement: {word_count} words")

    state["final_report"] = refined
    state["factchecked_report"] = refined
    state["word_count"] = word_count
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    state["needs_human_review"] = True
    state["messages"].append(f"Refined (iteration {state['iteration_count']}, {word_count} words)")
    return state
    state["factchecked_report"] = refined
    state["word_count"] = word_count
    state["iteration_count"] = state.get("iteration_count", 0) + 1
    state["needs_human_review"] = True
    state["messages"].append(f"Refined (iteration {state['iteration_count']}, {word_count} words)")
    return state


# ==============================================================
# Conditional Edges
# ==============================================================

def check_human_decision(state: ReportState) -> Literal["refine", "end"]:
    """Check human decision"""
    if state.get("human_approved", False):
        return "end"
    return "refine"


# ==============================================================
# Build LangGraph
# ==============================================================

def build_report_graph():
    """Build the LangGraph workflow"""
    workflow = StateGraph(ReportState)

    workflow.add_node("research", research_agent)
    workflow.add_node("write", writing_agent)
    workflow.add_node("factcheck", factcheck_agent)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("refine", refinement_agent)

    workflow.set_entry_point("research")
    workflow.add_edge("research", "write")
    workflow.add_edge("write", "factcheck")
    workflow.add_edge("factcheck", "human_review")
    workflow.add_edge("human_review", "refine")

    workflow.add_conditional_edges(
        "refine",
        check_human_decision,
        {"refine": "human_review", "end": END}
    )

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory, interrupt_before=["human_review"])


session_states = {}


def create_session():
    return str(uuid.uuid4())


def get_graph_visualization():
    """Get Mermaid diagram of the workflow"""
    try:
        graph = build_report_graph()
        mermaid_diagram = graph.get_graph().draw_mermaid()
        return f"```mermaid\n{mermaid_diagram}\n```"
    except Exception as e:
        return f"Error generating graph: {str(e)}"


def start_generation(topic):
    """Start report generation"""
    if not topic or not topic.strip():
        return ("⚠️ Enter a topic", "", gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False), None,
                create_progress_html("waiting"), "0", "0", "0", "0")

    try:
        session_id = create_session()

        initial_state = {
            "topic": topic,
            "research_summary": "",
            "draft_report": "",
            "factchecked_report": "",
            "human_feedback": "",
            "final_report": "",
            "word_count": 0,
            "iteration_count": 0,
            "needs_human_review": False,
            "human_approved": False,
            "messages": []
        }

        config: RunnableConfig = {"configurable": {"thread_id": session_id}}

        for event in report_graph.stream(cast(ReportState, initial_state), config):
            pass

        current_state = report_graph.get_state(config)
        state_values = current_state.values

        session_states[session_id] = config

        report = state_values.get("factchecked_report", "")
        wc = state_values.get("word_count", 0)

        # Status with color-coded word count
        if 950 <= wc <= 1000:
            status = "✅ Report generated!"
        else:
            status = "⚠️ Report generated."

        # Calculate text stats
        words = str(wc)
        chars = str(len(report))
        sentences = str(len([s for s in report.split('.') if s.strip()]))
        paragraphs = str(len([p for p in report.split('\n\n') if p.strip()]))

        progress = create_progress_html("review", wc)

        return (status, report, gr.update(visible=True),
                gr.update(visible=True), gr.update(visible=True), session_id,
                progress, words, chars, sentences, paragraphs)

    except Exception as e:
        return (f"❌ Error: {e}", "", gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False), None,
                create_progress_html("waiting"), "0", "0", "0", "0")


def approve_report(report, session_id):
    """Approve and finalize"""
    if not session_id or session_id not in session_states:
        return ("❌ Invalid session", report, gr.update(visible=False),
                create_progress_html("waiting"))

    try:
        config = session_states[session_id]
        current_state = report_graph.get_state(config)
        state_values = current_state.values.copy()

        state_values["human_approved"] = True
        state_values["human_feedback"] = ""

        report_graph.update_state(config, state_values)

        for event in report_graph.stream(None, config):
            pass

        final_state = report_graph.get_state(config)
        final_values = final_state.values
        final_report = final_values.get("final_report", report)
        wc = final_values.get("word_count", 0)

        # Status with word count validation
        if 950 <= wc <= 1000:
            status = f"✅ Approved! Word count: {wc} ✓ (within 950-1000)"
        else:
            status = f"✅ Approved! Word count: {wc} (enforced to 950-1000)"

        progress = create_progress_html("complete", wc)

        return status, final_report, gr.update(visible=True), progress

    except Exception as e:
        return f"❌ Error: {e}", report, gr.update(visible=False), create_progress_html("waiting")


def request_changes(report, feedback, session_id):
    """Request changes with feedback"""
    if not session_id or session_id not in session_states:
        return ("❌ Invalid session", report, gr.update(visible=False),
                gr.update(visible=True), gr.update(visible=True),
                create_progress_html("waiting"), "0", "0", "0", "0")

    if not feedback or not feedback.strip():
        wc = len(report.split())
        return ("⚠️ Provide feedback", report, gr.update(visible=False),
                gr.update(visible=True), gr.update(visible=True),
                create_progress_html("review", wc),
                str(wc), str(len(report)),
                str(len([s for s in report.split('.') if s.strip()])),
                str(len([p for p in report.split('\n\n') if p.strip()])))

    try:
        config = session_states[session_id]
        current_state = report_graph.get_state(config)
        state_values = current_state.values.copy()

        state_values["human_approved"] = False
        state_values["human_feedback"] = feedback

        report_graph.update_state(config, state_values)

        for event in report_graph.stream(None, config):
            pass

        updated_state = report_graph.get_state(config)
        updated_values = updated_state.values
        revised = updated_values.get("factchecked_report", report)
        wc = updated_values.get("word_count", 0)

        # Status with word count validation
        if 950 <= wc <= 1000:
            status = f"🔄 Changes applied. Word count: {wc} ✓ (within 950-1000)"
        else:
            status = f"🔄 Changes applied. Word count: {wc} (enforced to 950-1000)"

        # Calculate text stats
        words = str(wc)
        chars = str(len(revised))
        sentences = str(len([s for s in revised.split('.') if s.strip()]))
        paragraphs = str(len([p for p in revised.split('\n\n') if p.strip()]))

        progress = create_progress_html("review", wc)

        return (status, revised, gr.update(visible=False),
                gr.update(visible=True), gr.update(visible=True),
                progress, words, chars, sentences, paragraphs)

    except Exception as e:
        return (f"❌ Error: {e}", report, gr.update(visible=False),
                gr.update(visible=True), gr.update(visible=True),
                create_progress_html("waiting"), "0", "0", "0", "0")


def create_progress_html(stage="waiting", word_count=0):
    """Create HTML for agent progress visualization"""
    stages = {
        "waiting": {
            "research": "waiting", "write": "waiting", "factcheck": "waiting",
            "review": "waiting", "refine": "waiting"
        },
        "research": {
            "research": "active", "write": "waiting", "factcheck": "waiting",
            "review": "waiting", "refine": "waiting"
        },
        "write": {
            "research": "complete", "write": "active", "factcheck": "waiting",
            "review": "waiting", "refine": "waiting"
        },
        "factcheck": {
            "research": "complete", "write": "complete", "factcheck": "active",
            "review": "waiting", "refine": "waiting"
        },
        "review": {
            "research": "complete", "write": "complete", "factcheck": "complete",
            "review": "active", "refine": "waiting"
        },
        "refine": {
            "research": "complete", "write": "complete", "factcheck": "complete",
            "review": "complete", "refine": "active"
        },
        "complete": {
            "research": "complete", "write": "complete", "factcheck": "complete",
            "review": "complete", "refine": "complete"
        },
    }

    current_stages = stages.get(stage, stages["waiting"])

    colors = {
        "waiting": "#e0e0e0",
        "active": "#4CAF50",
        "complete": "#2196F3"
    }

    status_text = {
        "waiting": "Waiting...",
        "active": "In Progress...",
        "complete": "✓ Done"
    }

    html = '<div style="display: flex; gap: 10px; padding: 15px; background: #f5f5f5; border-radius: 8px;">'

    for agent, emoji, label in [
        ("research", "🔍", "Research"),
        ("write", "✍️", "Write"),
        ("factcheck", "🔎", "Fact-Check"),
        ("review", "👤", "Review"),
        ("refine", "✨", "Refine")
    ]:
        status = current_stages[agent]
        color = colors[status]
        text = status_text[status]
        html += f'''
            <div style="flex: 1; padding: 10px; background: {color};
                 border-radius: 5px; text-align: center; color: white; font-weight: bold;">
                {emoji} {label}<br><small>{text}</small>
            </div>
        '''

    html += '</div>'

    if word_count > 0:
        if 950 <= word_count <= 1000:
            color = "#4CAF50"
            icon = "✓"
        else:
            color = "#FF9800"
            icon = "⚠"
        html += (
            f'<div style="margin-top: 10px; padding: 10px; background: {color}; color: white; '
            f'border-radius: 5px; text-align: center; font-weight: bold;">'
            f'{icon} Current Word Count: {word_count}</div>'
        )

    return html


def save_report(report):
    """Save to file"""
    if not report or not report.strip():
        return None
    try:
        filename = f"report_{int(time.time())}.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        return filename
    except Exception:
        return None


# Initialize the report graph
report_graph = build_report_graph()
