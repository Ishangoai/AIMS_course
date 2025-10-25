import getpass
import os
from typing import List, Optional, TypedDict

from langchain.schema import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .prompts import EDITOR_PROMPT, FACT_CHECK_PROMPT, QUALITY_CONTROL_PROMPT, RESEARCH_PROMPT, WRITER_PROMPT
from .vaidator import validate_structure
from .word_counter import count_words, get_word_count


class ReportState(TypedDict):
    topic: str
    temperature: float
    research: Optional[str]
    draft: Optional[str]
    fact_check: Optional[str]
    edited_report: Optional[str]
    final_report: Optional[str]
    word_count: int
    feedback: Optional[str]
    human_feedback: Optional[str]
    iteration: int
    max_iterations: int
    status: str
    messages: List[str]


def trim_content_if_needed(content: str, target_words: int = 1000) -> str:
    """
    Aggressively trim content if it's too long.
    This is a fallback to ensure word count compliance.
    """
    words = content.split()
    current_count = len(words)

    if current_count <= 1050:
        return content

    # Need to trim - aim for target_words
    # Keep the structure by preserving headers
    lines = content.split("\n")
    result_lines = []
    word_count = 0

    for line in lines:
        line_words = len(line.split())
        # Always keep headers
        if line.strip().startswith("#") or line.strip().startswith("**"):
            result_lines.append(line)
        elif word_count + line_words <= target_words:
            result_lines.append(line)
            word_count += line_words
        elif word_count < target_words:
            # Partial line to reach target
            remaining = target_words - word_count
            words_in_line = line.split()
            if remaining > 0:
                result_lines.append(" ".join(words_in_line[:remaining]))
            break

    return "\n".join(result_lines)


def get_llm(temp: float):
    """Initializes and returns the Google Gemini LLM."""
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API Key: ")
        raise Exception("No api_key, bro")

    api_key = os.getenv("GOOGLE_API_KEY")
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=temp,
        google_api_key=api_key,
    )


def research_agent(state: ReportState) -> ReportState:
    """
    Research agent that gathers information on the topic.

    Args:
        state: Current workflow state

    Returns:
        Updated state with research findings
    """
    state["messages"].append("🔍 Research Agent: Gathering information...")

    topic = state["topic"]

    prompt = RESEARCH_PROMPT.format(topic=topic, previous_research=state.get("research", "None"))

    # Note: Using LLM's built-in knowledge base instead of web search
    prompt += "\n\nNote: Use your comprehensive knowledge base to provide current, accurate information on this topic."

    # Use lower temperature for revisions to be more controlled
    temp = state["temperature"]

    llm = get_llm(temp)

    response = llm.invoke([HumanMessage(content=prompt)])
    research = response.content

    state["research"] = research
    state["messages"].append(f"✓ Research completed ({len(research.split())} words of research)")

    return state


def writer_agent(state: ReportState) -> ReportState:
    """
    Writer agent that creates the report draft.

    Args:
        state: Current workflow state

    Returns:
        Updated state with draft report
    """
    state["messages"].append("✍️ Writer Agent: Drafting report...")

    prompt = WRITER_PROMPT.format(
        topic=state["topic"],
        research=state.get("research", ""),
        draft=state.get("draft", ""),
        feedback=state.get("feedback", "None"),
    )

    iteration = state.get("iteration", 0)
    temp = state["temperature"] if iteration == 0 else max(0.3, state["temperature"] - 0.2)

    llm = get_llm(temp)

    response = llm.invoke([HumanMessage(content=prompt)])
    draft = response.content

    word_count = get_word_count(draft)

    state["draft"] = draft
    state["word_count"] = word_count
    state["messages"].append(f"✓ Draft completed ({word_count} words)")

    return state


def fact_checker_agent(state: ReportState) -> ReportState:
    """
    Fact checker agent that verifies report accuracy.

    Args:
        state: Current workflow state

    Returns:
        Updated state with fact-check feedback
    """
    state["messages"].append("🔎 Fact Checker: Verifying accuracy...")

    content = state.get("draft", "")

    prompt = FACT_CHECK_PROMPT.format(topic=state["topic"], content=content)

    temp = state["temperature"]

    llm = get_llm(temp)

    response = llm.invoke([HumanMessage(content=prompt)])
    fact_check = response.content

    state["fact_check"] = fact_check
    state["messages"].append("✓ Fact-checking completed")

    return state


def quality_control_agent(state: ReportState) -> ReportState:
    """
    Quality control agent that performs final review.

    Args:
        state: Current workflow state

    Returns:
        Updated state with QC decision
    """
    state["messages"].append("✅ Quality Control: Final review...")

    content = state.get("edited_report", state.get("draft", ""))

    # Perform automated checks
    word_count_info = count_words(content)  # type: ignore
    structure_info = validate_structure(content)  # type: ignore

    # Use LLM for qualitative assessment

    prompt = QUALITY_CONTROL_PROMPT.format(
        content=content, word_count=word_count_info["word_count"], topic=state["topic"]
    )

    temp = state["temperature"]

    llm = get_llm(temp)

    response = llm.invoke([HumanMessage(content=prompt)])
    qc_result = response.content

    # Determine if report passes
    passes_word_count = word_count_info["within_range"]
    passes_structure = structure_info["is_valid"]
    passes_llm_check = "PASS" in qc_result.upper()

    if passes_word_count and passes_structure and passes_llm_check:
        state["status"] = "COMPLETE"
        state["final_report"] = content
        state["messages"].append("✓ Quality control PASSED - Report is ready!")
    else:
        issues = []
        if not passes_word_count:
            issues.append(word_count_info["message"])
            # Add specific guidance for word count issues
            current_wc = word_count_info["word_count"]
            if current_wc > 1050:
                excess = current_wc - 1050
                issues.append(f"CRITICAL: Report is {excess} words too long. You MUST cut significant content.")
            elif current_wc < 950:
                deficit = 950 - current_wc
                issues.append(f"Report is {deficit} words too short. Add more relevant details.")
        if not passes_structure:
            issues.append(structure_info["message"])
        if not passes_llm_check:
            issues.append("LLM quality check failed")

        state["status"] = "NEEDS_REVISION"
        state["feedback"] = "Issues found:\n" + "\n".join(f"- {issue}" for issue in issues)
        state["feedback"] += f"\n\nQC Assessment:\n{qc_result}"
        state["iteration"] += 1
        state["messages"].append(f"⚠️ Revision needed (iteration {state['iteration']})")

    return state


def editor_agent(state: ReportState) -> ReportState:
    """
    Editor agent that refines and polishes the report.

    Args:
        state: Current workflow state

    Returns:
        Updated state with edited report
    """
    state["messages"].append("📝 Editor Agent: Refining report...")

    content = state.get("draft", "")
    word_count_info = count_words(content)  # type: ignore

    # Include human feedback if available
    fact_check_feedback = state.get("fact_check", "No fact-check performed yet")
    if state.get("human_feedback"):
        fact_check_feedback = fact_check_feedback + f"\n\nHuman Feedback:\n{state['human_feedback']}"  # type: ignore

    prompt = EDITOR_PROMPT.format(
        content=content,
        word_count=word_count_info["word_count"],
        fact_check=fact_check_feedback,
    )

    temp = state["temperature"]

    llm = get_llm(temp)

    response = llm.invoke([HumanMessage(content=prompt)])
    edited = response.content

    new_word_count = count_words(edited)

    # If still too long after LLM editing, apply aggressive trimming
    if new_word_count["word_count"] > 1050:
        state["messages"].append(f"⚠️ LLM edit still too long ({new_word_count['word_count']} words), applying trim...")
        edited = trim_content_if_needed(edited, target_words=1000)
        new_word_count = count_words(edited)

    state["edited_report"] = edited
    state["word_count"] = new_word_count["word_count"]
    state["messages"].append(f"✓ Editing completed ({new_word_count['word_count']} words, {new_word_count['status']})")

    return state
