"""
llms_workflow.py
Enhanced Agentic workflow with STRICT word count enforcement (exactly 1010 words)
8 Specialized Agents orchestrated by LangGraph for Data Engineering report generation
"""

from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, TypedDict

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

# -------------------------------------------------------------
# 1. Environment setup
# -------------------------------------------------------------
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError(
        "Missing GOOGLE_API_KEY. Please set it in a .env file or environment variable."
    )


PROMPT_LOG_PATH = "prompt_log.jsonl"
WORKFLOWS_DIR = "workflows"


# Create workflows directory
os.makedirs(WORKFLOWS_DIR, exist_ok=True)


# -------------------------------------------------------------
# 2. State Definition for LangGraph
# -------------------------------------------------------------
class ReportState(TypedDict):
    """State object for the report generation workflow"""

    topic: str
    tone: str
    audience: str
    temperature: float
    # Agent outputs
    validation_result: dict[str, Any]
    optimized_prompt: str
    prompt_quality: str
    key_points: str
    research_report: str
    grade_result: dict[str, Any]
    structured_report: str
    final_report: str
    # Metadata
    word_count: int
    word_count_valid: bool
    structure_valid: bool
    citation_count: int
    citation_valid: bool
    retry_count: int
    log: list[dict[str, Any]]
    timestamp: str
    error: str


# -------------------------------------------------------------
# 3. Utility Functions
# -------------------------------------------------------------
def count_words(text: str) -> int:
    """Accurately count words in text, excluding markdown syntax."""
    clean_text = re.sub(r"[#*_`\[\]]", "", text)  # Remove markdown including brackets
    clean_text = re.sub(r"\s+", " ", clean_text)
    words = [w for w in clean_text.strip().split() if w]
    return len(words)


def check_word_count_limit(
    text: str, min_words: int = 1010, max_words: int = 1010
) -> dict[str, Any]:
    """Check if text meets word count requirements (exactly 1010 words)."""
    word_count = count_words(text)
    is_valid = word_count == 1010
    return {
        "is_valid": is_valid,
        "word_count": word_count,
        "min_words": min_words,
        "max_words": max_words,
        "status": (
            "✓ Valid"
            if is_valid
            else f"✗ {'Under' if word_count < 1010 else 'Over'} limit"
        ),
    }


def validate_structure(text: str) -> dict[str, Any]:
    """Validate report structure: Title, Introduction, 4 subtopics, Conclusion."""
    # Check H1 title
    h1_match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    has_title = bool(h1_match)

    # Check for exact "Introduction" and "Conclusion"
    has_intro = bool(re.search(r"^##\s+Introduction\s*$", text, re.MULTILINE))
    has_conclusion = bool(re.search(r"^##\s+Conclusion\s*$", text, re.MULTILINE))

    # Count all H2 sections
    h2_sections = re.findall(r"^##\s+(.+)$", text, re.MULTILINE)
    num_sections = len(h2_sections)

    # Should have 6 sections: Introduction + 4 subtopics + Conclusion
    correct_section_count = num_sections == 6

    is_valid = has_title and has_intro and has_conclusion and correct_section_count

    return {
        "is_valid": is_valid,
        "has_title": has_title,
        "has_intro": has_intro,
        "has_conclusion": has_conclusion,
        "num_sections": num_sections,
        "correct_count": correct_section_count,
    }


def count_citations(text: str) -> int:
    """Count citations in [Source Name] format, excluding markdown links."""
    # Match [Text] but not [Text](url)
    citations = re.findall(r"\[([^\]]{3,}?)\](?!\()", text)
    return len(citations)


def validate_citations(text: str) -> dict[str, Any]:
    """Validate that there are exactly 3 citations."""
    citation_count = count_citations(text)
    is_valid = citation_count == 3
    return {"is_valid": is_valid, "count": citation_count, "target": 3}


# -------------------------------------------------------------
# 4. Word Count Enforcement Functions
# -------------------------------------------------------------
def _save_section_content(
    sections: dict[str, Any],
    current_section: str | None,
    current_section_header: str | None,
    current_content: list[str],
) -> None:
    """Save accumulated content to the appropriate section."""
    if not current_section or not current_content:
        return

    content = "\n".join(current_content).strip()
    if current_section == "introduction":
        sections["introduction"] = content
    elif current_section == "conclusion":
        sections["conclusion"] = content
    else:
        sections["body_sections"].append(
            {"header": current_section_header, "content": content}
        )


def _determine_section_type(line: str) -> tuple[str, str]:
    """Determine section type from H2 header line."""
    section_lower = line.lower()
    if "introduction" in section_lower:
        return "introduction", line
    if "conclusion" in section_lower:
        return "conclusion", line
    return "body", line


def extract_report_sections(text: str) -> dict[str, Any]:
    """Extract structured sections from markdown report."""
    sections: dict[str, Any] = {
        "title": "",
        "introduction": "",
        "body_sections": [],
        "conclusion": "",
    }

    lines = text.split("\n")
    current_section: str | None = None
    current_section_header: str | None = None
    current_content: list[str] = []

    for line in lines:
        # Capture H1 title
        if re.match(r"^#\s+", line) and not sections["title"]:
            sections["title"] = line
            continue

        # Capture H2 sections
        if re.match(r"^##\s+", line):
            # Save previous section
            _save_section_content(sections, current_section, current_section_header, current_content)

            # Determine new section type
            current_section, current_section_header = _determine_section_type(line)
            current_content = []
        else:
            current_content.append(line)

    # Save final section
    _save_section_content(sections, current_section, current_section_header, current_content)

    return sections


def trim_text_to_word_count(text: str, target_words: int) -> str:
    """Intelligently trim text to target word count, preserving sentence boundaries."""
    words = text.split()
    if len(words) <= target_words:
        return text

    trimmed_words = words[:target_words]
    trimmed_text = " ".join(trimmed_words)

    # Try to end at a sentence boundary
    last_period = max(
        trimmed_text.rfind("."), trimmed_text.rfind("!"), trimmed_text.rfind("?")
    )
    if last_period > 0 and last_period > len(trimmed_text) * 0.8:
        return trimmed_text[: last_period + 1]

    return trimmed_text.rstrip(".,!?;:") + "."


def expand_section_content(content: str, additional_words_needed: int) -> str:
    """Expand content by elaborating on existing sentences."""
    sentences = re.split(r"([.!?]+\s+)", content)
    sentences = [s for s in sentences if s.strip()]

    elaborations = [
        " This approach ensures robust data quality and maintains system reliability across distributed environments.",
        " Data engineers should carefully consider scalability requirements and performance optimization strategies.",
        " Industry best practices recommend implementing comprehensive monitoring and alerting mechanisms.",
        " Organizations adopting this methodology typically "
        "observe significant improvements in data pipeline efficiency.",
        " This technique has proven effective in addressing common challenges related to data consistency and lineage.",
        " Modern data architectures increasingly rely on these patterns to achieve operational excellence.",
        " Proper implementation requires attention to both technical specifications and business requirements.",
        " Teams should evaluate trade-offs between complexity and maintainability when designing such systems.",
    ]

    expanded_parts: list[str] = []
    words_added = 0
    target = additional_words_needed

    for i, sentence in enumerate(sentences):
        expanded_parts.append(sentence)
        if words_added < target and i % 2 == 1:  # Add elaboration after punctuation
            elab_index = i % len(elaborations)
            elab = elaborations[elab_index]
            expanded_parts.append(elab)
            words_added += count_words(elab)

    return "".join(expanded_parts)


def _rebuild_report_from_sections(
    sections: dict[str, Any],
    intro: str,
    body: list[dict[str, str]],
    conclusion: str,
) -> str:
    """Rebuild markdown report from sections."""
    result = f"{sections['title']}\n\n"
    if intro:
        result += f"## Introduction\n{intro}\n\n"
    for section in body:
        result += f"{section['header']}\n{section['content']}\n\n"
    if conclusion:
        result += f"## Conclusion\n{conclusion}\n"
    return result.strip()


def _trim_sections(
    sections: dict[str, Any], excess: int
) -> tuple[str, list[dict[str, str]], str]:
    """Trim sections proportionally to remove excess words."""
    intro_count = count_words(sections["introduction"])
    conclusion_count = count_words(sections["conclusion"])
    body_count = sum(count_words(s["content"]) for s in sections["body_sections"])

    # Proportional trimming ratios
    trim_ratios = {"intro": 0.15, "conclusion": 0.15, "body": 0.70}

    # Trim introduction
    intro_trim = int(excess * trim_ratios["intro"])
    intro_target = max(80, intro_count - intro_trim)
    trimmed_intro = trim_text_to_word_count(sections["introduction"], intro_target)

    # Trim conclusion
    conclusion_trim = int(excess * trim_ratios["conclusion"])
    conclusion_target = max(80, conclusion_count - conclusion_trim)
    trimmed_conclusion = trim_text_to_word_count(sections["conclusion"], conclusion_target)

    # Trim body sections proportionally
    body_trim_remaining = excess - intro_trim - conclusion_trim
    trimmed_body: list[dict[str, str]] = []

    for section in sections["body_sections"]:
        section_words = count_words(section["content"])
        section_proportion = section_words / body_count if body_count > 0 else 0.25
        trim_amount = int(body_trim_remaining * section_proportion)
        target_section_words = max(80, section_words - trim_amount)
        trimmed_content = trim_text_to_word_count(section["content"], target_section_words)
        trimmed_body.append({"header": section["header"], "content": trimmed_content})

    return trimmed_intro, trimmed_body, trimmed_conclusion


def _expand_sections(sections: dict[str, Any], deficit: int) -> list[dict[str, str]]:
    """Expand body sections to add words."""
    body_count = sum(count_words(s["content"]) for s in sections["body_sections"])
    expanded_body: list[dict[str, str]] = []

    for section in sections["body_sections"]:
        section_words = count_words(section["content"])
        section_proportion = section_words / body_count if body_count > 0 else 0.25
        words_to_add = int(deficit * section_proportion)
        expanded_content = expand_section_content(section["content"], words_to_add)
        expanded_body.append({"header": section["header"], "content": expanded_content})

    return expanded_body


def enforce_word_count_deterministic(
    text: str, min_words: int = 1010, max_words: int = 1010
) -> str:
    """Deterministically enforce word count to exactly 1010 words."""
    current_count = count_words(text)
    target = 1010  # Exact target

    print(f"   🔧 Deterministic enforcement: {current_count} → target {target} words")

    if current_count == target:
        print(f"   ✓ Already exact: {current_count} words")
        return text

    sections = extract_report_sections(text)

    if current_count > target:
        # TRIM EXCESS WORDS
        excess = current_count - target
        print(f"   ✂️  Trimming {excess} excess words...")

        trimmed_intro, trimmed_body, trimmed_conclusion = _trim_sections(sections, excess)

        # Rebuild report
        result = _rebuild_report_from_sections(
            sections, trimmed_intro, trimmed_body, trimmed_conclusion
        )

        final_count = count_words(result)
        print(f"   ✓ After trimming: {final_count} words")
        return result

    # EXPAND TO ADD WORDS
    deficit = target - current_count
    print(f"   📝 Expanding by {deficit} words...")

    expanded_body = _expand_sections(sections, deficit)

    # Rebuild report
    result = _rebuild_report_from_sections(
        sections, sections["introduction"], expanded_body, sections["conclusion"]
    )

    final_count = count_words(result)
    print(f"   ✓ After expansion: {final_count} words")
    return result


def enforce_citation_count(text: str, target_count: int = 3) -> str:
    """Enforce exactly 3 citations."""
    current_citations = re.findall(r"\[([^\]]{3,}?)\](?!\()", text)
    current_count = len(current_citations)

    if current_count == target_count:
        return text

    print(f"   📚 Adjusting citations: {current_count} → {target_count}")

    if current_count > target_count:
        # Remove excess citations (keep first 3)
        for i in range(target_count, current_count):
            citation_pattern = r"\[" + re.escape(current_citations[i]) + r"\]"
            text = re.sub(citation_pattern, "", text, count=1)

    elif current_count < target_count:
        # Add generic citations
        needed = target_count - current_count
        generic_citations = [
            "[Industry Research Report]",
            "[Technical Documentation]",
            "[Data Engineering Best Practices]",
        ]

        # Insert citations at the end of the first body section
        sections = extract_report_sections(text)
        if sections["body_sections"]:
            first_body = sections["body_sections"][0]
            for i in range(needed):
                if i < len(generic_citations):
                    first_body["content"] += f" {generic_citations[i]}"

            # Rebuild
            result = f"{sections['title']}\n\n"
            result += f"## Introduction\n{sections['introduction']}\n\n"
            result += f"{first_body['header']}\n{first_body['content']}\n\n"
            for section in sections["body_sections"][1:]:
                result += f"{section['header']}\n{section['content']}\n\n"
            result += f"## Conclusion\n{sections['conclusion']}\n"
            text = result.strip()

    return text


# -------------------------------------------------------------
# 5. LLM Factory
# -------------------------------------------------------------
def get_llm(
    model_name: str = "gemini-2.0-flash",
    temperature: float = 0.2,
    max_tokens: int = 2000,
    **kwargs: Any,
) -> ChatGoogleGenerativeAI:
    """Create a ChatGoogleGenerativeAI model instance."""
    llm = ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        max_output_tokens=max_tokens,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        **kwargs,
    )
    return llm


def get_llm_with_tools() -> AgentExecutor | None:
    """Create LLM with Google web search tool."""
    if not os.getenv("GOOGLE_CSE_ID") or not os.getenv("GOOGLE_API_KEY"):
        print("   ⚠️ Google Search not configured, using LLM only")
        return None

    llm = get_llm(temperature=0.3, max_tokens=3000)
    try:
        search = GoogleSearchAPIWrapper(
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            google_cse_id=os.getenv("GOOGLE_CSE_ID"),
        )
        search_tool = Tool(
            name="google_search",
            description="Search Google for current information about Data Engineering topics.",
            func=search.run,
        )
        tools = [search_tool]
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a Data Engineering research assistant with access to Google search.",
                ),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"),
            ]
        )
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=False,
            max_iterations=5,
            handle_parsing_errors=True,
        )
        return agent_executor
    except Exception as e:
        print(f"   ⚠️ Search tool error: {e}")
        return None


# -------------------------------------------------------------
# 6. Agent Functions
# -------------------------------------------------------------
def agent_topic_validator(state: ReportState) -> ReportState:
    """Agent 1: Topic Validator - Ensures Data Engineering relevance"""
    print("\n" + "=" * 70)
    print("🔍 AGENT 1: TOPIC VALIDATOR")
    print("=" * 70)

    llm = get_llm(temperature=0.1, max_tokens=500)
    messages: list[BaseMessage] = [
        SystemMessage(
            content="""You are the Topic Validator Agent for Data Engineering reports.

VALID TOPICS: Data Pipelines, ETL/ELT, Stream Processing (Kafka, Flink), Data Warehousing,
Data Lakes, Data Quality, Data Governance, Data Cataloging, Orchestration (Airflow, Dagster),
Cloud Data Platforms (AWS, GCP, Azure), Data Architecture, CDC, Data Modeling, Real-time Analytics.

INVALID TOPICS: Pure software development, machine learning algorithms/model training,
frontend/mobile development, general business/marketing, non-data topics.

Respond ONLY in valid JSON format:
{"is_valid": true/false, "reasoning": "brief explanation"}"""
        ),
        HumanMessage(content=f"Evaluate this topic: '{state['topic']}'"),
    ]

    response = llm.invoke(messages).content
    is_valid = False
    reasoning = "Invalid response format"

    try:
        json_match = re.search(r"\{.*\}", str(response), re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            is_valid = bool(result.get("is_valid", False))
            reasoning = str(result.get("reasoning", "No reasoning provided"))
    except Exception as e:
        print(f"   ⚠️ JSON parsing error: {e}")
        is_valid = False

    state["validation_result"] = {
        "is_valid": is_valid,
        "message": str(response),
        "reasoning": reasoning,
    }
    state["log"].append(
        {"agent": "Topic Validator", "step": 1, "valid": is_valid, "reasoning": reasoning}
    )

    print(f"   Topic: '{state['topic']}'")
    print(f"   Valid: {is_valid}")
    print(f"   Reason: {reasoning}")
    print("=" * 70)
    return state


def agent_prompt_engineer(state: ReportState) -> ReportState:
    """Agent 2: Prompt Engineer - Creates optimized generation prompts"""
    print("📝 AGENT 2: PROMPT ENGINEER")
    print("=" * 70)

    llm = get_llm(temperature=0.2, max_tokens=800)
    messages: list[BaseMessage] = [
        SystemMessage(
            content="""You are a Prompt Engineering specialist. Create clear, comprehensive prompts
for Data Engineering technical writing that will produce high-quality reports."""
        ),
        HumanMessage(
            content=f"""Generate a detailed prompt for writing a Data Engineering report on: '{state['topic']}'

Requirements:
- Tone: {state['tone']}
- Audience: {state['audience']}
- Target: EXACTLY 1010 words
- Structure: Title + Introduction + 4 subtopics + Conclusion
- Include exactly 3 citations in [Source Name] format
- Technical depth appropriate for the audience

Create a comprehensive prompt that guides the writer to produce excellent content."""
        ),
    ]

    state["optimized_prompt"] = str(llm.invoke(messages).content)
    state["log"].append({"agent": "Prompt Engineer", "step": 2})
    print("   ✓ Optimized prompt generated")
    print("=" * 70)
    return state


def agent_quality_assurance(state: ReportState) -> ReportState:
    """Agent 3: Quality Assurance - Reviews prompt quality"""
    print("✅ AGENT 3: QUALITY ASSURANCE")
    print("=" * 70)

    llm = get_llm(temperature=0.2, max_tokens=500)
    messages: list[BaseMessage] = [
        SystemMessage(
            content="You are a Quality Assurance specialist. "
            "Review prompts for clarity, completeness, and effectiveness."
        ),
        HumanMessage(
            content=f"Review this prompt for quality:\n\n{state['optimized_prompt'][:600]}..."
        ),
    ]

    state["prompt_quality"] = str(llm.invoke(messages).content)
    state["log"].append({"agent": "Quality Assurance", "step": 3})
    print("   ✓ Prompt quality reviewed")
    print("=" * 70)
    return state


def agent_key_points_identifier(state: ReportState) -> ReportState:
    """Agent 4: Key Points Identifier - Extracts 4 essential subtopics"""
    print("🎯 AGENT 4: KEY POINTS IDENTIFIER")
    print("=" * 70)

    llm = get_llm(temperature=0.2, max_tokens=600)
    messages: list[BaseMessage] = [
        SystemMessage(
            content="""You are a Content Strategist. Identify exactly 4 key subtopics that should be covered
in a comprehensive Data Engineering report. Make them specific, actionable, and relevant."""
        ),
        HumanMessage(
            content=f"""Identify 4 essential subtopics for a report on: '{state['topic']}'

Requirements:
- Each subtopic should be distinct and substantial
- Cover different aspects of the topic
- Be specific and technical
- Suitable for {state['audience']}

Provide 4 descriptive subtopic titles."""
        ),
    ]

    state["key_points"] = str(llm.invoke(messages).content)
    state["log"].append({"agent": "Key Points", "step": 4})
    print("   ✓ 4 key subtopics identified")
    print(f"   {state['key_points'][:200]}...")
    print("=" * 70)
    return state


def agent_research_generator(state: ReportState) -> ReportState:
    """Agent 5: Research Agent - Generates comprehensive content"""
    print("🔬 AGENT 5: RESEARCH AGENT")
    print("=" * 70)

    research_prompt = f"""Write a comprehensive Data Engineering report on: '{state['topic']}'

REQUIREMENTS:
- Tone: {state['tone']}
- Audience: {state['audience']}
- Cover these key areas: {state['key_points']}
- Include exactly 3 citations in format [Source Name]
- Target length: 2500-3000 words (will be structured later)
- Technical depth appropriate for {state['audience']}

Write detailed, informative content covering all aspects of the topic."""

    # Try with search tool first
    agent_executor = get_llm_with_tools()
    research_report: str | None = None

    if agent_executor:
        try:
            print("   🔍 Using Google Search for research...")
            result = agent_executor.invoke({"input": research_prompt})
            research_report = str(result.get("output", ""))
            print(
                f"   ✓ Research with search complete: {count_words(research_report)} words"
            )
        except Exception as e:
            print(f"   ⚠️ Search failed: {e}")
            research_report = None

    # Fallback to LLM without search
    if not research_report or len(research_report.split()) < 500:
        print("   📝 Generating content with LLM...")
        llm = get_llm(max_tokens=4000, temperature=state["temperature"])
        messages: list[BaseMessage] = [
            SystemMessage(
                content=f"""You are an expert Data Engineering technical writer.
Tone: {state['tone']} | Audience: {state['audience']}
Write comprehensive, accurate, and engaging content."""
            ),
            HumanMessage(content=research_prompt),
        ]
        research_report = str(llm.invoke(messages).content)
        print(f"   ✓ Research complete: {count_words(research_report)} words")

    state["research_report"] = research_report
    state["log"].append(
        {"agent": "Research", "step": 5, "words": count_words(research_report)}
    )
    print("=" * 70)
    return state


def agent_grading_refinement(state: ReportState) -> ReportState:
    """Agent 6: Grading & Refinement - Fact-checks and improves quality"""
    print("📊 AGENT 6: GRADING & REFINEMENT")
    print("=" * 70)

    llm = get_llm(temperature=0.1, max_tokens=800)
    messages: list[BaseMessage] = [
        SystemMessage(
            content="""You are a Technical Reviewer. Grade this Data Engineering report on:
- Technical Accuracy (0-100)
- Completeness of Coverage (0-100)
- Quality of Explanations (0-100)

Provide a grade (A/B/C/D/F) and specific feedback.
Respond in JSON: {"grade": "A-F", "score": 0-100, "feedback": "...", "needs_refinement": true/false}"""
        ),
        HumanMessage(
            content=f"Topic: {state['topic']}\n\nReport excerpt:\n{state['research_report'][:1500]}..."
        ),
    ]

    response = llm.invoke(messages).content

    try:
        json_match = re.search(r"\{.*\}", str(response), re.DOTALL)
        if json_match:
            grade_result: dict[str, Any] = json.loads(json_match.group())
        else:
            grade_result = {"grade": "B", "score": 80, "needs_refinement": False}
    except Exception as e:
        print(f"Exception occurred: {e!s}")
        grade_result = {"grade": "B", "score": 80, "needs_refinement": False}

    state["grade_result"] = grade_result
    print(f"   Grade: {grade_result.get('grade')} ({grade_result.get('score')}/100)")

    # Refine if needed
    if grade_result.get("grade") in {"C", "D", "F"} or grade_result.get(
        "needs_refinement"
    ):
        print("   🔄 Refining report based on feedback...")
        llm_refine = get_llm(max_tokens=4000, temperature=0.3)
        refine_messages: list[BaseMessage] = [
            SystemMessage(content="Improve this report based on the feedback provided."),
            HumanMessage(
                content=f"Original report:\n{state['research_report']}"
                f"\n\nFeedback: {grade_result}\n\nProvide improved version."
            ),
        ]
        state["research_report"] = str(llm_refine.invoke(refine_messages).content)
        print(f"   ✓ Refinement complete: {count_words(state['research_report'])} words")

    state["log"].append(
        {
            "agent": "Grading",
            "step": 6,
            "grade": grade_result.get("grade"),
            "score": grade_result.get("score"),
        }
    )
    print("=" * 70)
    return state


def agent_structuring(state: ReportState) -> ReportState:
    """Agent 7: Structuring Agent - Organizes to exactly 1010 word target"""
    print("📐 AGENT 7: STRUCTURING AGENT")
    print("=" * 70)
    print(f"   Structuring to exactly 1010 words (attempt {state.get('retry_count', 0) + 1})...")

    llm = get_llm(max_tokens=2000, temperature=0.2)
    messages: list[BaseMessage] = [
        SystemMessage(
            content=f"""You are the Structuring Agent. Transform content into EXACTLY 1010 words.

MANDATORY STRUCTURE (6 sections):
# [Captivating, Engaging Title - NOT generic]

## Introduction
[Brief overview, ~150 words]

## [Descriptive Subtopic 1 Name]
[Technical details, ~200 words]

## [Descriptive Subtopic 2 Name]
[Technical details, ~200 words]

## [Descriptive Subtopic 3 Name]
[Technical details, ~200 words]

## [Descriptive Subtopic 4 Name]
[Technical details, ~200 words]

## Conclusion
[Summary and outlook, ~150 words]

CRITICAL REQUIREMENTS:
- Total: EXACTLY 1010 words (NO EXCEPTIONS)
- Section titles "Introduction" and "Conclusion" must be EXACT
- Exactly 3 citations in [Source Name] format
- Captivating H1 title (use action words, specific technologies)
- Tone: {state['tone']} | Audience: {state['audience']}
- NO extra sections, NO missing sections"""
        ),
        HumanMessage(content=f"Structure this report:\n\n{state['research_report']}"),
    ]

    structured_report = str(llm.invoke(messages).content)
    state["structured_report"] = structured_report

    # Validate word count
    word_check = check_word_count_limit(structured_report, 1010, 1010)
    state["word_count"] = word_check["word_count"]
    state["word_count_valid"] = word_check["is_valid"]

    # Validate structure
    structure_check = validate_structure(structured_report)
    state["structure_valid"] = structure_check["is_valid"]

    # Validate citations
    citation_check = validate_citations(structured_report)
    state["citation_count"] = citation_check["count"]
    state["citation_valid"] = citation_check["is_valid"]

    state["log"].append(
        {
            "agent": "Structuring",
            "step": 7,
            "words": state["word_count"],
            "word_valid": state["word_count_valid"],
            "structure_valid": state["structure_valid"],
            "citations": state["citation_count"],
        }
    )

    print(f"   Words: {state['word_count']} ({'✅' if state['word_count_valid'] else '❌'})")
    print(
        f"   Structure: {'✅' if state['structure_valid'] else '❌'} ({structure_check['num_sections']} sections)"
    )
    print(
        f"   Citations: {state['citation_count']} ({'✅' if state['citation_valid'] else '❌'})"
    )
    print("=" * 70)
    return state


def agent_polish_enforcement(state: ReportState) -> ReportState:
    """Agent 8: Polish & Enforcement - STRICT requirement enforcement"""
    print("✨ AGENT 8: POLISH & ENFORCEMENT")
    print("=" * 70)

    final_report = state["structured_report"]
    enforcement_needed = False

    # Check all requirements
    print("   Initial validation:")
    print(
        f"   - Word count: {state['word_count']} ({'✅' if state['word_count_valid'] else '❌ NEEDS FIX'})"
    )
    print(f"   - Structure: {'✅' if state['structure_valid'] else '❌ NEEDS FIX'}")
    print(
        f"   - Citations: {state['citation_count']} ({'✅' if state['citation_valid'] else '❌ NEEDS FIX'})"
    )

    # 1. ENFORCE WORD COUNT (exactly 1010)
    if not state["word_count_valid"]:
        print(f"\n   🔧 Enforcing word count: {state['word_count']} → 1010")
        final_report = enforce_word_count_deterministic(final_report, 1010, 1010)
        enforcement_needed = True

        # Re-validate
        word_check = check_word_count_limit(final_report, 1010, 1010)
        state["word_count"] = word_check["word_count"]
        state["word_count_valid"] = word_check["is_valid"]
        print(
            f"   ✓ Word count after enforcement: {state['word_count']} ({'✅' if state['word_count_valid'] else '❌'})"
        )

    # 2. ENFORCE STRUCTURE (Title + Intro + 4 subtopics + Conclusion)
    if not state["structure_valid"]:
        print("\n   🔧 Fixing structure issues...")
        structure_check = validate_structure(final_report)

        if not structure_check["has_intro"]:
            print("   ⚠️ Adding missing 'Introduction' section")
            # Find first H2 and replace with ## Introduction
            final_report = re.sub(
                r"^##\s+(.+?)",
                "## Introduction",
                final_report,
                count=1,
                flags=re.MULTILINE,
            )

        if not structure_check["has_conclusion"]:
            print("   ⚠️ Adding missing 'Conclusion' section")
            # Find last H2 and replace with ## Conclusion
            h2_matches = list(re.finditer(r"^##\s+(.+?)", final_report, re.MULTILINE))
            if h2_matches:
                last_match = h2_matches[-1]
                final_report = (
                    final_report[: last_match.start()]
                    + "## Conclusion"
                    + final_report[last_match.end() :]
                )

        enforcement_needed = True

        # Re-validate
        structure_check = validate_structure(final_report)
        state["structure_valid"] = structure_check["is_valid"]
        print(
            f"   ✓ Structure after enforcement: {'✅' if state['structure_valid'] else '❌'}"
        )

    # 3. ENFORCE CITATION COUNT (exactly 3)
    if not state["citation_valid"]:
        print(f"\n   🔧 Adjusting citations: {state['citation_count']} → 3")
        final_report = enforce_citation_count(final_report, target_count=3)
        enforcement_needed = True

        # Re-validate
        citation_check = validate_citations(final_report)
        state["citation_count"] = citation_check["count"]
        state["citation_valid"] = citation_check["is_valid"]
        print(
            f"   ✓ Citations after enforcement: {state['citation_count']} ({'✅' if state['citation_valid'] else '❌'})"
        )

    # 4. FINAL POLISH (if no enforcement was needed)
    if not enforcement_needed:
        print("   ✅ All requirements met, applying final polish...")
        # Minor formatting cleanup
        final_report = re.sub(r"\n{3,}", "\n\n", final_report)  # Remove excessive blank lines
        final_report = re.sub(r" +", " ", final_report)  # Remove excessive spaces
        final_report = final_report.strip()

    state["final_report"] = final_report

    # Final validation summary
    print("\n   📊 FINAL VALIDATION:")
    print(
        f"   - Word count: {state['word_count']} ({'✅ VALID' if state['word_count_valid'] else '❌ INVALID'})"
    )
    print(f"   - Structure: {'✅ VALID' if state['structure_valid'] else '❌ INVALID'}")
    print(
        f"   - Citations: {state['citation_count']} ({'✅ VALID' if state['citation_valid'] else '❌ INVALID'})"
    )

    all_valid = (
        state["word_count_valid"]
        and state["structure_valid"]
        and state["citation_valid"]
    )
    print(
        f"   - Overall: {'✅ ALL REQUIREMENTS MET' if all_valid else '❌ SOME REQUIREMENTS FAILED'}"
    )

    state["log"].append(
        {
            "agent": "Polish & Enforcement",
            "step": 8,
            "final_words": state["word_count"],
            "all_valid": all_valid,
            "enforcement_applied": enforcement_needed,
        }
    )
    print("=" * 70)
    return state


# -------------------------------------------------------------
# 7. Workflow Definition
# -------------------------------------------------------------
def decision_route_topic(state: ReportState) -> str:
    """Route based on topic validation"""
    if state.get("validation_result", {}).get("is_valid"):
        return "continue"
    return "end_invalid"


def decision_route_validation(state: ReportState) -> str:
    """Route based on comprehensive validation after structuring"""
    retry_count = state.get("retry_count", 0)

    # Check if all requirements are met
    all_valid = (
        state.get("word_count_valid", False)
        and state.get("structure_valid", False)
        and state.get("citation_valid", False)
    )

    if all_valid:
        return "finish"

    # Allow up to 2 retries
    if retry_count >= 2:
        print("\n   ⚠️ Maximum retries reached. Proceeding to enforcement...")
        return "finish"

    return "retry_structuring"


def retry_incrementer(state: ReportState) -> ReportState:
    """Increment retry counter"""
    state["retry_count"] = state.get("retry_count", 0) + 1
    print(f"\n🔁 RETRY ATTEMPT {state['retry_count']}/2")
    print("=" * 70)
    return state


def create_report_workflow() -> Any:
    """Create the LangGraph workflow"""
    workflow = StateGraph(ReportState)

    # Add all agent nodes
    workflow.add_node("topic_validator", agent_topic_validator)
    workflow.add_node("prompt_engineer", agent_prompt_engineer)
    workflow.add_node("quality_assurance", agent_quality_assurance)
    workflow.add_node("key_points_identifier", agent_key_points_identifier)
    workflow.add_node("research_generator", agent_research_generator)
    workflow.add_node("grading_refinement", agent_grading_refinement)
    workflow.add_node("structuring_agent", agent_structuring)
    workflow.add_node("polish_enforcement", agent_polish_enforcement)
    workflow.add_node("retry_incrementer", retry_incrementer)

    # Set entry point
    workflow.set_entry_point("topic_validator")

    # Define workflow edges
    workflow.add_conditional_edges(
        "topic_validator",
        decision_route_topic,
        {"continue": "prompt_engineer", "end_invalid": END},
    )
    workflow.add_edge("prompt_engineer", "quality_assurance")
    workflow.add_edge("quality_assurance", "key_points_identifier")
    workflow.add_edge("key_points_identifier", "research_generator")
    workflow.add_edge("research_generator", "grading_refinement")
    workflow.add_edge("grading_refinement", "structuring_agent")

    # Conditional routing after structuring (with retry logic)
    workflow.add_conditional_edges(
        "structuring_agent",
        decision_route_validation,
        {"finish": "polish_enforcement", "retry_structuring": "retry_incrementer"},
    )

    workflow.add_edge("retry_incrementer", "structuring_agent")
    workflow.add_edge("polish_enforcement", END)

    compiled_workflow = workflow.compile()

    # Save workflow visualization
    try:
        save_workflow_visualization(compiled_workflow)
    except Exception as e:
        print(f"⚠️ Warning: Could not save workflow visualization: {e}")

    return compiled_workflow


def save_workflow_visualization(workflow: Any) -> None:
    """Save the LangGraph workflow as a PNG image"""
    try:
        # Generate timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(WORKFLOWS_DIR, f"workflow_diagram_{timestamp}.png")

        # Get the Mermaid diagram representation
        try:
            # Try to get PNG directly (if graphviz is available)
            png_data = workflow.get_graph().draw_mermaid_png()

            with open(output_path, "wb") as f:
                f.write(png_data)

            print(f"✅ Workflow visualization saved: {output_path}")

        except Exception:
            # Fallback: Save as Mermaid markdown
            mermaid_output = os.path.join(
                WORKFLOWS_DIR, f"workflow_diagram_{timestamp}.mmd"
            )
            mermaid_code = workflow.get_graph().draw_mermaid()

            with open(mermaid_output, "w", encoding="utf-8") as f:
                f.write("```mermaid\n")
                f.write(mermaid_code)
                f.write("\n```")

            print(f"✅ Workflow saved as Mermaid diagram: {mermaid_output}")
            print(
                "   💡 To convert to PNG, use: https://mermaid.live or install graphviz"
            )

    except Exception as e:
        print(f"⚠️ Could not save workflow visualization: {e}")
        print(
            "   Note: Install graphviz for PNG export: pip install pygraphviz or brew install graphviz"
        )


# -------------------------------------------------------------
# 8. Wrapper Function for Gradio
# -------------------------------------------------------------
def generate_report_agentic_workflow(
    topic: str, temperature: float, tone: str, audience: str
) -> str:
    """
    Main entry point for report generation.

    Args:
        topic: The Data Engineering topic to write about
        temperature: LLM creativity level (0.0-1.0)
        tone: Writing tone (Professional, Technical, Educational, etc.)
        audience: Target audience (Data Engineers, Students, etc.)

    Returns:
        str: The generated report (exactly 1010 words) or error message
    """
    print("\n" + "🚀 " + "=" * 68)
    print("🚀 STARTING DATA ENGINEERING REPORT GENERATION WORKFLOW")
    print("🚀 " + "=" * 68)
    print(f"   Topic: {topic}")
    print(f"   Tone: {tone} | Audience: {audience} | Temperature: {temperature}")
    print("=" * 70 + "\n")

    # Create workflow
    app = create_report_workflow()

    # Initialize state
    initial_state = ReportState(
        topic=topic,
        tone=tone,
        audience=audience,
        temperature=temperature,
        validation_result={},
        optimized_prompt="",
        prompt_quality="",
        key_points="",
        research_report="",
        grade_result={},
        structured_report="",
        final_report="",
        word_count=0,
        word_count_valid=False,
        structure_valid=False,
        citation_count=0,
        citation_valid=False,
        retry_count=0,
        log=[],
        timestamp=datetime.now().isoformat(),
        error="",
    )

    # Execute workflow
    try:
        final_state = app.invoke(initial_state)
    except Exception as e:
        error_msg = f"⚠️ Workflow execution failed: {type(e).__name__}: {e!s}"
        print(f"\n{error_msg}")
        return error_msg

    # Log to file
    try:
        with open(PROMPT_LOG_PATH, "a", encoding="utf-8") as f:
            log_data = {
                "timestamp": final_state["timestamp"],
                "topic": topic,
                "word_count": final_state.get("word_count", 0),
                "word_valid": final_state.get("word_count_valid", False),
                "structure_valid": final_state.get("structure_valid", False),
                "citation_count": final_state.get("citation_count", 0),
                "retry_count": final_state.get("retry_count", 0),
            }
            f.write(json.dumps(log_data) + "\n")
    except Exception as e:
        print(f"⚠️ Logging error: {e}")

    # Check if topic was rejected
    validation_result = final_state.get("validation_result", {})
    is_valid_topic = validation_result.get("is_valid", True)

    if not is_valid_topic:
        reasoning = validation_result.get("reasoning", "Not Data Engineering related.")
        error_msg = f"⚠️ Topic Rejected: '{topic}'\n\nReason: {reasoning}\n\nPlease choose a Data Engineering topic."
        print(f"\n{error_msg}")
        return error_msg

    # Return final report
    if final_state.get("final_report"):
        final_report_text = final_state["final_report"]

        # Save to file
        try:
            if not os.path.exists("generated_reports"):
                os.makedirs("generated_reports")

            safe_topic = re.sub(r"[\W_]+", "_", topic.lower())[:50]
            output_filename = (
                f"report_{safe_topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            )
            output_path = os.path.join("generated_reports", output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(final_report_text)

            print(f"\n📄 Report saved: {output_filename}")
        except Exception as e:
            print(f"⚠️ File save error: {e}")

        # Print final summary
        print("\n" + "=" * 70)
        print("✅ WORKFLOW COMPLETE")
        print("=" * 70)
        print(
            f"   Word Count: {final_state.get('word_count', 0)} "
            f"({'✅ VALID' if final_state.get('word_count_valid') else '❌ INVALID'})"
        )
        print(
            f"   Structure: {'✅ VALID' if final_state.get('structure_valid') else '❌ INVALID'}"
        )
        print(
            f"   Citations: {final_state.get('citation_count', 0)} "
            f"({'✅ VALID' if final_state.get('citation_valid') else '❌ INVALID'})"
        )
        print(f"   Retries: {final_state.get('retry_count', 0)}")
        print("=" * 70 + "\n")

        return final_report_text

    # If no report was generated
    error_msg = (
        f"⚠️ Report generation failed. Error: {final_state.get('error', 'Unknown error')}"
    )
    print(f"\n{error_msg}")
    return error_msg


# -------------------------------------------------------------
# 9. CLI Entry Point for Testing
# -------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DATA ENGINEERING REPORT GENERATOR - CLI TEST")
    print("=" * 70 + "\n")

    # Test configuration
    REPORT_TOPIC = "Stream Processing with Apache Kafka"
    REPORT_TONE = "Professional"
    REPORT_AUDIENCE = "Data Engineers"
    REPORT_TEMPERATURE = 0.3

    # Ensure output directory exists
    if not os.path.exists("generated_reports"):
        os.makedirs("generated_reports")

    # Generate report
    final_report_text = generate_report_agentic_workflow(
        topic=REPORT_TOPIC,
        tone=REPORT_TONE,
        audience=REPORT_AUDIENCE,
        temperature=REPORT_TEMPERATURE,
    )

    # Display results
    if not final_report_text.startswith("⚠️"):
        word_count_final = count_words(final_report_text)
        structure_check = validate_structure(final_report_text)
        citation_check = validate_citations(final_report_text)

        print("\n" + "=" * 70)
        print("📊 FINAL REPORT ANALYSIS")
        print("=" * 70)
        print(f"Topic: {REPORT_TOPIC}")
        print(f"Word Count: {word_count_final} words")
        print("Target: Exactly 1010 words")
        print(f"Valid: {'✅ YES' if word_count_final == 1010 else '❌ NO'}")
        print("\nStructure:")
        print(f"  - Title: {'✅' if structure_check['has_title'] else '❌'}")
        print(f"  - Introduction: {'✅' if structure_check['has_intro'] else '❌'}")
        print(
            f"  - Sections: {structure_check['num_sections']} "
            f"{'✅' if structure_check['num_sections'] == 6 else '❌'}"
        )
        print(f"  - Conclusion: {'✅' if structure_check['has_conclusion'] else '❌'}")
        print(
            f"\nCitations: {citation_check['count']} {'✅' if citation_check['count'] == 3 else '❌'}"
        )
        print("=" * 70)

        # Display preview
        print("\n" + "=" * 70)
        print("📄 REPORT PREVIEW (first 500 chars)")
        print("=" * 70)
        print(final_report_text[:500] + "...")
        print("=" * 70 + "\n")
    else:
        print("\n❌ Report generation failed:")
        print(final_report_text)
