"""
Agentic system for automated report generation.

Architecture:
1. Planner - Creates structured outline with word counts
2. Researcher - Gathers information from Wikipedia
3. Writer - Generates report content
4. Validator - Checks word count and structure
5. Reviewer - Provides quality feedback

The system uses a conditional loop: if validation fails, it revises the report.
"""

import os
from typing import List, Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .tools import calculate_word_count_difference, count_words, get_wikipedia_tool, validate_structure

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================


def get_llm(temperature: float = 0.7):
    """
    Initializes Google Gemini LLM.

    Design Decision: Use temperature=0.7 for balance between creativity
    and factual accuracy. Lower for fact-checking, higher for creative writing.
    """
    if "GOOGLE_API_KEY" not in os.environ:
        raise EnvironmentError(
            "GOOGLE_API_KEY not found. Set it as an environment variable."
        )
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=temperature
    )


# ============================================================================
# STATE DEFINITION
# ============================================================================

class ReportState(TypedDict):
    """
    State passed between agents in the graph.

    Design Decision: Keep state minimal but comprehensive. Each field
    represents output from a specific agent/node.
    """
    # Input
    topic: str
    temperature: float  # Only affects writer
    max_iterations: int

    # Planner outputs
    plan: dict

    # Researcher outputs
    research_content: str

    # Writer outputs
    draft_report: str
    iteration_count: int  # Track revision attempts

    # Validator outputs
    word_count: int
    validation_result: dict

    # Reviewer outputs
    review_feedback: str

    # Final output
    final_report: str
    metadata: dict  # Store stats about generation process


# ============================================================================
# PYDANTIC MODELS FOR STRUCTURED OUTPUT
# ============================================================================

class SectionPlan(BaseModel):
    """Schema for a single report section."""
    title: str = Field(description="Section title (clear and descriptive)")
    key_points: List[str] = Field(
        description="3-5 key points to cover in this section"
    )
    word_count: int = Field(
        description="Target word count for this section",
        ge=50,  # At least 50 words per section
    )


class ReportPlan(BaseModel):
    """
    Schema for complete report plan.

    Design Decision: Use structured output to ensure consistent planning.
    This makes the writer's job easier and more predictable.
    """
    title: str = Field(description="Compelling report title")
    sections: List[SectionPlan] = Field(
        description="List of sections (must include intro, body sections, conclusion)",
        min_items=4,  # Minimum: intro + 2 body + conclusion
    )
    total_word_count: int = Field(
        description="Sum of all section word counts (should be ~1000)",
        ge=800,
        le=1000,
    )


# ============================================================================
# AGENT NODES
# ============================================================================

def planner_node(state: ReportState) -> dict:
    """
    Creates a detailed plan for the report.

    Design Decision: Use structured output (Pydantic) to ensure the plan
    has all required fields. This prevents downstream errors.
    """
    print("\n" + "=" * 60)
    print("PLANNER AGENT: Creating report structure...")
    print("=" * 60)

    llm = get_llm(temperature=0.3)  # Lower temp for structured planning

    prompt = ChatPromptTemplate.from_template("""
You are an expert technical report planner specializing in software engineering and MLOps topics.

Create a detailed plan for a report on: "{topic}"

Requirements:
1. The report MUST have exactly 1000 words (±50 words tolerance)
2. Structure: Introduction → 2-3 Body Sections → Conclusion
3. Introduction: ~150 words (context, importance, scope)
4. Each body section: ~250-350 words (detailed technical content)
5. Conclusion: ~150 words (summary, future outlook)

For each section, specify:
- A clear, descriptive title
- 3-5 key points to cover
- Exact target word count

**CRITICAL: The total_word_count field MUST be between 950-1050.
Calculate the sum of all section word counts and ensure it falls in this range.**

Make the plan comprehensive enough to guide a technical writer.
Focus on practical, real-world aspects of the topic.
""")

    # Use structured output for reliability
    structured_llm = llm.with_structured_output(ReportPlan)
    chain = prompt | structured_llm

    plan = chain.invoke({"topic": state["topic"]})

    print(f"\n✓ Created plan with {len(plan.sections)} sections")
    print(f"  Title: {plan.title}")
    for i, section in enumerate(plan.sections, 1):
        print(f"  {i}. {section.title} ({section.word_count} words)")

    return {
        "plan": plan.dict(),
        "iteration_count": 0
    }


def researcher_node(state: ReportState) -> dict:
    """
    Gathers information using Wikipedia tool.
    Uses LLM to determine optimal search queries based on the report plan.
    """
    print("\n" + "=" * 60)
    print("RESEARCHER AGENT: Gathering information...")
    print("=" * 60)

    llm = get_llm(temperature=0.2)
    wikipedia_tool = get_wikipedia_tool()

    topic = state["topic"]
    sections = state["plan"]["sections"]

    # Use LLM to generate smart search queries
    query_prompt = ChatPromptTemplate.from_template("""
You are a research assistant planning information gathering for a technical report.

Topic: {topic}

Report sections that need coverage:
{section_titles}

Generate 3-5 specific Wikipedia search queries that will gather the information needed.
Make queries specific and technical. Return ONLY the search queries, one per line.

Example output:
MLOps
Continuous integration machine learning
Model monitoring production
""")

    section_titles = "\n".join([f"- {s['title']}" for s in sections])

    response = llm.invoke(
        query_prompt.format(topic=topic, section_titles=section_titles)
    )

    # Parse queries from LLM response
    research_queries = [
        q.strip() for q in response.content.strip().split('\n')
        if q.strip() and not q.startswith('#')
    ][:5]  # Limit to 5 queries

    print(f"\n  Generated {len(research_queries)} search queries:")
    for q in research_queries:
        print(f"    • {q}")

    # Execute searches
    research_results = []
    for query in research_queries:
        print(f"\n  Searching Wikipedia: {query}")
        result = wikipedia_tool._run(query)
        research_results.append(f"### Research: {query}\n\n{result}\n\n")

    research_content = "\n".join(research_results)

    print("\n✓ Research complete")
    print(f"  Total content gathered: {count_words(research_content)} words")

    return {"research_content": research_content}


def writer_node(state: ReportState) -> dict:
    """Writes or revises the report."""
    iteration = state.get("iteration_count", 0)

    print("\n" + "=" * 60)
    print(f"✍️  WRITER: {'Writing' if iteration == 0 else 'Revising'} report (Iteration {iteration + 1})...")
    print("=" * 60)

    llm = get_llm(temperature=state.get("temperature", 0.7))

    if iteration == 0:
        # Initial draft
        prompt = ChatPromptTemplate.from_template("""
You are an expert technical writer.

TOPIC: {title}

PLAN:
{plan_details}

RESEARCH:
{research_content}

INSTRUCTIONS:
1. Write a complete, well-structured technical report
2. Follow the plan exactly - use specified section titles
3. **CRITICAL**: Match the target word counts:
{word_count_breakdown}

4. Style: Professional, technical, accessible
5. Use specific examples from research
6. Use markdown headers (##) for sections
7. Structure: Introduction → Body Sections → Conclusion

Write the COMPLETE report now.

CRITICAL INSTRUCTIONS:
1. Output ONLY the report content
2. Do NOT include any meta-commentary like "I will now write..." or "New word count: X"
3. Start directly with ## Introduction
4. End with the last sentence of the Conclusion
5. No word counts, no explanations, just the report.
""")

        plan_details = "\n".join([
            f"**{s['title']}** ({s['word_count']} words)\n"
            f"Points: {', '.join(s['key_points'])}"
            for s in state["plan"]["sections"]
        ])

        word_count_breakdown = "\n".join([
            f"   - {s['title']}: {s['word_count']} words"
            for s in state["plan"]["sections"]
        ])

        response = llm.invoke(
            prompt.format(
                title=state["plan"]["title"],
                plan_details=plan_details,
                research_content=state["research_content"][:8000],
                word_count_breakdown=word_count_breakdown
            )
        )

    else:
        # Revision
        validation = state.get("validation_result", {})
        word_count_feedback = validation.get("word_count_feedback", {})
        difference = word_count_feedback.get("difference", 0)
        current_wc = state.get("word_count", 0)

        # Calculate per-section adjustment
        num_sections = len(state["plan"]["sections"])
        words_per_section = abs(difference) // num_sections

        if difference > 0:
            # Too long - condense
            revision_instructions = f"""
CONDENSING STRATEGY (Remove {abs(difference)} words total):
- Remove approximately {words_per_section} words from each section
- Focus on: redundant explanations, repetitive examples, verbose sentences
- DO NOT remove entire paragraphs or key technical concepts
- Keep all section headers unchanged
- Trim filler words and condense sentences instead of cutting content

Example of good condensing:
Before: "Continuous Integration and Continuous Delivery (CI/CD) pipelines are absolutely crucial and essential for modern software development"
After: "CI/CD pipelines are crucial for modern software development"
"""  # noqa: E501
        else:
            # Too short - expand
            revision_instructions = f"""
EXPANSION STRATEGY (Add {abs(difference)} words total):
- Add approximately {words_per_section} words to each section
- Focus on: concrete examples, technical details, tool names, specific metrics
- Use the research material below to add factual content
- DO NOT add filler or fluff
- Expand by adding NEW information, not rephrasing existing content

Example of good expansion:
Before: "Models must be monitored."
After: "Models must be monitored continuously using tools like Prometheus and Grafana,
        tracking metrics such as prediction accuracy (precision, recall, F1-score),
        latency (p95, p99), and throughput (requests per second)."

ADD CONTENT FROM RESEARCH:
{{state["research_content"][:2000]}}
"""

        prompt = ChatPromptTemplate.from_template("""
You are revising a technical report that missed the word count target.

CURRENT DRAFT:
{current_draft}

WORD COUNT ISSUE:
- Current: {current_word_count} words
- Target: 1000 words (±50 tolerance)
- Difference: {difference} words ({status})

{revision_instructions}

CRITICAL INSTRUCTIONS:
1. Keep the EXACT same structure and section headers
2. Make TARGETED edits - don't rewrite everything
3. {action} {words_needed} words TOTAL across all sections
4. Maintain technical accuracy and professional tone
5. Count carefully as you write
6. Output ONLY the report content
7. Do NOT include any meta-commentary like "I will now write..." or "New word count: X"
8. Start directly with ## Introduction
9. End with the last sentence of the Conclusion
10. No word counts, no explanations, just the report

OUTPUT: The complete revised report with all sections.
""")

        # More specific context
        response = llm.invoke(
            prompt.format(
                current_draft=state["draft_report"],
                current_word_count=current_wc,
                difference=abs(difference),
                status="too long" if difference > 0 else "too short",
                revision_instructions=revision_instructions,
                action="REMOVE" if difference > 0 else "ADD",
                words_needed=abs(difference),
                research_snippet=state["research_content"][:2000]
            )
        )

    draft = response.content
    wc = count_words(draft)

    print(f"\n✓ {'Draft' if iteration == 0 else 'Revision'} completed")
    print(f"  Word count: {wc} words")

    return {
        "draft_report": draft,
        "iteration_count": iteration + 1
    }


def validator_node(state: ReportState) -> dict:
    """
    Validates word count and structure.

    Design Decision: Use Python tools for validation, not LLM.
    This is 100% reliable and much faster than asking an LLM to count.
    """
    print("\n" + "=" * 60)
    print("VALIDATOR: Checking report quality...")
    print("=" * 60)

    draft = state["draft_report"]

    # Check word count
    word_count = count_words(draft)
    word_count_result = calculate_word_count_difference(word_count)

    # Check structure
    structure_result = validate_structure(draft)

    # Combined validation
    is_valid = (
        word_count_result["within_tolerance"] and
        structure_result["is_valid"]
    )

    validation_result = {
        "is_valid": is_valid,
        "word_count_result": word_count_result,
        "structure_result": structure_result,
        "word_count_feedback": word_count_result
    }

    # Print results
    print(f"\n  {word_count_result['feedback']}")
    print(f"  Sections found: {structure_result['section_count']}")
    print(f"  Has introduction: {'✓' if structure_result['has_introduction'] else '✗'}")
    # print(f"  Has conclusion: {'✓' if structure_result['has_conclusion'] else '✗'}")
    print(f"\n  Overall: {'✓ VALID' if is_valid else '✗ NEEDS REVISION'}")

    return {
        "word_count": word_count,
        "validation_result": validation_result
    }


def reviewer_node(state: ReportState) -> dict:
    """
    Provides qualitative feedback on the report.

    Design Decision: Reviewer runs after validation passes. It doesn't
    trigger revisions (to avoid infinite loops), but provides valuable
    feedback for the user.
    """
    print("\n" + "=" * 60)
    print("REVIEWER: Analyzing report quality...")
    print("=" * 60)

    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_template("""
You are an expert editor reviewing a technical report for publication.

REPORT TO REVIEW:
{report}

Provide a concise review covering:
1. **Content Quality**: Are the technical concepts accurate and well-explained?
2. **Structure & Flow**: Is the report well-organized with smooth transitions?
3. **Clarity**: Is the writing clear and accessible to the target audience?
4. **Completeness**: Does it adequately cover the topic?
5. **Overall Assessment**: Ready for publication or needs improvements?

Be specific and constructive. Limit your review to 150 words.
""")

    response = llm.invoke(prompt.format(report=state["draft_report"]))

    print("\n✓ Review completed")

    return {
        "review_feedback": response.content,
        "final_report": state["draft_report"]
    }


def finalize_node(state: ReportState) -> dict:
    """
    Adds metadata and prepares final output.

    Design Decision: Collect statistics about the generation process.
    This is useful for debugging and demonstrating the system's capabilities.
    """
    print("\n" + "=" * 60)
    print("FINALIZING: Preparing output...")
    print("=" * 60)

    metadata = {
        "topic": state["topic"],
        "title": state["plan"]["title"],
        "total_sections": len(state["plan"]["sections"]),
        "final_word_count": state["word_count"],
        "iterations_needed": state["iteration_count"],
        "validation_passed": state["validation_result"]["is_valid"],
        "structure_valid": state["validation_result"]["structure_result"]["is_valid"]
    }

    print("\n✓ Report generation complete!")
    print(f"  Title: {metadata['title']}")
    print(f"  Word count: {metadata['final_word_count']}")
    print(f"  Iterations: {metadata['iterations_needed']}")

    return {"metadata": metadata}


# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================

def should_revise(state: ReportState) -> Literal["revise", "review"]:
    """
    Determines if report needs revision based on validation.

    Design Decision: Allow up to 2 revisions. This prevents infinite loops
    while still giving the system a chance to self-correct.
    """
    validation = state["validation_result"]
    iteration = state["iteration_count"]
    max_iterations = state.get("max_iterations", 3)

    # If valid or max iterations reached, move to review
    if validation["is_valid"] or iteration >= max_iterations:
        return "review"
    else:
        print("\n⟲ Validation failed. Triggering revision...")
        return "revise"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_graph():
    """
    Builds the LangGraph workflow.

    Design Decision: Use conditional edges to create a revision loop.
    This allows the system to self-correct without human intervention.

    Flow:
    planner → researcher → writer → validator
                                ↓         ↓
                              (valid) (invalid & iterations < 3)
                                ↓         ↓
                              reviewer ← writer (revise)
                                ↓
                            finalize → END
    """
    workflow = StateGraph(ReportState)

    # Add nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("researcher", researcher_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("finalize", finalize_node)

    # Define edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "validator")

    # Conditional edge: validate → review OR revise
    workflow.add_conditional_edges(
        "validator",
        should_revise,
        {
            "review": "reviewer",
            "revise": "writer"  # Loop back to writer
        }
    )

    workflow.add_edge("reviewer", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================

def run_agent(topic: str, temperature: float = 0.7, max_iterations: int = 3) -> dict:
    """
    Executes the complete report generation workflow.

    Args:
        topic: The subject of the report (e.g., "MLOps Best Practices")
        temperature: Temperature for Writer agent (0.0-1.0, affects creativity)
        max_iterations: Maximum number of revision iterations

    Returns:
        dict containing:
            - final_report: The complete report text
            - review_feedback: Editor's assessment
            - metadata: Generation statistics
            - plan: The original plan
    """
    app = build_graph()
    final_state = app.invoke({
        "topic": topic,
        "temperature": temperature,
        "max_iterations": max_iterations
    })
    return final_state


def visualize_graph():
    """
    Generates a visualization of the workflow.

    Design Decision: Provide this as a separate function so users can
    generate the diagram without running the full pipeline.
    """
    app = build_graph()
    try:
        # Try to generate PNG (requires graphviz)
        return app.get_graph().draw_mermaid_png()
    except Exception:
        # Fallback to mermaid text
        return app.get_graph().draw_mermaid()
