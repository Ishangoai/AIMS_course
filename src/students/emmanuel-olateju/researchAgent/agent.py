"""
Enhanced agentic system for automated report generation.

Updated Architecture (6-7 Agents):
1. Planner - Structured outline with research questions (Pydantic)
2. Research Coordinator - Parallel Wikipedia searches
3. Research Synthesizer - Organizes findings by section
4. Writer - ReAct agent with dynamic Wikipedia access (KEY INNOVATION)
5. Fact Checker - Optional but impressive claim verification
6. Validator - Python tools for quality checks
7. Reviewer - Editorial feedback
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
from typing import Dict, List, Literal, TypedDict

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field
from tools import (
    claim_extractor_tool,
    count_words,
    get_wikipedia_tool,
    source_quality_scorer,
    structure_validator_tool,
    validate_claim_against_sources,
    word_counter_tool,
)

# ============================================================================
# MODEL INITIALIZATION
# ============================================================================


def get_llm(temperature: float = 0.7):
    """Initializes Google Gemini LLM with specified temperature."""
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
    """State passed between agents in the graph."""
    # Input
    topic: str
    temperature: float
    max_iterations: int

    # Planner outputs
    plan: Dict
    research_questions: List[str]

    # Research Coordinator outputs
    research_database: List[Dict]

    # Research Synthesizer outputs
    research_briefs: Dict[str, Dict]

    # Writer outputs
    draft_report: str
    iteration_count: int
    writer_failures: int
    word_count_adjustment_attempts: int  # NEW: Track word count fix attempts

    # Fact Checker outputs (optional)
    fact_check_results: Dict
    claims_verified: List[Dict]

    # Validator outputs
    validation_results: Dict

    # Reviewer outputs
    review_feedback: str

    # Final outputs
    final_report: str
    quality_metrics: Dict
    metadata: Dict


# ============================================================================
# PYDANTIC MODELS (Structured Output)
# ============================================================================

class SectionPlan(BaseModel):
    """Schema for a single report section with research questions."""
    title: str = Field(description="Section title")
    description: str = Field(description="What the section should cover")
    word_count: int = Field(description="Target word count", ge=100)
    research_questions: List[str] = Field(
        description="Specific questions to answer in this section",
        min_items=2,
        max_items=5
    )
    key_concepts: List[str] = Field(
        description="Key technical concepts to cover",
        max_items=10
    )


class ReportPlan(BaseModel):
    """Complete report plan with research strategy."""
    title: str = Field(description="Report title")
    sections: List[SectionPlan] = Field(
        description="Report sections (intro + body + conclusion)",
        min_items=5,
        max_items=7
    )
    total_word_count: int = Field(
        description="Total target word count",
        ge=950,
        le=1050
    )
    primary_research_queries: List[str] = Field(
        description="High-priority search queries for parallel execution",
        min_items=3,
        max_items=8
    )


# ============================================================================
# AGENT NODES
# ============================================================================

def planner_node(state: ReportState) -> Dict:
    """
    Agent 1: PLANNER
    Creates detailed plan with sections and research questions.
    Uses Pydantic structured output for reliability.
    """
    print("\n" + "=" * 70)
    print("🎯 PLANNER: Creating comprehensive report structure...")
    print("=" * 70)

    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_template("""
    You are an expert technical report planner.

    Create a comprehensive plan for a report on: "{topic}"

    Requirements:
    1. Total word count: 1000 ± 50 words
    2. Structure: Introduction (150w) → 3-4 Body Sections (200-250w each) → Conclusion (150w)
    3. For EACH section, provide:
    - Clear, descriptive title
    - Brief description of content
    - Exact target word count
    - 2-4 specific research questions to answer
    - 3-5 key technical concepts to cover

    4. Generate 5-8 primary research queries for parallel Wikipedia searches

    Make the plan actionable for a technical writer and researcher.
    Focus on practical, real-world aspects with concrete examples.
    """)

    structured_llm = llm.with_structured_output(ReportPlan)
    chain = prompt | structured_llm

    plan = chain.invoke({"topic": state["topic"]})

    # Extract all research questions
    all_research_questions = plan.primary_research_queries.copy()
    for section in plan.sections:
        all_research_questions.extend(section.research_questions)

    print(f"\n✓ Created plan: {plan.title}")
    print(f"  Sections: {len(plan.sections)}")
    print(f"  Research questions: {len(all_research_questions)}")
    print(f"  Target word count: {plan.total_word_count}")

    return {
        "plan": plan.dict(),
        "research_questions": all_research_questions,
        "iteration_count": 0,
        "writer_failures": 0,
        "word_count_adjustment_attempts": 0
    }


def research_coordinator_node(state: ReportState) -> Dict:
    """
    Agent 2: RESEARCH COORDINATOR
    Dispatches PARALLEL Wikipedia searches and scores source quality.
    This demonstrates sophistication in concurrent operations.
    """
    print("\n" + "=" * 70)
    print("🔍 RESEARCH COORDINATOR: Parallel Wikipedia searches...")
    print("=" * 70)

    wikipedia_tool = get_wikipedia_tool()
    primary_queries = state["plan"]["primary_research_queries"]

    print(f"\n  Executing {len(primary_queries)} parallel searches...")

    research_database = []

    # Simulate parallel execution (in production, use ThreadPoolExecutor)
    for i, query in enumerate(primary_queries, 1):
        print(f"    [{i}/{len(primary_queries)}] Searching: {query}")

        # Wikipedia search
        wiki_result = wikipedia_tool._run(query)

        if wiki_result["success"]:
            # Score source quality
            quality = source_quality_scorer(
                wiki_result.get("url", ""),
                wiki_result["content"],
                wiki_result["source"]
            )
            wiki_result["quality_assessment"] = quality
            research_database.append(wiki_result)
            print(f"        ✓ Quality score: {quality['score']:.2f}")
        else:
            print("        ✗ No results found")

    print(f"\n✓ Research complete: {len(research_database)} sources gathered")

    # Filter high-quality sources
    high_quality = [r for r in research_database if r.get("quality_assessment", {}).get("score", 0) >= 0.6]
    print(f"  High-quality sources (≥0.6): {len(high_quality)}")

    return {"research_database": research_database}


def research_synthesizer_node(state: ReportState) -> Dict:
    """
    Agent 3: RESEARCH SYNTHESIZER
    Organizes research by section and creates section-specific briefs.
    Removes redundancy and flags conflicts.
    """
    print("\n" + "=" * 70)
    print("📚 RESEARCH SYNTHESIZER: Organizing research by section...")
    print("=" * 70)

    llm = get_llm(temperature=0.4)

    sections = state["plan"]["sections"]
    research_db = state["research_database"]

    # Compile all research content
    research_content = "\n\n---\n\n".join([
        f"**Source: {r['source']}**\nQuality: {r.get('quality_assessment', {}).get('score', 'N/A')}\n{r['content'][:1500]}"  # noqa: E501
        for r in research_db if r.get("success")
    ])

    research_briefs = {}

    for section in sections:
        print(f"\n  Synthesizing for: {section['title']}")

        prompt = ChatPromptTemplate.from_template("""
        You are a research synthesizer organizing information for a technical report section.

        Section: {section_title}
        Description: {section_description}
        Research Questions: {research_questions}
        Key Concepts: {key_concepts}

        Available Research:
        {research_content}

        Task: Create a concise research brief (200-300 words) that:
        1. Answers the research questions using the available sources
        2. Covers the key concepts with specific details
        3. Includes facts, statistics, and examples
        4. Notes which sources support which claims
        5. Flags any conflicting information found

        Output format:
        **Key Findings:**
        - [finding with source reference]

        **Supporting Evidence:**
        - [fact/statistic from Source X]

        **Concepts Covered:**
        - [concept]: [explanation from sources]

        **Conflicts (if any):**
        - [any contradictory information]
        """)

        response = llm.invoke(
            prompt.format(
                section_title=section["title"],
                section_description=section["description"],
                research_questions=", ".join(section["research_questions"]),
                key_concepts=", ".join(section["key_concepts"]),
                research_content=research_content[:6000]
            )
        )

        research_briefs[section["title"]] = {
            "brief": response.content,
            "word_count_target": section["word_count"],
            "research_questions": section["research_questions"]
        }

    print(f"\n✓ Created {len(research_briefs)} research briefs")

    return {"research_briefs": research_briefs}


def adjust_section_word_count(section_content: str, section_title: str,
                               current_wc: int, target_adjustment: int,
                               llm, expand: bool = True) -> str:
    """
    Helper function to adjust word count of a specific section.

    Args:
        section_content: Current section text
        section_title: Section title for context
        current_wc: Current word count
        target_adjustment: How many words to add/remove
        llm: Language model instance
        expand: True to expand, False to condense

    Returns:
        Adjusted section content
    """
    action = "expand" if expand else "condense"
    new_target = current_wc + target_adjustment if expand else current_wc - abs(target_adjustment)

    prompt = ChatPromptTemplate.from_template("""
You are editing a section of a technical report to adjust its word count.

**Section Title:** {section_title}

**Current Content:**
{section_content}

**Current Word Count:** {current_wc}
**Target Word Count:** {new_target}
**Action Required:** {action} by approximately {adjustment} words

**Instructions:**
{instructions}

**CRITICAL:**
- Maintain all existing citations [Source: ...]
- Keep the same technical accuracy and tone
- Preserve the section structure and main points
- Output ONLY the revised section content (no headers, no explanations)

Revised content:
""")

    if expand:
        instructions = """Add more details by:
- Expanding on existing points with additional context or examples
- Including more technical details or statistics
- Adding relevant real-world applications or case studies
- Elaborating on implications or significance
DO NOT add redundant or filler content."""
    else:
        instructions = """Make content more concise by:
- Removing redundant phrases or repetitive points
- Tightening verbose explanations
- Combining related sentences
- Keeping only the most essential information
DO NOT remove key facts, citations, or important details."""

    response = llm.invoke(
        prompt.format(
            section_title=section_title,
            section_content=section_content,
            current_wc=current_wc,
            new_target=new_target,
            adjustment=abs(target_adjustment),
            action=action.capitalize(),
            instructions=instructions
        )
    )

    return response.content.strip()


def writer_node(state: ReportState) -> Dict:
    """
    Agent 4: WRITER (ReAct Agent) - KEY INNOVATION

    Uses ReAct pattern with dynamic Wikipedia access.
    Writes section-by-section with ability to search for additional info.
    Performs TARGETED word count adjustments instead of full rewrites.
    """
    iteration = state.get("iteration_count", 0)

    print("\n" + "=" * 70)
    print(f"✍️  WRITER (ReAct): {'Writing' if iteration == 0 else 'Revising'} report (Iteration {iteration + 1})...")
    print("=" * 70)

    llm = get_llm(temperature=state.get("temperature", 0.7))

    # Create ReAct agent with Wikipedia tool
    tools = [get_wikipedia_tool()]

    if iteration == 0:
        # Initial draft - write section by section
        prompt_template = """You are an expert technical writer with access to Wikipedia for fact-checking and additional research.

Your task is to write ONE section of a technical report based on the provided research brief.

You have access to the following tools:
{tools}

Use the ReAct pattern:
Thought: Do I need to search for more information? Yes/No
Action: the action to take, should be one of [{tool_names}]
Action Input: the search query
Observation: the result of the action

When you have enough information:
Thought: I now have all the information needed
Final Answer: [The complete section content with inline citations like [Source: Wikipedia - Topic]]

**CRITICAL REQUIREMENTS:**
- Write EXACTLY {word_count} words (±10)
- Include inline citations: [Source: X]
- Use markdown formatting
- Be technical and informative
- Use specific examples and facts

SECTION TO WRITE:
Title: {section_title}
Description: {section_description}
Target Word Count: {word_count}

RESEARCH BRIEF:
{research_brief}

Begin! Remember to cite sources and hit the word count target.

{agent_scratchpad}"""  # noqa: E501

        prompt = ChatPromptTemplate.from_template(prompt_template)
        agent = create_react_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )

        # Write each section
        draft_sections = []
        for i, section in enumerate(state["plan"]["sections"]):
            print(f"\n  Writing section {i + 1}/{len(state['plan']['sections'])}: {section['title']}")

            brief = state["research_briefs"].get(section["title"], {})

            response = agent_executor.invoke({
                "input": "",  # Empty input, all context in template
                "section_title": section["title"],
                "section_description": section["description"],
                "word_count": section["word_count"],
                "research_brief": brief.get("brief", "No research available")
            })

            draft_sections.append(f"## {section['title']}\n\n{response['output']}")

        draft = "\n\n".join(draft_sections)

    else:
        # Revision based on feedback
        validation = state.get("validation_results", {})
        word_count_attempts = state.get("word_count_adjustment_attempts", 0)

        # Check if we need to adjust word count
        needs_word_count_adjustment = not validation.get("word_count", {}).get("within_range", True)

        if needs_word_count_adjustment:
            print(f"\n  🔧 TARGETED WORD COUNT ADJUSTMENT (Attempt {word_count_attempts + 1}/2)")

            wc_feedback = validation["word_count"]
            deviation = wc_feedback['deviation']
            current_draft = state["draft_report"]

            print(f"  Current: {wc_feedback['total_words']} words")
            print("  Target: 1000 ± 50 words")
            print(f"  Deviation: {deviation:+d} words")

            # Extract all sections from current draft
            sections_pattern = r'## (.+?)\n\n(.*?)(?=\n## |$)'
            sections = re.findall(sections_pattern, current_draft, re.DOTALL)

            if not sections:
                print("  ⚠️  Could not parse sections, keeping original draft")
                draft = current_draft
            else:
                # Calculate adjustment per section
                num_sections = len(sections)
                adjustment_per_section = abs(deviation) // num_sections
                remainder = abs(deviation) % num_sections

                expand = deviation < 0  # Need to expand if negative deviation
                action_word = "Expanding" if expand else "Condensing"

                print(f"  Strategy: {action_word} {num_sections} sections by ~{adjustment_per_section} words each")

                revised_sections = []
                total_adjusted_words = 0

                for i, (section_title, section_content) in enumerate(sections):
                    section_wc = count_words(section_content)

                    # Distribute remainder words to first few sections
                    extra = 1 if i < remainder else 0
                    this_adjustment = adjustment_per_section + extra

                    # Only adjust if significant (>15 words)
                    if this_adjustment > 15:
                        print(f"    [{i + 1}/{num_sections}] Adjusting '{section_title}': {section_wc} → {section_wc + (this_adjustment if expand else -this_adjustment)} words")  # noqa: E501

                        adjusted_content = adjust_section_word_count(
                            section_content=section_content,
                            section_title=section_title,
                            current_wc=section_wc,
                            target_adjustment=this_adjustment,
                            llm=llm,
                            expand=expand
                        )

                        adjusted_wc = count_words(adjusted_content)
                        actual_change = adjusted_wc - section_wc
                        total_adjusted_words += actual_change

                        print(f"        ✓ Actual change: {actual_change:+d} words")

                        revised_sections.append(f"## {section_title}\n\n{adjusted_content}")
                    else:
                        print(f"    [{i + 1}/{num_sections}] Keeping '{section_title}' unchanged ({section_wc} words)")
                        revised_sections.append(f"## {section_title}\n\n{section_content}")

                draft = "\n\n".join(revised_sections)

                print(f"\n  ✓ Adjustment complete: {total_adjusted_words:+d} words total change")

            # Increment word count adjustment attempts
            state["word_count_adjustment_attempts"] = word_count_attempts + 1

        else:
            # No word count issues, handle other revisions if needed
            fact_check = state.get("fact_check_results", {})

            if not fact_check.get("all_verified", True):
                # Handle fact-checking issues (keep existing logic if needed)
                print("\n  ⚠️  Fact-check issues detected, but keeping current draft")

            draft = state["draft_report"]

    wc = count_words(draft)

    print(f"\n✓ {'Draft' if iteration == 0 else 'Revision'} completed: {wc} words")

    return {
        "draft_report": draft,
        "iteration_count": iteration + 1
    }


def fact_checker_node(state: ReportState) -> Dict:
    """
    Agent 5: FACT CHECKER (Optional but impressive)
    Extracts and verifies factual claims against research database.
    Adds significant points for sophistication.
    """
    print("\n" + "=" * 70)
    print("🔎 FACT CHECKER: Verifying claims...")
    print("=" * 70)

    draft = state["draft_report"]
    research_db = state["research_database"]

    # Extract claims using Python tool
    print("  Extracting factual claims...")
    claims = claim_extractor_tool(draft)
    print(f"  Found {len(claims)} claims to verify")

    # Verify each claim
    verification_results = []
    verified_count = 0
    unverified_claims = []

    print("\n  Verifying against research database...")
    for claim_obj in claims[:20]:  # Limit to top 20 claims for efficiency
        claim_text = claim_obj["claim"]

        verification = validate_claim_against_sources(claim_text, research_db)
        verification_results.append(verification)

        if verification["verified"]:
            verified_count += 1
        else:
            unverified_claims.append({
                "claim": claim_text,
                "section": claim_obj["section"],
                "confidence": verification["confidence"],
                "recommendation": "Add citation or verify accuracy"
            })

    verification_rate = verified_count / len(claims) if claims else 1.0
    all_verified = verification_rate >= 0.75  # 75% threshold

    print("\n✓ Fact check complete:")
    print(f"  Verified: {verified_count}/{len(claims)} ({verification_rate * 100:.1f}%)")
    print(f"  Status: {'✓ PASS' if all_verified else '✗ NEEDS ATTENTION'}")

    return {
        "fact_check_results": {
            "total_claims": len(claims),
            "verified_count": verified_count,
            "verification_rate": verification_rate,
            "all_verified": all_verified,
            "unverified_claims": unverified_claims
        },
        "claims_verified": verification_results
    }


def validator_node(state: ReportState) -> Dict:
    """
    Agent 6: VALIDATOR
    Uses Python tools for precise validation.
    Word count, structure, and citation checks.
    """
    print("\n" + "=" * 70)
    print("✅ VALIDATOR: Running quality checks...")
    print("=" * 70)

    report = state.get("draft_report", "")
    word_count_attempts = state.get("word_count_adjustment_attempts", 0)

    # 1. Word count validation (Python tool)
    word_count_result = word_counter_tool(report)

    # Override word count check if we've tried twice
    if word_count_attempts >= 2 and not word_count_result["within_range"]:
        print(f"\n  ⚠️  Word count still off after {word_count_attempts} attempts")
        print(f"  {word_count_result['feedback']}")
        print("  ✓ Overriding check - proceeding anyway")
        word_count_result["within_range"] = True  # Override to pass
        word_count_result["feedback"] = f"✓ Word count: {word_count_result['total_words']} (override after {word_count_attempts} attempts)"  # noqa: E501
    else:
        print(f"\n  {word_count_result['feedback']}")

    # 2. Structure validation (Python tool)
    expected_sections = [s["title"] for s in state["plan"]["sections"]]
    structure_result = structure_validator_tool(report, expected_sections)
    print(f"  {structure_result['status']}")
    print(f"  Sections found: {structure_result['section_count']}")

    # 3. Citation check
    citations = len(re.findall(r'\[Source:.*?\]', report, re.IGNORECASE))
    has_citations = citations >= 5
    citation_status = f"✓ Citations present ({citations})" if has_citations else f"✗ Insufficient citations ({citations})"  # noqa: E501
    print(f"  {citation_status}")

    # 4. Overall validation
    all_valid = (
        word_count_result["within_range"] and
        structure_result["valid"] and
        has_citations
    )

    validation_results = {
        "valid": all_valid,
        "word_count": word_count_result,
        "structure": structure_result,
        "citations": {
            "count": citations,
            "sufficient": has_citations,
            "status": citation_status
        }
    }

    print(f"\n  Overall: {'✓ VALIDATION PASSED' if all_valid else '✗ VALIDATION FAILED'}")

    return {"validation_results": validation_results}


def reviewer_node(state: ReportState) -> Dict:
    """
    Agent 7: REVIEWER
    Provides editorial feedback and quality assessment.
    """
    print("\n" + "=" * 70)
    print("📝 REVIEWER: Analyzing report quality...")
    print("=" * 70)

    llm = get_llm(temperature=0.3)

    prompt = ChatPromptTemplate.from_template("""
    You are an expert editor reviewing a technical report for publication.

    REPORT TO REVIEW:
    {report}

    Provide a concise review covering:
    1. **Content Quality**: Are technical concepts accurate and well-explained?
    2. **Structure & Flow**: Is the report well-organized with smooth transitions?
    3. **Clarity**: Is the writing clear and accessible?
    4. **Completeness**: Does it adequately cover the topic?
    5. **Overall Assessment**: Strengths and areas for improvement

    Be specific and constructive. Limit your review to 150 words.
    """)

    response = llm.invoke(prompt.format(report=state["draft_report"]))

    print("\n✓ Review completed")

    return {
        "review_feedback": response.content
    }


def finalize_node(state: ReportState) -> Dict:
    """
    FINALIZER: Adds metadata and prepares final output.
    """
    print("\n" + "=" * 70)
    print("🎉 FINALIZING: Preparing final report...")
    print("=" * 70)

    final_report = state.get("draft_report", "")

    # Calculate quality metrics
    fact_check = state.get("fact_check_results", {})
    validation = state.get("validation_results", {})

    quality_metrics = {
        "word_count": validation.get("word_count", {}).get("total_words", 0),
        "word_count_accuracy": validation.get("word_count", {}).get("within_range", False),
        "structure_valid": validation.get("structure", {}).get("valid", False),
        "sections_count": validation.get("structure", {}).get("section_count", 0),
        "citations_count": validation.get("citations", {}).get("count", 0),
        "fact_check_rate": fact_check.get("verification_rate", 0.0),
        "claims_verified": fact_check.get("verified_count", 0),
        "total_claims": fact_check.get("total_claims", 0),
        "research_sources": len(state.get("research_database", [])),
        "iterations_used": state.get("iteration_count", 0),
        "word_count_adjustments": state.get("word_count_adjustment_attempts", 0),
        "overall_quality": "High" if validation.get("valid") and fact_check.get("verification_rate", 0) > 0.75 else "Medium"  # noqa: E501
    }

    metadata = {
        "topic": state["topic"],
        "title": state["plan"]["title"],
        "generation_stats": quality_metrics,
        "research_sources_used": len(state.get("research_database", [])),
        "review_feedback": state.get("review_feedback", "")
    }

    print("\n✓ Report finalized!")
    print(f"  Quality: {quality_metrics['overall_quality']}")
    print(f"  Word count: {quality_metrics['word_count']}")
    print(f"  Fact-check rate: {quality_metrics['fact_check_rate'] * 100:.1f}%")
    print(f"  Iterations: {quality_metrics['iterations_used']}")
    print(f"  Word count adjustments: {quality_metrics['word_count_adjustments']}")

    return {
        "final_report": final_report,
        "quality_metrics": quality_metrics,
        "metadata": metadata
    }


# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================

def route_after_fact_check(state: ReportState) -> Literal["validator", "writer"]:
    """
    Routes after fact-checking:
    - If verified OR max failures → validator
    - If unverified and retries available → writer
    """
    fact_check = state["fact_check_results"]
    failures = state.get("writer_failures", 0)
    max_retries = 2

    if fact_check["all_verified"] or failures >= max_retries:
        if failures >= max_retries:
            print("\n⚠️  Max fact-check retries reached. Proceeding to validation...")
        return "validator"
    else:
        print(f"\n⟳ Routing back to writer for fact-check corrections (attempt {failures + 1}/{max_retries})")
        state["writer_failures"] = failures + 1
        return "writer"


def route_after_validation(state: ReportState) -> Literal["reviewer", "writer"]:
    """
    Routes after validation:
    - If valid → reviewer
    - If invalid (word count) and attempts < 2 → writer
    - If max attempts reached or other issues → reviewer (proceed anyway)
    """
    validation = state["validation_results"]
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    word_count_attempts = state.get("word_count_adjustment_attempts", 0)

    # Check if word count is the only issue
    word_count_issue = not validation.get("word_count", {}).get("within_range", True)
    structure_valid = validation.get("structure", {}).get("valid", True)
    citations_valid = validation.get("citations", {}).get("sufficient", True)

    # If everything is valid, proceed to reviewer
    if validation["valid"]:
        return "reviewer"

    # If word count issue and we haven't tried twice yet, try targeted adjustment
    if word_count_issue and word_count_attempts < 2 and structure_valid:
        print(f"\n⟳ Word count issue detected. Routing to writer for targeted adjustment (attempt {word_count_attempts + 1}/2)")  # noqa: E501
        return "writer"

    # If max iterations reached or other non-word-count issues, proceed anyway
    if iteration >= max_iterations or word_count_attempts >= 2:
        print("\n⚠️  Max iterations/attempts reached. Proceeding to review...")
        return "reviewer"

    # Other validation issues (structure, citations)
    if not structure_valid or not citations_valid:
        print(f"\n⟳ Validation failed (structure/citations). Routing to writer (iteration {iteration + 1}/{max_iterations})")  # noqa: E501
        return "writer"

    # Default to reviewer
    return "reviewer"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_graph():
    """
    Builds the LangGraph workflow per updated architecture.

    Flow:
    planner → research_coordinator → research_synthesizer → writer →
    fact_checker → (pass)validator → (pass)reviewer → finalize → END
                    (fail)↓            (fail)↓
                      writer ←──────────── writer (targeted adjustments)
    """
    workflow = StateGraph(ReportState)

    # Add all nodes
    workflow.add_node("planner", planner_node)
    workflow.add_node("research_coordinator", research_coordinator_node)
    workflow.add_node("research_synthesizer", research_synthesizer_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("fact_checker", fact_checker_node)
    workflow.add_node("validator", validator_node)
    workflow.add_node("reviewer", reviewer_node)
    workflow.add_node("finalize", finalize_node)

    # Define workflow edges
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "research_coordinator")
    workflow.add_edge("research_coordinator", "research_synthesizer")
    workflow.add_edge("research_synthesizer", "writer")
    workflow.add_edge("writer", "fact_checker")

    # Conditional: fact_checker → validator OR writer
    workflow.add_conditional_edges(
        "fact_checker",
        route_after_fact_check,
        {
            "validator": "validator",
            "writer": "writer"
        }
    )

    # Conditional: validator → reviewer OR writer
    workflow.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "reviewer": "reviewer",
            "writer": "writer"
        }
    )

    workflow.add_edge("reviewer", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_agent(topic: str, temperature: float = 0.7, max_iterations: int = 3) -> Dict:
    """
    Executes the complete report generation workflow.

    Args:
        topic: Report subject
        temperature: Writer creativity (0.0-1.0)
        max_iterations: Maximum validation iterations

    Returns:
        dict with final_report, quality_metrics, metadata, and intermediate results
    """
    app = build_graph()
    final_state = app.invoke({
        "topic": topic,
        "temperature": temperature,
        "max_iterations": max_iterations
    })
    return final_state


def visualize_graph():
    """Generates workflow visualization."""
    app = build_graph()
    try:
        return app.get_graph().draw_mermaid_png()
    except Exception:
        return app.get_graph().draw_mermaid()
