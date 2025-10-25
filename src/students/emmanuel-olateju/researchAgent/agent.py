"""
Enhanced agentic system for automated report generation.
NOW WITH GOOGLE SEARCH, ANALYTICAL REWRITING AND HUMAN-IN-THE-LOOP FEEDBACK

Updated Architecture (6-7 Agents + Optional Human Feedback):
1. Planner - Structured outline with research questions (Pydantic)
2. Research Coordinator - Parallel Google searches
3. Research Synthesizer - Organizes findings by section
4. Writer - ReAct agent with ANALYTICAL REWRITING (KEY INNOVATION)
5. Fact Checker - Optional but impressive claim verification
6. Validator - Python tools for quality checks
7. Reviewer - Editorial feedback
8. Human Feedback (OPTIONAL) - Takes human corrections and regenerates
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
from typing import Dict, List, Literal, Optional, TypedDict

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, field_validator
from tools import (
    claim_extractor_tool,
    count_words,
    get_fallback_search_tool,
    get_google_search_tool,
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


def get_search_tool():
    """
    Returns the appropriate search tool based on environment configuration.
    Tries Google Search first, falls back to DuckDuckGo if credentials unavailable.
    """
    try:
        if os.getenv("GOOGLE_CSE_ID"):
            print("  Using Google Search")
            return get_google_search_tool()
        else:
            print("  Using DuckDuckGo (fallback - no GOOGLE_CSE_ID found)")
            return get_fallback_search_tool()
    except Exception as e:
        print(f"  Warning: Error initializing Google Search ({e}), using DuckDuckGo fallback")
        return get_fallback_search_tool()


# ============================================================================
# STATE DEFINITION
# ============================================================================

class ReportState(TypedDict):
    """State passed between agents in the graph."""
    # Input
    topic: str
    temperature: float
    max_iterations: int
    enable_human_feedback: bool  # NEW: Toggle for human feedback

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
    word_count_adjustment_attempts: int

    # Fact Checker outputs (optional)
    fact_check_results: Dict
    claims_verified: List[Dict]

    # Validator outputs
    validation_results: Dict

    # Reviewer outputs
    review_feedback: str

    # Human Feedback (NEW)
    human_feedback: Optional[str]
    human_feedback_applied: bool
    awaiting_human_feedback: bool

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
        max_items=15  # Increased from 8 to allow more flexibility
    )

    @field_validator('primary_research_queries')
    @classmethod
    def limit_queries(cls, v: List[str]) -> List[str]:
        """Truncate to 12 queries if more are provided."""
        if len(v) > 12:
            print(f"  ⚠️  Truncating {len(v)} queries to 12")
            return v[:12]
        return v


# ============================================================================
# ANALYTICAL REWRITING HELPERS
# ============================================================================

def analyze_content_for_adjustment(draft: str, target_wc: int, current_wc: int,
                                   base_model_name: str, attempt: int = 0) -> Dict:
    """
    Phase 1: Analytical scan with adaptive temperature to identify adjustments.

    Temperature increases with attempts for more creative solutions.
    Returns strategic recommendations for what to condense/expand.
    """
    import random

    deviation = target_wc - current_wc
    action = "condense" if deviation < 0 else "expand"

    # Adaptive temperature: increases with attempts for more variety
    # Attempt 0: 0.1-0.15, Attempt 1: 0.2-0.3, Attempt 2+: 0.3-0.5
    if attempt == 0:
        temp = random.uniform(0.1, 0.15)
    elif attempt == 1:
        temp = random.uniform(0.2, 0.3)
    else:
        temp = random.uniform(0.3, 0.5)

    print(f"    Phase 1 temperature: {temp:.2f} (attempt {attempt})")

    # Create adaptive-temperature LLM for analysis
    analysis_llm = ChatGoogleGenerativeAI(
        model=base_model_name,
        temperature=temp
    )

    prompt = ChatPromptTemplate.from_template("""
You are analyzing a technical report to identify opportunities for word count adjustment.

**Current Draft:**
{draft}

**Statistics:**
- Current word count: {current_wc}
- Target word count: {target_wc}
- Required adjustment: {action} by {abs_deviation} words

**Task:** Perform a detailed content analysis and identify:

1. **Section Breakdown:** For each section, list:
   - Section title
   - Current word count estimate
   - Content density (verbose/concise/balanced)

2. **Adjustment Strategy:** For each section, recommend:
   - Priority for adjustment (high/medium/low)
   - Specific opportunities to {action}:
     {strategy_guidance}
   - Estimated word count change possible

3. **Overall Plan:** Provide a clear strategy for achieving exactly {target_wc} words while:
   - Maintaining technical accuracy
   - Preserving all citations and key facts
   - Keeping balanced section proportions
   - Ensuring smooth flow and transitions

Output your analysis in structured format with specific, actionable recommendations.
""")

    if action == "condense":
        strategy_guidance = """- Redundant phrases or repetitive points to remove
     - Overly verbose explanations that can be tightened
     - Examples or details that are less critical
     - Run-on sentences that can be split or shortened"""
    else:
        strategy_guidance = """- Key points that deserve more elaboration
     - Technical concepts that need deeper explanation
     - Missing examples or real-world applications
     - Areas where context would improve understanding"""

    response = analysis_llm.invoke(
        prompt.format(
            draft=draft,
            current_wc=current_wc,
            target_wc=target_wc,
            action=action,
            abs_deviation=abs(deviation),
            strategy_guidance=strategy_guidance
        )
    )

    return {
        "analysis": response.content,
        "action": action,
        "deviation": deviation
    }


def rewrite_with_strategy(draft: str, analysis: Dict, base_model_name: str, attempt: int = 0) -> str:
    """
    Phase 2: Strategic rewrite with adaptive temperature.

    Temperature increases with attempts for more creative rewriting approaches.
    The model rewrites the entire draft following the strategic plan.
    """
    import random

    # Adaptive temperature: increases with attempts
    # Attempt 0: 0.4-0.5, Attempt 1: 0.5-0.6, Attempt 2+: 0.6-0.75
    if attempt == 0:
        temp = random.uniform(0.4, 0.5)
    elif attempt == 1:
        temp = random.uniform(0.5, 0.6)
    else:
        temp = random.uniform(0.6, 0.75)

    print(f"    Phase 2 temperature: {temp:.2f} (attempt {attempt})")

    # Create adaptive-temperature LLM for rewriting
    rewrite_llm = ChatGoogleGenerativeAI(
        model=base_model_name,
        temperature=temp
    )

    prompt = ChatPromptTemplate.from_template("""
You are rewriting a technical report following a strategic content adjustment plan.

**Original Draft:**
{draft}

**Content Analysis & Strategy:**
{analysis}

**Objective:**
Rewrite the ENTIRE report to {action} by approximately {abs_deviation} words, following the strategic recommendations above.

**Critical Requirements:**
1. **Word Count:** Target EXACTLY the adjusted word count through intelligent rewriting
2. **Preserve:** All citations [Source: ...], technical accuracy, and key facts
3. **Maintain:** Section structure, professional tone, and logical flow
4. **Apply Strategy:** Follow the specific recommendations from the analysis
5. **Quality:** Ensure the rewritten version is coherent and well-integrated

**Rewriting Guidelines:**
{rewriting_guidelines}

**Output Instructions:**
- Return ONLY the complete rewritten report
- Include all section headers (##)
- Maintain markdown formatting
- Ensure smooth transitions between adjusted sections
- Double-check that all citations are preserved

Rewritten report:
""")  # noqa: E501

    if analysis["action"] == "condense":
        rewriting_guidelines = """- Remove redundancies identified in the analysis
- Tighten verbose explanations without losing meaning
- Combine related points more efficiently
- Cut less essential details while keeping core content
- Make every word count - be precise and direct"""
    else:
        rewriting_guidelines = """- Expand key points identified in the analysis
- Add technical depth where recommended
- Include relevant examples and applications
- Provide additional context for better understanding
- Elaborate on implications and significance"""

    response = rewrite_llm.invoke(
        prompt.format(
            draft=draft,
            analysis=analysis["analysis"],
            action=analysis["action"],
            abs_deviation=abs(analysis["deviation"]),
            rewriting_guidelines=rewriting_guidelines
        )
    )

    return response.content.strip()


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

    4. Generate 5-10 primary research queries for parallel Google searches
       (Focus on quality over quantity - prioritize the most important queries)
       (Make queries specific and targeted for better search results)

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
        "word_count_adjustment_attempts": 0,
        "human_feedback_applied": False,
        "awaiting_human_feedback": False
    }


def research_coordinator_node(state: ReportState) -> Dict:
    """
    Agent 2: RESEARCH COORDINATOR
    Dispatches PARALLEL Google searches and scores source quality.
    """
    print("\n" + "=" * 70)
    print("🔍 RESEARCH COORDINATOR: Parallel Google searches...")
    print("=" * 70)

    search_tool = get_search_tool()
    primary_queries = state["plan"]["primary_research_queries"]

    print(f"\n  Executing {len(primary_queries)} parallel searches...")

    research_database = []

    for i, query in enumerate(primary_queries, 1):
        print(f"    [{i}/{len(primary_queries)}] Searching: {query}")

        search_result = search_tool.func(query)

        if search_result["success"]:
            # Process each search result
            for result_item in search_result.get("results", []):
                # Create a structured entry for each result
                entry = {
                    "success": True,
                    "query": query,
                    "title": result_item.get("title", ""),
                    "content": result_item.get("snippet", ""),
                    "url": result_item.get("url", ""),
                    "source": f"Google Search - {result_item.get('title', query)}"
                }

                # Score quality
                quality = source_quality_scorer(
                    entry["url"],
                    entry["content"],
                    entry["source"]
                )
                entry["quality_assessment"] = quality
                research_database.append(entry)

            print(f"        ✓ Found {len(search_result.get('results', []))} results")
        else:
            print(f"        ✗ No results found: {search_result.get('error', 'Unknown error')}")

    print(f"\n✓ Research complete: {len(research_database)} sources gathered")

    high_quality = [r for r in research_database if r.get("quality_assessment", {}).get("score", 0) >= 0.6]
    print(f"  High-quality sources (≥0.6): {len(high_quality)}")

    return {"research_database": research_database}


def research_synthesizer_node(state: ReportState) -> Dict:
    """
    Agent 3: RESEARCH SYNTHESIZER
    Organizes research by section and creates section-specific briefs.
    """
    print("\n" + "=" * 70)
    print("📚 RESEARCH SYNTHESIZER: Organizing research by section...")
    print("=" * 70)

    llm = get_llm(temperature=0.4)

    sections = state["plan"]["sections"]
    research_db = state["research_database"]

    research_content = "\n\n---\n\n".join([
        f"**Source: {r['source']}**\nQuality: {r.get('quality_assessment', {}).get('score', 'N/A')}\nURL: {r.get('url', 'N/A')}\n{r['content'][:1000]}"  # noqa: E501
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
        3. Includes facts, statistics, and examples from the search results
        4. Notes which sources support which claims (include URLs when relevant)
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
                research_content=research_content[:8000]  # Increased limit for more context
            )
        )

        research_briefs[section["title"]] = {
            "brief": response.content,
            "word_count_target": section["word_count"],
            "research_questions": section["research_questions"]
        }

    print(f"\n✓ Created {len(research_briefs)} research briefs")

    return {"research_briefs": research_briefs}


def writer_node(state: ReportState) -> Dict:
    """
    Agent 4: WRITER (ReAct Agent with ANALYTICAL REWRITING)

    Initial draft: Uses ReAct pattern with dynamic Google Search access.
    Revisions: Uses TWO-PHASE analytical rewriting (low-temp analysis + strategic rewrite).
    Human feedback: Applies human corrections with full rewrite.
    """
    iteration = state.get("iteration_count", 0)
    human_feedback = state.get("human_feedback")

    print("\n" + "=" * 70)
    if human_feedback:
        print("✏️  WRITER (Human Feedback): Revising based on feedback...")
    else:
        print(f"✏️  WRITER (ReAct): {'Writing' if iteration == 0 else 'Revising'} report (Iteration {iteration + 1})...")
    print("=" * 70)

    llm = get_llm(temperature=state.get("temperature", 0.7))

    # Create ReAct agent with search tool
    tools = [get_search_tool()]

    if human_feedback:
        # ===== HUMAN FEEDBACK REVISION =====
        print("\n  📝 Applying human feedback corrections...")

        prompt = ChatPromptTemplate.from_template("""
You are an expert technical writer revising a report based on human feedback.

**Current Draft:**
{draft}

**Human Feedback:**
{feedback}

**Research Briefs Available:**
{research_briefs}

**Task:**
Revise the ENTIRE report to address the human feedback while maintaining:
- Technical accuracy and depth
- All citations [Source: ...]
- Professional tone and structure
- Target word count (~1000 words)

**Critical Instructions:**
1. Carefully address EACH point in the feedback
2. Make substantial improvements where requested
3. Preserve what's working well
4. Maintain markdown formatting (## headers)
5. Keep or improve citation quality

Output the complete revised report:
""")

        # Prepare research briefs summary
        briefs_summary = "\n\n".join([
            f"**{title}:**\n{brief['brief'][:500]}"
            for title, brief in state["research_briefs"].items()
        ])

        response = llm.invoke(
            prompt.format(
                draft=state["draft_report"],
                feedback=human_feedback,
                research_briefs=briefs_summary[:3000]
            )
        )

        draft = response.content.strip()
        print("  ✓ Human feedback applied")

    elif iteration == 0:
        # ===== INITIAL DRAFT: Section-by-section writing =====
        prompt_template = """You are an expert technical writer with access to Google Search for fact-checking and additional research.

Your task is to write ONE section of a technical report based on the provided research brief.

You have access to the following tools:
{tools}

Use the ReAct pattern:
Thought: Do I need to search for more information? Yes/No
Action: the action to take, should be one of [{tool_names}]
Action Input: the search query (be specific and targeted)
Observation: the result of the action

When you have enough information:
Thought: I now have all the information needed
Final Answer: [The complete section content with inline citations like [Source: Website Name - URL]]

**CRITICAL REQUIREMENTS:**
- Write EXACTLY {word_count} words (±10)
- Include inline citations: [Source: Website Name] or [Source: Website Name - URL]
- Use markdown formatting
- Be technical and informative
- Use specific examples and facts from search results

SECTION TO WRITE:
Title: {section_title}
Description: {section_description}
Target Word Count: {word_count}

RESEARCH BRIEF:
{research_brief}

Begin! Remember to cite sources properly and hit the word count target.

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
                "input": "",
                "section_title": section["title"],
                "section_description": section["description"],
                "word_count": section["word_count"],
                "research_brief": brief.get("brief", "No research available")
            })

            draft_sections.append(f"## {section['title']}\n\n{response['output']}")

        draft = "\n\n".join(draft_sections)

    else:
        # ===== REVISION: Analytical rewriting approach =====
        validation = state.get("validation_results", {})
        word_count_attempts = state.get("word_count_adjustment_attempts", 0)

        needs_word_count_adjustment = not validation.get("word_count", {}).get("within_range", True)

        if needs_word_count_adjustment:
            print(f"\n  🔬 ANALYTICAL REWRITE (Attempt {word_count_attempts + 1}/2)")

            current_draft = state["draft_report"]
            wc_feedback = validation["word_count"]
            current_wc = wc_feedback['total_words']
            target_wc = 1000

            print(f"  Current: {current_wc} words")
            print(f"  Target: {target_wc} words")
            print(f"  Deviation: {wc_feedback['deviation']:+d} words")

            # Get base model name from the LLM
            base_model = getattr(llm, 'model', getattr(llm, 'model_name', 'gemini-2.5-flash-lite'))

            # Phase 1: Adaptive-temperature analysis
            print("\n  📊 Phase 1: Content analysis (adaptive temperature)...")
            analysis = analyze_content_for_adjustment(
                draft=current_draft,
                target_wc=target_wc,
                current_wc=current_wc,
                base_model_name=base_model,
                attempt=word_count_attempts
            )
            print(f"  ✓ Strategy: {analysis['action']} by {abs(analysis['deviation'])} words")

            # Phase 2: Adaptive-temperature strategic rewrite
            print("\n  ✏️  Phase 2: Strategic rewrite (adaptive temperature)...")
            draft = rewrite_with_strategy(
                draft=current_draft,
                analysis=analysis,
                base_model_name=base_model,
                attempt=word_count_attempts
            )

            new_wc = count_words(draft)
            actual_change = new_wc - current_wc
            print(f"  ✓ Rewrite complete: {new_wc} words ({actual_change:+d} change)")

            # Increment word count adjustment attempts
            state["word_count_adjustment_attempts"] = word_count_attempts + 1

        else:
            # No word count issues, keep current draft
            draft = state["draft_report"]

    wc = count_words(draft)
    print(f"\n✓ {'Draft' if iteration == 0 else 'Revision'} completed: {wc} words")

    update_dict = {
        "draft_report": draft,
        "iteration_count": iteration + 1
    }

    # Mark human feedback as applied if it was used
    if human_feedback:
        update_dict["human_feedback_applied"] = True
        update_dict["human_feedback"] = None  # Clear feedback after applying

    return update_dict


def fact_checker_node(state: ReportState) -> Dict:
    """
    Agent 5: FACT CHECKER
    Extracts and verifies factual claims against research database.
    """
    print("\n" + "=" * 70)
    print("🔎 FACT CHECKER: Verifying claims...")
    print("=" * 70)

    draft = state["draft_report"]
    research_db = state["research_database"]

    print("  Extracting factual claims...")
    claims = claim_extractor_tool(draft)
    print(f"  Found {len(claims)} claims to verify")

    verification_results = []
    verified_count = 0
    unverified_claims = []

    print("\n  Verifying against research database...")
    for claim_obj in claims[:20]:
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
    all_verified = verification_rate >= 0.75

    print("\n✓ Fact check complete:")
    print(f"  Verified: {verified_count}/{len(claims)} ({verification_rate * 100:.1f}%)")
    print(f"  Status: {'✓ PASS' if all_verified else '✗ NEEDS ATTENTION'}")

    # Prepare return dict
    result = {
        "fact_check_results": {
            "total_claims": len(claims),
            "verified_count": verified_count,
            "verification_rate": verification_rate,
            "all_verified": all_verified,
            "unverified_claims": unverified_claims
        },
        "claims_verified": verification_results
    }

    # Increment writer_failures if fact check failed
    if not all_verified:
        current_failures = state.get("writer_failures", 0)
        result["writer_failures"] = current_failures + 1

    return result


def validator_node(state: ReportState) -> Dict:
    """
    Agent 6: VALIDATOR
    Uses Python tools for precise validation.
    """
    print("\n" + "=" * 70)
    print("✅ VALIDATOR: Running quality checks...")
    print("=" * 70)

    report = state.get("draft_report", "")
    word_count_attempts = state.get("word_count_adjustment_attempts", 0)

    # 1. Word count validation
    word_count_result = word_counter_tool(report)

    # Override word count check if we've tried twice
    if word_count_attempts >= 2 and not word_count_result["within_range"]:
        print(f"\n  ⚠️  Word count still off after {word_count_attempts} attempts")
        print(f"  {word_count_result['feedback']}")
        print("  ✓ Overriding check - proceeding anyway")
        word_count_result["within_range"] = True
        word_count_result["feedback"] = f"✓ Word count: {word_count_result['total_words']} (override after {word_count_attempts} attempts)"  # noqa: E501
    else:
        print(f"\n  {word_count_result['feedback']}")

    # 2. Structure validation
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
    print("📋 REVIEWER: Analyzing report quality...")
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


def human_feedback_gate_node(state: ReportState) -> Dict:
    """
    NEW: HUMAN FEEDBACK GATE
    Pauses workflow if human feedback is enabled and not yet applied.
    """
    enable_feedback = state.get("enable_human_feedback", False)
    feedback_applied = state.get("human_feedback_applied", False)

    if enable_feedback and not feedback_applied:
        print("\n" + "=" * 70)
        print("👤 HUMAN FEEDBACK: Awaiting human input...")
        print("=" * 70)
        print("\n  Status: Waiting for human to provide feedback")
        print("  Use apply_human_feedback() to continue workflow")

        return {
            "awaiting_human_feedback": True
        }

    return {
        "awaiting_human_feedback": False
    }


def finalize_node(state: ReportState) -> Dict:
    """
    FINALIZER: Adds metadata and prepares final output.
    """
    print("\n" + "=" * 70)
    print("🎉 FINALIZING: Preparing final report...")
    print("=" * 70)

    final_report = state.get("draft_report", "")

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
        "human_feedback_applied": state.get("human_feedback_applied", False),
        "overall_quality": "High" if validation.get("valid") and fact_check.get("verification_rate", 0) > 0.75 else "Medium"  # noqa: E501
    }

    metadata = {
        "topic": state["topic"],
        "title": state["plan"]["title"],
        "generation_stats": quality_metrics,
        "research_sources_used": len(state.get("research_database", [])),
        "review_feedback": state.get("review_feedback", ""),
        "human_feedback_enabled": state.get("enable_human_feedback", False)
    }

    print("\n✓ Report finalized!")
    print(f"  Quality: {quality_metrics['overall_quality']}")
    print(f"  Word count: {quality_metrics['word_count']}")
    print(f"  Fact-check rate: {quality_metrics['fact_check_rate'] * 100:.1f}%")
    print(f"  Iterations: {quality_metrics['iterations_used']}")
    print(f"  Word count adjustments: {quality_metrics['word_count_adjustments']}")
    if quality_metrics['human_feedback_applied']:
        print("  Human feedback: ✓ Applied")

    return {
        "final_report": final_report,
        "quality_metrics": quality_metrics,
        "metadata": metadata
    }


# ============================================================================
# CONDITIONAL ROUTING
# ============================================================================

def route_after_fact_check(state: ReportState) -> Literal["validator", "writer"]:
    """Routes after fact-checking."""
    fact_check = state["fact_check_results"]
    failures = state.get("writer_failures", 0)
    max_retries = 2

    if fact_check["all_verified"] or failures >= max_retries:
        if failures >= max_retries:
            print("\n⚠️  Max fact-check retries reached. Proceeding to validation...")
        return "validator"
    else:
        print(f"\n⟳ Routing back to writer for fact-check corrections (attempt {failures + 1}/{max_retries})")
        return "writer"


def route_after_validation(state: ReportState) -> Literal["reviewer", "writer"]:
    """Routes after validation."""
    validation = state["validation_results"]
    iteration = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 3)
    word_count_attempts = state.get("word_count_adjustment_attempts", 0)

    word_count_issue = not validation.get("word_count", {}).get("within_range", True)
    structure_valid = validation.get("structure", {}).get("valid", True)
    citations_valid = validation.get("citations", {}).get("sufficient", True)

    if validation["valid"]:
        return "reviewer"

    if word_count_issue and word_count_attempts < 2 and structure_valid:
        print(f"\n⟳ Word count issue detected. Routing to writer for analytical rewrite (attempt {word_count_attempts + 1}/2)")  # noqa: E501
        return "writer"

    if iteration >= max_iterations or word_count_attempts >= 2:
        print("\n⚠️  Max iterations/attempts reached. Proceeding to review...")
        return "reviewer"

    if not structure_valid or not citations_valid:
        print(f"\n⟳ Validation failed (structure/citations). Routing to writer (iteration {iteration + 1}/{max_iterations})")  # noqa: E501
        return "writer"

    return "reviewer"


def route_after_reviewer(state: ReportState) -> Literal["human_feedback_gate", "finalize"]:
    """Routes after reviewer - checks if human feedback is needed."""
    enable_feedback = state.get("enable_human_feedback", False)
    feedback_applied = state.get("human_feedback_applied", False)

    if enable_feedback and not feedback_applied:
        return "human_feedback_gate"
    return "finalize"


def route_after_human_gate(state: ReportState) -> Literal["writer", "finalize", "END"]:
    """Routes after human feedback gate."""
    awaiting = state.get("awaiting_human_feedback", False)
    has_feedback = state.get("human_feedback") is not None

    if awaiting and not has_feedback:
        # Still waiting for feedback - this will pause the workflow
        return "END"
    elif has_feedback:
        # Feedback received, route to writer
        print("\n⟳ Human feedback received. Routing to writer for revision...")
        return "writer"
    else:
        # No feedback needed, proceed to finalize
        return "finalize"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def build_graph():
    """
    Builds the LangGraph workflow with analytical rewriting and human feedback.

    Flow:
    planner → research_coordinator → research_synthesizer → writer →
    fact_checker → (pass)validator → (pass)reviewer → human_feedback_gate →
                    (fail)↓            (fail)↓                    ↓
                      writer ←──────────────── writer    (if feedback) writer
                                                         (no feedback) finalize → END
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
    workflow.add_node("human_feedback_gate", human_feedback_gate_node)
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

    # Conditional: reviewer → human_feedback_gate OR finalize
    workflow.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {
            "human_feedback_gate": "human_feedback_gate",
            "finalize": "finalize"
        }
    )

    # Conditional: human_feedback_gate → writer OR finalize OR END (pause)
    workflow.add_conditional_edges(
        "human_feedback_gate",
        route_after_human_gate,
        {
            "writer": "writer",
            "finalize": "finalize",
            "END": END
        }
    )

    workflow.add_edge("finalize", END)

    return workflow.compile()


# ============================================================================
# MAIN EXECUTION & HUMAN FEEDBACK FUNCTIONS
# ============================================================================

def run_agent(topic: str, temperature: float = 0.7, max_iterations: int = 3,
              enable_human_feedback: bool = False) -> Dict:
    """
    Executes the complete report generation workflow.

    Args:
        topic: Report subject
        temperature: Writer creativity (0.0-1.0)
        max_iterations: Maximum validation iterations
        enable_human_feedback: If True, pauses after review for human feedback

    Returns:
        dict with final_report, quality_metrics, metadata, and intermediate results
    """
    app = build_graph()
    config = {"recursion_limit": 50}
    final_state = app.invoke({
        "topic": topic,
        "temperature": temperature,
        "max_iterations": max_iterations,
        "enable_human_feedback": enable_human_feedback
    }, config=config)
    return final_state


def apply_human_feedback(current_state: Dict, feedback: str) -> Dict:
    """
    Applies human feedback and continues the workflow.

    Usage:
        # Initial run with human feedback enabled
        state = run_agent("AI Ethics", enable_human_feedback=True)

        # Review the draft
        print(state["draft_report"])

        # Provide feedback and regenerate
        feedback = "Add more examples of bias in AI systems and expand the conclusion"
        final_state = apply_human_feedback(state, feedback)

    Args:
        current_state: The state returned from run_agent (paused at human_feedback_gate)
        feedback: Human feedback text describing desired changes

    Returns:
        Updated state after applying feedback and completing workflow
    """
    if not current_state.get("awaiting_human_feedback"):
        print("⚠️  Warning: State is not awaiting human feedback")
        return current_state

    print("\n" + "=" * 70)
    print("👤 APPLYING HUMAN FEEDBACK")
    print("=" * 70)
    print(f"\nFeedback: {feedback}\n")

    # Update state with feedback
    current_state["human_feedback"] = feedback
    current_state["awaiting_human_feedback"] = False

    # Reset failure counters to give the revision a fresh start
    current_state["writer_failures"] = 0
    current_state["word_count_adjustment_attempts"] = 0

    # Continue workflow from human_feedback_gate
    app = build_graph()
    config = {"recursion_limit": 50}
    final_state = app.invoke(current_state, config=config)

    return final_state


def run_agent_with_interactive_feedback(topic: str, temperature: float = 0.7,
                                       max_iterations: int = 3) -> Dict:
    """
    Convenience function that runs the agent and prompts for interactive feedback.

    This function:
    1. Generates initial report
    2. Displays it to the user
    3. Asks if they want to provide feedback
    4. If yes, takes feedback and regenerates
    5. Returns final result

    Args:
        topic: Report subject
        temperature: Writer creativity (0.0-1.0)
        max_iterations: Maximum validation iterations

    Returns:
        Final state after optional human feedback
    """
    # Generate initial report
    print("\n🚀 Starting report generation with interactive feedback option...\n")
    state = run_agent(topic, temperature, max_iterations, enable_human_feedback=True)

    # Display report
    print("\n" + "=" * 70)
    print("📄 DRAFT REPORT GENERATED")
    print("=" * 70)
    print(state.get("draft_report", "No report generated"))
    print("\n" + "=" * 70)

    # Ask for feedback
    print("\nWould you like to provide feedback for revision?")
    response = input("Enter 'yes' to provide feedback, or 'no' to finalize: ").strip().lower()

    if response in ['yes', 'y']:
        print("\nPlease provide your feedback (press Enter twice when done):")
        feedback_lines = []
        while True:
            line = input()
            if line == "" and len(feedback_lines) > 0 and feedback_lines[-1] == "":
                break
            feedback_lines.append(line)

        feedback = "\n".join(feedback_lines).strip()

        if feedback:
            final_state = apply_human_feedback(state, feedback)
            return final_state
        else:
            print("\n⚠️  No feedback provided. Using original draft.")
            return state
    else:
        print("\n✓ Proceeding with original draft.")
        # Finalize without feedback
        state["enable_human_feedback"] = False
        app = build_graph()
        final_state = app.invoke(state)  # pyright: ignore[reportArgumentType]
        return final_state


def visualize_graph():
    """Generates workflow visualization."""
    app = build_graph()
    try:
        return app.get_graph().draw_mermaid_png()
    except Exception:
        return app.get_graph().draw_mermaid()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Standard run without human feedback
    print("Example 1: Standard run with Google Search")
    result = run_agent("Quantum Computing", enable_human_feedback=False)
    print(result["final_report"])

    # Example 2: Run with human feedback (programmatic)
    print("\n\nExample 2: Programmatic human feedback")
    state = run_agent("Artificial Intelligence Ethics", enable_human_feedback=True)
    feedback = "Add more concrete examples of AI bias and expand the regulatory section"
    final_result = apply_human_feedback(state, feedback)
    print(final_result["final_report"])

    # Example 3: Interactive feedback
    print("\n\nExample 3: Interactive feedback")
    interactive_result = run_agent_with_interactive_feedback("Blockchain Technology")
