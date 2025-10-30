from __future__ import annotations

from typing import Any, Dict, List, Optional

from agents.article_writer.helpers import safe_llm_invoke
from agents.article_writer.prompt_manager import PromptManager
from agents.article_writer.state import ArticleState
from langchain_google_genai import ChatGoogleGenerativeAI


# Agent nodes
def planner_node(
    state: ArticleState,
    llm: ChatGoogleGenerativeAI,
    pm: PromptManager
) -> ArticleState:
    """Creates article outline"""
    print(f"\n📋 PLANNER: Creating outline for '{state['topic']}'")

    prompt = pm.get(
        "planner",
        topic=state["topic"],
        word_count=state["word_count"]
    )
    outline = safe_llm_invoke(llm, prompt)

    state["outline"] = outline
    print(f"✓ Outline created ({len(outline.split())} words)")
    return state


def researcher_node(
    state: ArticleState,
    llm: ChatGoogleGenerativeAI,
    pm: PromptManager,
    searcher: Optional[Any] = None
) -> ArticleState:
    """Generates research insights using LLM"""
    print("\n🔍 RESEARCHER: Gathering insights")

    # Extract research queries
    query_prompt = pm.get("researcher_queries",
                          topic=state["topic"],
                          outline=state["outline"][:500])
    query_response = safe_llm_invoke(llm, query_prompt)
    queries = [q.strip() for q in query_response.split('\n') if q.strip()][:5]

    insights: List[str] = []
    sources_all: List[Dict[str, str]] = []

    # Generate insights for each query
    for i, query in enumerate(queries, 1):
        print(f"  Researching: {query[:60]}...")

        results = []

        if searcher and searcher.enabled:
            results = searcher.search(query, num_results=3)
        else:
            # No web search results, LLM will use prior knowledge
            results = []

        print("Results: ", results)

        # Record sources
        for r in results:
            sources_all.append({
                "title": r.get("title", ""),
                "link": r.get("link", ""),
                "content": r.get("content", ""),
                "query": query
            })

        if results:
            evidence_lines = []
            for r in results:
                title = r.get("title", "") or r.get("link", "")
                snippet = (r.get("content", "") or "")[:240]
                link = r.get("link", "")
                evidence_lines.append(f"- {title} — {snippet} [{link}]")
            evidence = "\n".join(evidence_lines)
        else:
            evidence = "(No external sources available; use domain knowledge.)"

        # Ask LLM to synthesize a succinct, sourced insight
        insight_prompt = pm.get("researcher_insights", query=query, evidence=evidence)
        insight = safe_llm_invoke(llm, insight_prompt)
        insights.append(f"Query {i}: {query}\n\n{insight}")

    state["research_sources"] = sources_all
    print(f"✓ Generated {len(insights)} research insights")

    return state


def writer_node(
    state: ArticleState,
    llm: ChatGoogleGenerativeAI,
    pm: PromptManager,
    searcher
) -> ArticleState:
    """Writes or revises article"""
    print(f"\n✍️  WRITER: Composing article (revision {state['revision_count'] + 1})")

    sources = searcher.search(state["topic"], num_results=3) if searcher else []
    state["research_sources"] = sources

    evidence = "\n".join(
        f"- {s['title']} — {s['content'][:200]} [{s['link']}]"
        for s in sources if s.get("link")
    )
    sources_md = "\n".join(
        f"- [{s['title']}]({s['link']})"
        for s in sources if s.get("link")
    )

    prompt = pm.get(
        "writer",
        topic=state["topic"],
        outline=state["outline"],
        min_words=state.get("min_words"),
        max_words=state.get("max_words"),
        sources=sources_md,
        evidence=evidence
    )

    article = safe_llm_invoke(llm, prompt)

    state["article_draft"] = article
    state["revision_count"] += 1
    state["word_count"] = len(article.split())

    print(f"✓ Article written ({state['word_count']} words)")
    return state


def validator_node(
    state: ArticleState,
    llm: ChatGoogleGenerativeAI,
    pm: PromptManager
) -> ArticleState:
    """Validates article quality and word count"""
    print("\n✅ VALIDATOR: Checking quality")

    word_count = len(state["article_draft"].split())
    min_words = state.get("min_words")
    max_words = state.get("max_words")

    feedback_parts = []

    # Word count check
    if word_count < min_words:
        feedback_parts.append(f"Too short: {word_count} words (need {min_words}-{max_words})")
    elif word_count > max_words:
        feedback_parts.append(f"Too long: {word_count} words (need {min_words}-{max_words})")

    # Quality check (if word count ok or last revision)
    if not feedback_parts or state["revision_count"] >= state.get("max_revisions"):
        prompt = pm.get(
            "validator",
            article=state["article_draft"],
            word_count=word_count,
            min_words=state["min_words"],
            max_words=state["max_words"]
        )

        quality_feedback = safe_llm_invoke(llm, prompt, max_retries=2)

        if "VALID" not in quality_feedback.upper():
            feedback_parts.append(quality_feedback)

    # Set validation result
    state["is_valid"] = len(feedback_parts) == 0
    state["validation_feedback"] = "\n\n".join(feedback_parts)

    if state["is_valid"]:
        print("✓ Article validated")
    else:
        print("⚠️  Issues found")

    return state


def rewriter_node(
    state: ArticleState,
    llm: ChatGoogleGenerativeAI,
    pm: PromptManager
) -> ArticleState:
    """Revises article based on feedback"""
    print(f"\n🔄 REWRITER: Revising (attempt {state['revision_count']})")

    prompt = pm.get(
        "rewriter",
        article=state["article_draft"],
        feedback=state["validation_feedback"],
        outline=state["outline"],
        min_words=state.get("min_words"),
        max_words=state.get("max_words")
    )

    revised = safe_llm_invoke(llm, prompt)

    state["article_draft"] = revised
    state["word_count"] = len(revised.split())

    state["revision_count"] += 1

    print(f"✓ Article revised ({state['word_count']} words)")
    return state
