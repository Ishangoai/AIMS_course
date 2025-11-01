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
    searcher: Optional[Any] = None
) -> ArticleState:
    """Generates research insights using LLM"""
    print("\n🔍 RESEARCHER: Searching for sources")

    raw = searcher.search(state["topic"], num_results=3) if searcher else []
    normalized: List[Dict[str, str]] = []
    seen = set()

    for r in raw:
        title = (r.get("title") or "").strip()
        link = (r.get("link") or "").strip()
        content = (r.get("content") or r.get("snippet") or "").strip()
        engine = r.get("engine") or r.get("source") or "unknown"

        # fallback key to avoid empty links causing duplicates
        key = link or (title + content[:80])
        if key in seen:
            continue
        seen.add(key)

        normalized.append({
            "title": title or "untitled",
            "link": link,
            "content": content,
            "engine": engine
        })

    state["research_sources"] = normalized
    print(f"✓ Collected {len(normalized)} research sources")
    return state


def writer_node(
    state: ArticleState,
    llm: ChatGoogleGenerativeAI,
    pm: PromptManager
) -> ArticleState:
    """Writes or revises article"""
    print(f"\n✍️  WRITER: Composing article (revision {state['revision_count'] + 1})")

    sources = state.get("research_sources") or []

    evidence = "\n".join(
        f"- {s['title']} — {s['content'][:200]} [{s['link']}]"
        for s in sources if s.get("link")
    ) or "No external sources available."

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
    state["revision_count"] = state.get("revision_count", 0) + 1
    state["word_count"] = len(article.split())

    print(f"✓ Article written ({state['word_count']} words)")
    return state


def word_counter_node(state: ArticleState) -> ArticleState:
    """Counts words and checks against limits"""
    print("\n🔢 WORD COUNTER: Checking word count")

    word_count = len(state["article_draft"].split())
    min_words = state.get("min_words")
    max_words = state.get("max_words")

    feedback_parts = []

    if word_count < min_words:
        feedback_parts.append(
            f"Too short: {word_count} words (need {min_words}-{max_words})"
        )
    elif word_count > max_words:
        feedback_parts.append(
            f"Too long: {word_count} words (need {min_words}-{max_words})"
        )

    state["word_count_valid"] = len(feedback_parts) == 0
    state["word_count_feedback"] = "\n".join(feedback_parts) if feedback_parts else ""

    status = "✓" if state["word_count_valid"] else "⚠️"
    print(f"{status} Word count: {word_count} words (target: {min_words}-{max_words})")

    return state


def quality_validator_node(
    state: ArticleState,
    llm: ChatGoogleGenerativeAI,
    pm: PromptManager
) -> ArticleState:
    """Validates article quality using LLM"""
    print("\n✅ QUALITY VALIDATOR: Checking content quality")

    word_count = len(state["article_draft"].split())

    prompt = pm.get(
        "validator",
        article=state["article_draft"],
        word_count=word_count,
        min_words=state["min_words"],
        max_words=state["max_words"]
    )

    quality_feedback = safe_llm_invoke(llm, prompt, max_retries=2)

    state["quality_valid"] = "VALID" in quality_feedback.upper()
    state["quality_feedback"] = quality_feedback if not state["quality_valid"] else ""

    # Overall validation: both word count and quality must pass
    state["is_valid"] = state.get("word_count_valid", False) and state["quality_valid"]

    # Combine all feedback for rewriter
    feedback_parts = []

    if state.get("word_count_feedback"):
        feedback_parts.append("**WORD COUNT ISSUES:**")
        feedback_parts.append(state["word_count_feedback"])

    if state.get("quality_feedback"):
        feedback_parts.append("**QUALITY ISSUES:**")
        feedback_parts.append(state["quality_feedback"])

    state["validation_feedback"] = "\n\n".join(feedback_parts)

    # Detailed logging
    if state["is_valid"]:
        print("✓ All checks passed - article is ready")
    else:
        print("⚠️  Issues found:")
        if not state.get("word_count_valid"):
            print("   - Word count issue")
        if not state.get("quality_valid"):
            print("   - Quality issue")

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
