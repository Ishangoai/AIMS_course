from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from agents.article_writer.helpers import sentence_reduce_random
from agents.article_writer.prompt_manager import PromptManager
from agents.article_writer.searcher import WikipediaSearcher
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from .config import Config, RuntimeConfig, merge_runtime
from .nodes import planner_node, quality_validator_node, researcher_node, rewriter_node, word_counter_node, writer_node
from .state import ArticleState


# Routing
def route_validator(state: ArticleState) -> str:
    """Route based on validation result"""
    max_revs = state.get("max_revisions", 5)

    if state["is_valid"]:
        return "end"
    elif state["revision_count"] >= max_revs:
        print("\n⚠️  Max revisions reached. Using current version.")
        return "end"
    else:
        return "revise"


# Workflow builder
def create_workflow(
    runtime_config: RuntimeConfig,
    prompt_manager: PromptManager,
    searcher
) -> StateGraph:
    """Create the LangGraph workflow"""
    # Initialize components
    llm_writer = ChatGoogleGenerativeAI(
        model=runtime_config.model_name,
        temperature=runtime_config.writer_temperature,
        google_api_key=Config.get_api_key(),
        max_tokens=runtime_config.writer_max_tokens
    )

    llm_rewriter = ChatGoogleGenerativeAI(
        model=runtime_config.model_name,
        temperature=runtime_config.rewriter_temperature,
        google_api_key=Config.get_api_key(),
        max_tokens=runtime_config.rewriter_max_tokens
    )

    llm_validator = ChatGoogleGenerativeAI(
        model=runtime_config.model_name,
        temperature=runtime_config.validator_temperature,
        google_api_key=Config.get_api_key(),
        max_tokens=runtime_config.validator_max_tokens
    )

    # Create graph
    workflow = StateGraph(ArticleState)

    # Add nodes
    workflow.add_node("planner",
                    lambda s: planner_node(s, llm_writer, prompt_manager))  # type: ignore
    workflow.add_node("researcher",
                    lambda s: researcher_node(s, searcher))  # type: ignore
    workflow.add_node("writer",
                    lambda s: writer_node(s, llm_writer, prompt_manager))  # type: ignore
    workflow.add_node("word_counter",
                    lambda s: word_counter_node(s))  # type: ignore
    workflow.add_node("quality_validator",
                    lambda s: quality_validator_node(s, llm_validator, prompt_manager))  # type: ignore
    workflow.add_node("rewriter",
                    lambda s: rewriter_node(s, llm_rewriter, prompt_manager))  # type: ignore

    # Define flow
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    workflow.add_edge("writer", "word_counter")
    workflow.add_edge("word_counter", "quality_validator")

    # Conditional routing after quality validator
    workflow.add_conditional_edges(
        "quality_validator",
        route_validator,
        {"revise": "rewriter", "end": END}
    )

    # After rewrite, go back through both validators
    workflow.add_edge("rewriter", "word_counter")

    return workflow.compile()  # Workflow builder  # type: ignore


class ArticleSystem:
    """
    Main interface for article generation system

    Usage:
        system = ArticleSystem()
        system.setup()
        result = system.generate("Your topic")

        # Save article
        with open("article.txt", "w") as f:
            f.write(result["article"])
    """

    def __init__(self):
        """Initialize system"""
        self.prompt_manager = None
        self.workflow = None
        self._is_ready = False

    def setup(
        self,
        searcher,
        runtime_overrides: dict | None = None,
        config: Config | None = None
    ):
        """Setup system - creates prompts and workflow"""
        print("🔧 Setting up Article Generation System...")

        self.config = config or Config()

        # Initialize prompts
        self.prompt_manager = PromptManager(prompts_path=self.config.PROMPTS_PATH)

        self.runtime_config = merge_runtime(runtime_overrides)

        # Create workflow
        self.workflow = create_workflow(
            prompt_manager=self.prompt_manager,
            runtime_config=self.runtime_config,
            searcher=searcher
        )

        self._is_ready = True

        print("\n✅ Setup complete!")
        print(f"   - Prompts: {self.config.PROMPTS_PATH}")
        print(f"   - Model:   {self.runtime_config.model_name}")
        print(f"   - Target:  {self.runtime_config.target_words} ± {self.runtime_config.tolerance}")
        print(f"   - Max revs:{self.runtime_config.max_revisions}")
        print("   - Ready to generate!")

    def generate(
        self,
        topic: str,
        runtime_overrides: dict | None = None
    ) -> Dict[str, Any]:
        """
        Generate article

        Args:
            topic: Article topic

        Returns:
            Dictionary with article and metadata
        """
        if not self._is_ready or runtime_overrides:
            self.setup(runtime_overrides)

        min_words = self.runtime_config.target_words - self.runtime_config.tolerance
        max_words = self.runtime_config.target_words + self.runtime_config.tolerance

        print(f"\n{'=' * 70}")
        print(f"🚀 Generating Article: {topic}")
        print(f"{'=' * 70}")

        # Initialize state
        initial_state = ArticleState(
            topic=topic,
            outline="",
            article_draft="",
            validation_feedback="",
            is_valid=False,
            revision_count=0,
            word_count=0,

            target_words=self.runtime_config.target_words,
            tolerance=self.runtime_config.tolerance,
            min_words=min_words,
            max_words=max_words,
            max_revisions=self.runtime_config.max_revisions,
        )  # type: ignore

        # Run workflow
        final_state = self.workflow.invoke(initial_state)  # type: ignore

        print(f"\n{'=' * 70}")
        print("✨ Generation Complete!")
        print(f"   - Word count: {final_state['word_count']}")
        print(f"   - Revisions: {final_state['revision_count']}")
        print(f"{'=' * 70}\n")

        safeguarded_article, safeguarded_wc = sentence_reduce_random(
            final_state["article_draft"],
            final_state["max_words"],
            preserve_first_last=True,
            seed=42
        )
        met_limits_without_safeguard = (
            final_state["word_count"] >= final_state["min_words"]
            and final_state["word_count"] <= final_state["max_words"]
        )

        # Build UI payload
        return {
            # 1) Final article from the LLM after max revisions
            "final_article_llm": final_state["article_draft"],

            # 2) If it was valid or not
            "was_valid": final_state["is_valid"],

            # 3) Other details
            "details": {
                "topic": final_state["topic"],
                "outline": final_state["outline"],
                "sources": final_state["research_sources"],
                "word_count": final_state["word_count"],
                "validation_feedback": final_state["validation_feedback"],
                "revisions_used": final_state["revision_count"],
                "limits": {
                    "target_words": final_state["target_words"],
                    "tolerance": final_state["tolerance"],
                    "min_words": final_state["min_words"],
                    "max_words": final_state["max_words"],
                },
                # runtime model settings exposed from self.runtime (not in state)
                "model": self.runtime_config.model_name,
                "temperatures": {
                    "writer": self.runtime_config.writer_temperature,
                    "rewriter": self.runtime_config.rewriter_temperature,
                    "validator": self.runtime_config.validator_temperature,
                },
                "max_tokens": {
                    "writer": self.runtime_config.writer_max_tokens,
                    "rewriter": self.runtime_config.rewriter_max_tokens,
                    "validator": self.runtime_config.validator_max_tokens,
                },
            },

            # 4) Max revisions
            "max_revisions": self.runtime_config.max_revisions,

            # 5) Revised article that meets the word count limit after applying safeguard
            "safeguarded_article": safeguarded_article,
            "safeguarded_word_count": safeguarded_wc,

            # convenience flag
            "met_limits_without_safeguard": met_limits_without_safeguard,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "ready": self._is_ready,
            "prompts_path": str(Path(Config.PROMPTS_PATH).absolute()),
            "config": {
                "model": self.runtime_config.model_name if self.runtime_config else None,
                "writer_temperature": getattr(self.runtime_config, "writer_temperature", None),
                "rewriter_temperature": getattr(self.runtime_config, "rewriter_temperature", None),
                "validator_temperature": getattr(self.runtime_config, "validator_temperature", None),
                "target_words": getattr(self.runtime_config, "target_words", None),
                "tolerance": getattr(self.runtime_config, "tolerance", None),
                "max_revisions": getattr(self.runtime_config, "max_revisions", None),
                "writer_max_tokens": getattr(self.runtime_config, "writer_max_tokens", None),
                "rewriter_max_tokens": getattr(self.runtime_config, "rewriter_max_tokens", None),
                "validator_max_tokens": getattr(self.runtime_config, "validator_max_tokens", None),
            }
        }


def run_article_writer(
    topic: str = "CI/CD Pipelines",
    target_words: int = 1000,
    tolerance: int = 50,
    max_revisions: int = 3,
    model_name: str = "gemini-2.0-flash-exp",
    searcher=WikipediaSearcher(),
    writer_temperature: float = 0.2,
    rewriter_temperature: float = 0.1,
    validator_temperature: float = 0.0,
    writer_max_tokens: int | None = None,
    rewriter_max_tokens: int | None = None,
    validator_max_tokens: int | None = None,
    save_files: bool = False
):
    print("=" * 70)
    print("ARTICLE GENERATION SYSTEM")
    print("=" * 70)

    runtime_overrides = {
        "model_name": model_name,
        "writer_temperature": writer_temperature,
        "rewriter_temperature": rewriter_temperature,
        "validator_temperature": validator_temperature,
        "writer_max_tokens": writer_max_tokens,
        "rewriter_max_tokens": rewriter_max_tokens,
        "validator_max_tokens": validator_max_tokens,
        "target_words": target_words,
        "tolerance": tolerance,
        "max_revisions": max_revisions,
    }

    system = ArticleSystem()
    system.setup(
        config=Config(),
        runtime_overrides=runtime_overrides,
        searcher=searcher
    )

    topic = (topic or "The Future of Artificial Intelligence").strip() or "The Future of Artificial Intelligence"
    result = system.generate(topic)

    if save_files:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(".")
        llm_path = out_dir / f"article_llm_{stamp}.md"
        safe_path = out_dir / f"article_safe_{stamp}.md"
        meta_path = out_dir / f"article_meta_{stamp}.json"

        llm_path.write_text(result["final_article_llm"], encoding="utf-8")
        safe_path.write_text(result["safeguarded_article"], encoding="utf-8")
    else:
        llm_path, safe_path, meta_path = None, None, None

    meta = {
        "was_valid": result["was_valid"],
        "max_revisions": result["max_revisions"],
        "met_limits_without_safeguard": result["met_limits_without_safeguard"],
        "details": result["details"],
        "files": {"final_article_llm": llm_path, "safeguarded_article": llm_path},
    }

    return {
        "llm_path": str(llm_path),
        "safe_path": str(safe_path),
        "meta_path": str(meta_path),
        "metadata": meta,
        "result": result
    }


if __name__ == "__main__":
    response = run_article_writer(
        topic="CI/CD Pipelines",
        target_words=1000,
        tolerance=50,
        max_revisions=3
    )

    print(response)
