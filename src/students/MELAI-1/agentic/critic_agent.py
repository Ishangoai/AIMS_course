# ============================================================================
# FILE: critic_agent.py
# Critic Agent - Validates and revises report for quality
# ============================================================================

import os
from typing import Dict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")

TARGET_WORD_COUNT = 1000
WORD_COUNT_MIN = 950
WORD_COUNT_MAX = 1050
MAX_REVISION_ROUNDS = 10
REPORT_TOPIC = "Agentic AI"
NUM_RESEARCH_BULLETS = 10
NUM_FACT_CHECKS = 6
MODEL_NAME = "gemini-2.0-flash-exp"


class CriticAgent:
    """
    Critic Agent with zero temperature for objective evaluation.
    Stops immediately when word count is within acceptable range.
    """

    def __init__(self, api_key: str, model: str = MODEL_NAME):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.0, max_output_tokens=1800, google_api_key=api_key)

    def count_words(self, text: str) -> int:
        """Count words in text."""
        return len(text.split())

    def validate_structure(self, draft: str) -> bool:
        """Validate that report has required sections."""
        required_sections = ["## Introduction", "## Main Body", "## Conclusion"]
        return all(section in draft for section in required_sections)

    def calculate_revision_strategy(self, word_count: int) -> Dict:
        """Calculate what type of revision is needed."""
        if word_count < WORD_COUNT_MIN:
            gap = WORD_COUNT_MIN - word_count
            return {
                "strategy": "EXPAND",
                "gap": gap,
                "guidelines": """- Add technical depth (explain concepts thoroughly)
- Include concrete examples (specific tools: LangGraph, LangChain, CrewAI)
- Elaborate on architectural details (perception-reasoning-action loops)
- Add real-world application scenarios
- Expand on implementation patterns""",
                "target_change": f"Need approximately {gap} more words to reach minimum {WORD_COUNT_MIN}",
            }
        elif word_count > WORD_COUNT_MAX:
            excess = word_count - WORD_COUNT_MAX
            return {
                "strategy": "CONDENSE",
                "gap": excess,
                "guidelines": """- Remove redundant phrases and repetitive content
- Simplify overly verbose explanations
- Merge overlapping points into concise statements
- Eliminate unnecessary qualifiers and adjectives
- Consolidate similar examples""",
                "target_change": f"Remove approximately {excess} words to reach maximum {WORD_COUNT_MAX}",
            }
        else:
            return {
                "strategy": "FIX_STRUCTURE",
                "gap": 0,
                "guidelines": """- Add missing sections with proper markdown headers
- Ensure: ## Introduction, ## Main Body, ## Conclusion
- Verify each section has appropriate content""",
                "target_change": "Add missing section headers",
            }

    def perform_single_revision(self, draft: str, strategy_info: Dict, revision_count: int) -> str:
        """Perform a single revision iteration."""

        prompt = f"""You are a Technical Editor providing precise revisions.

REVISION #{revision_count + 1}

ISSUES DETECTED:
- {strategy_info["target_change"]}
- Strategy required: {strategy_info["strategy"]}

REVISION GUIDELINES:
{strategy_info["guidelines"]}

CRITICAL REQUIREMENTS:
- TARGET WORD COUNT: {TARGET_WORD_COUNT} words (acceptable range: {WORD_COUNT_MIN}-{WORD_COUNT_MAX})
- Maintain ALL citations [1], [2], [3], etc. exactly as they are
- Preserve all factual accuracy
- Keep the three-section structure (Introduction, Main Body, Conclusion)
- Do NOT add placeholders or TODOs
- Make SIGNIFICANT changes to hit the word count target

CURRENT DRAFT TO REVISE ({self.count_words(draft)} words):
{draft}

Generate the COMPLETE revised report with proper markdown formatting.
IMPORTANT: Aim for exactly {TARGET_WORD_COUNT} words (Â±50 words acceptable)."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()

    def revise_until_valid(self, draft: str) -> Dict:
        """
        Main revision loop - STOPS IMMEDIATELY when word count is in range [950-1050].

        Returns final report with metadata about revision process.
        """
        current_draft = draft
        revision_count = 0

        print(f"\n{'=' * 60}")
        print("ðŸ” CRITIC AGENT: Starting Validation")
        print(f"{'=' * 60}\n")

        # Check initial state
        word_count = self.count_words(current_draft)
        structure_valid = self.validate_structure(current_draft)
        within_range = WORD_COUNT_MIN <= word_count <= WORD_COUNT_MAX

        print("ðŸ“Š Initial state:")
        print(f"   - Word count: {word_count} (target: {WORD_COUNT_MIN}-{WORD_COUNT_MAX})")
        print(f"   - Structure valid: {structure_valid}")
        print(f"   - Within range: {within_range}\n")

        # âœ… ARRÃŠT IMMÃ‰DIAT si dÃ©jÃ  dans l'intervalle
        if within_range and structure_valid:
            print(f"{'=' * 60}")
            print("âœ… VALIDATION RÃ‰USSIE - Aucune rÃ©vision nÃ©cessaire!")
            print(f"   Word count: {word_count}")
            print(f"{'=' * 60}\n")
            return {
                "draft": current_draft,
                "success": True,
                "revision_count": 0,
                "word_count": word_count,
                "within_range": True,
                "structure_valid": True,
            }

        # Boucle de rÃ©vision
        while revision_count < MAX_REVISION_ROUNDS:
            print(f"ðŸ”„ RÃ©vision #{revision_count + 1}...")
            strategy_info = self.calculate_revision_strategy(word_count)

            # Perform revision
            current_draft = self.perform_single_revision(current_draft, strategy_info, revision_count)

            revision_count += 1

            # Check new state
            word_count = self.count_words(current_draft)
            structure_valid = self.validate_structure(current_draft)
            within_range = WORD_COUNT_MIN <= word_count <= WORD_COUNT_MAX

            print(f"   âœ“ Nouveau word count: {word_count}")

            # âœ… ARRÃŠT IMMÃ‰DIAT dÃ¨s que dans l'intervalle
            if within_range and structure_valid:
                print(f"\n{'=' * 60}")
                print(f"âœ… VALIDATION RÃ‰USSIE aprÃ¨s {revision_count} rÃ©vision(s)!")
                print(f"   Word count final: {word_count}")
                print(f"{'=' * 60}\n")
                return {
                    "draft": current_draft,
                    "success": True,
                    "revision_count": revision_count,
                    "word_count": word_count,
                    "within_range": True,
                    "structure_valid": True,
                }

            print()

        # Max iterations reached without success
        print(f"\n{'=' * 60}")
        print(f"âš ï¸ Max revisions ({MAX_REVISION_ROUNDS}) atteint")
        print(f"   Word count final: {word_count}")
        print(f"   Dans l'intervalle: {within_range}")
        print(f"{'=' * 60}\n")

        return {
            "draft": current_draft,
            "success": False,
            "revision_count": revision_count,
            "word_count": word_count,
            "within_range": within_range,
            "structure_valid": structure_valid,
        }

    # Maintain backward compatibility
    def revise_if_needed(self, draft: str, word_count: int, structure_valid: bool, revision_count: int) -> Dict:
        """Legacy function - redirects to new revise_until_valid method."""
        return self.revise_until_valid(draft)


# Create agent
critic_agent = CriticAgent(api_key=GOOGLE_API_KEY)
print("âœ… Critic Agent initialized")


# Example usage
def validate_report(draft_report: str) -> Dict:
    """
    Validate and revise a report until it meets word count requirements.
    STOPS IMMEDIATELY when word count is between 950-1050.

    Returns:
        dict with 'success', 'draft', 'word_count', 'revision_count'
    """
    return critic_agent.revise_until_valid(draft_report)
