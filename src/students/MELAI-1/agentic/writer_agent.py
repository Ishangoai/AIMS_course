# ============================================================================
# FILE: writer_agent.py
# Writer Agent - Composes structured technical report with citations
# ============================================================================
import json
import os
import re
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Configuration ---
TARGET_WORD_COUNT = 1000
WORD_COUNT_MIN = 950
WORD_COUNT_MAX = 1050
MAX_REVISION_ROUNDS = 10
REPORT_TOPIC = "Agentic AI"
NUM_RESEARCH_BULLETS = 10
NUM_FACT_CHECKS = 6
MODEL_NAME = "gemini-2.0-flash-exp"

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")


class WriterAgent:
    """
    Writer Agent with moderate temperature for natural writing.
    Complexity: Structured output, inline citations, section requirements.
    """

    def __init__(self, api_key: str, model: str = MODEL_NAME):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.5, max_output_tokens=2500, google_api_key=api_key)

    def write_report(self, topic: str, research_bullets: List[Dict], target_words: int = None) -> Dict:
        """
        Write structured report with inline citations.

        BEST PRACTICE: Clear structure requirements, citation system.
        """

        # Use target_words if provided, otherwise use default TARGET_WORD_COUNT
        word_target = target_words if target_words is not None else TARGET_WORD_COUNT

        research_context = json.dumps(research_bullets, indent=2)

        prompt = f"""You are an Expert Technical Writer for AI/ML documentation.

ASSIGNMENT:
Write a {word_target}-word technical report on: {topic}

STRUCTURE (use these exact headers):
## Introduction
[Define {topic}, explain significance, preview key points - approximately 220 words]

## Main Body
[Technical discussion - approximately 600 words]

### Core Architecture
[How agentic systems work: perception, reasoning, action loops]

### Key Frameworks and Tools
[LangGraph, LangChain, AutoGPT - be specific about capabilities]

### Applications and Impact
[Real-world use cases with concrete examples]

## Conclusion
[Synthesize insights, discuss future directions - approximately 180 words]

CITATIONS:
- Use [1], [2], [3] format after claims
- Cite at least 8 different research bullets
- Example: "Agents use tool calling to interact with external APIs [1]."

RESEARCH BULLETS FOR CITATION:
{research_context}

CONSTRAINTS:
- ONLY use information from the research bullets above
- Target: {word_target} words (strict requirement, must be between {WORD_COUNT_MIN}-{WORD_COUNT_MAX}).
- Professional academic tone
- NO placeholders like [TODO] or [EXAMPLE]
- Write in flowing paragraphs (no bullet lists in main text)
- Maintain factual accuracy

IMPORTANT: The word count MUST be between {WORD_COUNT_MIN} and {WORD_COUNT_MAX} words. Count carefully.

Generate the complete report now."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        draft = response.content.strip()

        # Validate with Python tools
        word_count = len(draft.split())
        has_intro = bool(re.search(r"^##?\s+Introduction", draft, re.MULTILINE | re.IGNORECASE))
        has_body = bool(re.search(r"^##?\s+(Main Body|Main)", draft, re.MULTILINE | re.IGNORECASE))
        has_conclusion = bool(re.search(r"^##?\s+Conclusion", draft, re.MULTILINE | re.IGNORECASE))

        print(f"âœ… Writer: {word_count} words drafted")
        print(f"   Structure: Intro={has_intro}, Body={has_body}, Conclusion={has_conclusion}")

        return {
            "draft": draft,
            "word_count": word_count,
            "within_range": WORD_COUNT_MIN <= word_count <= WORD_COUNT_MAX,
            "structure_valid": has_intro and has_body and has_conclusion,
        }


# Create agent instance (don't test yet)
writer_agent = WriterAgent(api_key=GOOGLE_API_KEY)
print("âœ… Writer Agent initialized")
