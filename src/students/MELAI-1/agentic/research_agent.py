# ============================================================================
# FILE: research_agent.py
# Research Agent - Gathers comprehensive information on Agentic AI
# ============================================================================
import json
import os
from typing import Dict, List

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage  # FIXED: Use HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI

TARGET_WORD_COUNT = 1000
WORD_COUNT_MIN = 950
WORD_COUNT_MAX = 1050
MAX_REVISION_ROUNDS = 10
REPORT_TOPIC = "Agentic AI"
NUM_RESEARCH_BULLETS = 10
NUM_FACT_CHECKS = 6
MODEL_NAME = "gemini-2.0-flash-exp"

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")


class ResearchAgent:
    """
    Research Agent with low temperature for factual accuracy.
    Complexity: Structured JSON output, categorization, confidence levels.
    """

    def __init__(self, api_key: str, model: str = MODEL_NAME):
        self.llm = ChatGoogleGenerativeAI(model=model, temperature=0.5, max_output_tokens=2000, google_api_key=api_key)

    def gather_research(self, topic: str, num_bullets: int = NUM_RESEARCH_BULLETS) -> List[Dict]:
        """
        Gather structured research bullets with categorization.

        BEST PRACTICE: Structured prompt with clear sections and JSON output.
        """

        prompt = f"""You are a Research Agent specializing in {topic}.

MISSION:
Generate {num_bullets} technical research points about {topic}.

CONTEXT:
{topic} refers to AI systems that autonomously perceive, reason, and act using:
- Tool calling and external API integration
- Multi-step planning and task decomposition
- Memory systems (short-term, long-term, vector stores)
- Frameworks: LangGraph, LangChain, AutoGPT, CrewAI

REQUIREMENTS:
Cover these categories:
1. Definition & Core Concepts (2 points)
2. Technical Architecture (3 points)
3. Frameworks & Tools (2 points)
4. Applications (2 points)
5. Challenges & Future (1 point)

OUTPUT FORMAT (JSON only, no markdown):
{{
  "bullets": [
    {{
      "text": "20-50 word factual statement",
      "category": "definition|architecture|frameworks|applications|challenges",
      "technical_depth": "high|medium|low",
      "source_type": "academic|industry|documentation"
    }}
  ]
}}

QUALITY STANDARDS:
- The number of words should strictly be between 975-1025
- Specific and technical (avoid vague claims)
- Include concrete examples (framework names, techniques)
- Mutually distinct points
- Factually accurate

Generate exactly {num_bullets} bullets now."""

        # FIXED: Use HumanMessage instead of SystemMessage
        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            content = response.content.strip()
            if content.startswith("```"):
                # Remove markdown code block formatting
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)
            bullets = data.get("bullets", [])

            # Ensure IDs for citations
            for i, b in enumerate(bullets):
                b["id"] = i + 1

            print(f"âœ… Research: {len(bullets)} bullets gathered")
            return bullets

        except json.JSONDecodeError as e:
            print(f"âš ï¸ JSON parse error: {e}")
            print(f"   Raw response (first 200 chars): {response.content[:200]}")
            # Fallback with valid structure
            return [
                {
                    "id": 1,
                    "text": f"{topic} systems autonomously use tools and multi-step reasoning to achieve goals.",  # type: ignore
                    "category": "definition",
                    "technical_depth": "high",
                    "source_type": "academic",
                }
            ]


# Create agent (don't test yet - will test in workflow)
research_agent = ResearchAgent(api_key=GOOGLE_API_KEY)
print("âœ… Research Agent initialized")
