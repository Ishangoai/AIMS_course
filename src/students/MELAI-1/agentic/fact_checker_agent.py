# ============================================================================
# FILE: fact_checker_agent.py
# Fact-Checker Agent - Validates claims against research
# ============================================================================
from dotenv import load_dotenv
from typing import Dict, List
from langchain_core.messages import HumanMessage  # FIXED: Use HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import json
import re

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


class FactCheckerAgent:
    """
    Fact-Checker Agent with zero temperature for objective verification.
    Complexity: Claim extraction, verification logic, confidence scoring.
    """

    def __init__(self, api_key: str, model: str = MODEL_NAME):
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.2,
            max_output_tokens=1500,
            google_api_key=api_key
        )

    def extract_claims(self, report: str, n: int = NUM_FACT_CHECKS) -> List[str]:
        """
        Extract key factual claims using heuristics.

        BEST PRACTICE: Use Python tools for extraction, not LLMs.
        """
        sentences = [s.strip() for s in report.split('.') if len(s.strip()) > 50]

        # Heuristic scoring for factual claims
        factual_patterns = [
            r'\b(Agentic AI|agents?|LLM|tool|framework|LangGraph|LangChain)\b',
            r'\b(enables?|allows?|provides?|implements?|uses?)\b',
            r'\d+',
            r'\b(through|via|using|by)\b'
        ]

        scored = []
        for sent in sentences:
            score = sum(1 for p in factual_patterns if re.search(p, sent, re.IGNORECASE))
            if score > 0:
                scored.append((score, sent))

        scored.sort(reverse=True)
        top_claims = [s for _, s in scored[:n]]

        print(f"   Extracted {len(top_claims)} claims for verification")
        return top_claims

    def verify_claims(self, claims: List[str], research_bullets: List[Dict]) -> List[Dict]:
        """
        Verify claims against research bullets.

        BEST PRACTICE: Structured verification with confidence levels.
        """

        prompt = f"""You are a Fact-Verification Specialist.

MISSION:
Verify each claim against the research bullets provided.

CLAIMS TO VERIFY:
{json.dumps(claims, indent=2)}

RESEARCH BULLETS (GROUND TRUTH):
{json.dumps(research_bullets, indent=2)}

OUTPUT FORMAT (JSON only, no markdown):
{{
  "fact_checks": [
    {{
      "claim": "exact claim text from above",
      "status": "VERIFIED|SUPPORTED|QUESTIONABLE|UNSUPPORTED",
      "supporting_bullet_ids": [1, 3],
      "explanation": "brief one-sentence reasoning",
      "confidence": "HIGH|MEDIUM|LOW"
    }}
  ]
}}

STATUS DEFINITIONS:
- VERIFIED: Claim is directly stated in research bullets
- SUPPORTED: Claim is a reasonable inference from research
- QUESTIONABLE: Claim is partially supported but may be overstated
- UNSUPPORTED: Claim has no basis in the research bullets

CONFIDENCE LEVELS:
- HIGH: Multiple supporting bullets, direct evidence
- MEDIUM: Single supporting bullet or indirect support
- LOW: Weak or ambiguous support

Generate exactly {len(claims)} fact-checks now."""

        # FIXED: Use HumanMessage
        response = self.llm.invoke([HumanMessage(content=prompt)])

        try:
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            data = json.loads(content)
            checks = data.get("fact_checks", [])

            verified = sum(1 for c in checks if c.get("status") in ["VERIFIED", "SUPPORTED"])
            print(f"âœ… Fact-check: {verified}/{len(checks)} verified/supported")

            return checks

        except json.JSONDecodeError as e:
            print(f"âš ï¸ JSON parse error: {e}")
            return [{
                "claim": "Parse error occurred",
                "status": "UNSUPPORTED",
                "explanation": "JSON parsing failed",
                "confidence": "LOW"
            }]


# Create agent (don't test yet)
fact_checker = FactCheckerAgent(api_key=GOOGLE_API_KEY)
print("âœ… Fact-Checker Agent initialized")