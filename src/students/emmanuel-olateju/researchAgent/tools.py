"""
Tools module for the agentic report generation system.
Provides Google Search integration and validation utilities.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
from typing import Any, Dict, List

from langchain_core.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper

# ============================================================================
# GOOGLE SEARCH TOOL
# ============================================================================


def get_google_search_tool():
    """
    Creates a Google Search tool with custom configuration.

    Requires:
        - GOOGLE_API_KEY environment variable
        - GOOGLE_CSE_ID (Custom Search Engine ID) environment variable

    Returns:
        Tool: Configured Google Search tool
    """
    # Initialize Google Search wrapper
    search_wrapper = GoogleSearchAPIWrapper(
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        google_cse_id=os.getenv("GOOGLE_CSE_ID"),
        k=5  # Number of results to return
    )

    def enhanced_search(query: str) -> Dict[str, Any]:
        """Enhanced Google search with structured output."""
        try:
            # Perform search
            results = search_wrapper.results(query, num_results=5)

            if not results:
                return {
                    "success": False,
                    "query": query,
                    "results": [],
                    "content": "",
                    "source": "Google Search",
                    "error": "No results found"
                }

            # Structure the results
            structured_results = []
            combined_content = []

            for idx, result in enumerate(results, 1):
                title = result.get("title", "")
                snippet = result.get("snippet", "")
                link = result.get("link", "")

                structured_results.append({
                    "rank": idx,
                    "title": title,
                    "snippet": snippet,
                    "url": link
                })

                combined_content.append(f"[Result {idx}] {title}\n{snippet}\nURL: {link}\n")

            return {
                "success": True,
                "query": query,
                "results": structured_results,
                "content": "\n".join(combined_content),
                "source": "Google Search",
                "total_results": len(results),
                "top_url": results[0].get("link", "") if results else ""
            }

        except Exception as e:
            return {
                "success": False,
                "query": query,
                "results": [],
                "content": "",
                "source": "Google Search",
                "error": str(e)
            }

    # Create the tool
    google_search_tool = Tool(
        name="google_search",
        description=(
            "Search Google for factual information and current data. "
            "Input should be a search query string. "
            "Returns top search results with titles, snippets, and URLs."
        ),
        func=enhanced_search
    )

    return google_search_tool


def get_fallback_search_tool():
    """
    Creates a fallback search tool that uses DuckDuckGo (no API key required).
    Useful for development/testing when Google API credentials aren't available.

    Returns:
        Tool: Configured DuckDuckGo search tool
    """
    from langchain_community.tools import DuckDuckGoSearchRun
    from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

    search_wrapper = DuckDuckGoSearchAPIWrapper(
        max_results=5,
        region="wt-wt",
        time="y"  # Past year
    )

    def enhanced_ddg_search(query: str) -> Dict[str, Any]:
        """Enhanced DuckDuckGo search with structured output."""
        try:
            ddg_search = DuckDuckGoSearchRun(api_wrapper=search_wrapper)
            result = ddg_search.run(query)

            if not result or result.strip() == "":
                return {
                    "success": False,
                    "query": query,
                    "content": "",
                    "source": "DuckDuckGo Search",
                    "error": "No results found"
                }

            return {
                "success": True,
                "query": query,
                "content": result,
                "source": "DuckDuckGo Search",
                "results": [{"snippet": result}]
            }

        except Exception as e:
            return {
                "success": False,
                "query": query,
                "content": "",
                "source": "DuckDuckGo Search",
                "error": str(e)
            }

    fallback_tool = Tool(
        name="fallback_search",
        description=(
            "Fallback search using DuckDuckGo. "
            "Input should be a search query string."
        ),
        func=enhanced_ddg_search
    )

    return fallback_tool


# ============================================================================
# VALIDATION TOOLS
# ============================================================================

def count_words(text: str) -> int:
    """
    Counts words in text (excludes markdown syntax).

    Args:
        text: Input text

    Returns:
        int: Word count
    """
    # Remove markdown headers
    text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
    # Remove citations
    text = re.sub(r'\[Source:.*?\]', '', text, flags=re.IGNORECASE)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Count words
    words = text.strip().split()
    return len(words)


def word_counter_tool(text: str) -> Dict[str, Any]:
    """
    LangChain-compatible word counter with validation.

    Args:
        text: Report text

    Returns:
        dict: Word count analysis with feedback
    """
    total_words = count_words(text)
    target = 1000
    tolerance = 50

    within_range = abs(total_words - target) <= tolerance
    deviation = total_words - target

    if within_range:
        feedback = f"✓ Word count: {total_words} (target: {target} ± {tolerance})"
    else:
        feedback = f"✗ Word count: {total_words} (target: {target} ± {tolerance}, deviation: {deviation:+d})"

    return {
        "total_words": total_words,
        "target": target,
        "tolerance": tolerance,
        "deviation": deviation,
        "within_range": within_range,
        "feedback": feedback
    }


def structure_validator_tool(text: str, expected_sections: List[str]) -> Dict[str, Any]:
    """
    Validates report structure (section headers).

    Args:
        text: Report text
        expected_sections: List of expected section titles

    Returns:
        dict: Structure validation results
    """
    # Find all markdown headers (## Title format)
    found_headers = re.findall(r'^##\s+(.+)$', text, re.MULTILINE)

    missing_sections = []
    for expected in expected_sections:
        # Fuzzy match (case-insensitive, partial match)
        if not any(expected.lower() in header.lower() for header in found_headers):
            missing_sections.append(expected)

    valid = len(missing_sections) == 0

    status = "✓ Structure valid" if valid else f"✗ Missing sections: {', '.join(missing_sections)}"

    return {
        "valid": valid,
        "section_count": len(found_headers),
        "expected_count": len(expected_sections),
        "found_sections": found_headers,
        "missing_sections": missing_sections,
        "status": status
    }


# ============================================================================
# FACT-CHECKING TOOLS
# ============================================================================

def claim_extractor_tool(text: str) -> List[Dict[str, str]]:
    """
    Extracts factual claims from report text.

    Args:
        text: Report text

    Returns:
        list: Extracted claims with metadata
    """
    claims = []

    # Split by sections
    sections = re.split(r'^##\s+(.+)$', text, flags=re.MULTILINE)

    current_section = "Introduction"
    for i, part in enumerate(sections):
        if i % 2 == 1:  # Section title
            current_section = part.strip()
        else:  # Section content
            # Extract sentences with factual indicators
            sentences = re.split(r'[.!?]+', part)

            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) < 20:  # Skip very short sentences
                    continue

                # Heuristics for factual claims
                factual_indicators = [
                    r'\d+%',  # Percentages
                    r'\d{4}',  # Years
                    r'\d+\s+(million|billion|thousand)',  # Large numbers
                    r'(approximately|about|over|under|nearly)\s+\d+',  # Numeric estimates
                    r'(increase|decrease|grow|decline)d?\s+(by|to)',  # Trends
                    r'(according to|research shows|studies indicate)',  # Attribution
                ]

                if any(re.search(pattern, sentence, re.IGNORECASE) for pattern in factual_indicators):
                    claims.append({
                        "claim": sentence,
                        "section": current_section,
                        "type": "factual"
                    })

    return claims


def validate_claim_against_sources(claim: str, research_database: List[Dict]) -> Dict[str, Any]:
    """
    Validates a claim against the research database.

    Args:
        claim: Factual claim to verify
        research_database: List of research sources

    Returns:
        dict: Verification result with confidence score
    """
    # Extract key terms from claim (nouns, numbers, proper nouns)
    key_terms = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b\d+(?:\.\d+)?%?\b', claim)

    if not key_terms:
        key_terms = claim.lower().split()[:5]  # Fallback: first 5 words

    # Search for key terms in research sources
    matches = []
    for source in research_database:
        if not source.get("success"):
            continue

        content = source.get("content", "").lower()
        matched_terms = sum(1 for term in key_terms if term.lower() in content)

        if matched_terms > 0:
            matches.append({
                "source": source.get("source", "Unknown"),
                "matched_terms": matched_terms,
                "match_ratio": matched_terms / len(key_terms),
                "quality_score": source.get("quality_assessment", {}).get("score", 0.5)
            })

    # Sort by match quality
    matches.sort(key=lambda x: x["match_ratio"] * x["quality_score"], reverse=True)

    # Determine verification status
    if not matches:
        verified = False
        confidence = 0.0
        supporting_sources = []
    elif matches[0]["match_ratio"] >= 0.5:  # At least 50% of terms matched
        verified = True
        confidence = matches[0]["match_ratio"] * matches[0]["quality_score"]
        supporting_sources = [m["source"] for m in matches[:3]]
    else:
        verified = False
        confidence = matches[0]["match_ratio"] * 0.5
        supporting_sources = []

    return {
        "claim": claim,
        "verified": verified,
        "confidence": confidence,
        "supporting_sources": supporting_sources,
        "total_matches": len(matches)
    }


def source_quality_scorer(url: str, content: str, source_name: str) -> Dict[str, Any]:
    """
    Scores the quality of a research source.

    Args:
        url: Source URL
        content: Source content
        source_name: Source name/title

    Returns:
        dict: Quality assessment with score
    """
    score = 0.5  # Base score
    reasons = []

    # Check for authoritative domains
    authoritative_domains = [
        '.edu', '.gov', '.org',
        'wikipedia.org', 'britannica.com',
        'nature.com', 'science.org', 'sciencedirect.com',
        'nytimes.com', 'bbc.com', 'reuters.com'
    ]

    if any(domain in url.lower() for domain in authoritative_domains):
        score += 0.2
        reasons.append("Authoritative source")

    # Content length indicates depth
    content_length = len(content)
    if content_length > 2000:
        score += 0.15
        reasons.append("Comprehensive content")
    elif content_length < 500:
        score -= 0.1
        reasons.append("Limited content")

    # Check for citations/references in content
    citation_indicators = ['retrieved', 'reference', 'citation', 'source:', 'according to']
    if any(indicator in content.lower() for indicator in citation_indicators):
        score += 0.1
        reasons.append("Contains citations")

    # Check for data/statistics
    if re.search(r'\d+%|\d+\s+(million|billion|thousand)', content):
        score += 0.05
        reasons.append("Contains statistics")

    # Normalize score to 0-1
    score = max(0.0, min(1.0, score))

    return {
        "score": score,
        "reasons": reasons,
        "content_length": content_length,
        "url": url
    }


# ============================================================================
# EXPORT ALL TOOLS
# ============================================================================

__all__ = [
    'get_google_search_tool',
    'get_fallback_search_tool',
    'count_words',
    'word_counter_tool',
    'structure_validator_tool',
    'claim_extractor_tool',
    'validate_claim_against_sources',
    'source_quality_scorer',
]
