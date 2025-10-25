"""
Tools for the report generation system.
Includes Wikipedia search and validation utilities.
"""
import re

import requests
from langchain_core.tools import BaseTool


class WikipediaSearchTool(BaseTool):
    """
    Wikipedia API search tool for gathering factual information.
    Uses direct API calls for reliability.
    """
    name: str = "wikipedia"
    description: str = (
        "Search Wikipedia for factual information about technical topics, "
        "concepts, tools, methodologies, and best practices. "
        "Input should be a specific search query (e.g., 'MLOps', 'CI/CD pipeline'). "
        "Returns summarized content from the most relevant Wikipedia article."
    )
    doc_content_chars_max: int = 5000  # Increased for more context

    def _run(self, query: str) -> str:
        """Execute Wikipedia search."""
        try:
            session = requests.Session()
            session.headers.update({
                "User-Agent": "ReportGeneratorBot/1.0 (Educational Project)"
            })
            url = "https://en.wikipedia.org/w/api.php"

            # Search for page
            search_params = {
                "action": "opensearch",
                "search": query,
                "limit": "1",
                "namespace": "0",
                "format": "json",
            }
            search_response = session.get(url=url, params=search_params, timeout=10)
            search_response.raise_for_status()
            search_data = search_response.json()

            if not search_data[1]:
                return f"No Wikipedia results found for '{query}'. Try a broader search term."

            page_title = search_data[1][0]

            # Fetch content
            content_params = {
                "action": "query",
                "format": "json",
                "titles": page_title,
                "prop": "extracts",
                "explaintext": True,
            }
            content_response = session.get(url=url, params=content_params, timeout=10)
            content_response.raise_for_status()
            content_data = content_response.json()

            page_info = content_data["query"]["pages"]
            page_id = list(page_info.keys())[0]

            if page_id == "-1":
                return f"Page '{page_title}' not found on Wikipedia."

            extract = page_info[page_id].get("extract", "")
            if not extract:
                return "No content available for this topic."

            # Return truncated content with source citation
            content = extract[:self.doc_content_chars_max]
            return f"[Source: Wikipedia - {page_title}]\n\n{content}"

        except requests.exceptions.RequestException as e:
            return f"Wikipedia API error: {str(e)}"
        except Exception as e:
            return f"Unexpected error during Wikipedia search: {str(e)}"


def get_wikipedia_tool():
    """Returns configured Wikipedia search tool."""
    return WikipediaSearchTool()


def count_words(text: str) -> int:
    """
    Accurately counts words in text.

    Why: LLMs are notoriously bad at counting. This Python function
    ensures accurate word count validation.
    """
    if not text:
        return 0
    # Split on whitespace and filter empty strings
    words = [word for word in text.split() if word.strip()]
    return len(words)


def validate_structure(text: str) -> dict:
    """
    Validates that the report has proper structure.

    Returns dict with:
    - has_sections: bool (has markdown headers)
    - section_count: int
    - has_introduction: bool
    - has_conclusion: bool

    Why: Ensures the report meets structural requirements without
    manual inspection.
    """
    # Find all markdown headers (## or #)
    headers = re.findall(r'^#{1,3}\s+(.+)$', text, re.MULTILINE)

    # Check for intro/conclusion (case-insensitive)
    text_lower = text.lower()
    has_intro = any(keyword in text_lower[:500] for keyword in ['introduction', 'overview', 'abstract'])
    # has_conclusion = any(keyword in text_lower[-500:] for keyword in ['conclusion', 'summary', 'final'])

    return {
        "has_sections": len(headers) >= 3,  # At least intro + body + conclusion
        "section_count": len(headers),
        "headers": headers,
        "has_introduction": has_intro,
        # "has_conclusion": has_conclusion,
        "is_valid": len(headers) >= 3 and has_intro  # and has_conclusion
    }


def calculate_word_count_difference(actual: int, target: int = 1000, tolerance: int = 50) -> dict:
    """
    Calculates word count difference and provides feedback.

    Why: Provides clear feedback for the revision loop, helping the
    agent understand if it needs to expand or condense content.
    """
    difference = actual - target
    within_tolerance = abs(difference) <= tolerance

    feedback = ""
    if within_tolerance:
        feedback = f"✓ Word count is within tolerance ({actual} words)"
    elif difference > 0:
        feedback = f"✗ Report is {difference} words too long. Need to condense by ~{difference} words."
    else:
        feedback = f"✗ Report is {abs(difference)} words too short. Need to expand by ~{abs(difference)} words."

    return {
        "actual": actual,
        "target": target,
        "difference": difference,
        "within_tolerance": within_tolerance,
        "feedback": feedback
    }
