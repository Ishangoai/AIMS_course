# report_agent_app/web_search.py
"""
Web Search Integration Module - Enhanced
Provides latest information from web for report generation
"""
import logging
import os

from langchain_core.tools import tool
from langchain_google_community import GoogleSearchAPIWrapper

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize search wrapper only if credentials provided
search = None
if GOOGLE_API_KEY and GOOGLE_CSE_ID:
    try:
        search = GoogleSearchAPIWrapper(
            google_api_key=GOOGLE_API_KEY,
            google_cse_id=GOOGLE_CSE_ID,
            k=10
        )
        logger.info("✅ Google Search initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize Google Search: {e}")
        search = None
else:
    logger.info("ℹ️ Google Search disabled (no credentials provided)")


@tool
def search_google(query: str) -> str:
    """
    Search Google for latest information on topic.
    Returns top 10 results formatted for AI context.

    Args:
        query: Search query string

    Returns:
        Formatted search results or error message
    """
    if not search:
        return "⚠️ Web search not configured. Please add Google API credentials to enable this feature."

    logger.info(f"🔍 Searching Google for: {query}")

    try:
        # Perform search
        results = search.results(query, num_results=10)

        if not results:
            return "No search results found. Try a different query."

        # Format results for AI consumption
        formatted_results = []
        formatted_results.append(f"Web Search Results for: '{query}'\n")
        formatted_results.append("=" * 60 + "\n")

        for i, result in enumerate(results, 1):
            title = result.get('title', 'No title')
            snippet = result.get('snippet', 'No description')
            link = result.get('link', '')

            formatted_results.append(
                f"\n[Result {i}]\n"
                f"Title: {title}\n"
                f"Description: {snippet}\n"
                f"Source: {link}\n"
                f"{'-' * 60}"
            )

        output = "\n".join(formatted_results)
        logger.info(f"✅ Found {len(results)} search results")

        return output

    except Exception as e:
        logger.error(f"❌ Search failed: {e}")
        return f"Search error: {str(e)}. Continuing without web search."


def get_search_context(topic: str) -> str:
    """
    Get formatted search context for essay generation.
    Optimized query for MLOps and technical content.

    Args:
        topic: Essay topic

    Returns:
        Formatted search results or empty string if search fails
    """
    try:
        # Create optimized search query
        query = f"{topic} MLOps best practices 2024 2025 latest trends"
        result = search_google.invoke({"query": query})

        # If search failed, return empty string
        if "error" in result.lower() or "not configured" in result.lower():
            logger.warning("Search unavailable, continuing without web context")
            return ""

        return result

    except Exception as e:
        logger.error(f"Failed to get search context: {e}")
        return ""


# For testing
if __name__ == "__main__":
    test_topic = "CI/CD pipelines for machine learning"
    print("Testing web search...")
    result = get_search_context(test_topic)
    print(result)
