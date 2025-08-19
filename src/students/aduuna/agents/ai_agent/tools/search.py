import logging
import typing

from langchain_core.tools import tool
from langchain_google_community import GoogleSearchAPIWrapper

logger = logging.getLogger(__name__)


# Initialize the Google Search wrapper (uses same env vars as graph.py)
search_wrapper = GoogleSearchAPIWrapper()


@tool
def web_search_tool(query: str) -> str:
    """
    Search the web for information related to the essay topic.

    This tool is designed for the Planner agent to gather initial research
    and background information when creating an essay outline.

    Args:
        query: The search query string. Should be specific and relevant
               to the essay topic being researched.
    Returns:
        str: Search results containing relevant information, or an error message
             if search is not available.

    Example:
        >>> web_search_tool("artificial intelligence in education benefits")
        "Search results about AI in education..."
    """
    if not query or not isinstance(query, str):
        logger.error("Invalid query provided to web_search_tool")
        return "Error: Invalid or empty search query provided."

    logger.info(f"Searching web for: {query}")
    # Clean up the query and perform search
    cleaned_query = query.strip()
    results = search_wrapper.run(cleaned_query)

    if not results:
        return f"No search results found for query: {query}"

    logger.info(f"Successfully retrieved search results for: {query}")
    return results


@tool
def research_topic_tool(topic: str, num_queries: int = 3) -> typing.Dict[str, str]:
    """
    Perform comprehensive research on an essay topic using multiple search queries.
    This is a higher-level tool that combines multiple web searches to gather
    comprehensive background information for essay planning.

    Args:
        topic: The main essay topic to research
        num_queries: Number of different search angles to explore (default: 3)
    Returns:
        Dict mapping search aspects to their results
    """
    if not topic or not isinstance(topic, str):
        return {"error": "Invalid topic provided"}

    # Generate different search angles for comprehensive research
    search_angles = [
        f"{topic} overview",
        f"{topic} benefits advantages",
        f"{topic} challenges problems issues",
        f"{topic} current trends 2024 2025",
        f"{topic} future implications",
        f"{topic} case studies examples",
    ]

    # Limit to requested number of queries
    search_angles = search_angles[:num_queries]

    research_results = {}

    for angle in search_angles:
        result = web_search_tool.invoke({"query": angle})
        research_results[angle] = result
        logger.info(f"Completed research for angle: {angle}")

    return research_results


# Utility functions for easier agent integration
def search_for_topic(topic: str) -> str:
    """
    Simple utility to search for a topic.
    This is a direct function (not a tool) for internal use by agents.
    """
    return search_wrapper.run(topic.strip())


# Export search tools
SEARCH_TOOLS = [web_search_tool, research_topic_tool]
