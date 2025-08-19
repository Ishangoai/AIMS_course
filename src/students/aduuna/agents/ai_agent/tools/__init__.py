from .counter import (
    COUNTER_TOOLS,
    count_essay_words,
    count_words_in_sections,
    word_counter_tool,
)
from .search import (
    SEARCH_TOOLS,
    research_topic_tool,
    search_for_topic,
    web_search_tool,
)

# Combine all tools for easy access
ESSAY_TOOLS = COUNTER_TOOLS + SEARCH_TOOLS

# Export all tools and utilities
__all__ = [
    # Tools
    "word_counter_tool",
    "count_words_in_sections",
    "web_search_tool",
    "research_topic_tool",
    # Tool collections
    "COUNTER_TOOLS",
    "SEARCH_TOOLS",
    "ESSAY_TOOLS",
    # Utility functions
    "count_essay_words",
    "search_for_topic",
]


def get_available_tools():
    """
    Return a list of available tools with their availability status.
    """
    tools_status = {
        "word_counter_tool": True,
        "count_words_in_sections": True,
        "web_search_tool": True,
        "research_topic_tool": True,
    }

    return tools_status
