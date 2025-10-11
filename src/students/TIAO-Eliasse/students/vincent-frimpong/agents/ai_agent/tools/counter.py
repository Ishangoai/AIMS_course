import json
import logging
import re
import typing

from langchain_core.tools import tool

logger = logging.getLogger(__name__)


@tool
def word_counter_tool(text: str) -> int:
    """
    Count the number of words in a given text block.
    This tool splits text into words and returns an accurate count,
    handling punctuation and whitespace appropriately.

    Args:
        text: The text to count words in. Can be a single string or
              combined text from multiple sections.
    Returns:
        int: The total number of words in the text.
    Example:
        >>> word_counter_tool("Hello world! This is a test.")
        6
    """
    if not text or not isinstance(text, str):
        logger.warning("Invalid input to word_counter_tool: empty or non-string text")
        return 0

    # Remove extra whitespace and split into words
    # This regex splits on whitespace and handles punctuation appropriately
    words = re.findall(r"\b\w+\b", text.lower())
    word_count = len(words)

    logger.info(f"Counted {word_count} words in text of {len(text)} characters")
    return word_count


@tool
def count_words_in_sections(draft_sections: str) -> typing.Dict[str, typing.Union[int, typing.Dict[str, int]]]:
    """
    Count words in each section of the draft and return detailed statistics.

    This tool is specifically designed for the essay state's draft_sections
    and provides both individual section counts and total count.

    Args:
        draft_sections: JSON string representation of dictionary mapping
                       section titles to their content
    Returns:
        Dict containing:
        - 'total_words': Total word count across all sections
        - 'section_counts': Dictionary mapping section titles to their word counts
        - 'sections_completed': Number of sections that have content
    Example:
        >>> import json
        >>> sections = {"intro": "Hello world", "body": "This is longer content"}
        >>> count_words_in_sections(json.dumps(sections))
        {'total_words': 6, 'section_counts': {'intro': 2, 'body': 4}, 'sections_completed': 2}
    """

    try:
        # Parse the JSON string into a dictionary
        sections_dict = json.loads(draft_sections)
    except (json.JSONDecodeError, TypeError):
        logger.warning("Invalid input to count_words_in_sections: not valid JSON")
        return {"total_words": 0, "section_counts": {}, "sections_completed": 0}

    if not isinstance(sections_dict, dict):
        logger.warning("Invalid input to count_words_in_sections: not a dictionary")
        return {"total_words": 0, "section_counts": {}, "sections_completed": 0}

    section_counts = {}
    total_words = 0

    for section_title, content in sections_dict.items():
        if content and isinstance(content, str):
            word_count = word_counter_tool.invoke({"text": content})
            section_counts[section_title] = word_count
            total_words += word_count
        else:
            section_counts[section_title] = 0

    sections_completed = sum(1 for count in section_counts.values() if count > 0)

    result = {"total_words": total_words, "section_counts": section_counts, "sections_completed": sections_completed}

    logger.info(f"Word count analysis: {sections_completed} sections, {total_words} total words")
    return result


# Utility functions for easier agent integration
def count_essay_words(draft_sections_dict: typing.Dict[str, str]) -> int:
    """
    Simple utility to count total words in essay sections.
    This is a direct function (not a tool) for internal use by agents.
    """
    total_words = 0
    for content in draft_sections_dict.values():
        if content and isinstance(content, str):
            words = re.findall(r"\b\w+\b", content.lower())
            total_words += len(words)
    return total_words


# Export counter tools
COUNTER_TOOLS = [word_counter_tool, count_words_in_sections]
