import langchain.tools as tools


@tools.tool("count_words")
def count_words(text: str) -> int:
    """Count the number of words in a given text."""
    return len(text.split()) if text else 0
