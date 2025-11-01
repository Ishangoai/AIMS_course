"""Word counting tool for ensuring report length requirements."""


def count_words(text: str) -> dict:
    """
    Count words in text and check if within target range.

    Args:
        text: The text to count words in

    Returns:
        Dictionary with word count and status
    """
    # Remove markdown headers and extra whitespace
    words = text.split()
    word_count = len(words)

    target = 1000
    min_words = 950
    max_words = 1050

    if min_words <= word_count <= max_words:
        status = "PASS"
        message = f"Word count {word_count} is within target range ({min_words}-{max_words})"
    elif word_count < min_words:
        status = "TOO_SHORT"
        deficit = min_words - word_count
        message = f"Word count {word_count} is {deficit} words short of minimum ({min_words})"
    else:
        status = "TOO_LONG"
        excess = word_count - max_words
        message = f"Word count {word_count} is {excess} words over maximum ({max_words})"

    return {
        "word_count": word_count,
        "target": target,
        "min": min_words,
        "max": max_words,
        "status": status,
        "message": message,
        "within_range": status == "PASS",
    }


def get_word_count(text: str) -> int:
    """Simple word count function."""
    return len(text.split())
