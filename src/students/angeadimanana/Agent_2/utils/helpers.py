"""
Utility functions for the agentic report system
"""

import re


def count_words(text: str) -> int:
    """
    Count the number of words in text
    Args:
        text (str): Input text to count words
    Returns:
        int: Number of words in the text
    """
    # Remove characters that are not counted
    words = re.findall(r"\b[\w'-]+\b", text, flags=re.UNICODE)

    # Filter out empty strings
    words = [word for word in words if word.strip()]

    return len(words)


def extract_qa_score(qa_results: str) -> float:
    """
    Extract the overall QA score from the QA results
    Args:
        qa_results (str): QA evaluation text
    Returns:
        float: QA score between 0 and 1
    """
    # Look for pattern like "Overall Score: 0.85/1.0" or "Overall Score: 0.85"
    match = re.search(r'Overall Score:\s*([0-9.]+)', qa_results)
    if match:
        return float(match.group(1))
    return 0.0  # Default to 0 if score not found


def is_report_approved(qa_score: float, threshold: float = 0.999999) -> bool:
    """
    Check if the report is approved based on QA score
    Args:
        qa_score (float): QA score
        threshold (float): Minimum score for approval
    Returns:
        bool: True if approved, False otherwise
    """
    return qa_score >= threshold
