import random
from collections import Counter
from typing import Tuple, List
import gradio as gr
import re


def case_converter(text: str, case_type: str) -> str:
    """
    Convert input text to the specified case type.

    Args:
        text (str): The input text to be converted.
        case_type (str): The type of case conversion to apply.
            Options: "Uppercase", "Lowercase", "Title Case", "Alternate Case"

    Returns:
        str: The converted text.
    """
    if case_type == "Uppercase":
        return text.upper()
    elif case_type == "Lowercase":
        return text.lower()
    elif case_type == "Title Case":
        return text.title()
    elif case_type == "Alternate Case":
        return ''.join(
            c.upper() if i % 2 == 0 else c.lower()
            for i, c in enumerate(text)
        )
    return text


def text_reverser(text: str, reverse_type: str) -> str:
    """
    Reverse text based on the specified reversal type.

    Args:
        text (str): The input text to be reversed.
        reverse_type (str): The type of reversal to apply.
            Options: "Word-wise", "Character-wise", "Scrambled"

    Returns:
        str: The reversed text.
    """
    if reverse_type == "Word-wise":
        return ' '.join(text.split()[::-1])
    elif reverse_type == "Character-wise":
        return text[::-1]
    elif reverse_type == "Scrambled":
        text_list = list(text)
        random.shuffle(text_list)
        return ''.join(text_list)
    return text


def text_cleanup(text: str) -> str:
    """
    Clean up text by normalizing whitespace and capitalizing sentences.

    Args:
        text (str): The input text to be cleaned.

    Returns:
        str: The cleaned text with normalized whitespace and proper
            capitalization.
    """
    words = text.split()
    text = ' '.join(words)
    text = text.strip()
    
    if not text:
        return text
    
    result = []
    new_sentence = True
    
    for char in text:
        if new_sentence and char.isalpha():
            result.append(char.upper())
            new_sentence = False
        else:
            result.append(char)
            if char in '.!?':
                new_sentence = True
    
    return ''.join(result)


def word_frequency_analyzer(text: str) -> str:
    """
    Analyze word frequency in the text and return formatted results.

    Args:
        text (str): The input text to analyze.

    Returns:
        str: A formatted string showing word frequencies in descending order.
    """
    if not text.strip():
        return "No text to analyze."
    
    words = []
    current_word = []
    
    for char in text.lower():
        if char.isalnum():
            current_word.append(char)
        elif current_word:
            words.append(''.join(current_word))
            current_word = []
    
    if current_word:
        words.append(''.join(current_word))
    
    if not words:
        return "No words found."
    
    word_counts = Counter(words)
    
    result = []
    for word, count in word_counts.most_common():
        result.append(f"{word}: {count}")
    
    return '\n'.join(result)


def top_words_analyzer(text: str, top_n: int = 5) -> str:
    """
    Get the top N most frequent words in the text.

    Args:
        text (str): The input text to analyze.
        top_n (int): Number of top words to return. Default is 5.

    Returns:
        str: A formatted string showing the top N words with their frequencies.
    """
    if not text.strip():
        return "No text to analyze."
    
    # Convert to lowercase and split into words
    words = []
    current_word = []
    
    for char in text.lower():
        if char.isalnum():
            current_word.append(char)
        elif current_word:
            words.append(''.join(current_word))
            current_word = []
    
    if current_word:
        words.append(''.join(current_word))
    
    if not words:
        return "No words found."
    
    # Count word frequencies
    word_counts = Counter(words)
    
    # Get top N words
    top_words = word_counts.most_common(top_n)
    
    # Format output
    result = []
    for i, (word, count) in enumerate(top_words, 1):
        result.append(f"{i}. {word}: {count}")
    
    return '\n'.join(result)


def unique_words_count(text: str) -> int:
    """
    Count the number of unique words in the text.

    Args:
        text (str): The input text to analyze.

    Returns:
        int: The number of unique words.
    """
    if not text.strip():
        return 0
    
    words = []
    current_word = []
    
    for char in text.lower():
        if char.isalnum():
            current_word.append(char)
        elif current_word:
            words.append(''.join(current_word))
            current_word = []
    
    if current_word:
        words.append(''.join(current_word))
    
    return len(set(words))


def remove_punctuation(text: str) -> str:
    """
    Remove all punctuation from the text.

    Args:
        text (str): The input text.

    Returns:
        str: Text with all punctuation removed.
    """
    result = []
    for char in text:
        if char.isalnum() or char.isspace():
            result.append(char)
    return ''.join(result)


def extract_emails(text: str) -> str:
    """
    Extract all email addresses from the text using regex.

    Args:
        text (str): The input text.

    Returns:
        str: A formatted list of email addresses found, or a message
            if none found.
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)
    
    if emails:
        return '\n'.join(emails)
    return "No email addresses found."


def text_analyzer(text: str) -> Tuple[int, int, int, float]:
    """
    Analyze text and return various statistics.

    Args:
        text (str): The input text to analyze.

    Returns:
        Tuple[int, int, int, float]: A tuple containing:
            - Number of characters (excluding spaces)
            - Number of words
            - Number of sentences
            - Average word length
    """
    num_chars = len(text.replace(" ", ""))
    num_words = len(text.split())
    
    num_sentences = 0
    for char in text:
        if char in '.!?':
            num_sentences += 1
    
    avg_len = num_chars / num_words if num_words > 0 else 0
    return num_chars, num_words, num_sentences, round(avg_len, 2)


def clear_all() -> List:
    """
    Clear all input and output fields.

    Returns:
        List: A list of empty/zero values for all fields.
    """
    return ["", "", 0, 0, 0, 0]