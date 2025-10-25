import re

import google.generativeai as genai


def clean_word_count(text):
    """
    Count words in text after removing special characters like *, :, ., and newlines.
    Args:
        text (str): The input text to be processed.
    Returns:
        int: The count of words in the cleaned text.
    """
    # Remove special characters
    cleaned_text = re.sub(r'[\*\:\.\,\n]', ' ', text)

    # Remove extra spaces
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    # Split into words
    words = cleaned_text.split(' ')

    # Return word count
    return len([w for w in words if w])  # filters out empty strings


word_counter_tool = genai.protos.FunctionDeclaration(
    name="word_counter_tool",
    descrition="Counts the number of words in a given text after cleaning it from special characters.",
    parameters={
        "type": "object",
        "properties": {
            "text": {
                "type": "string",
                "description": "The input text to be processed for word count."
            }
        },
        "required": ["text"]
    }
)
