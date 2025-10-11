import gradio as gr


def case_converter(text: str, type: str) -> str:
    if type == "Uppercase":
        return text.upper()
    elif type == "Lowercase":
        return text.lower()
    elif type == "Title Case":
        return text.title()
    else:
        return text


def text_reverser(text: str, type: str) -> str:
    if type == "Word-wise":
        return ' '.join(text.split()[::-1])
    else:  # Character-wise
        return text[::-1]


def text_analyzer(text: str) -> str:
    num_chars = len(text)
    num_words = len(text.split())
    num_sentences = text.count('.') + text.count('!') + text.count('?')

    # spaces are not counted as characters in words
    num_char_in_words = num_chars - text.count(' ')
    average_word_length = 0
    if num_words > 0:
        average_word_length = num_char_in_words / num_words

    return f"Characters: {num_chars}, Words: {num_words}, Sentences: {num_sentences}, Average Word Length: {average_word_length:.2f}" 

    