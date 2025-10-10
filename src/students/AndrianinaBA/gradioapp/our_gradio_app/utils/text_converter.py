def case_converter(text, option=None):
    """
    Convert the entered text to a certain style
    Args :
        - text (str): the text that we want to convert to a certain style
        - option (str) : the mode that we want to convert the text to

    Returns :
        - The converted text
    """
    if option == "Uppercase":
        return text.upper()
    elif option == "Lower":
        return text.lower()
    elif option == "Title Case":
        return text.title()


def text_reverser(text, option=None):
    """
    Do we want to reverse the text or not?
    Args :
        - text (str): the text that we want to convert to a certain style
        - option (str) : the mode that we want to convert the text to

    Returns :
        - The converted text
    """
    treated = text
    if option == "Reverse Word Order":
        return " ".join(treated.split()[::-1])
    elif option == "Reverse all characters":
        return treated[::-1]
    else:
        return treated


def text_analyzer(text, selection=None):
    """
    Selection is a list of the analyzes that we want to do
    Args :
     - text (str) : the text that we want to analyze
     - selection (list) : a list of the the analyzes that we want to do

    Returns :
     - A list of word count, character count and the average word per word in the analyzed text
    """
    splitted = text.split()

    count = 0
    avg = 0
    for word in splitted:
        count += 1
        avg += len(word)

    # we will not count the blank lines
    stripped = text.strip()
    char_count = 0
    for char in stripped:
        if char == ' ':
            continue
        char_count += 1

    final_char_count = char_count
    word_count = len(splitted)
    average_char_per_word = avg / count
    return word_count, final_char_count, average_char_per_word
