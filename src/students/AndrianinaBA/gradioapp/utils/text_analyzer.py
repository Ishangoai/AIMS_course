def case_converter(text, option=None):
    """
    Convert the entered text to a certain style
    Args :
        - text (str): the text that we want to convert to a certain style
        - option (str) : the mode that we want to convert the text to
        those choices are : "Upper Case", "Lower Case", "Title Case"

    Returns :
        - The converted text with the specified option
    """
    if option == "Upper Case":
        return text.upper()
    elif option == "Lower Case":
        return text.lower()
    elif option == "Title Case":
        return text.title()


def text_reverser_word(text, reverse=False) -> str:
    """
    Do we want to reverse the text or not?
    Args :
        - text (str): the text that we want to convert to a certain style
        - option (Bool) : the mode that we want to convert the text to

    Returns :
        - The converted text with the specified option
    """
    treated = text
    if reverse:
        return " ".join(treated.split()[::-1])
    else:
        return treated


def text_reverser_all_character(text, reverse=False) -> str:
    """"
    Do we want to reverse all the charachters in the text or not?
    Args :
        - text (str): the text that we want to convert to a certain style
        - option (Bool) : the mode that we want to convert the text to
    Returns :
        - The converted text with the specified option
    """
    treated = text
    if reverse:
        return treated[::-1]
    else:
        return treated


def text_analyzer(text):
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
    if count == 0:
        count = 1  # to avoid division by zero error

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


if __name__ == "__main__":
    import gradio as gr

    with gr.Blocks() as app:
        gr.Markdown("## Text analyzer")
        text = gr.Textbox(placeholder="Enter the text you want to analyze here : ", label="Your text here")
        with gr.Tabs():
            with gr.Tab(label="Case converter"):
                pass

            with gr.Tab(label="Text reverser"):
                pass

            with gr.Tab(label="Text analysis"):
                pass

    app.launch()
