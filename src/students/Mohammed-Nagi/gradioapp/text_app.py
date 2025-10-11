import gradio as gr


def case_converter(text, case_option):
    """
    converts text to specified case format.

    Args:
        text (str): The input text to be converted.
        case_option (str): The case conversion option:
            - "Uppercase": Convert all characters to uppercase
            - "Lowercase": Convert all characters to lowercase
            - "Titlecase": Convert to title case (first letter of each word capitalized)
            - "Squigglecase": Alternate between uppercase and lowercase characters

    Returns:
        str: The converted text, or empty string if input is empty.
    """
    if not text:
        return ""

    if case_option == "Uppercase":
        text = text.upper()
    elif case_option == "Lowercase":
        text = text.lower()
    elif case_option == "Titlecase":
        text = text.title()
    elif case_option == "Squigglecase":
        text = "".join(
            c.upper() if i % 2 == 0 else c.lower()
            for i, c in enumerate(text)
        )

    return text


def text_reverser(text, reverse_word, reverse_char):
    """
    reverse text based on word order and/or character order.

    Args:
        text (str): The input text to be reversed.
        reverse_word (bool): If True, reverse the order of words.
        reverse_char (bool): If True, reverse the order of characters.

    Returns:
        str: The reversed text based on the specified options, or empty
            string if input is empty.
    """
    if not text:
        return ""

    if reverse_word and not reverse_char:
        words = text.split()
        text = " ".join(reversed(words))
    elif reverse_char and not reverse_word:
        text = text[::-1]
    elif reverse_word and reverse_char:
        words = text.split()
        text = " ".join(reversed(words))[::-1]

    return text


def text_analyser(text):
    """
    analyse text and calculate statistics.

    Args:
        text (str): The input text to analyze.

    Returns:
        list: A list containing three values:
            - word_count (int): Number of words in the text
            - char_count (int): Number of characters excluding spaces
            - avg_len (float): Average word length rounded to 2 decimal places
        Returns [0, 0, 0] if input is empty.
    """
    if not text:
        return [0, 0, 0]

    char_count = len(text.replace(" ", ""))
    words = text.split()
    word_count = len(words)
    avg_len = round(sum(len(w) for w in words) / word_count, 2)

    return word_count, char_count, avg_len


def pyramid_text(text):
    """
    format text into a pyramid shape with increasing word counts per line.

    The pyramid starts with 1 word on the first line, then 3 words on the
    second line, 5 words on the third line, and so on (incrementing by 2).
    Each line is centered based on the longest line.

    Args:
        text (str): The input text to format as a pyramid.

    Returns:
        str: The pyramid-formatted text with each line centered and separated
            by newlines, or empty string if input is empty.
    """
    if not text:
        return ""

    words = text.split()
    pyramid_lines = []
    count = 1   # start with 1 word on the first line
    index = 0

    while index < len(words):
        line_words = words[index : index + count]
        line = " ".join(line_words)
        pyramid_lines.append(line)
        index += count
        count += 2

    # centeralise the text
    max_length = max(len(line) for line in pyramid_lines)
    centered_lines = [line.center(max_length) for line in pyramid_lines]

    return "\n".join(centered_lines)


def process_and_analyse(text, case_option, reverse_word, reverse_char):
    """
    applies case conversion and text reversal transformations
    sequentially, then analyzes the resulting text.

    Args:
        text (str): The input text to process.
        case_option (str): The case conversion option to apply.
        reverse_word (bool): If True, reverse word order.
        reverse_char (bool): If True, reverse character order.

    Returns:
        tuple: A tuple containing:
            - processed_text (str): The transformed text
            - word_count (int): Number of words
            - char_count (int): Number of characters
            - avg_len (float): Average word length
    """
    text = case_converter(text, case_option)
    text = text_reverser(text, reverse_word, reverse_char)
    words, chars, length = text_analyser(text)

    return text, words, chars, length


def clear_all():
    """
    reset all text fields and analysis counters to their default values.

    Returns:
        list: A list of default values for clearing the interface:
            - Empty string for input text
            - Empty string for output text
            - 0 for word count
            - 0 for character count
            - 0 for average word length
    """
    return ["", "", 0, 0, 0]


# Text Editor App
with gr.Blocks(theme=gr.Theme()) as text_app:
    gr.Markdown("# Text Editor App")

    # Input Box
    input_text = gr.Textbox(label="Enter Your Text", lines=5, placeholder="type or paste text here...")

    # Control Tabs
    with gr.Tabs():
        with gr.Tab("Case Converter"):
            case_option = gr.Radio(
                ["Uppercase", "Lowercase", "Titlecase", "Squigglecase"],
                label="Choose Case",
                value="Uppercase")

        with gr.Tab("Text Reverser"):
            reverse_word = gr.Checkbox(label="Reverse Word Order")
            reverse_char = gr.Checkbox(label="Reverse Character Order")

        with gr.Tab("Visual Effects"):
            pyramid_button = gr.Button("Generate Pyramid Text")

    # Analysis Display
    with gr.Group():
        gr.Markdown("**Text Analysis**")

        with gr.Row():
            word_count = gr.Label(label="Word Count", value=0)
            char_count = gr.Label(label="Character Count", value=0)
            avg_len = gr.Label(label="Average Word Length", value=0)

    # Dynamic Analysis
    input_text.input(fn=text_analyser, inputs=input_text, outputs=[word_count, char_count, avg_len])

    # Output Box
    output_text = gr.Textbox(label="Output Text", lines=5, interactive=False,)

    # Pyramid Visual
    pyramid_button.click(fn=pyramid_text, inputs=input_text, outputs=output_text)

    # Run Button
    run_button = gr.Button("Apply Changes")

    run_button.click(
        fn=process_and_analyse,
        inputs=[input_text, case_option, reverse_word, reverse_char],
        outputs=[output_text, word_count, char_count, avg_len]
    )

    # Clear Button
    clear_button = gr.Button("Clear")

    clear_button.click(
        fn=clear_all,
        inputs=None,
        outputs=[input_text, output_text, word_count, char_count, avg_len]
    )
