import gradio as gr


def process_text(text, case_option, reverse_words, reverse_chars):
    """
    This function makes some operations on the text_input

    Args:
        text (str): The input text to be processed
        case_option (str): The case conversion option {"Uppercase", "Lowercase", "Title Case"}
        reverse_words (bool): =?True, reverses the order of the words in the text.
        reverse_chars (bool): =?True, reverses the order of all characters

    Returns:
        str: The transformed text after applying the selected operations.
    """
    if not text:
        return ""

    if case_option == "Uppercase":
        text = text.upper()
    elif case_option == "Lowercase":
        text = text.lower()
    elif case_option == "Title Case":
        text = text.title()

    if reverse_words:
        text = " ".join(text.split()[::-1])
    if reverse_chars:
        text = text[::-1]

    return text


def analyze_text(text):
    """
    Analyzes a given text and returns statistics about its content.

    Args:
        text (str): The input text to analyze.

    Returns:
        str: A formatted string containing:
            - Word Count: the number of words in the text.
            - Character Count: the total number of characters in the text.
            - Average Word Length: the average length of the words, rounded to two decimals.
    """
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    avg_word_length = (sum(len(w) for w in words) / word_count) if word_count > 0 else 0
    return word_count, char_count, avg_word_length
    # f"Word Count: {word_count}\nCharacter Count: {char_count}\nAverage Word Length: {avg_word_length:.2f}"


def main_fn(text, case_option, reverse_words, reverse_chars):
    """
    Processes the input text based on selected transformations and returns both
    the processed text and its statistical analysis

    Args:
        text (str): input text
        case_option (str): case conversion option.
        reverse_words (bool): If True, reverses of words
        reverse_chars (bool): If True, reverses the all characters in the text.

    Returns:
        tuple[str, str]
            - text after transformations
            - (word count, character count, and average word length).
    """
    processed = process_text(text, case_option, reverse_words, reverse_chars)
    word, character, average = analyze_text(text)
    return processed, word, character, average


with gr.Blocks(theme=gr.themes.Ocean()) as app:  # type: ignore
    # gr.Markdown("# Textfy - A text editor app")
    gr.HTML(
        """
         <div style="
        width:100%;
        background-color:#007BFF;
        padding:20px 0;
        text-align:center;
        margin-bottom:30px;
        box-shadow:0 4px 6px rgba(0,0,0,0.1);
    ">
        <h1 style="
            color:white;
            font-size:36px;
            margin:0;
            font-family:Arial, sans-serif;
            letter-spacing:1px;
        ">
            Textfy Assistant
        </h1>
        <p style="
            color:white;
            font-size:18px;
            margin-top:8px;
            font-family:Arial, sans-serif;
        ">
            A companion that analyzes, transforms, and uncovers the hidden structure of your texts.
        </p>
    </div>
        """
    )

    with gr.Row():
        text_input = gr.Textbox(label="Enter your text here", placeholder="Type or paste text...", lines=7)

    with gr.Row():
        with gr.Column(scale=2, min_width=300):
            with gr.Tabs():
                # Tab 1: Case Converter
                with gr.Tab("Case Converter"):
                    case_option = gr.Radio(
                        ["Uppercase", "Lowercase", "Title Case"],
                        label="Choose a case conversion",
                        value="Uppercase"
                    )

                # Tab 2: Text Reverser
                with gr.Tab("Text Reverser"):
                    reverse_words = gr.Checkbox(label="Reverse Word Order", value=False)
                    reverse_chars = gr.Checkbox(label="Reverse All Characters", value=False)

        with gr.Column(scale=2, min_width=300):
            with gr.Tab("Text Statistics"):
                with gr.Row():
                    word = gr.Textbox(label="Total of words", interactive=False)
                    character = gr.Textbox(label="Total of characters", interactive=False)
                    average = gr.Textbox(label="Average", interactive=False)

    output_box = gr.Textbox(label="Processed Output", lines=5)

    process = [text_input, case_option, reverse_words, reverse_chars]

    for ctrl in process:
        ctrl.change(main_fn,
        inputs=process,
        outputs=[output_box, word, character, average]
        )
    reset_btn = gr.Button("Reset", variant='primary')

    reset_btn.click(
        fn=lambda: ("", "Uppercase", False, False, "", "", "", ""),  # reset all controls
        inputs=None,
        outputs=[text_input, case_option, reverse_words, reverse_chars, output_box, word, character, average]
    )

# app.launch(share=False)
