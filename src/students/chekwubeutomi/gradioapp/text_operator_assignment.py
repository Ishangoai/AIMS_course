import gradio as gr

# ---- Functions for text operations ----


def case_converter(text, case_option):
    if not text:
        return ""
    if case_option == "Uppercase":
        return text.upper()
    elif case_option == "Lowercase":
        return text.lower()
    elif case_option == "Title Case":
        return text.title()
    return text


def text_reverser(text, reverse_words, reverse_chars):
    if not text:
        return ""
    result = text
    if reverse_words:
        result = " ".join(result.split()[::-1])
    if reverse_chars:
        result = result[::-1]
    return result


def text_analyzer(text):
    if not text.strip():
        return "Word Count: 0 | Character Count: 0 | Average Word Length: 0"
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    avg_word_len = sum(len(w) for w in words) / word_count
    return f"Word Count: {word_count} | Character Count: {char_count} | Average Word Length: {avg_word_len:.2f}"


# ---- Main function that integrates all sections ----

def process_text(text, case_option, reverse_words, reverse_chars, analyze):
    """
    Combines all transformations into one output box.
    """
    if not text:
        return ""

    result = text

    # Apply case conversion
    if case_option:
        result = case_converter(result, case_option)

    # Apply reversals
    if reverse_words or reverse_chars:
        result = text_reverser(result, reverse_words, reverse_chars)

    # Perform analysis (append info)
    if analyze:
        analysis = text_analyzer(result)
        result += f"\n\n🔍 {analysis}"

    return result


# ---- Gradio Interface ----

with gr.Blocks() as app:
    gr.Markdown("# Text Operations App")
    gr.Markdown("Type or paste text, choose your operations, and see the result below.")

    # Text input at the top
    text_input = gr.Textbox(label="Enter your text here", lines=4, placeholder="Type or paste text...")

    # Accordions for operation sections
    with gr.Accordion("Case Converter", open=True):
        case_option = gr.Radio(
            ["Uppercase", "Lowercase", "Title Case"],
            label="Select case conversion",
            value=None
        )

    with gr.Accordion("Text Reverser", open=False):
        reverse_words = gr.Checkbox(label="Reverse Word Order", value=False)
        reverse_chars = gr.Checkbox(label="Reverse All Characters", value=False)

    with gr.Accordion("Text Analyzer", open=False):
        analyze = gr.Checkbox(label="Show text analysis", value=False)

    # Output textbox
    output_box = gr.Textbox(label="Result", lines=6)

    # Button to apply operations
    apply_btn = gr.Button("Apply Operations ")

    # Connect button click to processing function
    apply_btn.click(
        fn=process_text,
        inputs=[text_input, case_option, reverse_words, reverse_chars, analyze],
        outputs=output_box
    )
