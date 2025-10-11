import gradio as gr
import pandas as pd
import numpy as np

# Fixed the text_configs initialization and removed the unnecessary State wrapper
text_configs = {
    "text_case": None,
    "reverse_word_order": False,
    "reverse_char_order": False
}

def to_uppercase(text):
    return text.upper()

def to_lowercase(text):
    return text.lower()

def to_titlecase(text):
    return text.title()

def get_word_tokens(text):
    return text.split()

def reverse_word(text):
    tokens = get_word_tokens(text)
    reversed_tokens = tokens[::-1]
    return " ".join(reversed_tokens)

def reverse_characters(text):
    tokens = get_word_tokens(text)
    reversed_words = []
   
    for token in tokens:
        reversed_chars = token[::-1]
        reversed_words.append(reversed_chars)
   
    return " ".join(reversed_words)

def word_count(text):
    tokens = get_word_tokens(text)
    return len(tokens)

def character_count(text):
    return len(text.replace(" ", ""))  # Excluding spaces for character count

def average_word_length(text):
    tokens = get_word_tokens(text)
    if not tokens:
        return 0
    total_chars = sum(len(word) for word in tokens)
    return round(total_chars / len(tokens), 2)

def generate_table(word_count, char_count, avg_word_length):
    data = pd.DataFrame({
        "Metric": ["Word Count", "Character Count", "Average Word Length"],
        "Value": [word_count, char_count, avg_word_length]
    })
    return data

def selected_textcase(option):
    """Update text case configuration"""
    if option == "Uppercase":
        text_configs["text_case"] = "upper"
    elif option == "Titlecase":
        text_configs["text_case"] = "title"
    elif option == "Lowercase":
        text_configs["text_case"] = "lower"
    else:
        text_configs["text_case"] = None

def set_reverse_word(is_checked):
    text_configs["reverse_word_order"] = is_checked

def set_reverse_char(is_checked):
    text_configs["reverse_char_order"] = is_checked

def text_actions(text):
    """Main function to process text based on selected operations"""
    if not text:
        return "", generate_table(0, 0, 0)
   
    processed_text = text
    text_case = text_configs["text_case"]
    reverse_word_order = text_configs["reverse_word_order"]
    reverse_char_order = text_configs["reverse_char_order"]
   
    # Apply text case transformations
    if text_case == "upper":
        processed_text = to_uppercase(processed_text)
    elif text_case == "lower":
        processed_text = to_lowercase(processed_text)
    elif text_case == "title":
        processed_text = to_titlecase(processed_text)
   
    # Apply reverse operations
    if reverse_word_order:
        processed_text = reverse_word(processed_text)
   
    if reverse_char_order:
        processed_text = reverse_characters(processed_text)
   
    # Calculate text statistics
    no_of_words = word_count(text)
    no_of_characters = character_count(text)
    avg_word_length = average_word_length(text)
   
    stats_table = generate_table(no_of_words, no_of_characters, avg_word_length)
   
    return processed_text, stats_table

def update_statistics(text):
    """Update statistics in real-time as user types"""
    if not text:
        return generate_table(0, 0, 0)
   
    wc = word_count(text)
    cc = character_count(text)
    awl = average_word_length(text)
   
    return generate_table(wc, cc, awl)


def clear_actions():
    """Clear all selected actions"""
    text_configs["text_case"] = None
    text_configs["reverse_word_order"] = False
    text_configs["reverse_char_order"] = False
    return gr.Radio(value=None), gr.Checkbox(value=False), gr.Checkbox(value=False)

def clear_all():
    """Clear both text and actions"""
def clear_actions():
    """Clear all selected actions"""
    text_configs["text_case"] = None
    text_configs["reverse_word_order"] = False
    text_configs["reverse_char_order"] = False
    return gr.Radio(value=None), gr.Checkbox(value=False), gr.Checkbox(value=False)

def clear_all():
    """Clear both text and actions"""
    clear_actions()
    return "", generate_table(0, 0, 0), None, False, False

# Create the Gradio interface
with gr.Blocks(css="""
    #spaced-row {margin:30px 0px;}
    .stat-box {border: 1px solid #e2e8f0; border-radius: 8px; padding: 15px;}
""") as dard:
    gr.Markdown("# Text Processing Tool")
   
    with gr.Row():
        # --- Sidebar ---
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### Text Actions")
           
            with gr.Accordion("Case Converter", open=False):
                case_choice = gr.Radio(
                    ["Uppercase", "Lowercase", "Titlecase"],
                    label="Select Case",
                    info="Choose text case transformation"
                )
           
            with gr.Accordion("Reverse Operations", open=False):
                reverse_word_checkbox = gr.Checkbox(
                    label="Reverse Word Order",
                    info="Reverse the order of words"
                )
                reverse_char_checkbox = gr.Checkbox(
                    label="Reverse All Characters",
                    info="Reverse characters in each word"
                )
           
            with gr.Row():
                clear_actions_btn = gr.Button("Clear Actions", variant="secondary")
                clear_all_btn = gr.Button("Clear All", variant="stop")

        # --- Main Content Area ---
        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    user_input = gr.TextArea(
                        label="Input Text",
                        placeholder="Enter or paste your text here...",
                        lines=5
                    )
                   
                    with gr.Row():
                        send_btn = gr.Button("Process Text", variant="primary")
                        clear_text_btn = gr.Button("Clear Text", variant="secondary")

                with gr.Column():
                    output = gr.TextArea(
                        label="Processed Output",
                        lines=5,
                        interactive=False
                    )

            with gr.Row(elem_id="spaced-row"):
                with gr.Column():    
                    stats_display = gr.Dataframe(
                        label='Text Statistics',
                        headers=["Metric", "Value"],
                        datatype=["str", "number"],
                        row_count=3,
                        col_count=2,
                        value=generate_table(0, 0, 0)
                    )

    # --- Event Handlers ---
   
    # Update statistics in real-time as user types
    user_input.change(
        fn=update_statistics,
        inputs=user_input,
        outputs=stats_display
    )
   
    # Update configuration when options change
    case_choice.change(
        fn=selected_textcase,
        inputs=case_choice
    )
   
    reverse_word_checkbox.change(
        fn=set_reverse_word,
        inputs=reverse_word_checkbox
    )
   
    reverse_char_checkbox.change(
        fn=set_reverse_char,
        inputs=reverse_char_checkbox
    )
   
    # Process text when send button is clicked
    send_btn.click(
        fn=text_actions,
        inputs=user_input,
        outputs=[output, stats_display]
    )
   
    # Clear text only
    clear_text_btn.click(
        fn=lambda: ("", generate_table(0, 0, 0)),
        outputs=[user_input, stats_display]
    )
   
    # Clear actions only
    clear_actions_btn.click(
        fn=clear_actions,
        outputs=[case_choice, reverse_word_checkbox, reverse_char_checkbox]
    )
   
    # Clear everything
    clear_all_btn.click(
        fn=clear_all,
        outputs=[user_input, stats_display, case_choice, reverse_word_checkbox, reverse_char_checkbox]
    )

# Launch the app
if __name__ == "__main__":
    dard.launch()