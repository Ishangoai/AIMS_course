
import gradio as gr
from gradioapp.utils.text_operations import correct_text
def update_corrected_text(text):
    """Update the corrected text box"""
    return correct_text(text)


def update_statistics(text):
    """Update text statistics dynamically"""
    word_count, char_count, avg_word_length = analyze_text(text)
    return (
        f"**Word Count:** {word_count}",
        f"**Character Count:** {char_count}",
        f"**Average Word Length:** {avg_word_length}"
    )


def update_output(original_text, corrected_text, case_type, reverse_words, reverse_chars):
    """Update the final output based on all operations"""
    result = process_text(original_text, corrected_text, case_type, reverse_words, reverse_chars)
    
    # Update statistics based on corrected text
    word_count, char_count, avg_word_length = analyze_text(corrected_text)
    stats = (
        f"**Word Count:** {word_count}\n\n"
        f"**Character Count:** {char_count}\n\n"
        f"**Average Word Length:** {avg_word_length}"
    )
    
    return result, stats


def clear_all():
    """Clear all fields"""
    return "", "", "", "Uppercase", False, False, ""


# Create the Gradio interface
with gr.Blocks(css="body {background: #f2f7ff;}") as text_app:
    
    gr.Markdown("# 📝 Text Operations & Analysis Tool")
    gr.Markdown("Enter your text below and apply various transformations and corrections.")
    
    # ========== STEP 1: TEXT INPUT AND CORRECTION ==========
    gr.Markdown("## Step 1: Input & Text Correction")
    
    with gr.Row():
        # Column 1: Original Text Input with formatting options
        with gr.Column():
            gr.Markdown("### Original Text Input")
            
            # Text formatting options (similar to Word)
            with gr.Row():
                font_size = gr.Dropdown(
                    choices=["Small", "Normal", "Large", "Extra Large"],
                    value="Normal",
                    label="Font Size",
                    scale=1
                )
                text_style = gr.Dropdown(
                    choices=["Body", "Title", "Subtitle", "Heading"],
                    value="Body",
                    label="Text Style",
                    scale=1
                )
            
            original_input = gr.Textbox(
                label="Type or paste your text here",
                placeholder="Enter your text...",
                lines=8,
                max_lines=15
            )
        
        # Column 2: Corrected Text Output
        with gr.Column():
            gr.Markdown("### Corrected Text")
            gr.Markdown("*Auto-corrected version used for operations below*")
            
            corrected_output = gr.Textbox(
                label="Corrected Text",
                placeholder="Corrected text will appear here...",
                lines=8,
                max_lines=15,
                interactive=True  # Allow manual editing if needed
            )
            
            correct_btn = gr.Button("🔄 Correct Text", variant="primary")
    
    gr.Markdown("---")
    
    # ========== SECTIONS 1, 2, 3: OPERATIONS ==========
    gr.Markdown("## Text Operations")
    
    with gr.Tabs():
        # Section 1: Case Converter
        with gr.Tab("🔤 Case Converter"):
            gr.Markdown("### Convert text case")
            case_type = gr.Radio(
                choices=["Uppercase", "Lowercase", "Title Case"],
                value="Uppercase",
                label="Select Case Type"
            )
        
        # Section 2: Text Reverser
        with gr.Tab("🔄 Text Reverser"):
            gr.Markdown("### Reverse text order")
            reverse_word_order = gr.Checkbox(
                label="Reverse Word Order",
                value=False
            )
            reverse_all_chars = gr.Checkbox(
                label="Reverse All Characters",
                value=False
            )
        
        # Section 3: Text Analyzer
        with gr.Tab("📊 Text Analyzer"):
            gr.Markdown("### Text Statistics (based on corrected text)")
            with gr.Column():
                stats_display = gr.Markdown("**Word Count:** 0\n\n**Character Count:** 0\n\n**Average Word Length:** 0.0")
    
    gr.Markdown("---")
    
    # ========== OUTPUT ==========
    gr.Markdown("## Final Output")
    
    final_output = gr.Textbox(
        label="Processed Result",
        placeholder="Your processed text will appear here...",
        lines=8,
        max_lines=15
    )
    
    # Buttons
    with gr.Row():
        process_btn = gr.Button("✨ Apply Operations", variant="primary", scale=2)
        clear_btn = gr.Button("🗑️ Clear All", scale=1)
    
    # ========== EVENT HANDLERS ==========
    
    # Auto-correct when text is entered
    correct_btn.click(
        fn=update_corrected_text,
        inputs=[original_input],
        outputs=[corrected_output]
    )
    
    # Update statistics dynamically when corrected text changes
    corrected_output.change(
        fn=lambda text: update_statistics(text)[0] + "\n\n" + update_statistics(text)[1] + "\n\n" + update_statistics(text)[2],
        inputs=[corrected_output],
        outputs=[stats_display]
    )
    
    # Process all operations and update output
    process_btn.click(
        fn=update_output,
        inputs=[original_input, corrected_output, case_type, reverse_word_order, reverse_all_chars],
        outputs=[final_output, stats_display]
    )
    
    # Clear all fields
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[original_input, corrected_output, final_output, case_type, 
                reverse_word_order, reverse_all_chars, stats_display]
    )
    
    # ========== EXTRA FEATURES ==========
    gr.Markdown("---")
    gr.Markdown("### 💡 Tips")
    gr.Markdown(
        "- The corrected text is automatically cleaned (spacing, punctuation, capitalization)\n"
        "- You can manually edit the corrected text before applying operations\n"
        "- Statistics update in real-time as you type\n"
        "- All operations are applied to the corrected version of your text"
    )


# Launch the app
if __name__ == "__main__":
    text_app.launch()