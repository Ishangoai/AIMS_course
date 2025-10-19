"""
A Gradio-based text formatting and mini Scrabble game interface.

This module provides tools to:
- Perform text transformations (case conversion, reversal).
- Display Statistics about the current texts.
- Integrate a Scrabble-inspired mini word game.

Authors:
    Daniel Ayomide and Elisha Komolafe
Date:
    2025-10-12
"""

from __future__ import annotations

import re
from typing import Any

import gradio as gr
from gradioapp.utils import text_app_utils


def process_text(
    text: str, case_option: str, reverse_words: bool, reverse_chars: bool
) -> str:
    """
    Apply user-selected text transformations.

    Args:
        text (str): The input text string.
        case_option (str): Case transformation option ("Uppercase", "Lowercase", "Title Case").
        reverse_words (bool): Whether to reverse the order of words.
        reverse_chars (bool): Whether to reverse all characters.

    Returns:
        str: The processed text.
    """
    if not text:
        return ""

    # Case conversion
    match case_option:
        case "Uppercase":
            text = text.upper()
        case "Lowercase":
            text = text.lower()
        case "Title Case":
            text = text.title()

    # Reversal options
    if reverse_words:
        text = " ".join(text.split()[::-1])
    if reverse_chars:
        text = text[::-1]

    return text


def text_stats(text: str) -> dict[str, Any]:
    """
    Compute various linguistic and statistical features of the input text.

    Includes:
        - Word Count
        - Character Count
        - Average Word Length
        - Type–Token Ratio (TTR)
        - Longest and Shortest Word

    Args:
        text (str): The input text string.

    Returns:
        dict[str, Any]: A dictionary containing text statistics.
    """
    if not text.strip():
        return {
            "Word Count": 0,
            "Character Count": 0,
            "Average Word Length": 0.0,
            "Type–Token Ratio (TTR)": 0.0,
            "Lexical Density": 0.0,
            "Longest Word": "",
            "Shortest Word": "",
        }

    # Tokenize words (alphanumeric only)
    words = re.findall(r"\b\w+\b", text)
    word_count = len(words)
    char_count = len(text)
    unique_words = len(set(w.lower() for w in words))

    avg_word_len = sum(len(w) for w in words) / word_count if word_count else 0.0
    ttr = unique_words / word_count if word_count else 0.0

    longest_word = max(words, key=len, default="")
    shortest_word = min(words, key=len, default="")

    return {
        "Word Count": word_count,
        "Character Count": char_count,
        "Average Word Length": round(avg_word_len, 2),
        "Type–Token Ratio (TTR)": round(ttr, 3),
        "Longest Word": longest_word,
        "Shortest Word": shortest_word,
    }


def clear_all() -> tuple[str, str, dict[str, Any]]:
    """
    Reset the text fields and statistics for the text analyzer.

    Returns:
        tuple[str, str, dict[str, Any]]: Empty strings and reset statistics.
    """
    return (
        "",
        "",
        {
            "Word Count": 0,
            "Character Count": 0,
            "Average Word Length": 0.0,
            "Type–Token Ratio (TTR)": 0.0,
            "Lexical Density": 0.0,
            "Longest Word": "",
            "Shortest Word": "",
        },
    )


# ------------------- BUILD MAIN APP ------------------- #
with gr.Blocks() as text_app:
    gr.Markdown(
        "# 🨁 Text Formatting Application and Scrabble-ish Gamemode 🨃\n"
        "### 🨂 Perform text operations and test your might against the computer 😎 🨀"
    )

    # --- SIDE-BY-SIDE Input + Analyzer ---
    with gr.Row(equal_height=True):
        input_box = gr.Textbox(
            label="Enter your text here",
            lines=6,
            placeholder="Type or paste text...",
        )

        # Live statistics display
        with gr.Column():
            gr.Markdown("### 📊 Live Text Analyzer")
            stats_output = gr.JSON(
                label="Text Statistics",
                value={
                    "Word Count": 0,
                    "Character Count": 0,
                    "Average Word Length": 0.0,
                    "Type–Token Ratio (TTR)": 0.0,
                    "Lexical Density": 0.0,
                    "Longest Word": "",
                    "Shortest Word": "",
                },
            )

    # --- TEXT OPERATIONS ---
    with gr.Accordion("Text Operations", open=True):
        with gr.Tab("Case Converter"):
            case_option = gr.Radio(
                ["Uppercase", "Lowercase", "Title Case"],
                label="Choose Case",
                value="Lowercase",
            )

        with gr.Tab("Text Reverser"):
            reverse_words = gr.Checkbox(label="Reverse Word Order")
            reverse_chars = gr.Checkbox(label="Reverse All Characters")

    # --- OUTPUT AREA AND CONTROLS ---
    output_box = gr.Textbox(label="Output", lines=5)
    with gr.Row():
        clear_btn = gr.Button("Clear", variant="stop")

    gr.Markdown("---")

    # --- SCRABBLE GAME SECTION ---
    gr.Markdown("## 🮖 Mini Scrabble Game")
    gr.Markdown("Test your word-composing skills by playing against the computer!")

    scrabble_game = text_app_utils.scrabble_demo
    scrabble_game.render()

    # --- EVENT CONNECTIONS ---
    for component in (input_box, case_option, reverse_words, reverse_chars):
        component.change(
            process_text,
            inputs=[input_box, case_option, reverse_words, reverse_chars],
            outputs=output_box,
        )

    input_box.change(text_stats, inputs=input_box, outputs=stats_output)
    clear_btn.click(clear_all, outputs=[input_box, output_box, stats_output])
