from collections import Counter
from typing import List, Optional, Tuple

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

# Global configuration
text_configs = {
    "text_case": None,
    "reverse_word_order": False,
    "reverse_char_order": False
}


def to_uppercase(text: str) -> str:
    """Convert text to uppercase."""
    return text.upper()


def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def to_titlecase(text: str) -> str:
    """Convert text to title case."""
    return text.title()


def get_word_tokens(text: str) -> List[str]:
    """Split text into word tokens."""
    return text.split()


def reverse_word(text: str) -> str:
    """Reverse the order of words in text."""
    tokens = get_word_tokens(text)
    reversed_tokens = tokens[::-1]
    return " ".join(reversed_tokens)


def reverse_characters(text: str) -> str:
    """Reverse characters in each word of the text."""
    tokens = get_word_tokens(text)
    reversed_words = []

    for token in tokens:
        reversed_chars = token[::-1]
        reversed_words.append(reversed_chars)

    return " ".join(reversed_words)


def word_count(text: str) -> int:
    """Count the number of words in text."""
    tokens = get_word_tokens(text)
    return len(tokens)


def character_count(text: str) -> int:
    """Count characters excluding spaces."""
    return len(text.replace(" ", ""))


def average_word_length(text: str) -> float:
    """Calculate average word length."""
    tokens = get_word_tokens(text)
    if not tokens:
        return 0.0
    total_chars = sum(len(word) for word in tokens)
    return round(total_chars / len(tokens), 2)


def get_word_length_distribution(text: str) -> Counter:
    """Get distribution of word lengths."""
    tokens = get_word_tokens(text)
    if not tokens:
        return Counter()
    word_lengths = [len(word) for word in tokens]
    return Counter(word_lengths)


def get_character_frequency(text: str) -> Counter:
    """Get frequency of characters (case insensitive, excluding spaces)."""
    text_no_spaces = text.replace(" ", "").lower()
    if not text_no_spaces:
        return Counter()
    return Counter(text_no_spaces)


def generate_table(word_count_val: int, char_count_val: int, avg_word_length_val: float) -> pd.DataFrame:
    """Generate statistics table."""
    data = pd.DataFrame({
        "Metric": ["Word Count", "Character Count", "Average Word Length"],
        "Value": [word_count_val, char_count_val, avg_word_length_val]
    })
    return data


def create_statistics_visualizations(text: str) -> Tuple[Figure, Figure, Figure]:
    """
    Create three matplotlib visualizations for text statistics.

    Returns:
        Tuple containing: basic stats figure, word length distribution figure, character frequency figure
    """
    # Create figures with consistent styling
    plt.style.use('seaborn-v0_8')

    # Initialize empty figures for the case of no data
    if not text or text.isspace():
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.text(0.5, 0.5, "No data to display",
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title("Basic Text Statistics")
        ax1.axis('off')

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.text(0.5, 0.5, "No data to display",
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title("Word Length Distribution")
        ax2.axis('off')

        fig3, ax3 = plt.subplots(figsize=(6, 4))
        ax3.text(0.5, 0.5, "No data to display",
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title("Character Frequency")
        ax3.axis('off')

        return fig1, fig2, fig3

    # Calculate statistics
    wc = word_count(text)
    cc = character_count(text)
    awl = average_word_length(text)

    # Figure 1: Basic Statistics Bar Chart
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    metrics = ['Words', 'Characters', 'Avg Word Length']
    values = [wc, cc, awl]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    bars = ax1.bar(metrics, values, color=colors, alpha=0.8)
    ax1.set_title("Basic Text Statistics", fontsize=14, fontweight='bold')
    ax1.set_ylabel("Count", fontsize=12)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                f'{value}', ha='center', va='bottom', fontweight='bold')

    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=0)
    fig1.tight_layout()

    # Figure 2: Word Length Distribution
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    word_length_dist = get_word_length_distribution(text)

    if word_length_dist:
        lengths = list(word_length_dist.keys())
        counts = list(word_length_dist.values())

        bars = ax2.bar(lengths, counts, color='#96CEB4', alpha=0.8)
        ax2.set_title("Word Length Distribution", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Word Length", fontsize=12)
        ax2.set_ylabel("Frequency", fontsize=12)
        ax2.set_xticks(lengths)

        # Add value labels on bars
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                    f'{count}', ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, "No words to analyze",
                ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title("Word Length Distribution")

    fig2.tight_layout()

    # Figure 3: Character Frequency (Top 10)
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    char_freq = get_character_frequency(text)

    if char_freq:
        # Get top 10 most common characters
        common_chars = char_freq.most_common(10)
        chars, freqs = zip(*common_chars) if common_chars else ([], [])

        if chars and freqs:
            bars = ax3.bar(range(len(chars)), freqs, color='#FFA07A', alpha=0.8)
            ax3.set_title("Top 10 Character Frequency", fontsize=14, fontweight='bold')
            ax3.set_xlabel("Characters", fontsize=12)
            ax3.set_ylabel("Frequency", fontsize=12)
            ax3.set_xticks(range(len(chars)))
            ax3.set_xticklabels(chars)

            # Add value labels on bars
            for bar, freq in zip(bars, freqs):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                        f'{freq}', ha='center', va='bottom', fontsize=9)
        else:
            ax3.text(0.5, 0.5, "No characters to analyze",
                    ha='center', va='center', transform=ax3.transAxes, fontsize=12)
    else:
        ax3.text(0.5, 0.5, "No characters to analyze",
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)

    fig3.tight_layout()

    return fig1, fig2, fig3


def create_advanced_visualizations(text: str) -> Figure:
    """Create advanced visualization for word frequency analysis."""
    fig, ax = plt.subplots(figsize=(10, 6))

    if not text or text.isspace():
        ax.text(0.5, 0.5, "No data to display",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title("Word Frequency Analysis")
        ax.axis('off')
        return fig

    tokens = get_word_tokens(text)
    if not tokens:
        ax.text(0.5, 0.5, "No words to analyze",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title("Word Frequency Analysis")
        ax.axis('off')
        return fig

    word_freq = Counter(tokens)
    common_words = word_freq.most_common(8)

    if not common_words:
        ax.text(0.5, 0.5, "No frequent words found",
                ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title("Word Frequency Analysis")
        ax.axis('off')
        return fig

    words, counts = zip(*common_words)

    # Create line plot with markers
    x_pos = range(len(words))
    ax.plot(x_pos, counts, marker='o', linestyle='-',
            color='#6A0DAD', linewidth=2, markersize=8, markerfacecolor='#FFA07A')

    ax.set_title("Most Frequent Words", fontsize=14, fontweight='bold')
    ax.set_xlabel("Words", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(words, rotation=45, ha='right')

    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add value labels on points
    for i, (word, count) in enumerate(common_words):
        ax.annotate(f'{count}', (i, count),
                   textcoords="offset points", xytext=(0, 10),
                   ha='center', fontweight='bold')

    fig.tight_layout()
    return fig


def selected_textcase(option: Optional[str]) -> None:
    """Update text case configuration."""
    if option == "Uppercase":
        text_configs["text_case"] = "upper"
    elif option == "Titlecase":
        text_configs["text_case"] = "title"
    elif option == "Lowercase":
        text_configs["text_case"] = "lower"
    else:
        text_configs["text_case"] = None


def set_reverse_word(is_checked: bool) -> None:
    """Update reverse word order configuration."""
    text_configs["reverse_word_order"] = is_checked


def set_reverse_char(is_checked: bool) -> None:
    """Update reverse character order configuration."""
    text_configs["reverse_char_order"] = is_checked


def text_actions(text: str) -> Tuple[str, pd.DataFrame, Figure, Figure, Figure, Figure]:
    """Process text and generate all outputs."""
    if not text:
        empty_fig1, empty_fig2, empty_fig3 = create_statistics_visualizations("")
        empty_advanced = create_advanced_visualizations("")
        return "", generate_table(0, 0, 0), empty_fig1, empty_fig2, empty_fig3, empty_advanced

    processed_text = text
    text_case = text_configs["text_case"]
    reverse_word_order = text_configs["reverse_word_order"]
    reverse_char_order = text_configs["reverse_char_order"]

    # Apply text transformations
    if text_case == "upper":
        processed_text = to_uppercase(processed_text)
    elif text_case == "lower":
        processed_text = to_lowercase(processed_text)
    elif text_case == "title":
        processed_text = to_titlecase(processed_text)

    if reverse_word_order:
        processed_text = reverse_word(processed_text)

    if reverse_char_order:
        processed_text = reverse_characters(processed_text)

    # Generate statistics and visualizations
    wc = word_count(text)
    cc = character_count(text)
    awl = average_word_length(text)

    stats_table = generate_table(wc, cc, awl)
    stats_fig, word_len_fig, char_freq_fig = create_statistics_visualizations(text)
    advanced_fig = create_advanced_visualizations(text)

    return processed_text, stats_table, stats_fig, word_len_fig, char_freq_fig, advanced_fig


def update_statistics(text: str) -> Tuple[pd.DataFrame, Figure, Figure, Figure, Figure]:
    """Update statistics and visualizations based on input text."""
    if not text or text.isspace():
        empty_fig1, empty_fig2, empty_fig3 = create_statistics_visualizations("")
        empty_advanced = create_advanced_visualizations("")
        return generate_table(0, 0, 0), empty_fig1, empty_fig2, empty_fig3, empty_advanced

    wc = word_count(text)
    cc = character_count(text)
    awl = average_word_length(text)

    stats_table = generate_table(wc, cc, awl)
    stats_fig, word_len_fig, char_freq_fig = create_statistics_visualizations(text)
    advanced_fig = create_advanced_visualizations(text)

    return stats_table, stats_fig, word_len_fig, char_freq_fig, advanced_fig


def clear_actions() -> Tuple[gr.Radio, gr.Checkbox, gr.Checkbox]:
    """Clear action configurations."""
    text_configs["text_case"] = None
    text_configs["reverse_word_order"] = False
    text_configs["reverse_char_order"] = False
    return gr.Radio(value=None), gr.Checkbox(value=False), gr.Checkbox(value=False)


def clear_all() -> Tuple[str, pd.DataFrame, Figure, Figure, Figure, Figure, gr.Radio, gr.Checkbox, gr.Checkbox]:
    """Clear all inputs and outputs."""
    clear_actions()
    empty_fig1, empty_fig2, empty_fig3 = create_statistics_visualizations("")
    empty_advanced = create_advanced_visualizations("")
    return "", generate_table(0, 0, 0), empty_fig1, empty_fig2, empty_fig3, empty_advanced, None, False, False  # type: ignore


# Create Gradio interface
with gr.Blocks(
    css="""
    #spaced-row {margin:30px 0px;}
    .viz-container {background: white; border-radius: 10px; padding: 10px; margin: 5px;}
    """
) as dard:
    gr.Markdown("# Text Processing & Analytics Tool")

    with gr.Row():
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

            with gr.Row():
                with gr.Column():
                    stats_display = gr.Dataframe(
                        label='Basic Statistics',
                        headers=["Metric", "Value"],
                        datatype=["str", "number"],
                        row_count=3,
                        col_count=2,
                        value=generate_table(0, 0, 0)
                    )

            with gr.Tabs():
                with gr.TabItem("Overview"):
                    with gr.Row():
                        stats_plot = gr.Plot(label="Basic Statistics Chart")
                        word_len_plot = gr.Plot(label="Word Length Distribution")

                with gr.TabItem("Character Analysis"):
                    with gr.Row():
                        char_freq_plot = gr.Plot(label="Character Frequency")
                        advanced_plot = gr.Plot(label="Word Frequency Analysis")

    # Event handlers
    user_input.change(
        fn=update_statistics,
        inputs=user_input,
        outputs=[stats_display, stats_plot, word_len_plot, char_freq_plot, advanced_plot]
    )

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

    send_btn.click(
        fn=text_actions,
        inputs=user_input,
        outputs=[output, stats_display, stats_plot, word_len_plot, char_freq_plot, advanced_plot]
    )

    clear_text_btn.click(
        fn=lambda: (
            "",
            generate_table(0, 0, 0),
            *create_statistics_visualizations(""),
            create_advanced_visualizations("")
        ),
        outputs=[user_input, stats_display, stats_plot, word_len_plot, char_freq_plot, advanced_plot]
    )

    clear_actions_btn.click(
        fn=clear_actions,
        outputs=[case_choice, reverse_word_checkbox, reverse_char_checkbox]
    )

    clear_all_btn.click(
        fn=clear_all,
        outputs=[
            user_input, stats_display, stats_plot, word_len_plot, char_freq_plot,
            advanced_plot, case_choice, reverse_word_checkbox, reverse_char_checkbox
        ]
    )

if __name__ == "__main__":
    dard.launch()
