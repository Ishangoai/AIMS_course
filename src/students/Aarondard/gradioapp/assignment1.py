import re
import string
from collections import Counter
from typing import NamedTuple, cast

import gradio as gr
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure
from textblob import TextBlob

text_configs = {
    "text_case": None,
    "reverse_word_order": False,
    "reverse_char_order": False,
}

# Common function words to exclude from analysis (manual list instead of NLTK stopwords)
FUNCTION_WORDS = {
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "can", "shall", "i", "you", "he", "she",
    "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his",
    "its", "our", "their", "this", "that", "these", "those", "am", "not", "so",
    "than", "too", "very", "what", "when", "where", "which", "who", "whom", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "some", "such",
    "no", "nor", "only", "own", "same", "just", "also", "now", "then", "here",
    "there", "up", "down", "out", "off", "over", "under", "again", "further",
    "once", "every", "because", "since", "until", "while", "after", "before",
    "above", "below", "between", "through", "during", "without", "under",
    "about", "against", "into", "like", "through", "until", "upon", "within"
}

# Add punctuation for exclusion
FUNCTION_WORDS = FUNCTION_WORDS.union(set(string.punctuation))


def to_uppercase(text: str) -> str:
    return text.upper()


def to_lowercase(text: str) -> str:
    return text.lower()


def to_titlecase(text: str) -> str:
    return text.title()


def get_word_tokens(text: str) -> list[str]:
    """Simple word tokenization without NLTK."""
    # Convert to lowercase and split, then clean up tokens
    words = text.lower().split()
    # Remove punctuation from words and filter empty strings
    cleaned_words = []
    for word in words:
        # Remove punctuation from start and end of words
        cleaned_word = word.strip(string.punctuation)
        if cleaned_word:  # Only add non-empty strings
            cleaned_words.append(cleaned_word)
    return cleaned_words


def get_sentences(text: str) -> list[str]:
    """Simple sentence splitting without NLTK."""
    # Split on sentence endings and filter empty strings
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    return sentences


def reverse_word(text: str) -> str:
    tokens = get_word_tokens(text)
    reversed_tokens = tokens[::-1]
    return " ".join(reversed_tokens)


def reverse_characters(text: str) -> str:
    tokens = get_word_tokens(text)
    reversed_words = []

    for token in tokens:
        reversed_chars = token[::-1]
        reversed_words.append(reversed_chars)

    return " ".join(reversed_words)


def word_count(text: str) -> int:
    tokens = get_word_tokens(text)
    return len([token for token in tokens if token.isalpha()])


def character_count(text: str) -> int:
    return len(text.replace(" ", ""))


def paragraph_count(text: str) -> int:
    paragraphs = [p for p in text.split("\n") if p.strip()]
    return len(paragraphs)


def sentence_count(text: str) -> int:
    sentences = get_sentences(text)
    return len(sentences)


def average_word_length(text: str) -> float:
    tokens = get_word_tokens(text)
    alpha_tokens = [token for token in tokens if token.isalpha()]
    if not alpha_tokens:
        return 0.0
    total_chars = sum(len(word) for word in alpha_tokens)
    return round(total_chars / len(alpha_tokens), 2)


def average_sentence_length(text: str) -> float:
    sentences = get_sentences(text)
    if not sentences:
        return 0.0
    total_words = sum(
        len([word for word in get_word_tokens(sent) if word.isalpha()])
        for sent in sentences
    )
    return round(total_words / len(sentences), 2)


def vocabulary_size(text: str) -> int:
    tokens = get_word_tokens(text)
    alpha_tokens = [token for token in tokens if token.isalpha()]
    return len(set(alpha_tokens))


def type_token_ratio(text: str) -> float:
    tokens = get_word_tokens(text)
    alpha_tokens = [token for token in tokens if token.isalpha()]
    if not alpha_tokens:
        return 0.0
    return round(len(set(alpha_tokens)) / len(alpha_tokens), 3)


def stopword_ratio(text: str) -> float:
    tokens = get_word_tokens(text)
    alpha_tokens = [token for token in tokens if token.isalpha()]
    if not alpha_tokens:
        return 0.0
    stopword_count = sum(1 for token in alpha_tokens if token in FUNCTION_WORDS)
    return round(stopword_count / len(alpha_tokens), 3)


def lexical_density(text: str) -> float:
    tokens = get_word_tokens(text)
    alpha_tokens = [token for token in tokens if token.isalpha()]
    if not alpha_tokens:
        return 0.0

    # Consider all non-stopwords as content words
    content_words = [token for token in alpha_tokens if token not in FUNCTION_WORDS]
    return round(len(content_words) / len(alpha_tokens), 3)


def flesch_reading_ease(text: str) -> float:
    """Flesch Reading Ease score - higher = easier to read."""
    sentences = get_sentences(text)
    if not sentences:
        return 0.0

    total_sentences = len(sentences)
    total_words = sum(
        len([word for word in get_word_tokens(sent) if word.isalpha()])
        for sent in sentences
    )
    total_syllables = sum(
        count_syllables(word)
        for sent in sentences
        for word in get_word_tokens(sent)
        if word.isalpha()
    )

    if total_words == 0:
        return 0.0

    return round(
        206.835
        - 1.015 * (total_words / total_sentences)
        - 84.6 * (total_syllables / total_words),
        1,
    )


def count_syllables(word: str) -> int:
    """Simple syllable counter."""
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"):
        count -= 1
    if count == 0:
        count += 1
    return count


class Sentiment(NamedTuple):
    polarity: float
    subjectivity: float


def sentiment_analysis(text: str) -> dict[str, float]:
    """Basic sentiment analysis using TextBlob."""
    if not text.strip():
        return {"polarity": 0.0, "subjectivity": 0.0}

    try:
        blob = TextBlob(text)
        sentiment = cast(Sentiment, blob.sentiment)
        return {
            "polarity": round(sentiment.polarity, 3),
            "subjectivity": round(sentiment.subjectivity, 3),
        }
    except Exception as e:
        print(f"Unable to get polarity and subjectivity {str(e)}")
        return {"polarity": 0.00, "subjectivity": 0.00}


def filter_function_words(words: list[str]) -> list[str]:
    """Remove common function words from the word list."""
    return [word for word in words if word.lower() not in FUNCTION_WORDS]


def create_word_frequency_chart(text: str) -> Figure:
    """Create a word frequency bar chart instead of word cloud."""
    if not text or text.isspace():
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=16)
        ax.axis("off")
        return fig

    tokens = get_word_tokens(text)
    filtered_tokens = filter_function_words(tokens)

    if not filtered_tokens:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(
            0.5,
            0.5,
            "No meaningful words to display\n(after filtering common words)",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        return fig

    # Count word frequencies and get top 15 words
    word_freq = Counter(filtered_tokens)
    common_words = word_freq.most_common(15)

    words, counts = zip(*common_words)

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    y_pos = range(len(words))

    bars = ax.barh(y_pos, counts, color='skyblue', alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()  # Highest frequency at top
    ax.set_xlabel('Frequency')
    ax.set_title('Top 15 Most Frequent Words (Excluding Common Words)', pad=20, fontsize=14)

    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width, bar.get_y() + bar.get_height() / 2,
                f' {width}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    return fig


def generate_comprehensive_table(text: str) -> pd.DataFrame:
    """Generate comprehensive statistics table."""
    if not text or text.isspace():
        return pd.DataFrame(
            {"Category": ["No data available"], "Statistic": [""], "Value": [""]}
        )

    # Basic Statistics
    wc = word_count(text)
    cc = character_count(text)
    pc = paragraph_count(text)
    sc = sentence_count(text)
    awl = average_word_length(text)
    asl = average_sentence_length(text)

    # Readability & Sentiment
    fre = flesch_reading_ease(text)
    sentiment = sentiment_analysis(text)

    data = [
        # Basic Statistics
        ("Basic", "Word Count", wc),
        ("Basic", "Character Count", cc),
        ("Basic", "Paragraph Count", pc),
        ("Basic", "Sentence Count", sc),
        ("Basic", "Average Word Length", awl),
        ("Basic", "Average Sentence Length", asl),
        # Readability & Sentiment
        ("Readability", "Flesch Reading Ease", fre),
        ("Sentiment", "Sentiment Polarity", sentiment["polarity"]),
        ("Sentiment", "Subjectivity", sentiment["subjectivity"]),
    ]

    columns_list: tuple[str, str, str] = ("Category", "Statistic", "Value")
    return pd.DataFrame(data, columns=list(columns_list))  # type: ignore


def create_statistics_visualizations(text: str) -> tuple[Figure, Figure]:
    """Create statistics visualizations using matplotlib instead of plotly."""
    if not text or text.isspace():
        empty_fig, empty_ax = plt.subplots(figsize=(8, 4))
        empty_ax.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=14)
        empty_ax.axis("off")
        empty_fig.suptitle("No data available")
        return empty_fig, empty_fig

    # Basic statistics bar chart
    wc = word_count(text)
    cc = character_count(text)
    pc = paragraph_count(text)
    sc = sentence_count(text)

    # Create basic stats figure
    stats_fig, stats_ax = plt.subplots(figsize=(10, 6))
    metrics = ["Words", "Characters", "Paragraphs", "Sentences"]
    values = [wc, cc, pc, sc]
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"]

    bars = stats_ax.bar(metrics, values, color=colors, alpha=0.7)
    stats_ax.set_title("Basic Text Statistics", fontsize=14, pad=20)
    stats_ax.set_ylabel("Count")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        stats_ax.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{int(height)}', ha='center', va='bottom')

    plt.setp(stats_ax.xaxis.get_majorticklabels(), rotation=45)
    stats_fig.tight_layout()

    # Create readability and sentiment gauge figure
    fre = flesch_reading_ease(text)
    sentiment = sentiment_analysis(text)

    gauge_fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Reading ease gauge (simplified as bar)
    ax1.barh([0], [fre], color='lightblue', alpha=0.7, height=0.5)
    ax1.set_xlim(0, 100)
    ax1.set_xlabel('Score')
    ax1.set_title('Reading Ease Score')
    ax1.text(fre / 2, 0, f'{fre}', ha='center', va='center', fontweight='bold')

    # Add readability indicators
    ax1.axvline(x=30, color='red', linestyle='--', alpha=0.5)
    ax1.axvline(x=60, color='yellow', linestyle='--', alpha=0.5)
    ax1.axvline(x=90, color='green', linestyle='--', alpha=0.5)

    # Sentiment gauge (simplified as bar)
    sentiment_score = (sentiment["polarity"] + 1) * 50  # Convert -1 to +1 to 0-100 scale
    ax2.barh([0], [sentiment_score], color='lightcoral', alpha=0.7, height=0.5)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('Score')
    ax2.set_title('Sentiment Score')
    ax2.text(sentiment_score / 2, 0, f'{sentiment_score:.1f}', ha='center', va='center', fontweight='bold')

    # Add sentiment indicators
    ax2.axvline(x=40, color='red', linestyle='--', alpha=0.5)
    ax2.axvline(x=60, color='yellow', linestyle='--', alpha=0.5)
    ax2.axvline(x=80, color='green', linestyle='--', alpha=0.5)

    gauge_fig.suptitle('Readability & Sentiment Scores', fontsize=14)
    gauge_fig.tight_layout()

    return stats_fig, gauge_fig


def create_advanced_visualizations(text: str) -> Figure:
    """Create advanced word frequency visualization using matplotlib."""
    if not text or text.isspace():
        empty_fig, empty_ax = plt.subplots(figsize=(10, 6))
        empty_ax.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=16)
        empty_ax.axis("off")
        empty_fig.suptitle("No data available")
        return empty_fig

    tokens = get_word_tokens(text)
    filtered_tokens = filter_function_words(tokens)

    if not filtered_tokens:
        empty_fig, empty_ax = plt.subplots(figsize=(10, 6))
        empty_ax.text(0.5, 0.5, "No meaningful words after filtering",
                     ha="center", va="center", fontsize=14)
        empty_ax.axis("off")
        empty_fig.suptitle("Word Frequency Analysis")
        return empty_fig

    word_freq = Counter(filtered_tokens)
    common_words = word_freq.most_common(8)

    words, counts = zip(*common_words)

    advanced_fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(words, counts, color="#FFA07A", alpha=0.7)
    ax.set_title("Most Frequent Meaningful Words", fontsize=14, pad=20)
    ax.set_xlabel("Words")
    ax.set_ylabel("Frequency")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{int(height)}', ha='center', va='bottom')

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    advanced_fig.tight_layout()

    return advanced_fig


def selected_textcase(option: str) -> None:
    """Set the selected text case transformation."""
    if option == "Uppercase":
        text_configs["text_case"] = "upper"
    elif option == "Titlecase":
        text_configs["text_case"] = "title"
    elif option == "Lowercase":
        text_configs["text_case"] = "lower"
    else:
        text_configs["text_case"] = None


def set_reverse_word(is_checked: bool) -> None:
    """Set reverse word order flag."""
    text_configs["reverse_word_order"] = is_checked


def set_reverse_char(is_checked: bool) -> None:
    """Set reverse character order flag."""
    text_configs["reverse_char_order"] = is_checked


def text_actions(text: str) -> tuple[str, pd.DataFrame, Figure, Figure, Figure, Figure]:
    """Process text and generate all analyses."""
    if not text:
        empty_fig, empty_ax = plt.subplots(figsize=(8, 4))
        empty_ax.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=14)
        empty_ax.axis("off")
        empty_fig.suptitle("No data available")
        empty_wordchart = create_word_frequency_chart("")
        empty_table = generate_comprehensive_table("")
        return "", empty_table, empty_fig, empty_fig, empty_fig, empty_wordchart

    processed_text = text
    text_case = text_configs["text_case"]
    reverse_word_order = text_configs["reverse_word_order"]
    reverse_char_order = text_configs["reverse_char_order"]

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

    stats_table = generate_comprehensive_table(text)
    stats_fig, gauge_fig = create_statistics_visualizations(text)
    advanced_fig = create_advanced_visualizations(text)
    word_freq_fig = create_word_frequency_chart(text)

    return (
        processed_text,
        stats_table,
        stats_fig,
        gauge_fig,
        advanced_fig,
        word_freq_fig,
    )


def update_statistics(text: str) -> tuple[pd.DataFrame, Figure, Figure, Figure, Figure]:
    """Update statistics based on text input."""
    if not text or text.isspace():
        empty_fig, empty_ax = plt.subplots(figsize=(8, 4))
        empty_ax.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=14)
        empty_ax.axis("off")
        empty_fig.suptitle("No data available")
        empty_wordchart = create_word_frequency_chart("")
        empty_table = generate_comprehensive_table("")
        return empty_table, empty_fig, empty_fig, empty_fig, empty_wordchart

    stats_table = generate_comprehensive_table(text)
    stats_fig, gauge_fig = create_statistics_visualizations(text)
    advanced_fig = create_advanced_visualizations(text)
    word_freq_fig = create_word_frequency_chart(text)

    return stats_table, stats_fig, gauge_fig, advanced_fig, word_freq_fig


def clear_actions() -> tuple[gr.Radio, gr.Checkbox, gr.Checkbox]:
    """Clear all text actions and reset UI elements."""
    text_configs["text_case"] = None
    text_configs["reverse_word_order"] = False
    text_configs["reverse_char_order"] = False
    return gr.Radio(value=None), gr.Checkbox(value=False), gr.Checkbox(value=False)


def clear_all() -> tuple[str, pd.DataFrame, Figure, Figure, Figure, Figure, None, bool, bool]:
    """Clear all inputs and outputs."""
    clear_actions()

    empty_fig, empty_ax = plt.subplots(figsize=(8, 4))
    empty_ax.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=14)
    empty_ax.axis("off")
    empty_fig.suptitle("No data available")
    empty_wordchart = create_word_frequency_chart("")
    empty_table = generate_comprehensive_table("")
    return (
        "",
        empty_table,
        empty_fig,
        empty_fig,
        empty_fig,
        empty_wordchart,
        None,
        False,
        False,
    )


with gr.Blocks(
    css="""
    body {background-color: #ffffff;}
    #spaced-row {margin:30px 0px;}
    .viz-container {background: white; border-radius: 10px; padding: 10px; margin: 5px;}
    .stats-table {max-height: 400px; overflow-y: auto;}
"""
) as dard:
    gr.Markdown("# Text Analysis")

    with gr.Row():
        with gr.Column(scale=1, min_width=300):
            gr.Markdown("### Text Actions")

            with gr.Accordion("Case Converter", open=False):
                case_choice = gr.Radio(
                    ["Uppercase", "Lowercase", "Titlecase"],
                    label="Select Case",
                    info="Choose text case transformation",
                )

            with gr.Accordion("Reverse Operations", open=False):
                reverse_word_checkbox = gr.Checkbox(
                    label="Reverse Word Order", info="Reverse the order of words"
                )
                reverse_char_checkbox = gr.Checkbox(
                    label="Reverse All Characters", info="Reverse characters in each word"
                )

            with gr.Row():
                clear_actions_btn = gr.Button("Clear Actions", variant="secondary")
                clear_all_btn = gr.Button("Clear All", variant="secondary")

        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    user_input = gr.TextArea(
                        label="Input Text",
                        placeholder="Enter or paste your text here for comprehensive analysis...",
                        lines=5,
                    )

            with gr.Row():
                send_btn = gr.Button("Process Text", variant="primary")
                clear_text_btn = gr.Button("Clear Text", variant="secondary")

            with gr.Column():
                output = gr.TextArea(label="Processed Output", lines=5, interactive=False)

    with gr.Row():
        with gr.Column():
            stats_display = gr.Dataframe(
                label="Comprehensive Text Statistics",
                headers=["Category", "Statistic", "Value"],
                datatype=["str", "str", "str"],
                row_count=14,
                col_count=3,
                wrap=True,
                elem_classes=["stats-table"],
            )

    with gr.Tabs():
        with gr.TabItem("Overview"):
            with gr.Row():
                stats_plot = gr.Plot(label="Basic Statistics")
                gauge_plot = gr.Plot(label="Readability & Sentiment")

        with gr.TabItem("Word Analysis"):
            with gr.Row():
                advanced_plot = gr.Plot(label="Word Frequency Analysis")
                wordcloud_display = gr.Plot(label="Word Frequency Chart")

    user_input.change(
        fn=update_statistics,
        inputs=user_input,
        outputs=[stats_display, stats_plot, gauge_plot, advanced_plot, wordcloud_display],
    )

    case_choice.change(fn=selected_textcase, inputs=case_choice)

    reverse_word_checkbox.change(fn=set_reverse_word, inputs=reverse_word_checkbox)

    reverse_char_checkbox.change(fn=set_reverse_char, inputs=reverse_char_checkbox)

    send_btn.click(
        fn=text_actions,
        inputs=user_input,
        outputs=[
            output,
            stats_display,
            stats_plot,
            gauge_plot,
            advanced_plot,
            wordcloud_display,
        ],
    )

    def clear_text_function() -> tuple[str, pd.DataFrame, Figure, Figure, Figure, Figure]:
        """Clear text input and reset outputs."""
        empty_fig, empty_ax = plt.subplots(figsize=(8, 4))
        empty_ax.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=14)
        empty_ax.axis("off")
        empty_fig.suptitle("No data available")
        empty_wordchart = create_word_frequency_chart("")
        empty_table = generate_comprehensive_table("")
        return "", empty_table, empty_fig, empty_fig, empty_fig, empty_wordchart

    clear_text_btn.click(
        fn=clear_text_function,
        outputs=[user_input, stats_display, stats_plot, gauge_plot, advanced_plot, wordcloud_display],
    )

    clear_actions_btn.click(
        fn=clear_actions,
        outputs=[case_choice, reverse_word_checkbox, reverse_char_checkbox],
    )

    def clear_all_function() -> tuple[str, pd.DataFrame, Figure, Figure, Figure, Figure, None, bool, bool]:
        """Clear all inputs and outputs."""
        clear_actions()
        empty_fig, empty_ax = plt.subplots(figsize=(8, 4))
        empty_ax.text(0.5, 0.5, "No data to display", ha="center", va="center", fontsize=14)
        empty_ax.axis("off")
        empty_fig.suptitle("No data available")
        empty_wordchart = create_word_frequency_chart("")
        empty_table = generate_comprehensive_table("")
        return (
            "",
            empty_table,
            empty_fig,
            empty_fig,
            empty_fig,
            empty_wordchart,
            None,
            False,
            False,
        )

    clear_all_btn.click(
        fn=clear_all_function,
        outputs=[
            user_input,
            stats_display,
            stats_plot,
            gauge_plot,
            advanced_plot,
            wordcloud_display,
            case_choice,
            reverse_word_checkbox,
            reverse_char_checkbox,
        ],
    )
