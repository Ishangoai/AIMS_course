import re
import ssl
import string
from collections import Counter
from typing import NamedTuple, cast

import gradio as gr
import matplotlib.pyplot as plt
import nltk
import pandas as pd
import plotly.graph_objects as go
from matplotlib.figure import Figure
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from textblob import TextBlob
from wordcloud import WordCloud

# Handle SSL certificate issues for NLTK downloads
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


# Download required NLTK data with error handling
def download_nltk_resources():
    resources = ["punkt", "punkt_tab", "stopwords"]

    for resource in resources:
        try:
            if resource in ["punkt", "punkt_tab"]:
                nltk.data.find(f"tokenizers/{resource}")
            elif resource == "stopwords":
                nltk.data.find(f"corpora/{resource}")
            else:
                nltk.data.find(f"taggers/{resource}")
        except LookupError:
            print(f"Downloading NLTK resource: {resource}")
            try:
                nltk.download(resource, quiet=True)
            except Exception as e:
                print(f"Failed to download {resource}: {e}")


# Download resources at startup
download_nltk_resources()


text_configs = {
    "text_case": None,
    "reverse_word_order": False,
    "reverse_char_order": False,
}

# Common function words to exclude from analysis
FUNCTION_WORDS = set(stopwords.words("english")).union(
    {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "can",
        "shall",
    }
)

# Add punctuation for exclusion
FUNCTION_WORDS = FUNCTION_WORDS.union(set(string.punctuation))


def to_uppercase(text: str) -> str:
    return text.upper()


def to_lowercase(text: str) -> str:
    return text.lower()


def to_titlecase(text: str) -> str:
    return text.title()


def get_word_tokens(text: str) -> list[str]:
    try:
        return word_tokenize(text.lower())
    except Exception as e:
        # Fallback to simple split if tokenization fails
        print(f"Unable to get word tokens {str(e)}")
        return text.lower().split()


def get_sentences(text: str) -> list[str]:
    try:
        return sent_tokenize(text)
    except Exception as e:
        # Fallback to simple sentence splitting
        print(f"Unable to fetch sentences {str(e)}")
        return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]


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

    # Fallback: consider all non-stopwords as content words
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


def create_wordcloud_figure(text: str) -> Figure:
    """Create a word cloud matplotlib figure."""
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

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
        max_words=50,
        relative_scaling=0.5,  # type: ignore
    ).generate(" ".join(filtered_tokens))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("Word Cloud (Excluding Common Words)", pad=20, fontsize=14)

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


def create_statistics_visualizations(text: str) -> tuple[go.Figure, go.Figure]:
    """Create statistics and gauge visualizations."""
    if not text or text.isspace():
        empty_plot = go.Figure()
        empty_plot.add_annotation(
            text="No data to display", x=0.5, y=0.5, showarrow=False
        )
        empty_plot.update_layout(title="No data available")
        return empty_plot, empty_plot

    # Basic statistics bar chart
    wc = word_count(text)
    cc = character_count(text)
    pc = paragraph_count(text)
    sc = sentence_count(text)

    stats_fig = go.Figure(
        data=[
            go.Bar(
                name="Count",
                x=["Words", "Characters", "Paragraphs", "Sentences"],
                y=[wc, cc, pc, sc],
                marker_color=["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A"],
            )
        ]
    )
    stats_fig.update_layout(
        title="Basic Text Statistics",
        xaxis_title="Metric",
        yaxis_title="Count",
        showlegend=False,
    )

    # Readability and sentiment gauge chart
    fre = flesch_reading_ease(text)
    sentiment = sentiment_analysis(text)

    gauge_fig = go.Figure()

    # Flesch Reading Ease gauge
    gauge_fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=fre,
            title={"text": "Reading Ease"},
            domain={"row": 0, "column": 0},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 30], "color": "red"},
                    {"range": [30, 60], "color": "yellow"},
                    {"range": [60, 100], "color": "green"},
                ],
            },
        )
    )

    # Sentiment gauge
    gauge_fig.add_trace(
        go.Indicator(
            mode="gauge+number",
            value=(sentiment["polarity"] + 1) * 50,  # Convert -1 to +1 to 0-100 scale
            title={"text": "Sentiment"},
            domain={"row": 0, "column": 1},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 40], "color": "red"},
                    {"range": [40, 60], "color": "yellow"},
                    {"range": [60, 100], "color": "green"},
                ],
            },
        )
    )

    gauge_fig.update_layout(
        grid={"rows": 1, "columns": 2, "pattern": "independent"},
        title="Readability & Sentiment Scores",
    )

    return stats_fig, gauge_fig


def create_advanced_visualizations(text: str) -> go.Figure:
    """Create advanced word frequency visualization."""
    if not text or text.isspace():
        empty_plot = go.Figure()
        empty_plot.add_annotation(
            text="No data to display", x=0.5, y=0.5, showarrow=False
        )
        empty_plot.update_layout(title="No data available")
        return empty_plot

    tokens = get_word_tokens(text)
    filtered_tokens = filter_function_words(tokens)

    if not filtered_tokens:
        empty_plot = go.Figure()
        empty_plot.add_annotation(
            text="No meaningful words after filtering", x=0.5, y=0.5, showarrow=False
        )
        empty_plot.update_layout(title="Word Frequency Analysis")
        return empty_plot

    word_freq = Counter(filtered_tokens)
    common_words = word_freq.most_common(8)

    words, counts = zip(*common_words)

    advanced_fig = go.Figure(
        data=[go.Bar(x=words, y=counts, marker_color="#FFA07A", name="Word Frequency")]
    )
    advanced_fig.update_layout(
        title="Most Frequent Meaningful Words",
        xaxis_title="Words",
        yaxis_title="Frequency",
        showlegend=False,
    )

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


def text_actions(text: str) -> tuple[str, pd.DataFrame, go.Figure, go.Figure, go.Figure, Figure]:
    """Process text and generate all analyses."""
    if not text:
        empty_plot = go.Figure()
        empty_plot.add_annotation(
            text="No data to display", x=0.5, y=0.5, showarrow=False
        )
        empty_plot.update_layout(title="No data available")
        empty_wordcloud = create_wordcloud_figure("")
        empty_table = generate_comprehensive_table("")
        return "", empty_table, empty_plot, empty_plot, empty_plot, empty_wordcloud

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
    wordcloud_fig = create_wordcloud_figure(text)

    return (
        processed_text,
        stats_table,
        stats_fig,
        gauge_fig,
        advanced_fig,
        wordcloud_fig,
    )


def update_statistics(text: str) -> tuple[pd.DataFrame, go.Figure, go.Figure, go.Figure, Figure]:
    """Update statistics based on text input."""
    if not text or text.isspace():
        empty_plot = go.Figure()
        empty_plot.add_annotation(
            text="No data to display", x=0.5, y=0.5, showarrow=False
        )
        empty_plot.update_layout(title="No data available")
        empty_wordcloud = create_wordcloud_figure("")
        empty_table = generate_comprehensive_table("")
        return empty_table, empty_plot, empty_plot, empty_plot, empty_wordcloud

    stats_table = generate_comprehensive_table(text)
    stats_fig, gauge_fig = create_statistics_visualizations(text)
    advanced_fig = create_advanced_visualizations(text)
    wordcloud_fig = create_wordcloud_figure(text)

    return stats_table, stats_fig, gauge_fig, advanced_fig, wordcloud_fig


def clear_actions() -> tuple[gr.Radio, gr.Checkbox, gr.Checkbox]:
    """Clear all text actions and reset UI elements."""
    text_configs["text_case"] = None
    text_configs["reverse_word_order"] = False
    text_configs["reverse_char_order"] = False
    return gr.Radio(value=None), gr.Checkbox(value=False), gr.Checkbox(value=False)


def clear_all() -> tuple[str, pd.DataFrame, go.Figure, go.Figure, go.Figure, Figure, None, bool, bool]:
    """Clear all inputs and outputs."""
    clear_actions()

    empty_plot = go.Figure()
    empty_plot.add_annotation(
        text="No data to display", x=0.5, y=0.5, showarrow=False
    )
    empty_plot.update_layout(title="No data available")
    empty_wordcloud = create_wordcloud_figure("")
    empty_table = generate_comprehensive_table("")
    return (
        "",
        empty_table,
        empty_plot,
        empty_plot,
        empty_plot,
        empty_wordcloud,
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
                wordcloud_display = gr.Plot(label="Word Cloud")

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

    def clear_text_function() -> tuple[str, pd.DataFrame, go.Figure, go.Figure, go.Figure, Figure]:
        """Clear text input and reset outputs."""
        empty_plot = go.Figure()
        empty_plot.add_annotation(
            text="No data to display", x=0.5, y=0.5, showarrow=False
        )
        empty_plot.update_layout(title="No data available")
        empty_wordcloud = create_wordcloud_figure("")
        empty_table = generate_comprehensive_table("")
        return "", empty_table, empty_plot, empty_plot, empty_plot, empty_wordcloud

    clear_text_btn.click(
        fn=clear_text_function,
        outputs=[user_input, stats_display, stats_plot, gauge_plot, advanced_plot, wordcloud_display],
    )

    clear_actions_btn.click(
        fn=clear_actions,
        outputs=[case_choice, reverse_word_checkbox, reverse_char_checkbox],
    )

    def clear_all_function() -> tuple[str, pd.DataFrame, go.Figure, go.Figure, go.Figure, Figure, None, bool, bool]:
        """Clear all inputs and outputs."""
        clear_actions()
        empty_plot = go.Figure()
        empty_plot.add_annotation(
            text="No data to display", x=0.5, y=0.5, showarrow=False
        )
        empty_plot.update_layout(title="No data available")
        empty_wordcloud = create_wordcloud_figure("")
        empty_table = generate_comprehensive_table("")
        return (
            "",
            empty_table,
            empty_plot,
            empty_plot,
            empty_plot,
            empty_wordcloud,
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
