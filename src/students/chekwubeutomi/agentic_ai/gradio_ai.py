import gradio as gr

from .agent import get_article


def fetch_article(topic):
    """Fetch article from the agent with error handling."""
    if not topic:
        return gr.update(value="⚠️ Please select a topic first."), ""

    try:
        article = get_article(topic)
        if not article:
            return gr.update(value="⚠️ No article found for this topic."), ""
        words = article.split()
        word_length = len(words)
        if word_length > 1050:
            article = "".join(words[:1050])
        return article, article
    except Exception as e:
        error_msg = f"❌ Error fetching article: {str(e)}"
        return gr.update(value=error_msg), ""


def reset_fields():
    """Reset all interface fields to default state."""
    return gr.update(value=None), gr.update(value=""), gr.update(value="")


# Topic configuration
TOPICS = [
    "Agentic AI",
    "MLOps",
    "CI/CD",
    "Software Development Best Practices",
    "Gradio"
]

# Build Gradio interface
with gr.Blocks(title="Article Fetcher", theme=gr.themes.Soft()) as article_fetcher:
    gr.Markdown(
        """
        ## 📰 Article Fetcher
        Select a topic from the dropdown and click **Fetch Article** to load content.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            topic_dropdown = gr.Dropdown(
                choices=TOPICS,
                label="Select a Topic",
                info="Choose a topic to fetch related article"
            )

            with gr.Row():
                fetch_button = gr.Button("Fetch Article", variant="primary")
                reset_button = gr.Button("Reset Screen", variant="secondary")

        with gr.Column(scale=2):
            markdown_output = gr.Markdown(
                label="Article Content",
                value="*Select a topic and click Fetch Article to begin.*"
            )

    # Hidden text component for storing raw article data
    article_output = gr.Text(visible=False)

    # Event handlers
    fetch_button.click(
        fn=fetch_article,
        inputs=topic_dropdown,
        outputs=[markdown_output, article_output]
    )

    reset_button.click(
        fn=reset_fields,
        outputs=[topic_dropdown, markdown_output, article_output]
    )

    # Dropdown change event - automatically fetch when topic is selected
    topic_dropdown.change(
        fn=fetch_article,
        inputs=topic_dropdown,
        outputs=[markdown_output, article_output]
    )

# Export the interface
if __name__ == "__main__":
    article_fetcher.launch()
