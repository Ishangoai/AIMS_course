import gradio as gr


#import the generator function as this is a mockup for the actual implementation
def simulate_article_generation(topic, config):
    pass

def run_article_generation(topic, model_name, writer_temp, rewriter_temp, validator_temp,
                           target_words, tolerance, max_revisions,
                           writer_max_tokens, rewriter_max_tokens, validator_max_tokens):
    config = {
        "model_name": model_name,
        "writer_temperature": writer_temp,
        "rewriter_temperature": rewriter_temp,
        "validator_temperature": validator_temp,
        "target_words": int(target_words),
        "tolerance": int(tolerance),
        "max_revisions": int(max_revisions),
        "writer_max_tokens": int(writer_max_tokens) if writer_max_tokens else None,
        "rewriter_max_tokens": int(rewriter_max_tokens) if rewriter_max_tokens else None,
        "validator_max_tokens": int(validator_max_tokens) if validator_max_tokens else None
    }

    result = simulate_article_generation(topic, config)
    output = result.get("result", {})
    details = output.get("details", {})

    safeguarded_result = output.get("safeguarded_result", "⚠️ No safeguarded result found.")
    article = output.get("final_article_llm", "⚠️ No article text generated.")
    outline = details.get("outline", "⚠️ No outline generated.")
    insights = details.get("research_insights", [])

    insights_text = "\n\n---\n\n".join(insights) if insights else "No research insights available."

    return safeguarded_result, article, outline, insights_text


custom_css = """
.main-header {
    text-align: center;
    padding: 2rem 0 1rem 0;
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    border-radius: 12px;
    margin-bottom: 2rem;
    color: white;
}

.main-header h1 {
    margin: 0;
    font-size: 2.5rem;
    font-weight: 700;
}

.main-header p {
    margin: 0.5rem 0 0 0;
    font-size: 1.1rem;
    opacity: 0.95;
}

.example-topics {
    margin-top: 0.5rem;
    margin-bottom: 1rem;
}

.example-topics .gallery {
    gap: 0.75rem !important;
}

.example-topics .gallery button {
    border: 2px solid #f5576c !important;
    border-radius: 8px !important;
    padding: 0.75rem 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 2px 4px rgba(245, 87, 108, 0.1) !important;
}

.example-topics .gallery button:hover {
    border-color: #f093fb !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 8px rgba(245, 87, 108, 0.2) !important;
}

/* Remove blue background from labels */
label {
    background: transparent !important;
    color: #333 !important;
    font-weight: 600 !important;
}

.label-wrap {
    background: transparent !important;
}

span.label-wrap span {
    background: transparent !important;
    color: #333 !important;
}

/* Primary button - Dark Green */
button.primary {
    background: linear-gradient(135deg, #2d5016 0%, #3d6b1f 100%) !important;
    border: none !important;
    color: white !important;
}

button.primary:hover {
    background: linear-gradient(135deg, #3d6b1f 0%, #4d7c2f 100%) !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(45, 80, 22, 0.3) !important;
}

/* Secondary button - Orange */
button.secondary {
    background: linear-gradient(135deg, #ff6b35 0%, #f7931e 100%) !important;
    border: none !important;
    color: white !important;
}

button.secondary:hover {
    background: linear-gradient(135deg, #f7931e 0%, #ffaa4d 100%) !important;
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(255, 107, 53, 0.3) !important;
}

/* Small action buttons */
button[size="sm"] {
    background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%) !important;
    border: none !important;
    color: white !important;
    padding: 0.5rem 1rem !important;
}

button[size="sm"]:hover {
    background: linear-gradient(135deg, #4b5563 0%, #374151 100%) !important;
    transform: translateY(-1px);
    box-shadow: 0 2px 8px rgba(75, 85, 99, 0.3) !important;
}

.action-buttons {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.5rem;
}

.generate-section {
    margin: 1.5rem 0;
}

.output-section {
    margin-top: 2rem;
}

.main-container {
    display: flex;
    gap: 2rem;
    align-items: flex-start;
}

.input-column {
    flex: 1;
    min-width: 400px;
}

.output-column {
    flex: 1.5;
    min-width: 500px;
}

#component-0 {
    max-width: 1600px;
    margin: 0 auto;
}
"""

with gr.Blocks(theme=gr.themes.Soft(primary_hue="green"), css=custom_css) as demo:  # type: ignore
    gr.HTML("""
        <div class="main-header">
            <h1>🧾 AI Article Generator</h1>
            <p>Advanced multi-agent system that autonomously researches, 
            writes, validates, and safeguards academic-quality essays</p>
        </div>
    """)

    # Main container with side-by-side layout
    with gr.Row(elem_classes="main-container"):
        # Left Column - Inputs
        with gr.Column(elem_classes="input-column"):
            gr.Markdown("## 📝 Input Configuration")

            with gr.Group():
                topic = gr.Textbox(
                    label="📝 Article Topic",
                    placeholder="Enter your article topic here...",
                    lines=2,
                    scale=1
                )

                with gr.Row(elem_classes="example-topics"):
                    gr.Examples(
                        examples=[
                            ["CI/CD"],
                            ["MLOps"],
                            ["Microservices Architecture"],
                            ["Kubernetes"],
                            ["DevSecOps"]
                        ],
                        inputs=topic,
                        label="💡 Example Topics"
                    )

            with gr.Accordion("⚙️ Advanced Configuration", open=False):
                with gr.Row():
                    model_name = gr.Dropdown(
                        choices=["gemini-2.0-flash-exp", "gemini-2.5-pro"],
                        value="gemini-2.0-flash-exp",
                        label="🤖 Model",
                        info="Select the AI model to use"
                    )

                gr.Markdown("### Temperature Settings")
                with gr.Row():
                    writer_temp = gr.Slider(0.0, 1.0, value=0.45,
                                            step=0.05,
                                            label="Writer",
                                            info="Creativity level")
                with gr.Row():
                    rewriter_temp = gr.Slider(0.0, 1.0, value=0.25,
                                              step=0.05,
                                              label="Rewriter",
                                              info="Refinement focus")
                with gr.Row():
                    validator_temp = gr.Slider(0.0, 1.0, value=0.20,
                                               step=0.05,
                                               label="Validator",
                                               info="Strictness level")

                gr.Markdown("### Content Parameters")
                with gr.Row():
                    target_words = gr.Number(value=1000, label="Target Words", info="Desired article length")
                with gr.Row():
                    tolerance = gr.Number(value=50, label="Tolerance (±)", info="Acceptable word variance")
                with gr.Row():
                    max_revisions = gr.Number(value=5, label="Max Revisions", info="Quality iterations")

                gr.Markdown("### Token Limits (Optional)")
                with gr.Row():
                    writer_max_tokens = gr.Number(value=2048, label="Writer Max Tokens")
                with gr.Row():
                    rewriter_max_tokens = gr.Number(value=2048, label="Rewriter Max Tokens")
                with gr.Row():
                    validator_max_tokens = gr.Number(value=2048, label="Validator Max Tokens")

            with gr.Row(elem_classes="generate-section"):
                generate_btn = gr.Button("🚀 Generate Article", variant="primary", size="lg", scale=2)
                clear_btn = gr.Button("🔄 Clear", variant="secondary", size="lg", scale=1)

        # Right Column - Outputs
        with gr.Column(elem_classes="output-column"):
            gr.Markdown("## 📤 Generated Results")

            with gr.Tabs():
                with gr.Tab("🏁 Final Result"):
                    safeguarded_box = gr.Markdown(label="Safeguarded Final Result")
                    with gr.Row():
                        copy_safeguarded = gr.Button("📋 Copy", size="sm", scale=1)
                        with gr.Row():
                            download_format = gr.Dropdown(
                                choices=["Markdown (.md)", "Text (.txt)", "JSON (.json)"],
                                value="Markdown (.md)",
                                label="Download Format",
                                scale=2
                            )
                            download_safeguarded = gr.Button("⬇️ Download", size="sm", scale=1)

                with gr.Tab("📄 Article Draft"):
                    with gr.Row():
                        copy_article = gr.Button("📋 Copy", size="sm")
                    article_box = gr.Markdown(label="Original Article Draft")

                with gr.Tab("📋 Outline"):
                    with gr.Row():
                        copy_outline = gr.Button("📋 Copy", size="sm")
                    outline_box = gr.Markdown(label="Generated Outline")

                with gr.Tab("🔍 Sources"):
                    with gr.Row():
                        copy_insights = gr.Button("📋 Copy", size="sm")
                    insights_box = gr.Markdown(label="Research Insights")
                with gr.Tab("📊 Metadata"):
                    with gr.Row():
                        copy_metadata = gr.Button("📋 Copy", size="sm")
                    metadata_box = gr.Markdown(label="Validation Details")

    # Buttons Section - After outputs are defined
    with gr.Row(elem_classes="generate-section"):
        generate_btn = gr.Button("🚀 Generate Article", variant="primary", size="lg", scale=2)
        clear_btn = gr.Button("🔄 Clear", variant="secondary", size="lg", scale=1)

    # Main generation action
    generate_btn.click(
        fn=run_article_generation,
        inputs=[
            topic, model_name,
            writer_temp, rewriter_temp, validator_temp,
            target_words, tolerance, max_revisions,
            writer_max_tokens, rewriter_max_tokens, validator_max_tokens
        ],
        outputs=[safeguarded_box, article_box, outline_box, insights_box, metadata_box]
    )

    # Clear button action
    clear_btn.click(
        lambda: ("", None, None, None, None),
        outputs=[topic, safeguarded_box, article_box, outline_box, insights_box, metadata_box]
    )

    # Copy functionality only for final result
    copy_safeguarded.click(lambda x: gr.Info("Copied to clipboard!"), inputs=[safeguarded_box])

    # Download functionality with format selection
    def download_with_format(content, format_choice):
        import json
        if "Markdown" in format_choice:
            return (content, "final_result.md")
        elif "Text" in format_choice:
            return (content, "final_result.txt")
        elif "JSON" in format_choice:
            json_data = {"final_result": content}
            return (json.dumps(json_data, indent=2), "final_result.json")
        return (content, "final_result.txt")

    download_safeguarded.click(
        lambda x, fmt: download_with_format(x, fmt),
        inputs=[safeguarded_box, download_format],
        outputs=[gr.File(visible=False)]
    )

    gr.Markdown("""
    ---
    <div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">
        <p>Powered by Multi-Agent AI Architecture | Research → Write → Validate → Safeguard</p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch()
