import gradio as gr

from utils.gem import do_report  # your LLM function


# --- Your chatbot logic ---
def chat_fn(user_message, history):
    """
    Calls your custom LLM (do_report) and updates the chat history and Markdown.
    """
    # Call your LLM to generate a report
    response = do_report(user_message)

    # Extract the final assistant reply from the response object
    report_text = response["review_data"]["final_text"]

    # Format nicely for Markdown
    markdown_response = f"### 🤖 AI Report on: {user_message}\n\n{report_text}"

    # Add to chat history if you want
    history.append((user_message, markdown_response))

    # Return both chat history and Markdown text
    return history, markdown_response


# --- Gradio App Layout ---
css = """
/* Overall app background */
.gradio-container {
    background: linear-gradient(135deg, #001F3F, #003566, #001845);
    color: #FFFFFF;
    font-family: 'Poppins', sans-serif;
}

/* Card styling for chatbot and report */
.gr-box, .gr-chatbot {
    background: rgba(255, 255, 255, 0.08) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

/* Markdown text */
.custom-center .gr-markdown {
    text-align: center !important;
    font-size: 1.1rem !important;
    color: white !important;
}

/* Header markdown */
h1, h2, h3 {
    color: #FFD60A !important;
}

/* Buttons */
button {
    border-radius: 10px !important;
    background: linear-gradient(90deg, #0077B6, #0096C7);
    color: white !important;
    font-weight: bold !important;
    transition: 0.3s ease;
    border: none !important;
}
button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #00B4D8, #48CAE4);
}

/* Textbox */
textarea, input {
    background: rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    color: white !important;
}

/* Download button alignment */
.custom-center {
    text-align: center;
}
"""

with gr.Blocks(title="ML Reporter",
               css=css) as llm_chat:

    gr.Markdown("# 🧠 Ask your favorite topic in MLOps, CI/CD, Gradio, or Machine Learning 😄",
                elem_classes="custom-center")

    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot(label="Simple AI Assistant")

        with gr.Column():
            result_display = gr.Markdown("## Your formatted report will appear here ✅✅✅",
                                         elem_classes="custom-center")

    with gr.Row():
        with gr.Column():
            topic_prompt = gr.Textbox(
                label="Enter your topic here",
                placeholder="E.g., Explain MLOps.",
                lines=2,
            )
        with gr.Column():
            submit_btn = gr.Button("Generate Report")
            download = gr.DownloadButton(label="Download as .md", elem_classes="custom-center")
            download.click(
                fn=lambda md: md,
                inputs=result_display,
            )

    # --- Connect interactions ---
    submit_btn.click(
        fn=chat_fn,
        inputs=[topic_prompt, chatbot],
        outputs=[chatbot, result_display],
    )

    topic_prompt.submit(
        fn=chat_fn,
        inputs=[topic_prompt, chatbot],
        outputs=[chatbot, result_display],
    )

# --- Run the app ---
if __name__ == "__main__":
    llm_chat.launch()
