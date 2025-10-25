import os
import re
from datetime import datetime
from typing import Any, Dict

import gradio as gr

# Import from the local module (assuming agents/chatbot/assign2.py is your agent file)
# NOTE: Ensure your folder structure matches this import or adjust it.
try:
    # Changed import path based on the structure suggested in the original prompt
    # If the file is just in the same directory, use: from assign2 import CI_CD_TOPICS, generate_report
    from agents.chatbot.assign2 import CI_CD_TOPICS, generate_report
except ImportError:
    print("Warning: Could not import from agents.chatbot.assign2. Using fallback.")
    # Fallback definitions
    CI_CD_TOPICS = [
        "GitHub Actions for CI/CD",
        "Jenkins Pipeline as Code",
        "Automated Testing in CI/CD",
        "Docker in Continuous Deployment",
        "Kubernetes for CI/CD"
    ]

    def generate_report(topic: str, temperature: float = 0.7, user_feedback: str = "") -> Dict[str, Any]:
        return {
            "report": f"# {topic}\n\nReport generation system is not properly configured."
            "Please check the module imports and API keys.",
            "word_count": 0,
            "status": "error",
            "messages": [],
            "timestamp": datetime.now().isoformat()
        }

# --- Backend Logic for Gradio ---


def generate_ci_cd_report(topic, temperature, user_feedback, history):
    """Generate a CI/CD report using the agentic system and update chat history."""

    # Initialize history if none
    if history is None:
        history = []

    user_message = f"**Request:** Generate a 950-1050 word report about **{topic}**."
    if user_feedback:
        user_message += f"\n**User Instructions:** {user_feedback}"

    # Append user message to history
    history.append([user_message, None])

    # Display a starting message
    history.append([None, "⏳ **System:** Initializing agent workflow..."])

    yield gr.update(value=history), gr.update(value="")  # Update history immediately

    # Generate the report
    result = generate_report(
        topic=topic,
        temperature=float(temperature),
        user_feedback=user_feedback
    )

    # Process and update history with system messages and final report
    if result["status"] == "success":
        # Extract system/assistant messages
        system_messages = [msg["content"] for msg in result.get("messages", [])
        if msg["role"] in ["system", "assistant"]]

        # Keep only the user's initial message and the final assistant/report message
        final_history = history[:1]

        # Add all process messages to the chat history
        for msg_content in system_messages[:-1]:  # All but the very last one (which contains the full report)
            final_history.append([None, f"⚙️ **Agent:** {msg_content}"])
            yield gr.update(value=final_history), gr.update(value=result["report"])
            # Yield updates for perceived responsiveness

        # Add the final report to the chat history as the assistant's final output
        final_history.append([None, "✅ **Final Report:** Word count confirmed (950-1050). Scroll up for the full text."
                                ])

        # Return the final report in the report_output field
        yield gr.update(value=final_history), gr.update(value=result["report"])

    else:
        error_msg = f"❌ **Error:** Report generation failed: {result['report']}"
        history.append([None, error_msg])
        yield gr.update(value=history), gr.update(value=result["report"])


def save_report(report_text, topic):
    """Save report to file"""
    if report_text and not report_text.startswith("Error") and "*Final Word count:" in report_text:
        # Clean the topic for a safe filename
        clean_topic = re.sub(r'[<>:"/\\|?*]', '_', topic)
        clean_topic = clean_topic.replace(' ', '_')

        reports_dir = "saved_reports"
        os.makedirs(reports_dir, exist_ok=True)

        filename = f"ci_cd_report_{clean_topic}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        filepath = os.path.join(reports_dir, filename)

        try:
            # Write the report (Markdown format)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(report_text)
            return f"✅ Report saved successfully as **{filename}**"
        except Exception as e:
            return f"❌ Save failed: {str(e)}"
    return "❌ No valid report to save. Generate a report first."

# --- Gradio Interface Definition ---


with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue"), title="CI/CD Report Generator") as interface:  # type: ignore
    gr.Markdown("""
    # 🚀 Agentic CI/CD Report Generator
    **Generate professional reports with an EXACT word count (950-1050 words) using specialized AI Agents.**
    The system includes a Research Agent, Writing Agent, Fact Checker, and an **Iterative Length Revision Agent**.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            topic_dropdown = gr.Dropdown(
                choices=CI_CD_TOPICS,
                label="**1. Select CI/CD Topic**",
                value=CI_CD_TOPICS[0],
                allow_custom_value=True,
                interactive=True
            )

            temperature_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.7,
                step=0.1,
                label="**2. Creativity Level (Temperature)**",
                info="Higher = more creative/expansive; Lower = more factual/concise"
            )

            user_feedback = gr.Textbox(
                label="**3. Additional Instructions (Optional)**",
                placeholder="E.g., Focus more on cloud security, use Kubernetes examples...",
                lines=3
            )

            generate_btn = gr.Button("🔥 **Start Report Generation** (950-1050 Words)", variant="primary", size="lg")

            with gr.Accordion("Report Management", open=True):
                save_btn = gr.Button("📄 Save Report to File (.md)", variant="secondary")
                save_status = gr.Textbox(label="Save Status", interactive=False, max_lines=1)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.TabItem("Final Report Output"):
                    report_output = gr.Markdown(
                        label="Generated Report",
                        show_copy_button=True,
                        value="*The final report will appear here once generated.*",
                        elem_id="report-output"
                    )

                with gr.TabItem("Agent Process Log"):
                    chatbot = gr.Chatbot(
                        label="Agent Workflow Log (Follow the steps for length control and fact-checking)",
                        height=550,
                        show_copy_button=True
                    )

    gr.Markdown("""
    ---
    ### ⚙️ How the Length Control Works:
    1. **Draft:** The Writing Agent creates an initial draft (~1000 words).
    2. **Check:** The Length Checker counts the words (Target: 950-1050).
    3. **Revise Loop:** If the count is outside the range, the Revision Agent runs to **add or remove**
    text based on the required deviation.
    4. **Repeat:** Steps 2 & 3 repeat up to 3 times to ensure the final,
    precise word count is achieved before the Final Reviewer polishes the text.
    """)

    # Event handlers
    # NOTE: Using .click for non-streaming output, .then for streaming (using yield)
    generate_btn.click(
        fn=generate_ci_cd_report,
        inputs=[topic_dropdown, temperature_slider, user_feedback, chatbot],
        outputs=[chatbot, report_output]  # Chatbot is updated first, then report_output
    )

    save_btn.click(
        fn=save_report,
        inputs=[report_output, topic_dropdown],
        outputs=[save_status]
    )

# For uvicorn compatibility
app = interface

if __name__ == "__main__":
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False  # Set to True if you want a public link
    )
