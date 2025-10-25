import gradio as gr
from agents.report.graph import generate_report as gen_report
from agents.report.graph import planning


def llm_call(topic: str, temperature: float, outline):
    return gen_report(topic, temperature, outline)


# Enhanced custom CSS with improved colors and polish
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

.gradio-container {
    width: 70%;
    margin: auto;
    font-family: 'Inter', sans-serif;
}

/* Modern gradient buttons with depth */
.generate-btn {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    border: none;
    color: white;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
    transition: all 0.3s ease;
}

.generate-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4);
}

/* Header styling with gradient text */
.gradio-container h1 {
    background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
}

.gradio-container h2 {
    color: #64748b;
    font-weight: 500;
    font-size: 1.125rem;
    margin-top: 0;
}

/* Enhanced input field */
.gradio-container input[type="text"] {
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    transition: all 0.3s ease;
    font-size: 0.95rem;
}

.gradio-container input[type="text"]:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

/* Enhanced textbox areas */
.gradio-container textarea {
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    transition: all 0.3s ease;
}

.gradio-container textarea:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

/* Label styling */
.gradio-container label {
    color: #334155;
    font-weight: 600;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
}

/* Secondary button styling */
button[variant="secondary"] {
    background: linear-gradient(135deg, #64748b 0%, #475569 100%);
    border: none;
    color: white;
    font-weight: 600;
    box-shadow: 0 4px 15px rgba(100, 116, 139, 0.2);
    transition: all 0.3s ease;
}

button[variant="secondary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(100, 116, 139, 0.3);
}

/* Card-like appearance for main sections */
.gradio-container > div {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
}

/* Copy button enhancement */
button.copy {
    color: #6366f1;
    transition: all 0.2s ease;
}

button.copy:hover {
    color: #8b5cf6;
    transform: scale(1.05);
}

/* Temperature display with color gradient */
#temp-display {
    font-size: 2rem;
    font-weight: 700;
    text-align: center;
    padding: 1.5rem;
    border-radius: 12px;
    transition: all 0.5s ease;
    border: 3px solid;
}

#temp-display.cold {
    background: linear-gradient(135deg, #e0f2fe 0%, #bae6fd 100%);
    border-color: #0ea5e9;
    color: #0369a1;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.3);
}

#temp-display.cool {
    background: linear-gradient(135deg, #dbeafe 0%, #93c5fd 100%);
    border-color: #3b82f6;
    color: #1e40af;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
}

#temp-display.moderate {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-color: #f59e0b;
    color: #b45309;
    box-shadow: 0 4px 15px rgba(245, 158, 11, 0.3);
}

#temp-display.warm {
    background: linear-gradient(135deg, #fed7aa 0%, #fdba74 100%);
    border-color: #f97316;
    color: #c2410c;
    box-shadow: 0 4px 15px rgba(249, 115, 22, 0.3);
}

#temp-display.hot {
    background: linear-gradient(135deg, #fecaca 0%, #fca5a5 100%);
    border-color: #ef4444;
    color: #b91c1c;
    box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
}
"""

with gr.Blocks(css=custom_css) as iface:
    gr.Markdown("# 📊 Report Writer")
    gr.Markdown("## Write a report about an MLOps related topic")

    with gr.Row():
        with gr.Column(scale=3):
            topic = gr.Text(
                label="Choose the topic",
                placeholder="Enter an MLOps topic",
                max_length=25
            )
            temperature = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.7,
                step=0.1,
                label="🌡️ Temperature Control",
                info="Adjust model creativity"
            )
        with gr.Column(scale=1):
            generate = gr.Button(
                "Generate Outline",
                variant="primary",
                elem_classes="generate-btn"
            )
            temp_display = gr.HTML(
                value='<div id="temp-display" class="moderate" style="font-size: 0.85rem;'
                'padding: 0.5rem; text-align: center;">🌤️ 0.7°</div>'
            )
    with gr.Row():
        with gr.Column():
            result = gr.Textbox(
                lines=20,
                label="📄 Outline",
                placeholder="Your generated outline will appear here...",
                show_copy_button=True,
                interactive=False
            )
            with gr.Row():
                generate_report = gr.Button(
                    "Generate Report",
                    scale=1,
                    variant="primary",
                    elem_classes="generate-btn"
                )
                regenerate_outline = gr.Button(
                    "🔄 Regenerate Outline",
                    scale=1,
                    variant="secondary"
                )

        with gr.Column():
            result1 = gr.Textbox(
                lines=20,
                label="📑 Report",
                placeholder="Report will appear here...",
                show_copy_button=True
            )
            regenerate_report = gr.Button(
                "🔄 Regenerate Report",
                scale=1,
                variant="secondary"
            )

    # Event handlers - Fixed to follow proper Gradio patterns
    def update_temp_display(temp_value):
        """Update temperature display with color based on value"""
        if temp_value < 0.4:
            css_class = "cold"
            emoji = "🧊"
        elif temp_value < 0.6:
            css_class = "cool"
            emoji = "❄️"
        elif temp_value < 0.9:
            css_class = "moderate"
            emoji = "🌤️"
        elif temp_value < 1.3:
            css_class = "warm"
            emoji = "🔥"
        else:
            css_class = "hot"
            emoji = "🌋"

        return f'<div id="temp-display" class="{css_class}" style="font-size: 0.85rem; padding: 0.5rem;'
        f'text-align: center;">{emoji} {temp_value:.1f}°</div>'

    temperature.change(
        fn=update_temp_display,
        inputs=temperature,
        outputs=temp_display
    )

    def generate_and_enable_edit(topic, temp):
        """Generate outline and make it editable"""
        outline = planning(topic, temp)
        return gr.update(value=outline, interactive=True)

    generate.click(
        fn=generate_and_enable_edit,
        inputs=[topic, temperature],
        outputs=result
    )

    regenerate_outline.click(
        fn=generate_and_enable_edit,
        inputs=[topic, temperature],
        outputs=result
    )

    generate_report.click(
        fn=llm_call,
        inputs=[topic, temperature, result],
        outputs=result1
    )

    regenerate_report.click(
        fn=llm_call,
        inputs=topic,
        outputs=result1
    )

if __name__ == "__main__":
    iface.launch()
