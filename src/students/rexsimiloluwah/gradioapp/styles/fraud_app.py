custom_css = """
/* ===== Global Font Scaling ===== */
body, .gradio-container, .gr-block, .gr-text, .gr-html, .gr-markdown, .gr-button, .gr-slider, .gr-number {
    font-size: 18px !important;
    line-height: 1.6 !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

.gradio-container {
    padding-top: 40px !important; /* space between top and content */
    padding-bottom: 20px !important;
}

/* ===== Background and Layout ===== */
body {
    background: #f2f7ff;
    color: #111;
}

/* ===== Tabs Styling ===== */
.gradio-tabs, .gradio-tabs button {
    font-size: 30px !important;
    font-weight: 600 !important;
    padding: 10px 20px !important;
}

.gradio-tabs button {
    color: #1a237e;
}

.gradio-tabs button.selected {
    background: #e3f2fd !important;
    border-bottom: 3px solid #1976d2 !important;
    color: #0d47a1 !important;
}

/* ===== Details (collapsible) Styling ===== */
details {
    margin-bottom: 12px;
    border-radius: 8px;
    border: 1px solid #ddd;
    padding: 10px 15px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

summary {
    font-size: 17px;
    font-weight: 600;
    color: #1a237e;
    cursor: pointer;
    outline: none;
    margin-bottom: 5px;
}

details[open] summary {
    color: #0d47a1;
}

details p, details li {
    font-size: 16px;
    line-height: 1.6;
    margin-left: 10px;
}

/* ===== Slider Accent ===== */
.orange-slider input[type='range'] {
    accent-color: #ff9800;
}
.orange-slider input[type='range']::-webkit-slider-thumb {
    background-color: #ff9800;
}
.orange-slider input[type='range']::-moz-range-thumb {
    background-color: #ff9800;
}
"""
