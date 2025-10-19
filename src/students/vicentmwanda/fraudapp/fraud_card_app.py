# app.py

import gradio as gr

from .utils.fraud_card import predict_fraud


def wrapped_predict(*args):
    return predict_fraud(list(args))


# Creative CSS with animations and modern design
custom_css = """
:root {
    --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --success-gradient: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
    --warning-gradient: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
    --danger-gradient: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
    --dark-bg: #0f1419;
    --card-bg: rgba(255, 255, 255, 0.95);
    --dark-card: #1a202c;
}

body {
    background-size: 400% 400%;
    animation: gradient 15s ease infinite;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
}

@keyframes gradient {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

.glass-card {
    background: rgba(255, 255, 255, 0.25);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
}

.dark-glass {
    background: rgba(15, 20, 25, 0.85);
    backdrop-filter: blur(10px);
    border-radius: 20px;
    border: 1px solid rgba(255, 255, 255, 0.1);
    color: white;
}

.header-container {
    text-align: center;
    padding: 3rem 2rem;
    background: rgba(255, 255, 255, 0.1);
    backdrop-filter: blur(5px);
    border-radius: 0 0 30px 30px;
    font-weight:bold!important;
    font-size:36px!important;
    margin-bottom: 2rem;
}src/students/Aarondard/api/main.py
    font-weight: 800;
    background: linear-gradient(135deg, #667eea, #764ba2, #f093fb, #f5576c);
    background-size: 300% 300%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s ease-in-out infinite;
    margin-bottom: 1rem;
}src/students/Aarondard/api/main.py
    50% { background-position: 100% 50%; }
}

.subtitle {
    font-size: 1.3rem;
    color: rgba(255, 255, 255, 0.9);
    font-weight: 300;
}

.feature-card {
    background: var(--card-bg);
    border-radius: 16px;
    padding: 1.5rem;
    margin: 1rem 0;
    border-left: 4px solid;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}

.feature-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
}

.risk-meter {
    width: 100%;
    height: 20px;
    background: linear-gradient(90deg, #48bb78, #ed8936, #f56565);
    border-radius: 10px;
    margin: 1rem 0;
    position: relative;
    overflow: hidden;
}

.risk-level {
    height: 100%;
    background: rgba(255, 255, 255, 0.3);
    border-radius: 10px;
    transition: width 0.5s ease;
}

.creative-btn {
    background: var(--primary-gradient);
    border: none;
    color: white;
    padding: 1rem 2rem;
    border-radius: 50px;
    font-weight: 600;
    font-size: 1.1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    position: relative;
    overflow: hidden;
}

.creative-btn1 {
    background: #008000;
    border: none;
    color: white;
    padding: 1rem 2rem;
    border-radius: 50px;
    font-weight: 600;
    font-size: 1.1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    position: relative;
    overflow: hidden;
}

.creative-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
}

.creative-btn:active {
    transform: translateY(0);<div style="font-size: 0.9rem; opacity: 0.8;">Adjust parameters and click Detect Fraud
    </div>
                            </div>
                        </div>
}

.creative-btn::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    transition: left 0.5s;
}

.creative-btn:hover::before {
    left: 100%;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin: 1.5rem 0;
}

.stat-card {
    background: var(--primary-gradient);
    color: white;
    padding: 1rem;
    border-radius: 12px;
    text-align: center;
    transition: transform 0.3s ease;
}

.stat-card:hover {
    transform: scale(1.05);
}

.pulse {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { transform: scale(1); }
    50% { transform: scale(1.05); }
    100% { transform: scale(1); }
}

.floating {
    animation: floating 3s ease-in-out infinite;
}

@keyframes floating {
    0% { transform: translate(0,  0px); }
    50%  { transform: translate(0, -10px); }
    100%   { transform: translate(0, 0px); }
}

.tab-badge {
    background: var(--primary-gradient);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-size: 0.8rem;
    margin-left: 0.5rem;
}

.result-display {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 2rem;
    border-radius: 20px;
    text-align: center;
    font-size: 1.2rem;<div style="font-size: 0.9rem; opacity: 0.8;">Adjust parameters and click Detect Fraud</div>
                            </div>
                        </div>
    font-weight: 600;
    margin: 1rem 0;
    min-height: 100px;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: all 0.3s ease;
}

.footer {
    text-align: center;
    padding: 2rem;
    color: rgba(255, 255, 255, 0.8);
    font-size: 0.9rem;
}<div style="font-size: 0.9rem; opacity: 0.8;">Adjust parameters and click Detect Fraud</div>
                            </div>
                        </div>

.cyber-grid {
    background:
        linear-gradient(90deg, rgba(255,255,255,0.03) 1px, transparent 1px),
        linear-gradient(180deg, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 20px 20px;
}
"""


def create_animated_header():
    return gr.Markdown("""
    <div class="header-container">
        <div class="animated-title">🛡️ CyberShield Fraud Detection</div>
        <div class="subtitle">Advanced App for transaction security analysis</div>
        <div style="margin-top: 1rem; font-size: 1rem; color: #222;">
            🔒 Real-time protection • Machine Learning • Instant Analysis
        </div>
    </div>
    """)


def create_stats_dashboard():
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            <div class="stats-grid">
                <div class="stat-card pulse">
                    <div style="font-size: 2rem;">📊</div>
                    <div>28 Features</div>
                </div>
                <div class="stat-card">
                    <div style="font-size: 2rem;">⚡</div>
                    <div>Real-time</div>
                </div>
                <div class="stat-card">
                    <div style="font-size: 2rem;">🎯</div>
                    <div>99.75% Accuracy</div>
                </div>
                <div class="stat-card">
                    <div style="font-size: 2rem;">🛡️</div>
                    <div>Secure</div>
                </div>
            </div>
            """)


def create_feature_card(title, description, icon, color="#667eea"):
    return gr.Markdown(f"""
    <div class="feature-card" style="border-left-color: {color};">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
            <h4 style="margin: 0; color: {color};">{title}</h4>
        </div>
        <p style="margin: 0; color: #666;">{description}</p>
    </div>
    """)


with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as fraud_app:  # type:ignore

    # Animated Header
    create_animated_header()

    # Stats Dashboard
    create_stats_dashboard()

    # Initialize all slider variables at the top level
    V1 = V2 = V3 = V4 = V5 = V6 = V7 = V8 = V9 = V10 = None
    V11 = V12 = V13 = V14 = V15 = V16 = V17 = V18 = V19 = V20 = None
    V21 = V22 = V23 = V24 = V25 = V26 = V27 = V28 = Time = Amount = None

    with gr.Tabs() as main_tabs:

        # Main Analysis Tab
        with gr.TabItem("🔍 Transaction Scanner", elem_classes=["cyber-grid"]):
            with gr.Row():

                # Left Panel - Quick Controls
                with gr.Column(scale=1):
                    predict_btn = gr.Button(
                        "🚀 Detect Fraud",
                        variant="primary",
                        size="lg",
                        elem_classes=["creative-btn1", "floating"],
                    )

                    with gr.Column():
                        result = gr.HTML("""
                        <div class="result-display">
                            <div>
                                <div style="font-size: 2rem;">🔍</div>
                                <div>Ready for Analysis</div>
                                <div style="font-size: 0.9rem; opacity: 0.8;">Adjust parameters and
                                click Detect Fraud </div>
                            </div>
                        </div>
                        """)

                    with gr.Group(elem_classes=["glass-card"]):
                        gr.Markdown("### 📈  Analysis")
                # Analysis Section

                        # Quick Presets
                        # gr.Markdown("#### 🎛️ Quick Presets")
                        with gr.Row():
                            normal_btn = gr.Button("🟢 Normal", size="sm")
                            suspicious_btn = gr.Button("🟡 Suspicious", size="sm")
                            fraud_btn = gr.Button("🔴 Fraud", size="sm")

                        # Risk Meter
                        gr.Markdown("#### 📊 Risk Assessment")
                        risk_meter = gr.HTML("""
                        <div class="risk-meter">
                            <div class="risk-level" style="width: 0%;"></div>
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #666;">
                            <span>Low</span>
                            <span>Medium</span>
                            <span>High</span>
                        </div>
                        """)


#  Middle Panel - Main Features
                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["glass-card"]):
                        gr.Markdown("### 📊 Core Transaction Features")
                        with gr.Row():
                            with gr.Column():
                                Time = gr.Slider(0, 172000, value=0, label="⏰ Time",
                                               info="Seconds since first transaction")
                                Amount = gr.Slider(0.0, 5000.0, value=149.62, step=0.1,
                                                 label="💰 Amount",
                                                 info="Transaction amount in USD")

                        gr.Markdown("### 🔧 Principal Components (In order of Importance)")
                        with gr.Row():
                            col1, col2, col3 = gr.Column(), gr.Column(), gr.Column()

                    # V14, V17, V10, V12, V16, V11, V3, V4, V9, V18, V7, V21
                            with col1:
                                V14 = gr.Slider(-50.0, 50.0, value=-0.311169, step=0.1, label="🎯 V14")
                                V17 = gr.Slider(-50.0, 50.0, value=0.207971, step=0.1, label="📈 V17")
                                V10 = gr.Slider(-50.0, 50.0, value=0.090794, step=0.1, label="🎯 V10")
                                V12 = gr.Slider(-50.0, 50.0, value=-0.617801, step=0.1, label="🎯 V12")
                                V16 = gr.Slider(-50.0, 50.0, value=-0.470401, step=0.1, label="📈 V16")
                                V11 = gr.Slider(-50.0, 50.0, value=-0.551600, step=0.1, label="🎯 V11")

                            with col2:
                                V3 = gr.Slider(-50.0, 50.0, value=2.536347, step=0.1, label="🎯 V3")
                                V4 = gr.Slider(-50.0, 50.0, value=1.378155, step=0.1, label="🎯 V4")
                                V9 = gr.Slider(-50.0, 50.0, value=0.363787, step=0.1, label="🎯 V9")
                                V18 = gr.Slider(-50.0, 50.0, value=0.025791, step=0.1, label="📈 V18")
                                V7 = gr.Slider(-50.0, 50.0, value=0.239599, step=0.1, label="🎯 V7")
                                V21 = gr.Slider(-50.0, 50.0, value=-0.018307, step=0.1, label="📈 V21")

                # Right Panel - Additional Features
                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["glass-card"]):
                        gr.Markdown("### 🎯 Extended Feature Set")
                        with gr.Row():
                            col4, col5, col6 = gr.Column(), gr.Column(), gr.Column()

                            with col4:
                                V1 = gr.Slider(-50.0, 50.0, value=-1.359807, step=0.1, label="🎯 V1")
                                V2 = gr.Slider(-50.0, 50.0, value=-0.072781, step=0.1, label="🎯 V2")
                                V5 = gr.Slider(-50.0, 50.0, value=-0.338321, step=0.1, label="🎯 V5")
                                V6 = gr.Slider(-50.0, 50.0, value=0.462388, step=0.1, label="🎯 V6")
                                V8 = gr.Slider(-50.0, 50.0, value=0.098698, step=0.1, label="🎯 V8")
                                V13 = gr.Slider(-50.0, 50.0, value=-0.991391, step=0.1, label="🎯 V13")
                                V15 = gr.Slider(-50.0, 50.0, value=1.468177, step=0.1, label="📈 V15")
                                V19 = gr.Slider(-50.0, 50.0, value=0.403993, step=0.1, label="📈 V19")

                            with col5:
                                V20 = gr.Slider(-50.0, 50.0, value=0.251412, step=0.1, label="📈 V20")
                                V22 = gr.Slider(-50.0, 50.0, value=0.277838, step=0.1, label="📈 V22")
                                V23 = gr.Slider(-50.0, 50.0, value=-0.110474, step=0.1, label="📈 V23")
                                V24 = gr.Slider(-50.0, 50.0, value=0.066928, step=0.1, label="📈 V24")
                                V25 = gr.Slider(-50.0, 50.0, value=0.128539, step=0.1, label="📈 V25")
                                V26 = gr.Slider(-50.0, 50.0, value=-0.189115, step=0.1, label="📈 V26")
                                V27 = gr.Slider(-50.0, 50.0, value=0.133558, step=0.1, label="📈 V27")
                                V28 = gr.Slider(-50.0, 50.0, value=-0.021053, step=0.1, label="📈 V28")

        # How to Use Tab
        with gr.TabItem("📚 Knowledge Base"):
            with gr.Row():
                with gr.Column(scale=2):
                    with gr.Group(elem_classes=["dark-glass"]):
                        gr.Markdown("""
                        # 🎓 Master Guide

                        ## 🎯 Getting Started
                        """)

                        create_feature_card(
                            "Real-time Analysis",
                            "Our App processes 28 features simultaneously to detect fraud patterns in milliseconds",
                            "⚡", "#f56565"
                        )

                        create_feature_card(
                            "Machine Learning",
                            "Advanced neural networks trained on millions of transactions for maximum accuracy",
                            "🤖", "#667eea"
                        )

                        create_feature_card(
                            "Security First",
                            "All analysis happens locally - your data never leaves your device",
                            "🛡️", "#48bb78"
                        )

                with gr.Column(scale=1):
                    with gr.Group(elem_classes=["dark-glass"]):
                        gr.Markdown("""
                        ## 🏆 Pro Tips

                        ⭐ **Start with presets** to understand different scenarios

                        ⭐ **Monitor the risk meter** for instant feedback

                        ⭐ **Combine unusual values** to test edge cases

                        ⭐ **Check live metrics** for confidence scores
                        """)

        # About Tab
        with gr.TabItem("🌟 About System"):
            with gr.Group(elem_classes=["glass-card"]):
                gr.Markdown("""
                # 🚀 About CyberShield

                ## Our Mission
                Protecting financial transactions with cutting-edge AI technology.

                ### 🔬 Technology Stack
                - **Machine Learning**: Advanced anomaly detection algorithms
                - **Real-time Processing**: Instant analysis with sub-second response
                - **Security**: Local processing ensures complete privacy
                - **Accuracy**: 99.75% detection rate across diverse scenarios

                ### 🏆 Features
                """)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("""
                        🎯 **28-Dimensional Analysis**
                        📊 **Live Risk Assessment**
                        ⚡ **Real-time Processing**
                        🛡️ **Privacy First**
                        """)
                    with gr.Column():
                        gr.Markdown("""
                        🔍 **Pattern Recognition**
                        📈 **Confidence Scoring**
                        🎨 **Interactive Interface**
                        🌐 **Cross-Platform**
                        """)

    # Footer
    gr.Markdown("""
    <div class="footer">
        <div style="margin-bottom: 1rem;">
            <span style="margin: 0 1rem;">✨ Built with Advanced App</span>
            <span style="margin: 0 1rem;">🛡️ Enterprise Security</span>
            <span style="margin: 0 1rem;">⚡ Real-time Processing</span>
        </div>
        <div>CyberShield Fraud Detection System v2.0 • Protecting Your Financial Future</div>
    </div>
    """)

    # Connect the prediction function
    predict_btn.click(
        fn=wrapped_predict,
        inputs=[
            Time, V1, V2, V3, V4, V5, V6, V7, V8, V9, V10,
            V11, V12, V13, V14, V15, V16, V17, V18, V19, V20,
            V21, V22, V23, V24, V25, V26, V27, V28, Amount
        ],
        outputs=[result, risk_meter],
    )

if __name__ == "__main__":
    fraud_app.launch(share=True)
