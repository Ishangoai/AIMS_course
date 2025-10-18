import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import random

# Mock model prediction function - replace with your actual trained model
def predict_fraud(features):
    """
    Replace this with your actual model.predict() function
    Example: return model.predict_proba([features])[0]
    """
    # Simulate prediction for demo purposes
    fraud_score = np.random.random()
    return fraud_score

# Fraud detection jokes for entertainment
FRAUD_JOKES = [
    "Why did the fraudster go to art school? To master forgery! 🎨",
    "What's a fraudster's favorite game? Hide and cheat! 🕵️",
    "Why don't fraudsters like gardening? They prefer money laundering! 💰",
    "What do you call a dishonest credit card? A con-card! 💳",
    "Why was the fraudster bad at poker? Everyone could see through their schemes! 🃏",
    "How do fraudsters stay in shape? By running scams! 🏃",
    "What's a fraudster's favorite music? Anything that's a steal! 🎵"
]

def get_random_joke():
    return random.choice(FRAUD_JOKES)

def create_gauge_chart(fraud_probability):
    """Create an animated gauge chart for fraud probability"""
    
    # Determine color based on risk level
    if fraud_probability < 0.3:
        color = "green"
        risk_level = "LOW RISK"
    elif fraud_probability < 0.7:
        color = "orange"
        risk_level = "MEDIUM RISK"
    else:
        color = "red"
        risk_level = "HIGH RISK"
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = fraud_probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': f"Fraud Risk: {risk_level}", 'font': {'size': 24, 'color': color}},
        delta = {'reference': 50, 'increasing': {'color': "red"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'lightgreen'},
                {'range': [30, 70], 'color': 'lightyellow'},
                {'range': [70, 100], 'color': 'lightcoral'}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70}}))
    
    fig.update_layout(
        paper_bgcolor = "lavender",
        font = {'color': "darkblue", 'family': "Arial"},
        height=400
    )
    
    return fig

def create_feature_importance_chart(features_dict):
    """Create a bar chart showing the most important features"""
    # Get top contributing features (simplified - in real scenario use SHAP or feature importance)
    important_features = {k: abs(v) for k, v in list(features_dict.items())[:10]}
    
    df = pd.DataFrame(list(important_features.items()), columns=['Feature', 'Magnitude'])
    df = df.sort_values('Magnitude', ascending=True)
    
    fig = px.bar(df, x='Magnitude', y='Feature', orientation='h',
                 title='Top 10 Feature Contributions',
                 labels={'Magnitude': 'Absolute Value', 'Feature': 'Feature Name'},
                 color='Magnitude',
                 color_continuous_scale='Viridis')
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def predict_transaction(time, amount, *v_features):
    """Main prediction function"""
    
    # Combine all features
    features_dict = {
        'Time': time,
        'Amount': amount
    }
    
    # Add V1-V28 features
    for i, val in enumerate(v_features, 1):
        features_dict[f'V{i}'] = val
    
    # Create feature array
    feature_array = [time, *v_features, amount]
    
    # Get prediction
    fraud_probability = predict_fraud(feature_array)
    
    # Determine verdict
    if fraud_probability > 0.7:
        verdict = "🚨 FRAUD ALERT! 🚨"
        verdict_color = "red"
        recommendation = "⛔ BLOCK this transaction immediately and contact cardholder!"
    elif fraud_probability > 0.3:
        verdict = "⚠️ SUSPICIOUS"
        verdict_color = "orange"
        recommendation = "⚡ Additional verification recommended before processing."
    else:
        verdict = "✅ LEGITIMATE"
        verdict_color = "green"
        recommendation = "✨ Transaction appears safe. Proceed normally."
    
    # Create visualizations
    gauge_fig = create_gauge_chart(fraud_probability)
    importance_fig = create_feature_importance_chart(features_dict)
    
    # Format result message
    result_html = f"""
    <div style='text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                border-radius: 15px; color: white; box-shadow: 0 10px 25px rgba(0,0,0,0.3);'>
        <h1 style='font-size: 3em; margin: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>{verdict}</h1>
        <h2 style='font-size: 1.8em; margin: 10px;'>Fraud Probability: {fraud_probability*100:.2f}%</h2>
        <p style='font-size: 1.3em; margin: 15px; padding: 15px; background: rgba(255,255,255,0.2); 
                  border-radius: 10px;'>{recommendation}</p>
    </div>
    """
    
    # Transaction details
    details = f"""
    ### 📊 Transaction Details:
    - **Amount:** ${amount:,.2f}
    - **Time:** {time:.0f} seconds from first transaction
    - **Total Features Analyzed:** 30
    
    ### 🎯 Risk Assessment:
    - **Fraud Score:** {fraud_probability*100:.2f}%
    - **Confidence Level:** {(1-abs(0.5-fraud_probability)*2)*100:.1f}%
    """
    
    return result_html, details, gauge_fig, importance_fig

# Create the Gradio interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="purple", secondary_hue="blue")) as app:
    
    # Header
    gr.HTML("""
        <div style='text-align: center; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    padding: 30px; border-radius: 15px; margin-bottom: 20px;'>
            <h1 style='color: white; font-size: 3em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);'>
                🛡️ Credit Card Fraud Detection System
            </h1>
            <p style='color: white; font-size: 1.3em; margin-top: 10px;'>
                AI-Powered Transaction Security Analysis
            </p>
        </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 📝 Transaction Information")
            
            with gr.Group():
                time_input = gr.Number(
                    label="⏱️ Time (seconds since first transaction)",
                    value=0,
                    info="Elapsed time between transactions"
                )
                
                amount_input = gr.Number(
                    label="💵 Transaction Amount ($)",
                    value=100.0,
                    info="Enter the transaction amount"
                )
            
            gr.Markdown("## 🔢 PCA Features (V1-V28)")
            gr.Markdown("*These are anonymized features from PCA transformation*")
            
            # Create V1-V28 inputs in a compact grid
            v_inputs = []
            with gr.Accordion("Feature Inputs (V1-V28)", open=False):
                for i in range(0, 28, 4):
                    with gr.Row():
                        for j in range(4):
                            if i + j < 28:
                                v_input = gr.Number(
                                    label=f"V{i+j+1}",
                                    value=0.0,
                                    scale=1
                                )
                                v_inputs.append(v_input)
            
            # Quick fill options
            with gr.Row():
                gr.Markdown("**Quick Fill:**")
            with gr.Row():
                random_btn = gr.Button("🎲 Random Values", variant="secondary", size="sm")
                zero_btn = gr.Button("0️⃣ Reset to Zero", variant="secondary", size="sm")
                sample_btn = gr.Button("📋 Sample Transaction", variant="secondary", size="sm")
            
            predict_btn = gr.Button("🔍 Analyze Transaction", variant="primary", size="lg")
            
        with gr.Column(scale=1):
            gr.Markdown("## 🎪 Fun Corner")
            joke_display = gr.Textbox(label="😄 Fraud Detection Joke", interactive=False, lines=3)
            joke_btn = gr.Button("🎭 Get Another Joke", variant="secondary")
            
            gr.Markdown("## ℹ️ System Info")
            gr.Markdown("""
            **Features:**
            - 30 input features
            - Real-time prediction
            - Risk visualization
            - Feature importance
            
            **Risk Levels:**
            - 🟢 Low: 0-30%
            - 🟡 Medium: 30-70%
            - 🔴 High: 70-100%
            """)
    
    # Results section
    gr.Markdown("---")
    gr.Markdown("## 📈 Analysis Results")
    
    result_display = gr.HTML()
    
    with gr.Row():
        with gr.Column():
            details_display = gr.Markdown()
        with gr.Column():
            gauge_plot = gr.Plot(label="Fraud Risk Gauge")
    
    importance_plot = gr.Plot(label="Feature Importance Analysis")
    
    # Event handlers
    def fill_random():
        return [np.random.randn() for _ in range(28)]
    
    def fill_zeros():
        return [0.0 for _ in range(28)]
    
    def fill_sample():
        # Sample from actual dataset range
        return [np.random.randn() * np.random.uniform(0.5, 3) for _ in range(28)]
    
    random_btn.click(fill_random, outputs=v_inputs)
    zero_btn.click(fill_zeros, outputs=v_inputs)
    sample_btn.click(fill_sample, outputs=v_inputs)
    
    joke_btn.click(get_random_joke, outputs=joke_display)
    
    predict_btn.click(
        predict_transaction,
        inputs=[time_input, amount_input] + v_inputs,
        outputs=[result_display, details_display, gauge_plot, importance_plot]
    )
    
    # Initial joke on load
    app.load(get_random_joke, outputs=joke_display)
    
    # Footer
    gr.HTML("""
        <div style='text-align: center; padding: 20px; margin-top: 30px; 
                    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    border-radius: 10px;'>
            <p style='color: white; margin: 0;'>
                ⚡ Powered by Advanced Machine Learning | 🔒 Secure Transaction Processing
            </p>
        </div>
    """)

# Launch the app
if __name__ == "__main__":
    app.launch(share=True)