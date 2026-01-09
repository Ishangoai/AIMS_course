import gradio as gr

# Fonction simple pour tester
def greet(name):
    return f"Hello, {name}! 🎬"

# Crée l'app Gradio
with gr.Blocks(title="🎬 TEST MOVIE APP") as app_scale:
    gr.Markdown("""
    # 🎬 Dummy Movie App
    This is just a simple test app for deployment.
    """)
    
    with gr.Row():
        name_input = gr.Textbox(label="Your Name", placeholder="Type your name here...")
        greet_button = gr.Button("Say Hello")
        output = gr.Textbox(label="Output")
    
    # Connecter le bouton
    greet_button.click(fn=greet, inputs=name_input, outputs=output)

# Lancer localement si besoin
if __name__ == "__main__":
    app_scale.launch()
