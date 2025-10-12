import gradio as gr
from transformers import pipeline

pipe = pipeline("translation", model="Helsinki-NLP/opus-mt-en-es")


def predict(text):
    return pipe(text)[0]["translation_text"]


demo = gr.Interface(
    fn=predict,
    inputs='text',
    outputs='text',
)

demo.launch()
