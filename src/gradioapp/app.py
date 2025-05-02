import gradio as gr
import requests

API_URL = "http://localhost:8000"  # Adjust the URL if your FastAPI app runs on a different host/port


def greet_user(username):
    response = requests.get(f"{API_URL}/hello")
    if response.status_code == 200:
        return response.json()["message"]
    return "Error fetching greeting"


def update_username(new_username):
    response = requests.put(f"{API_URL}/user", json={"username": new_username})
    if response.status_code == 200:
        return response.json()["message"]
    return "Error updating username"


def evaluate_expression(expression):
    response = requests.post(f"{API_URL}/evaluate", json={"expression": expression})
    if response.status_code == 200:
        return response.json()["result"]
    return "Error evaluating expression"


with gr.Blocks() as app:
    gr.Markdown("# AIMS Course Gradio App")
    with gr.Row():
        username_input = gr.Textbox(label="Enter your username")
        update_button = gr.Button("Update Username")
        update_output = gr.Textbox(label="Update Status", interactive=False)

        update_button.click(fn=update_username, inputs=username_input, outputs=update_output)

    greet_button = gr.Button("Greet Me")
    greet_output = gr.Textbox(label="Greeting", interactive=False)

    greet_button.click(fn=greet_user, inputs=None, outputs=greet_output)

    expression_input = gr.Textbox(label="Enter a mathematical expression")
    evaluate_button = gr.Button("Evaluate Expression")
    evaluate_output = gr.Textbox(label="Evaluation Result", interactive=False)

    evaluate_button.click(fn=evaluate_expression, inputs=expression_input, outputs=evaluate_output)

# app.launch()
