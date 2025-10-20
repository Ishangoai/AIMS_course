import os

import gradio as gr
import requests

API_URL = f"http://127.0.0.1:{os.environ.get('PORT', '8080')}"


def greet_user():
    response = requests.get(f"{API_URL}/hello")
    if response.status_code == 200:
        return response.json()["message"]
    return "Error fetching greeting"


def evaluate_expression(expression):
    response = requests.get(f"{API_URL}/evaluate", params={"expression": expression})
    if response.status_code == 200:
        return response.json()["result"]
    return "Error evaluating expression"


def register_user(username, name=None):
    response = requests.post(f"{API_URL}/register", json={"username": username, "name": name})
    if response.status_code == 200:
        return response.json()["message"]
    return "Error registering user"


def update_user(username, new_name):
    response = requests.put(f"{API_URL}/register/{username}", json={"name": new_name})
    if response.status_code == 200:
        return response.json()["message"]
    return "Error updating username"


def get_registered_users():
    response = requests.get(f"{API_URL}/register")
    if response.status_code == 200:
        return list(response.json()["users"].items())
    return "Error fetching registered users"


def delete_user(username):
    response = requests.delete(f"{API_URL}/register/{username}/delete")
    if response.status_code == 200:
        return response.json()["message"]
    return "Error deleting user"


with gr.Blocks() as app:
    gr.Markdown("# AIMS Course Gradio App")

    with gr.Tab("Greet"):
        greet_button = gr.Button("Greet Me")
        greet_output = gr.Textbox(label="Greeting", interactive=False)
        greet_button.click(fn=greet_user, inputs=None, outputs=greet_output)

    with gr.Tab("Evaluate Expression"):
        expression_input = gr.Textbox(label="Enter a mathematical expression")
        evaluate_button = gr.Button("Evaluate Expression")
        evaluate_output = gr.Textbox(label="Evaluation Result", interactive=False)
        evaluate_button.click(fn=evaluate_expression, inputs=expression_input, outputs=evaluate_output)

    with gr.Tab("Register User"):
        register_input_username = gr.Textbox(label="Enter username to register")
        register_input_name = gr.Textbox(label="Enter name (optional)")
        register_button = gr.Button("Register User")
        register_output = gr.Textbox(label="Registration Status", interactive=False)
        register_button.click(
            fn=register_user,
            inputs=[register_input_username, register_input_name],
            outputs=register_output
        )

    with gr.Tab("Update User"):
        username_options = gr.Textbox(label="Select a user to update")
        user_name_input = gr.Textbox(label="Enter your new name")
        update_button = gr.Button("Update User")
        update_output = gr.Textbox(label="Update Status", interactive=False)
        update_button.click(fn=update_user, inputs=[username_options, user_name_input], outputs=update_output)

    with gr.Tab("View Registered Users"):
        fetch_users_button = gr.Button("Fetch Registered Users")
        users_output = gr.DataFrame(label="Registered Users", interactive=False, headers=["Username", "Name"])
        fetch_users_button.click(fn=get_registered_users, inputs=None, outputs=users_output)

    with gr.Tab("Delete User"):
        delete_input = gr.Textbox(label="Enter username to delete")
        delete_button = gr.Button("Delete User")
        delete_output = gr.Textbox(label="Deletion Status", interactive=False)
        delete_button.click(fn=delete_user, inputs=delete_input, outputs=delete_output)


# import requests
