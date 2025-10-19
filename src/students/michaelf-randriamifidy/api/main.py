import csv
import os
import shutil
import textwrap

import gradio as gr

# from agents.chatbot.llm_gradio import llm_chat
from api.models import UpdateUserRequest, UserRequest
from api.safe_eval import safe_eval
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import FileResponse
from gradioapp.app import app as demo
from gradioapp.fraud_app import fraud_detection_app as fraud
from gradioapp.heart_disease_app import heart_app
from gradioapp.image_editor_app import image_transformation
from gradioapp.utils.fraud_detection import is_valid_csv_file, predict_fraud

app = FastAPI(
    title="AIMS Course API",
    description=textwrap.dedent("""
    ## Mounted Apps
    ----
    1. [**General Gradio Demo**](/gradio/)
    2. [**Heart Disease Prediction App**](/heart-disease/)
    3. [**Simple LLM Chatbot**](/llm-chat/)
    4. [**Image Transformation**](/image-transformation/)
    5. [**Fraud Detection Model**](/fraud-detection/)
    -----
    """),
    version="1.0.0",
    contact={"name": "Support Team", "email": "vincent@ishango.ai"},
    redirect_slashes=False,
)

# Global variable to store the usernames
current_user = os.environ.get("GITHUB_USER", "default")
users = {}


@app.get("/", include_in_schema=False)
def root():
    """
    Redirect the root path `/` to the Swagger UI documentation.
    """
    return get_swagger_ui_html(openapi_url="/openapi.json", title="AIMS Course API Docs")


@app.get("/hello", summary="Greet the user", description="Returns a greeting message.")
def hello():
    return {"message": f"Hello from {current_user}! Ready!"}


@app.get(
    "/evaluate",
    summary="Evaluate an expression",
    description="Evaluates a Math expression provided in the request body.",
    response_description="The result of the evaluated expression.",
)
def evaluate(expression: str):
    """
    Evaluate the given arguments and return the result.
    """
    try:
        # Evaluate the expression
        # print(args.expression)
        # print(type(args.expression))
        result = safe_eval(expression)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


@app.post("/register", summary="Register a new user", description="Registers a new user with the given username.")
def register_user(request: UserRequest):
    """
    Register a new user with the given username.
    """
    # Extract the username from the request
    username = request.username
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")

    users[username] = request.model_dump().get("name", None)
    return {"message": f"User {username} registered successfully"}


@app.get("/register", summary="Get registered users", description="Returns a list of registered users.")
def get_registered_users():
    """
    Get the list of registered users.
    """
    # Here you would typically fetch the users from a database
    return {"users": users}


@app.get("/register/{username}", summary="Get user details", description="Returns details of a specific user.")
def get_user_details(username: str):
    """
    Get the details of a specific user.
    """
    # Here you would typically fetch the user from a database
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    return {"username": username, "name": users[username]}


@app.delete(
    "/register/{username}/delete",
    summary="Delete a user",
    description="Deletes a user with the given username."
)
def delete_user(username: str):
    """
    Delete a user with the given username.
    """
    # Here you would typically delete the user from a database
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    del users[username]
    return {"message": f"User {username} deleted successfully"}


@app.put("/register/{username}", summary="Update user details", description="Updates the details of a specific user.")
def update_user_details(username: str, request: UpdateUserRequest):
    """
    Update the details of a specific user.
    """
    # Here you would typically update the user in a database
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    users[username] = request.model_dump().get("name", None)
    return {"message": f"User {username} updated successfully"}


@app.post("/predict", summary="Predict user input features")
# async def make_predictions(file: UploadFile = File(...)):
#     """
#     Prediction API
#     """
#     FOLDER = os.getcwd()
#     file_path = os.path.join(FOLDER, file.filename)
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)
#     # check input features file validity
#     if not is_valid_csv_file(file_path):
#         raise HTTPException(status_code=400, detail="Invalid file input, please upload a valid csv file")
#     # Run prediction
#     predictions = predict_fraud(file_path)
#     if predictions is None:
#         raise HTTPException(status_code=400, detail="Prediction failed: input was invalid.")
#     # Save predictions to CSV
#     file_name = os.path.basename(file.filename)
#     result_filename = f"prediction_{file_name}.csv"
#     result_path = os.path.join(FOLDER, file_name)
#     with open(result_path, "w", newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(["sample_index", "prediction"])
#         idx = 1
#         for pred in predictions:
#             if pred == 0:
#                 writer.writerow([idx, "legitimate"])
#             else:
#                 writer.writerow([idx, "fraud"])
#             idx += 1
#     return FileResponse(
#         path=result_path,
#         filename=result_filename,
#         media_type="text/csv"
#     )
async def make_predictions(file: UploadFile = File(...)):
    """
    Prediction API
    """
    # Ensure filename is not None
    if file.filename is None:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    folder = os.getcwd()
    file_path = os.path.join(folder, file.filename)

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Check input CSV validity
    if not is_valid_csv_file(file_path):
        raise HTTPException(status_code=400, detail="Invalid file input, please upload a valid CSV file.")

    # Run prediction
    predictions = predict_fraud(file_path)
    if predictions is None:
        raise HTTPException(status_code=400, detail="Prediction failed: input was invalid.")

    # Save predictions to CSV
    file_name = os.path.basename(file.filename)
    result_filename = f"prediction_{file_name}.csv"
    result_path = os.path.join(folder, result_filename)

    with open(result_path, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["sample_index", "prediction"])
        for idx, pred in enumerate(predictions, start=1):
            writer.writerow([idx, "legitimate" if pred == 0 else "fraud"])

    return FileResponse(
        path=result_path,
        filename=result_filename,
        media_type="text/csv"
    )


fraud_detector: gr.Blocks = fraud.build_interface()

gr.mount_gradio_app(app, demo, path="/gradio")
gr.mount_gradio_app(app, heart_app, path="/heart-disease")
# gr.mount_gradio_app(app, llm_chat, path="/llm-chat")
gr.mount_gradio_app(app, image_transformation, path="/image-transformation")
gr.mount_gradio_app(app, fraud_detector, path="/fraud-detection")
