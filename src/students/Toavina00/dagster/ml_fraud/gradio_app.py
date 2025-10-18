import os

import gradio as gr
import mlflow
import mlflow.sklearn as ms
import numpy as np

# Configuration
MODEL_NAME = "tuned-fraud-detector"
MODEL_STAGE = "Production"

# Set the tracking URI to find the local MLflow database
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..", "mlflow_local_tracking.db")

# Set the tracking URI to find the local MLflow database
mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(DB_PATH)}")

# Construct the model URI for the registry
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

# Load the model into memory
model = ms.load_model(model_uri)


# Prediction Function
# This function is what Gradio will call when a user interacts with the UI
def predict_fn(*inputs):
    # Create a Pandas DataFrame that matches the model's expected input format
    model_input = np.array(list(inputs))
    prediction_class = model.predict(model_input)  # pyright: ignore

    # The result is typically a NumPy array or list; get the first element
    predicted_class = int(prediction_class[0])

    # Format the output for display
    return f"{'Genuine' if predicted_class == 0 else 'Fraud'}"


column_names = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]

# Gradio Interface Definition
iface = gr.Interface(fn=predict_fn, inputs=[gr.Number(value=0.0, label=l) for l in column_names], outputs=gr.Text())
# Launch the App
if __name__ == "__main__":
    print("Launching Gradio app to interact with the model")
    iface.launch(server_name="0.0.0.0", server_port=7861)
