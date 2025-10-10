import os

import gradio as gr
import mlflow
import pandas as pd

# Configuration
MODEL_NAME = "tuned-temp-forecaster"
MODEL_STAGE = "Production"

# Construct an absolute path to the database file.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(SCRIPT_DIR, "..", "..", "..", "..", "..", "mlflow_local_tracking.db")

# Set the tracking URI to find the local MLflow database
mlflow.set_tracking_uri(f"sqlite:///{os.path.abspath(DB_PATH)}")

# Construct the model URI for the registry
model_uri = f"models:/{MODEL_NAME}/{MODEL_STAGE}"

# Load the model into memory
model = mlflow.pyfunc.load_model(model_uri)


# Prediction Function
# This function is what Gradio will call when a user interacts with the UI
def predict_temperature(lagged_temp):

    # Create a Pandas DataFrame that matches the model's expected input format
    input_df = pd.DataFrame(
        data={"t2m_celsius_lag1": [lagged_temp]}
    )
    input_df['t2m_celsius_lag1'] = input_df['t2m_celsius_lag1'].astype('float32')

    prediction_result = model.predict(input_df)

    # The result is typically a NumPy array or list; get the first element
    predicted_value = prediction_result[0]

    # Format the output for display
    return f"{predicted_value:.2f} °C"


# Gradio Interface Definition

with gr.Blocks() as iface:
    gr.Markdown(
        """
        # Temperature Forecaster
        Enter the temperature from the previous time step (°C) to predict the next temperature.
        This model is loaded directly from the 'Production' stage in the MLflow Model Registry.
        """
    )
    with gr.Row():
        input_temp = gr.Number(
            label="Previous Temperature (°C)",
            value=10.0  # Default example value
        )
        output_temp = gr.Textbox(
            label="Predicted Temperature",
            interactive=False  # User cannot edit this box
        )

    # Button to trigger the prediction
    predict_btn = gr.Button("Predict")
    predict_btn.click(
        fn=predict_temperature,
        inputs=input_temp,
        outputs=output_temp
    )

# Launch the App
if __name__ == "__main__":
    print("Launching Gradio app to interact with the model")
    iface.launch(server_name="0.0.0.0", server_port=7861)
