import joblib
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

# Load the trained model
model = joblib.load("fraud_rf_model.pkl")


# Define the expected input format
class ModelInput(BaseModel):
    features: list[float]  # A list of numerical features


app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Fraud model API is running"}


@app.post("/predict")
def predict(input: ModelInput):
    X = np.array([input.features])  # Convert to 2D array
    prediction = model.predict(X)
    return {"prediction": int(prediction[0])}
