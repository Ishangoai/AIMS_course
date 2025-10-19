import joblib
import numpy as np
from fastapi import FastAPI, Request

app = FastAPI()
base_dir = "/var/autofs/misc/home/eliud/Desktop/AIMS_course"
model_path = f"{base_dir}/mlruns/1/562ddd67cc6b49619f8e7aea68cd6cfd/artifacts/tuned_random_forest_v1760881924.pkl"

model = joblib.load(model_path)
features = list(model.feature_names_in_)

# print(list.modefeature_names_in)


@app.post("/predict/")
async def predict(request: Request):
    json_data = await request.json()
    try:
        input_values = [float(json_data[feat]) for feat in features]
    except KeyError as e:
        return {"error": f"Invalid input: missing feature '{e.args[0]}'"}

    input_array = np.array([input_values])
    prediction = model.predict(input_array)

    # If you want the class label, map here (optional)
    class_mapping = {0: "Not Fraud", 1: "Fraud"}
    pred_label = class_mapping.get(prediction[0], "Unknown")

    return {
        "prediction": int(prediction[0]),
        "label": pred_label
    }
