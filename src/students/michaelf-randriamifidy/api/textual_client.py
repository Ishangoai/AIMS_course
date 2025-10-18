import os

import pandas as pd
import requests


def consume_predict_api(api_url: str, file_path: str):
    # Open the CSV file to upload
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, "text/csv")}
        response = requests.post(api_url, files=files)

    if response.status_code == 200:
        # Save returned CSV file (predictions)
        output_filename = f"predictions_{os.path.basename(file_path)}"
        with open(output_filename, "wb") as out_file:
            out_file.write(response.content)
        print(f"Prediction file saved as: {output_filename}")

        if input("Display sample predictions ? [y/n]") == 'y':
            df = pd.read_csv(output_filename)
            print(df.head(min(10, df.shape[0])))

    else:
        # Handle errors (bad input, etc.)
        print(f"Failed to get predictions: {response.status_code} - {response.text}")


if __name__ == "__main__":
    api_endpoint = "http://127.0.0.1:8000/predict"  # Change if needed
    input_csv = "your_input_file.csv"  # Change to your CSV file path

    if os.path.isfile(input_csv):
        consume_predict_api(api_endpoint, input_csv)
    else:
        print("Input file does not exist.")
