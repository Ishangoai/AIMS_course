import os

import pandas as pd
import requests


def ping_server(url: str) -> bool:
    try:
        response = requests.get(url)
        # If the server responds with status code 200 (OK), it's available
        if response.status_code == 200:
            print(f"COOL!!!! Server is running at {url}\n")
            return True
        else:
            print(f"Server responded with status code: {response.status_code}\n")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Failed to connect to server: {e}\n")
        return False


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
        print(f"Prediction file saved as: {output_filename}\n")

        if input("Display sample predictions ? [y/n] ") == 'y':
            df = pd.read_csv(output_filename)
            print(df.head(min(10, df.shape[0])).to_string(index=False))

    else:
        # Handle errors (bad input, etc.)
        print(f"Failed to get predictions: {response.status_code} - {response.text}\n")


if __name__ == "__main__":
    server_url = "http://0.0.0.0:8080"

    is_server_running = ping_server(server_url)

    if is_server_running:
        input_csv = input("Enter the location of your data: ")  # Change to your CSV file path

        if os.path.isfile(input_csv):
            consume_predict_api(f"{server_url}/predict", input_csv)
        else:
            print("Input file does not exist.")

    else:
        print("End of program")
