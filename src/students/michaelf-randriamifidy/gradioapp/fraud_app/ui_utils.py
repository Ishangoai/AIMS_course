import os
import requests

SERVER_URL = "http://0.0.0.0:8080"

def upload_file_and_get_predictions(file):
    try:
        if file is None:
            return "Please upload a file.", "", None

        file_name = os.path.basename(file.name)    

        with open(file.name, "rb") as f:
            files = {'file': (file_name, f, 'text/csv')}
            response = requests.post(f"{SERVER_URL}/predict", files=files)

        if response.status_code != 200:
            error_detail = response.json().get("detail", "Unknown error occurred.")
            return f"Error: {error_detail}", "", None

        # Save downloaded prediction file
        prediction_filename = f"predictions_{file_name}"
        with open(prediction_filename, "wb") as out_file:
            out_file.write(response.content)

        prediction_text = response.content.decode("utf-8")
        return None, prediction_text, prediction_filename

    except Exception as e:
        return f"Exception occurred: {str(e)}", "", None



def get_csv_preview(file_path, num_rows=5):
    try:
        df = pd.read_csv(file_path)
        preview_df = df.head(num_rows)
        return preview_df
    except Exception:
        return None        
