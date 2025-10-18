from datetime import datetime
import requests
import numpy as np
from sklearn.metrics import confusion_matrix
import dagster_slack

class ClientDownloader:
    def __init__(self, url=None):
        self.url = url

    def download_and_save(self, output_filename):
        if not isinstance(self.url, str) or not self.url.strip():
            raise ValueError("URL is not set or invalid.")

        if not isinstance(output_filename, str) or not output_filename.strip():
            raise ValueError("Invalid output filename.")

        try:
            response = requests.get(self.url, stream=True)
            response.raise_for_status()

        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Request failed: {e}")

        try:
            with open(output_filename, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        except IOError as e:
            raise RuntimeError(f"Failed to write file: {e}")


def post_message_in_slack(slack: dagster_slack.SlackResource,
                            message: str,
                            channel: str="aims_course_october2025"
                            ):
        
        slack.get_client().chat_postMessage(
            channel='aims_course_october2025',
            text=message
        )
        
def calculate_false_positive_rate(y_true:np.ndarray, y_pred:np.ndarray) -> float:
    """
    Calculate False Positive Rate
    Args:
        y_true (np.ndarray): true value
        y_pred (np.ndarray): predicted value 

    Returns:
        float: False Positive Rate
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn)


def to_native(val):
    import numpy as np
    if isinstance(val, np.generic):
        return val.item()
    return val

def random_forest_summary_message(authors, accuracy, recall, fpr, n_estimators):
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    

    message = (
        f"**Fraud Detection**\n"
        f"**Model Training Summary**\n"
        f"----------------------------------\n"
        f" Model Type: Random Forest\n"
        # f" Dataset: {data_name}\n"
        f" Authors: {authors}\n"
        f" n_estimators: {n_estimators}\n"
        f" Accuracy: {accuracy:.4f}\n"
        f" Recall: {recall:.4f}\n"
        f" False Positive Rate (FPR): {fpr:.4f}\n"
        f" Timestamp: {time_now}\n"
        f"----------------------------------"
    )
    return message
