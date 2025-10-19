import os
import tempfile
from datetime import datetime
import dagster_slack
import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import requests
import seaborn as sns
from sklearn.metrics import confusion_matrix


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

def get_experiment(mlflow_client, name):
    try:
        experiment = mlflow_client.get_experiment_by_name("name")
        if experiment is None:
            experiment_id = mlflow_client.create_experiment("name")
        else:
            experiment_id = experiment.experiment_id
    except Exception:  # Handle cases where get_experiment_by_name might raise error if not found
        experiment_id = mlflow_client.create_experiment("name")
    experiment_id
    
def post_message_in_slack(slack: dagster_slack.SlackResource,
                            message: str,
                            channel: str = "aims_course_october2025"
                            ):

    slack.get_client().chat_postMessage(
        channel='aims_course_october2025',
        text=message
    )


def was_model_promoted_to_staging(promote_to_staging: dict) -> bool:
    return promote_to_staging.get("status") == "promoted_to_staging"


def get_latest_staging_version(model_name: str, mlflow_client) -> object | None:
    latest_staging_version = None
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        if mv.current_stage == "Staging":
            if latest_staging_version is None or mv.version > latest_staging_version.version:
                latest_staging_version = mv
    return latest_staging_version


def archive_existing_production_models(model_name: str, mlflow_client, context) -> None:
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        if mv.current_stage == "Production":
            context.log.info(f"Archiving previous Production model '{mv.name}' (version {mv.version})")
            mlflow_client.transition_model_version_stage(
                name=mv.name,
                version=mv.version,
                stage="Archived"
            )


def promote_model_to_production(model_name: str, model_version: str, mlflow_client, context) -> None:
    context.log.info(f"Promoting model '{model_name}' (version {model_version}) to Production")
    mlflow_client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Production"
    )


def get_model_by_name(model_name: str, model_version: str):
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.sklearn.load_model(model_uri)
    return model


def dump_model_to_pickle(model_name: str, model_version: str, context) -> None:
    model = get_model_by_name(model_name, model_version)

    DUMP_PATH = os.path.join(os.getcwd(), "fraud_detector.pkl")
    context.log.info(f"Dump promoted model to pickle file at {DUMP_PATH}")

    try:
        joblib.dump(model, DUMP_PATH)
    except Exception as e:
        context.log.info(f"Failed to dump model, reason: {e}")


def calculate_false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate False Positive Rate
    Args:
        y_true (np.ndarray): true value
        y_pred (np.ndarray): predicted value

    Returns:
        float: False Positive Rate
    """
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return float(fp / (fp + tn))


def to_native(val):
    import numpy as np
    if isinstance(val, np.generic):
        return val.item()
    return val


def random_forest_summary_message(authors, accuracy, recall, fpr, n_estimators):
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    message = (
    f"*Fraud Detection — Model Test Summary*\n"
    f"-----------------------------\n"
    f"*👩‍💻 Authors:* {authors}\n"
    f"*🧩 Model Type:* `RandomForest`\n"
    f"*🌲 n_estimators:* `{n_estimators}`\n"
    f"*✅ Accuracy:* `{accuracy:.4f}`\n"
    f"*🎯 Recall:* `{recall:.4f}`\n"
    f"*🚨 False Positive Rate (FPR):* `{fpr:.4f}`\n"
    f"*🕒 Timestamp:* `{time_now}`\n"
    f"-----------------------------"
)
    return message


def log_confusion_matrix(y_true, y_pred, labels=None, artifact_name="confusion_matrix.png"):
    """
    Plots and logs a confusion matrix to the current MLflow run.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, artifact_name)
        plt.savefig(path)
        mlflow.log_artifact(path, artifact_path="plots")

    plt.close()
