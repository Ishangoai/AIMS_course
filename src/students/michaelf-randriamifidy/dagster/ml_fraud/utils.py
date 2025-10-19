import os
import tempfile
from datetime import datetime
from typing import Any, List, Optional, Sequence, Union

import dagster_slack
import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn as ms
import numpy as np
import requests
import seaborn as sns
from dagster import AssetExecutionContext

# from dagster_slack import SlackResource
from mlflow import MlflowClient
from mlflow.entities.model_registry import ModelVersion
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


def get_experiment(mlflow_client: MlflowClient, name: str) -> str:
    """
    Retrieve or create an MLflow experiment by name.

    Args:
        mlflow_client (MlflowClient): The MLflow client.
        name (str): The name of the experiment to retrieve or create.

    Returns:
        str: The experiment ID of the retrieved or newly created experiment.
    """
    try:
        # Try to get the experiment by name
        experiment = mlflow_client.get_experiment_by_name(name)
        if experiment is None:
            experiment_id = mlflow_client.create_experiment(name)
        else:
            experiment_id = experiment.experiment_id
    except Exception:
        # Handle unexpected errors gracefully by attempting creation
        experiment_id = mlflow_client.create_experiment(name)

    return experiment_id


def post_message_in_slack(
    slack: dagster_slack.SlackResource,
    message: str,
    channel: str = "aims_course_october2025"
) -> None:
    """
    Post a message to a specified Slack channel using a Dagster Slack resource.

    Args:
        slack (dagster_slack.SlackResource): The Slack resource instance used to access the Slack API.
        message (str): The text message to send to the channel.
        channel (str, optional): The Slack channel ID or name where the message will be sent.
            Defaults to "aims_course_october2025".

    Returns:
        None
    """
    slack.get_client().chat_postMessage(
        channel=channel,
        text=message
    )

# def was_model_promoted_to_staging(promote_to_staging: dict) -> bool:
#     return promote_to_staging.get("status") == "promoted_to_staging"


def was_model_promoted_to_staging(promote_to_staging: dict[str, str]) -> bool:
    """
    Check whether a model was promoted to the staging environment.

    Args:
        promote_to_staging (dict[str, str]): A dictionary containing the promotion status information.

    Returns:
        bool: True if the model's status is "promoted_to_staging", otherwise False.
    """
    return promote_to_staging.get("status") == "promoted_to_staging"

# def get_latest_staging_version(model_name: str, mlflow_client) -> object | None:
#     latest_staging_version = None
#     for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
#         if mv.current_stage == "Staging":
#             if latest_staging_version is None or mv.version > latest_staging_version.version:
#                 latest_staging_version = mv
#     return latest_staging_version


def get_latest_staging_version(model_name: str, mlflow_client: MlflowClient) -> Optional[ModelVersion]:
    """
    Retrieve the latest model version currently in the 'Staging'.

    Args:
        model_name (str): The name of the registered MLflow model.
        mlflow_client (MlflowClient): The MLflow client.

    Returns:
        Optional[ModelVersion]: The latest model version object in 'Staging' if found,
        otherwise None.
    """
    latest_staging_version: Optional[ModelVersion] = None

    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        if mv.current_stage == "Staging":
            if latest_staging_version is None or int(mv.version) > int(latest_staging_version.version):
                latest_staging_version = mv

    return latest_staging_version

# def archive_existing_production_models(model_name: str, mlflow_client, context) -> None:
#     for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
#         if mv.current_stage == "Production":
#             context.log.info(f"Archiving previous Production model '{mv.name}' (version {mv.version})")
#             mlflow_client.transition_model_version_stage(
#                 name=mv.name,
#                 version=mv.version,
#                 stage="Archived"
#             )


def archive_existing_production_models(
    model_name: str,
    mlflow_client: MlflowClient,
    context: AssetExecutionContext
) -> None:
    """
    Archive all existing models in the 'Production' stage.

    Args:
        model_name (str): Registered MLflow model name.
        mlflow_client (MlflowClient): MLflow tracking client.
        context (OpExecutionContext): Dagster logging context.

    Returns:
        None
    """
    for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
        if mv.current_stage == "Production":
            context.log.info(
                f"Archiving previous Production model '{mv.name}' (version {mv.version})"
            )
            mlflow_client.transition_model_version_stage(
                name=mv.name,
                version=mv.version,
                stage="Archived"
            )


def promote_model_to_production(
    model_name: str,
    model_version: str,
    mlflow_client: MlflowClient,
    context: AssetExecutionContext
) -> None:
    """
    Promote a model version to the 'Production' stage.

    Args:
        model_name (str): Registered model name.
        model_version (str): Model version to promote.
        mlflow_client (MlflowClient): MLflow tracking client.
        context (OpExecutionContext): Dagster logging context.
    """
    context.log.info(f"Promoting model '{model_name}' (version {model_version}) to Production")
    mlflow_client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage="Production"
    )


def get_model_by_name(model_name: str, model_version: str) -> Any:
    """
    Load a registered model by name and version.

    Args:
        model_name (str): Registered model name.
        model_version (str): Model version to load.

    Returns:
        Any: Loaded ML model object.
    """
    model_uri = f"models:/{model_name}/{model_version}"

    return ms.load_model(model_uri)


def dump_model_to_pickle(
    model_name: str,
    model_version: str,
    context: AssetExecutionContext
) -> None:
    """
    Dump a promoted model to a local pickle file.

    Args:
        model_name (str): Registered model name.
        model_version (str): Model version to dump.
        context (OpExecutionContext): Dagster logging context.
    """
    model = get_model_by_name(model_name, model_version)
    dump_path = os.path.join(os.getcwd(), "fraud_detector.pkl")

    context.log.info(f"Dumping promoted model to: {dump_path}")

    try:
        joblib.dump(model, dump_path)
    except Exception as e:
        context.log.info(f"Failed to dump model: {e}")


def calculate_false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the false positive rate (FPR).

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        float: False positive rate.
    """
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return float(fp / (fp + tn))


def to_native(val: Any) -> Any:
    """
    Convert numpy scalars to native Python types.

    Args:
        val (Any): Input value.

    Returns:
        Any: Native Python type.
    """
    if isinstance(val, np.generic):
        return val.item()
    return val


def random_forest_summary_message(
    authors: str,
    accuracy: float,
    recall: float,
    fpr: float,
    n_estimators: int
) -> str:
    """
    Create a formatted RandomForest summary message.

    Args:
        authors (str): Model authors.
        accuracy (float): Model accuracy.
        recall (float): Model recall.
        fpr (float): False positive rate.
        n_estimators (int): Number of trees.

    Returns:
        str: Markdown-formatted summary message.
    """
    time_now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return (
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


def log_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: Optional[List[Union[int, str]]] = None,
    artifact_name: str = "confusion_matrix.png"
) -> None:
    """
    Plot and log a confusion matrix to MLflow.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.
        labels (Optional[List[Union[int, str]]]): Label names.
        artifact_name (str): Saved artifact name.
    """
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))

    if labels is not None:
        # Convert all labels to strings for DataFrame / heatmap
        labels_str = [str(label) for label in labels]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels_str, yticklabels=labels_str)
    else:
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, artifact_name)
        plt.savefig(path, bbox_inches="tight")
        mlflow.log_artifact(path, artifact_path="plots")

    plt.close()


def get_top_features_by_importance(
    feature_names: Sequence[str],
    importances: np.ndarray,
    cumulative_threshold: float = 0.8
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get top features contributing to the cumulative importance threshold.

    Args:
        feature_names (Sequence[str]): List of feature names.
        importances (np.ndarray): Feature importance values.
        cumulative_threshold (float): Fraction of total importance to include.

    Returns:
        tuple of (top_features, top_importances)
    """
    if len(feature_names) != len(importances):
        raise ValueError("feature_names and importances must have the same length.")

    sorted_idx = np.argsort(importances)[::-1]
    sorted_features = np.array(feature_names)[sorted_idx]
    sorted_importances = importances[sorted_idx]

    cumulative_importance = np.cumsum(sorted_importances) / sorted_importances.sum()
    top_n_idx = np.searchsorted(cumulative_importance, cumulative_threshold) + 1

    return sorted_features[:top_n_idx], sorted_importances[:top_n_idx]


def log_feature_importance(
    feature_names: Sequence[str],
    importances: np.ndarray,
    cumulative_threshold: Optional[float] = 0.8,
    artifact_name: str = "feature_importance.png"
) -> None:
    """
    Plot and log feature importances as a bar plot to MLflow.

    Args:
        feature_names (Sequence[str]): List of feature names.
        importances (np.ndarray): Feature importance values.
        cumulative_threshold (Optional[float]): Fraction of importance to include.
        artifact_name (str): File name for the saved artifact.
    """
    # Select top features if cumulative_threshold is set
    if cumulative_threshold is not None:
        top_features, top_importances = get_top_features_by_importance(
            feature_names, importances, cumulative_threshold
        )
        title = f"Top Feature Importance. Cumulative Contribution of {cumulative_threshold * 100}%"
    else:
        top_features = np.array(feature_names)
        top_importances = importances
        title = "Top Feature Importance."

    # Plot
    plt.figure(figsize=(8, max(4, 0.4 * len(top_features))))
    sns.barplot(x=top_importances, y=top_features, palette="viridis")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title(title)

    # Save and log to MLflow
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, artifact_name)
        plt.savefig(path, bbox_inches="tight")
        mlflow.log_artifact(path, artifact_path="plots")
