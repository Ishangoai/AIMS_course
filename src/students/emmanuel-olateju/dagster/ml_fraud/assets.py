import os
import sys
from datetime import datetime  # noqa: F401

# import requests

sys.path.append(".")

from typing import Dict

import dagster as dg
import dagster_slack
import matplotlib.pyplot as plt
import pandas as pd
import yaml
from dagster import asset

from .custom_modules.modelling import model_training_testing
from .custom_modules.preprocessing import clean_data, data_splitting, split_features_labels

# from .resources import mlflow_resource

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(CONFIG_PATH, 'r') as file:
    CONFIGS = yaml.safe_load(file)


slack_resource = dagster_slack.SlackResource(token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN"))


# ----------------------------- DATA PREPARATION ASSETS ----------------------------- #
@asset(
    group_name="Extraction"
)
def ingest_dataset() -> pd.DataFrame:
    "Ingest Dataset from remote CSV file"
    dataset_link = CONFIGS['training']['dataset_link']
    df = pd.read_csv(dataset_link)
    return df


@asset(
    group_name="Transformation"
)
def preprocess_data(ingest_dataset: pd.DataFrame) -> pd.DataFrame:
    "Preprocess the ingested dataset"
    df = clean_data(ingest_dataset)
    return df


@asset(
        group_name="Transformation"
)
def splitting_data(preprocess_data: pd.DataFrame) -> Dict:
    "Split the preprocessed data into training and testing sets"
    X, y = split_features_labels(preprocess_data, label_column='Class')

    test_size = CONFIGS["training"]["test_size"]
    random_state = CONFIGS["training"]["random_state"]
    return data_splitting(X, y, test_size, random_state)


@asset(
        required_resource_keys={"mlflow"},
        group_name="Storage"
)
def save_data_artifacts(context: dg.AssetExecutionContext, preprocess_data: pd.DataFrame, splitting_data: Dict) -> None:
    "Save preprocessed data and split data artifacts to local storage or bucket"
    mlflow = context.resources.mlflow

    data_dir = CONFIGS["artifacts"]["data_dir"]
    if os.path.exists(data_dir) is False:
        os.makedirs(data_dir, exist_ok=True)

    # Save preprocessed data locally
    preprocess_data_path = f"{data_dir}/preprocessed_data.csv"
    preprocess_data.to_csv(f"{data_dir}/preprocessed_data.csv", index=False)
    # Log preprocessed data to MLflow
    mlflow.log_artifact(preprocess_data_path, artifact_path="data")
    context.log.info("✅ Preprocessed data logged to MLflow")

    for key, value in splitting_data.items():
        file_path = f"{data_dir}/{key}.csv"
        pd.DataFrame(value).to_csv(f"{data_dir}/{key}.csv", index=False)

        # Log each split to MLflow
        mlflow.log_artifact(file_path, artifact_path="data")
        context.log.info(f"✅ {key} logged to MLflow")
    context.log.info("🎉 All data artifacts saved locally and logged to MLflow")


# ----------------------- MODELING ASSETS --------------------------------- #
@asset(
        resource_defs={
            "slack": slack_resource,
            },
        required_resource_keys={"mlflow"},
        group_name="Modeling"
)
def train_model(context: dg.AssetExecutionContext, preprocess_data: pd.DataFrame, splitting_data: Dict):

    mlflow = context.resources.mlflow
    X_train = splitting_data['X_train']
    X_test = splitting_data['X_test']
    y_train = splitting_data['y_train']
    y_test = splitting_data['y_test']

    context.log.info("🚀 Starting model training")

    # Get the active run object
    active_run = mlflow.active_run()

    # Access run information
    run_id = active_run.info.run_id
    experiment_id = active_run.info.experiment_id
    run_name = active_run.info.run_name

    context.log.info(f"Current MLflow run ID: {run_id}")
    context.log.info(f"Experiment ID: {experiment_id}")
    context.log.info(f"Run name: {run_name}")

    # ============= LOG DATA ARTIFACTS TO MLFLOW =============
    context.log.info("📊 Logging data artifacts to MLflow...")

    # Create temporary directory for artifacts
    import tempfile
    temp_dir = tempfile.mkdtemp()

    try:
        # Save and log preprocessed data
        preprocess_data_path = os.path.join(temp_dir, "preprocessed_data.csv")
        preprocess_data.to_csv(preprocess_data_path, index=False)
        mlflow.log_artifact(preprocess_data_path)
        context.log.info("✅ Preprocessed data logged to MLflow")

        # Save and log split data
        for key, value in splitting_data.items():
            file_path = os.path.join(temp_dir, f"{key}.csv")
            pd.DataFrame(value).to_csv(file_path, index=False)
            mlflow.log_artifact(file_path)
            context.log.info(f"✅ {key} logged to MLflow")

        context.log.info("🎉 All data artifacts logged to MLflow")

    finally:
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

    # ============= TRAIN MODEL =============
    performance = model_training_testing(
        X_train, y_train, X_test, y_test,
        CONFIGS['training']['random_forest_params_space'],
        CONFIGS['training']['random_state']
    )

    trained_model = performance["model"]

    # Log metrics
    mlflow.log_metrics({
        "accuracy": performance["accuracy"],
        "recall": performance["recall"],
        "roc_auc": performance["roc_auc"],
    })

    # Generate confusion matrix
    cm = performance["cm"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Add text annotations
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                            ha="center", va="center", color="black")

    ax.set_title("Confusion Matrix on Test Data")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # Set ticks
    ax.set_xticks(range(cm.shape[1]))
    ax.set_yticks(range(cm.shape[0]))

    plt.tight_layout()

    mlflow.log_figure(fig, "confusion_matrix.png")
    plt.close(fig)

    # ✅ Log model itself (works for sklearn models)
    mlflow.sklearn.log_model(
        sk_model=trained_model,
        artifact_path="model",
        registered_model_name="fraud_detection_model"
    )
    context.log.info("✅ Model successfully logged to MLflow")

    # Log the run URL for easy access
    run_url = f"{mlflow.tracking_uri}/#/experiments/{experiment_id}/runs/{run_id}"
    context.log.info(f"View run at: {run_url}")

    return performance
# def train_model(context: dg.AssetExecutionContext, splitting_data: Dict):

#     mlflow = context.resources.mlflow
#     X_train = splitting_data['X_train']
#     X_test = splitting_data['X_test']
#     y_train = splitting_data['y_train']
#     y_test = splitting_data['y_test']

#     context.log.info("🚀 Starting model training")

#     # Method 1: Get the active run object
#     active_run = mlflow.active_run()

#     # Access run information
#     run_id = active_run.info.run_id
#     experiment_id = active_run.info.experiment_id
#     run_name = active_run.info.run_name
#     start_time = active_run.info.start_time

#     context.log.info(f"Current MLflow run ID: {run_id}")
#     context.log.info(f"Experiment ID: {experiment_id}")
#     context.log.info(f"Run name: {run_name}")

#     # Method 2: Access run data (parameters, metrics, tags)
#     run_data = active_run.data

#     # Method 3: Get run info dict
#     run_info = active_run.info

#     # Start an MLflow run via the resource context
#     # with mlflow.start_run(run_name="fraud_detection_model") as run:
#     performance = model_training_testing(
#         X_train, y_train, X_test, y_test,
#         CONFIGS['training']['random_forest_params_space'],
#         CONFIGS['training']['random_state']
#     )

#     trained_model = performance["model"]

#     # Log metrics
#     mlflow.log_metrics({
#         "accuracy": performance["accuracy"],
#         "recall": performance["recall"],
#         "roc_auc": performance["roc_auc"],
#     })

#     # Generate confusion matrix
#     cm = performance["cm"]

#     fig, ax = plt.subplots(figsize=(6, 5))
#     im = ax.imshow(cm, cmap='Blues', interpolation='nearest')

#     # Add colorbar
#     plt.colorbar(im, ax=ax)

#     # Add text annotations
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, str(cm[i, j]),
#                             ha="center", va="center", color="black")

#     ax.set_title("Confusion Matrix on Test Data")
#     ax.set_xlabel("Predicted Label")
#     ax.set_ylabel("True Label")

#     # Set ticks
#     ax.set_xticks(range(cm.shape[1]))
#     ax.set_yticks(range(cm.shape[0]))

#     plt.tight_layout()

#     mlflow.log_figure(fig, "confusion_matrix.png")
#     plt.close(fig)

#     # ✅ Log model itself (works for sklearn models)
#     mlflow.sklearn.log_model(
#         sk_model=trained_model,
#         artifact_path="model",
#         registered_model_name="fraud_detection_model"
#     )
#     context.log.info("✅ Model successfully logged to MLflow")

#     # Log the run URL for easy access
#     run_url = f"{mlflow.tracking_uri}/#{experiment_id}/{run_id}"
#     context.log.info(f"View run at: {run_url}")
#     # context.log.info(f"✅ MLflow run complete: {run.info.run_id}")
#     return performance


@asset(
        resource_defs={"slack": slack_resource},
        group_name="Modeling"
)
def notify_modelling_results(context: dg.AssetExecutionContext, train_model: Dict) -> None:

    accuracy = train_model['accuracy']
    recall = train_model['recall']
    roc_auc = train_model['roc_auc']

    message = (
        ":bar_chart: **Model Performance:**\n"
        f"        • Accuracy: {accuracy:.4f}\n"
        f"        • Recall: {recall:.4f}\n"
        f"        • ROC-AUC: {roc_auc:.4f}\n"
        ":bust_in_silhouette: Trained by: Emmanuel Olateju + Mohammed\n"
        ":wrench: Model: RandomForest Classifier\n"
        ":chart_with_upwards_trend: Experiment: fraud_detection_ml"
    )
    slack: dagster_slack.SlackResource = context.resources.slack
    slack.get_client().chat_postMessage(
        channel='aims_course_october2025',
        text=f"{os.environ.get('GITHUB_USER', 'default')}'s \n {message}"
    )


# ------ Train_model asset - v1 ----------------
# def train_model(context: dg.AssetExecutionContext, splitting_data: Dict) -> Dict:

#     X_train = splitting_data['X_train']
#     X_test = splitting_data['X_test']
#     y_train = splitting_data['y_train']
#     y_test = splitting_data['y_test']

#     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     message = f"Training Session Started at {current_time} {':rocket:'}"
#     # slack: dagster_slack.SlackResource = context.resources.slack
#     # slack.get_client().chat_postMessage(
#     #     channel='aims_course_october2025',
#     #     text=f"Emmanuel Olateju + Mohammed \n {message}"
#     # )
#     print(message)

#     performance = model_training_testing(
#         X_train, y_train, X_test, y_test,
#         CONFIGS['training']['random_forest_params_space'], CONFIGS['training']['random_state']
#         )

#     current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     message = f"Training Session Done at {current_time} {':rocket:'}"
#     # slack: dagster_slack.SlackResource = context.resources.slack
#     # slack.get_client().chat_postMessage(
#     #     channel='aims_course_october2025',
#     #     text=f"Emmanuel Olateju + Mohammed \n {message}"
#     # )
#     print(message)

#     return performance
