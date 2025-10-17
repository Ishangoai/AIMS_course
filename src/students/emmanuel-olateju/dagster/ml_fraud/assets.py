import os
import sys
from datetime import datetime

# import requests

sys.path.append(".")

from typing import Dict

import dagster as dg
import dagster_slack
import pandas as pd
import yaml
from dagster import asset

from .custom_modules.modelling import model_training_testing
from .custom_modules.preprocessing import clean_data, data_splitting, split_features_labels

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
        group_name="Loading"
)
def save_data_artifacts(preprocess_data: pd.DataFrame, splitting_data: Dict) -> None:
    "Save preprocessed data and split data artifacts to local storage or bucket"
    data_dir = CONFIGS["artifacts"]["data_dir"]
    if os.path.exists(data_dir) is False:
        os.mkdir(data_dir)
    preprocess_data.to_csv(f"{data_dir}/preprocessed_data.csv", index=False)

    for key, value in splitting_data.items():
        pd.DataFrame(value).to_csv(f"{data_dir}/{key}.csv", index=False)


# ----------------------- MODELING ASSETS --------------------------------- #
@asset(
        resource_defs={"slack": slack_resource},
        group_name="Modeling"
)
def train_model(context: dg.AssetExecutionContext, splitting_data: Dict) -> Dict:

    X_train = splitting_data['X_train']
    X_test = splitting_data['X_test']
    y_train = splitting_data['y_train']
    y_test = splitting_data['y_test']

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"Training Session Started at {current_time} {':rocket:'}"
    slack: dagster_slack.SlackResource = context.resources.slack
    slack.get_client().chat_postMessage(
        channel='aims_course_october2025',
        text=f"Emmanuel Olateju + Mohammed \n {message}"
    )

    performance = model_training_testing(
        X_train, y_train, X_test, y_test,
        CONFIGS['training']['random_forest_params_space'], CONFIGS['training']['random_state']
        )

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"Training Session Done at {current_time} {':rocket:'}"
    slack: dagster_slack.SlackResource = context.resources.slack
    slack.get_client().chat_postMessage(
        channel='aims_course_october2025',
        text=f"Emmanuel Olateju + Mohammed \n {message}"
    )

    return performance


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


# ----------------------------- NOTIFICATION ASSETS ----------------------------- #
# @asset
# def send_ml_performance_to_slack(train_model: float) -> float:
#     SLACK_WEBHOOK_URL = CONFIGS['hooks']['slack_webhook_url']
#     # Dummy performance values
#     roc_auc = train_model  # replace with your real value later

#     # Choose metric to display
#     metric_name = "ROC_AUC"
#     metric_value = roc_auc

#     # Your favorite emoji
#     emoji = ":rocket:"

#     # Construct the Slack message
#     message = f"ML model performance on test set:\n*{metric_name}*: {metric_value:.2f} {emoji}"

#     # Send message to Slack
#     response = requests.post(SLACK_WEBHOOK_URL, json={"text": message})

#     if response.status_code != 200:
#         pass
#         # Notify in Email
#         # context.log.error(f"Failed to send Slack message: {response.text}")
#     else:
#         pass
#         # context.log.info("Slack message sent successfully!")

#     return metric_value
