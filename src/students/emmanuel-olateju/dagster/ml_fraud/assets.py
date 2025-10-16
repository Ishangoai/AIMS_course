import os
import sys

import requests

sys.path.append(".")

from typing import Dict

import pandas as pd
import yaml
from dagster import asset

from .custom_modules.preprocessing import clean_data, data_splitting, split_features_labels

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.yaml')
with open(CONFIG_PATH, 'r') as file:
    CONFIGS = yaml.safe_load(file)


# ----------------------------- DATA PREPARATION ASSETS ----------------------------- #
@asset
def ingest_dataset() -> pd.DataFrame:
    "Ingest Dataset from remote CSV file"
    dataset_link = CONFIGS['training']['dataset_link']
    df = pd.read_csv(dataset_link)
    return df


@asset
def preprocess_data(ingest_dataset: pd.DataFrame) -> pd.DataFrame:
    "Preprocess the ingested dataset"
    df = clean_data(ingest_dataset)
    return df


@asset
def splitting_data(preprocess_data: pd.DataFrame) -> Dict:
    "Split the preprocessed data into training and testing sets"
    X, y = split_features_labels(preprocess_data, label_column='Class')

    test_size = CONFIGS["training"]["test_size"]
    random_state = CONFIGS["training"]["random_state"]
    return data_splitting(X, y, test_size, random_state)


@asset
def save_data_artifacts(preprocess_data: pd.DataFrame, splitting_data: Dict) -> None:
    "Save preprocessed data and split data artifacts to local storage or bucket"
    data_dir = CONFIGS["artifacts"]["data_dir"]
    preprocess_data.to_csv(f"{data_dir}/preprocessed_data.csv", index=False)

    for key, value in splitting_data.items():
        pd.DataFrame(value).to_csv(f"{data_dir}/{key}.csv", index=False)


# ----------------------- MODELING ASSETS --------------------------------- #
@asset
def train_model(splitting_data: Dict) -> float:
    # X_train = splitting_data['X_train']
    # X_test = splitting_data['X_test']
    # y_train = splitting_data['y_train']
    # y_test = splitting_data['y_test']

    return 0.95


# ----------------------------- NOTIFICATION ASSETS ----------------------------- #
@asset
def send_ml_performance_to_slack(train_model: float) -> float:
    SLACK_WEBHOOK_URL = CONFIGS['hooks']['slack_webhook_url']
    # Dummy performance values
    accuracy = train_model  # replace with your real value later

    # Choose metric to display
    metric_name = "Accuracy"
    metric_value = accuracy

    # Your favorite emoji
    emoji = ":rocket:"

    # Construct the Slack message
    message = f"ML model performance on test set:\n*{metric_name}*: {metric_value:.2f} {emoji}"

    # Send message to Slack
    response = requests.post(SLACK_WEBHOOK_URL, json={"text": message})

    if response.status_code != 200:
        pass
        # Notify in Email
        # context.log.error(f"Failed to send Slack message: {response.text}")
    else:
        pass
        # context.log.info("Slack message sent successfully!")

    return metric_value
