import os

import joblib
import mlflow
import requests


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


def dump_model_to_pickle(model_name: str, model_version: str, context) -> None:
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    DUMP_PATH = os.path.join(os.getcwd(), "fraud_detector.pkl")
    context.log.info(f"Dump promoted model to pickle file at {DUMP_PATH}")

    try:
        joblib.dump(model, DUMP_PATH)
    except Exception as e:
        context.log.info(f"Failed to dump model, reason: {e}")
