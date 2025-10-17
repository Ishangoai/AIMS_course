import os

import dagster as dg

BASE_DIR = os.path.abspath(os.getenv("DAGSTER_HOME", "."))
SQLITE_DB_FILENAME = "mlflow_local_tracking.db"
SQLITE_DB_PATH = os.path.join(BASE_DIR, SQLITE_DB_FILENAME)

EXPERIMENT_NAME = "fraud_detection"


# Define the MLflow resource
# mlflow_resource = mlflow_tracking.configured(
#     {
#         # Point MLflow to use the local SQLite database
#         "mlflow_tracking_uri": f"sqlite:///{SQLITE_DB_PATH}",
#         "experiment_name": EXPERIMENT_NAME,
#     }
# )

class FraudDataAPI(dg.ConfigurableResource):
    api_url: str = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"

    @property
    def url(self) -> str:
        return self.api_url
