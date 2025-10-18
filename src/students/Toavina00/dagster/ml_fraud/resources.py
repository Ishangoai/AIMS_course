import os

import dagster as dg
import mlflow

BASE_DIR = os.path.abspath(os.getenv("DAGSTER_HOME", "."))
DB_FILE = "mlflow_local_tracking.db"
DB_PATH = os.path.join(BASE_DIR, DB_FILE)
TRACKING_URI = f"sqlite:///{DB_PATH}"
EXPERIMENT_NAME = "ml_fraud_detection"


class FraudMlflow:
    def __init__(self, experiment_name: str, tracking_uri: str) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)

        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = self.client.create_experiment(self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id


class FraudMlflowResource(dg.ConfigurableResource):
    _experiment_name: str = EXPERIMENT_NAME
    _tracking_uri: str = TRACKING_URI

    def create_resource(self, context: dg.InitResourceContext):
        return FraudMlflow(self._experiment_name, self._tracking_uri)


mlflow_resource = FraudMlflowResource()


class FraudDataAPI(dg.ConfigurableResource):
    api_url: str = (
        "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"
    )

    @property
    def url(self) -> str:
        return self.api_url
