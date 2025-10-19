import dagster as dg
import mlflow
import pydantic as pyd

from .constant import EXPERIMENT_NAME, REGISTERED_MODEL_NAME, TRACKING_URI


class FraudMlflow:
    def __init__(self, experiment_name: str, tracking_uri: str, registered_model_name: str) -> None:
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.client = mlflow.MlflowClient(tracking_uri=self.tracking_uri)
        self.registered_model_name = registered_model_name

        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = self.client.create_experiment(self.experiment_name)
        else:
            self.experiment_id = experiment.experiment_id


class FraudMlflowResource(dg.ConfigurableResource):
    """Mlflow API resource"""

    _experiment_name: str = EXPERIMENT_NAME
    _tracking_uri: str = TRACKING_URI
    _registered_model_name: str = REGISTERED_MODEL_NAME

    def create_resource(self, context: dg.InitResourceContext):
        return FraudMlflow(self._experiment_name, self._tracking_uri, self._registered_model_name)


mlflow_resource = FraudMlflowResource()


class FraudDataAPI(dg.ConfigurableResource):
    """Data source API resource"""

    api_url: str = (
        "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"
    )

    @property
    def url(self) -> str:
        return self.api_url


class FraudPromotionConfig(dg.Config):
    """Configuration for model promotion thresholds"""

    staging_f1_threshold: float = pyd.Field(
        default=0.8, description="Minimum acceptable F1-score for promoting a model to Staging."
    )
    staging_acc_threshold: float = pyd.Field(
        default=0.85, description="Minimum acceptable Accuracy for promoting a model to Staging."
    )
