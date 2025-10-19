import dagster as dg
import pydantic as pyd

from .utils import ClientDownloader


class FraudResourceConfig(dg.ConfigurableResource):
    data_url: str = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"

    @property
    def client(self) -> ClientDownloader:
        return ClientDownloader(url=self.data_url)


class FraudTuningConfig(dg.Config):
    """Configuration for model tuning and optimization"""
    max_hyperopt_evals: int = pyd.Field(
        default=4,
        description="Maximum number of evaluations allowed by Hyperopt during model tuning."
    )


class FraudPromotionConfig(dg.Config):
    """Configuration for model promotion thresholds"""
    staging_accuracy_threshold: float = pyd.Field(
        default=0.7,
        description="Minimum acceptable Accuracy for promoting a model to Staging"
    )
    staging_recall_threshold: float = pyd.Field(
        default=0.5,
        description="Minimum acceptable Recall for promoting a model to Staging"
    )
    staging_fpr_threshold: float = pyd.Field(
        default=0.2,
        description="Maximum acceptable False Positive Rate for promoting a model to Staging"
    )
