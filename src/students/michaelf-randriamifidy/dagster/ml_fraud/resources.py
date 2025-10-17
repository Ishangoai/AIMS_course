import dagster as dg

from .utils import ClientDownloader


class FraudResourceConfig(dg.ConfigurableResource):
    data_url: str = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"

    @property
    def client(self) -> ClientDownloader:
        return ClientDownloader(url=self.data_url)


class TuningConfig(dg.Config):
    """Configuration for model tuning and optimization"""
    max_hyperopt_evals: int = pyd.Field(
        default=20,
        description="Maximum number of evaluations allowed by Hyperopt during model tuning."
    )
