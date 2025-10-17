import dagster as dg

from .utils import ClientDownloader


class FraudResourceConfig(dg.ConfigurableResource):
    data_url: str = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"

    @property
    def client(self) -> ClientDownloader:
        return ClientDownloader(url=self.data_url)
