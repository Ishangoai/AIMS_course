import dagster as dg
from dagster_slack import SlackResource


class FraudDataConfig(dg.ConfigurableResource):
    data_source: str = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"


fraud_data_source = FraudDataConfig()
