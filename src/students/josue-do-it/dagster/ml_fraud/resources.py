import dagster as dg
from dagster_slack import SlackResource


class FraudDataConfig(dg.ConfigurableResource):
    data_source: str = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"


fraud_data_source = FraudDataConfig()


slack_resource = SlackResource(
    token="xoxb-2422279561408-9303546575606-kvjli9hRzdk0AIUphprXpMQa"
)
