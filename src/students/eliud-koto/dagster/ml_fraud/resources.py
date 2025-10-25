import dagster as dg
from dagster_slack import SlackResource


class FraudDataConfig(dg.ConfigurableResource):
    data_source: str = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"


slack_resource = SlackResource(token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN"))
