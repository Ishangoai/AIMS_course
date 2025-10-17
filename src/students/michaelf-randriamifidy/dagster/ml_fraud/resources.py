import dagster as dg

class FraudResourceConfig(dg.ConfigurableResource):
    data_url: str = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"