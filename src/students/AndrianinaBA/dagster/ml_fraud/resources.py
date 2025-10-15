import dagster as dg

class FraudDataConfig(dg.ConfigurableResource):
    data_source: str = "https://r"
