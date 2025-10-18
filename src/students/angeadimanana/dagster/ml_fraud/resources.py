import dagster as dg
import pydantic as pyd


class FraudDataConfig(dg.Config):
    test_size: float = pyd.Field(
        default=0.2,
        description="Proportion of dataset to include in the test split (0.2 = 20%)"
    )
    random_state: int = pyd.Field(
        default=42,
        description="Random seed for reproducibility"
    )
    data_url: str = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"


data_link = FraudDataConfig()
