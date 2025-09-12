import dagster as dg
import pydantic as pyd


class FraudDataResource(dg.ConfigurableResource):
    """Resource for fraud data configuration"""
    data_url: str = pyd.Field(
        default="https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv",
        description="URL to the credit card fraud dataset"
    )
    test_size: float = pyd.Field(
        default=0.2,
        description="Proportion of data to use for testing"
    )
    random_state: int = pyd.Field(
        default=42,
        description="Random state for reproducibility in data splitting"
    )


class RandomForestConfig(dg.Config):
    """Configuration for RandomForest hyperparameter tuning"""
    param_grid: dict = pyd.Field(
        default={
            'n_estimators': [50, 100, 200],
        },
        description="Parameter grid for RandomForest hyperparameter tuning"
    )
    cv_folds: int = pyd.Field(
        default=3,
        description="Number of cross-validation folds for hyperparameter tuning"
    )
    scoring: str = pyd.Field(
        default='f1',
        description="Scoring metric for cross-validation"
    )
    random_state: int = pyd.Field(
        default=42,
        description="Random state for reproducibility in model training"
    )


fraud_data_resource = FraudDataResource()
