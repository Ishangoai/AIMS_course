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


class RandomForestTuningConfig(dg.Config):
    """Configuration for RandomForest hyperparameter tuning"""
    # hyperparameter_to_tune: str = "n_estimators"  # Choose: n_estimators, max_depth, or min_samples_split
    n_estimators_options: list[int] = [50, 100, 200]
    max_depth_options: list[int] = [10, 20, 30]
    # min_samples_split_options: list[int] = [2, 5, 10]
    cv_folds: int = 3  # 3-fold cross-validation


class FraudPromotionConfig(dg.Config):
    """Configuration for model promotion thresholds"""
    staging_accuracy_threshold: float = 0.95
    staging_precision_threshold: float = 0.90
    staging_recall_threshold: float = 0.75


data_link = FraudDataConfig()
tuning_config = RandomForestTuningConfig()
promotion_config = FraudPromotionConfig()
