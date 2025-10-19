import dagster as dg
import pydantic as pyd


class PromotionConfig(dg.Config):
    """Configuration for model promotion thresholds"""
    staging_accuracy_threshold: float = pyd.Field(
        default=0.9,
        description="Minimum acceptable accuracy for promoting a model to Staging."
    )
    staging_precision_threshold: float = pyd.Field(
        default=0.75,
        description="Minimum acceptable precision for promoting a model to Staging."
    )
    staging_recall_threshold: float = pyd.Field(
        default=0.75,
        description="Minimum acceptable recall for promoting a model to Staging."
    )
    stagging_f1_score_threshold: float = pyd.Field(
        default=0.7,
        description="Minimum acceptable F1 score for promoting a model to Staging"
    )
