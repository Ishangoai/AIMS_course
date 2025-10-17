import dagster as dg

from ..ml.resources import mlflow_client, mlflow_resource
from .resources import PromotionConfig


@dg.asset(
    description="Promotes the newly trained model to Staging if it meets performance criteria.",
    resource_defs={"mlflow_tracking": mlflow_resource, "mlflow_client": mlflow_client},
    compute_kind="python",
    group_name="promote_model"
)
def promote_model_to_staging(
    context: dg.AssetExecutionContext,
    config: PromotionConfig,
    test_model: dict
) -> dg.MaterializeResult:
    # Get the MLflow client from the context to interact with the model registry
    mlflow_client = context.resources.mlflow_client
    context.log.info("Starting model promotion to Staging.")

    # If the evaluation step was skipped, we also skip promotion
    if test_model.get("status") == "skipped_evaluation":
        context.log.info("Evaluation was skipped in the previous step. Skipping staging promotion.")
        return dg.MaterializeResult(
            value={"status": "skipped_promotion", "reason": "evaluation_skipped_upstream"},
            metadata={"status": "skipped_promotion_due_to_upstream_skip"}
        )
    # Extract metrics and model version info from evaluation result
    eval_metrics = test_model.get("eval_metrics", {})
    model_version_info = test_model.get("model_version_info")

    # If no model version info was returned, skip promotion.
    # model_version_info might be None due to an upstream failure
    if not model_version_info:
        context.log.warning("No valid model version info found from evaluation. Skipping staging promotion.")
        return dg.MaterializeResult(
            value={"status": "skipped_promotion", "reason": "no_model_version_info_from_evaluation"},
            metadata={"status": "skipped_promotion_no_model_info"}
        )
    # Get performance metrics (default to infinity if missing)
    current_accuracy = eval_metrics.get("test_accuracy", float('inf'))
    current_recall = eval_metrics.get("test_recall", float('inf'))
    current_fpr = eval_metrics.get("test_fpr", float('inf'))

    STAGING_ACCURACY = config.staging_accuracy_threshold
    STAGING_RECALL = config.staging_recall_threshold
    STAGING_FPR = config.staging_fpr_threshold
    # Log the evaluation metrics and threshold criteria
    context.log.info(f"Model evaluated with Accuracy: {current_accuracy:.4f},"
                     f"Recall: {current_recall:.4f}, FPR: {current_fpr}"
                     )
    context.log.info(f"Staging promotion thresholds: Accuracy > {STAGING_ACCURACY},"
                     f"Recall > {STAGING_RECALL}, FPR > {STAGING_FPR}"
                     )

    # Check if model meets promotion criteria
    if (
        current_accuracy >= STAGING_ACCURACY
        and current_recall >= STAGING_RECALL
        and current_fpr < STAGING_FPR
    ):
        try:
            # Extract the model name and version for promotion
            model_name = model_version_info["name"]
            model_version = model_version_info["version"]

            # Promote the model to the 'Staging' stage
            context.log.info(f"Model '{model_name}' (version {model_version}) meets criteria. Promoting to Staging")
            mlflow_client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Staging"
            )

            # Return successful result with status and relevant metadata
            context.log.info(f"Model '{model_name}' (version {model_version}) promoted to Staging.")
            return dg.MaterializeResult(
                value={
                    "status": "promoted_to_staging",
                    "model_name": model_name,
                    "model_version": model_version,
                    "metrics": eval_metrics
                },
                metadata={
                    "status": "promoted_to_staging",
                    "model_name": dg.MetadataValue.text(model_name),
                    "model_version": dg.MetadataValue.text(str(model_version)),
                    "accuracy_at_promotion": dg.MetadataValue.float(current_accuracy),
                    "recall_at_promotion": dg.MetadataValue.float(current_recall),
                    "fpr_at_promotion": dg.MetadataValue.float(current_fpr)
                }
            )
        except Exception as e:
            # Handle any exception during promotion and log the error
            context.log.error(f"Error promoting model to Staging: {e}")
            return dg.MaterializeResult(
                value={"status": "failed_promotion_to_staging", "error": str(e)},
                metadata={"status": "failed_promotion_to_staging", "error_message": dg.MetadataValue.text(str(e))}
            )
    # If model doesn't meet criteria, log and return "not promoted"
    else:
        context.log.info("Model does not meet performance criteria for Staging promotion. Skipping.")
        return dg.MaterializeResult(
            value={
                "status": "not_promoted_to_staging",
                "reason": "criteria_not_met",
                "metrics": eval_metrics
            },
            metadata={
                "status": "not_promoted_to_staging",
                "accuracy": dg.MetadataValue.float(current_accuracy),
                "recall": dg.MetadataValue.float(current_recall),
                "fpr": dg.MetadataValue.float(current_fpr)
            }
        )
