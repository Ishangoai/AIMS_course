# We have removed the other ml projects from here and using a separete database

import dagster as dg
import mlflow

# ml_fraud assets & resources
from .ml_fraud import assets as ml_fraud_assets
from .ml_fraud.resources import FraudDataConfig, mlflow_fraud_resource

all_ml_fraud_assets = dg.load_assets_from_modules([ml_fraud_assets])
all_ml_fraud_checks = dg.load_asset_checks_from_modules([ml_fraud_assets])


@dg.failure_hook(required_resource_keys={"mlflow_track"})
def mlflow_failure_hook(context):
    # Compatible failure hook: avoid using context.failure_event which may not exist
    step = getattr(context, "step_key", "unknown_step")
    run_id = getattr(context, "run_id", "unknown_run")
    error_message = f"Dagster job failed in step {step}; run_id={run_id}"
    try:
        # Start a nested mlflow run (or top-level if none active)
        if mlflow.active_run() is None:
            mlflow.start_run(run_name="failure_log")
        else:
            mlflow.start_run(run_name="failure_log", nested=True)
        mlflow.set_tag("dagster_job_status", "failed")
        mlflow.set_tag("dagster_error_message", error_message)
        mlflow.log_param("dagster_failed_step", step)
        mlflow.log_param("dagster_run_id", run_id)
    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()


# Define Fraud Detection ML Job
fraud_ml_job = dg.define_asset_job(
    name="fraud_detection_ml_pipeline",
    selection=dg.AssetSelection.groups(
        "ml_fraud_ingest",
        "ml_fraud_transform",
        "ml_fraud_model",
        "ml_fraud_training",
        "ml_fraud_notification",
    ),
    hooks={mlflow_failure_hook},
)

# Compose Definitions (only include the ml_fraud assets to focus on fraud pipeline)
assets = [*all_ml_fraud_assets]
asset_checks = [*all_ml_fraud_checks]
jobs = [fraud_ml_job]

defs = dg.Definitions(
    assets=assets,
    resources={
        "io_manager": dg.FilesystemIOManager(base_dir="./tmp_dg_storage"),
        "config": FraudDataConfig(),
        "mlflow_track": mlflow_fraud_resource,
    },
    jobs=jobs,
    asset_checks=asset_checks,
)
