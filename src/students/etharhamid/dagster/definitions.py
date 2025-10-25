# definitions file
import dagster as dg

from .de import assets as de_assets
from .ml import assets as ml_assets
from .ml.resources import (
    Era5RequestConfig,
    PromotionConfig,
    TuningConfig,
)

# Import fraud detection resources from ml.resources (the unified file)
from .ml_fraud import assets as ml_fraud_assets

# Load all assets from modules
all_de_assets = dg.load_assets_from_modules([de_assets])
all_de_checks = dg.load_asset_checks_from_modules([de_assets])
all_ml_assets = dg.load_assets_from_modules([ml_assets])
all_ml_checks = dg.load_asset_checks_from_modules([ml_assets])
all_ml_fraud_assets = dg.load_assets_from_modules([ml_fraud_assets])


# MLflow failure hook
@dg.failure_hook(required_resource_keys={"mlflow_tracking"})
def mlflow_failure_hook(context):
    mlflow_client = context.resources.mlflow_tracking
    error_message = f"Dagster job failed: {context.failure_event.message}"
    mlflow_client.set_tag("dagster_job_status", "failed")
    mlflow_client.set_tag("dagster_error_message", error_message)
    mlflow_client.log_param("dagster_failed_step", context.step_key)
    mlflow_client.log_param("dagster_run_id", context.run_id)


# Data Engineering Job
de_job = dg.define_asset_job(
    name="simple_data_engineering_example",
    selection=dg.AssetSelection.groups("de_ingest", "de_transform"),
)


# Machine Learning Job (ERA5)
ml_job = dg.define_asset_job(
    name="era5_machine_learning_with_mlflow",
    selection=dg.AssetSelection.groups("ml_ingest", "ml_transform", "ml_model", "ml_evaluate", "ml_promote"),
    hooks={mlflow_failure_hook},
    config={
        "ops": {
            "raw_xarray_dataset": {
                "config": Era5RequestConfig().model_dump()
            },
            "promote_model_to_production": {
                "config": PromotionConfig().model_dump()
            },
            "tune_ridge_hyperparameters": {
                "config": TuningConfig().model_dump()
            }
        }
    }
)


# Fraud Detection Job
fraud_detection = dg.define_asset_job(
    name="fraud_detection",
    description=(
        "End-to-end fraud detection pipeline:\n"
        "data ingestion, transformation, training, evaluation, and notification"
    ),
    selection=dg.AssetSelection.groups(
        "ml_fraud_ingest",      # raw_fraud_data
        "ml_fraud_transform",   # clean_fraud_data, transformed_fraud_data, split_fraud_data
        "ml_fraud_train",             # tune_random_forest
        "ml_fraud_evaluate",          # evaluate_model
        "ml_fraud_promote"            # send_slack_notification
    )
)


# Schedule for ERA5 job
era5_daily_schedule = dg.ScheduleDefinition(
    job=ml_job,
    cron_schedule="0 7 * * *",  # Every day at 7:00 AM
    name="era5_daily_schedule"
)


# Define all assets and resources for Dagster to discover
defs = dg.Definitions(
    assets=[*all_ml_assets, *all_de_assets, *all_ml_fraud_assets],
    resources={
        "io_manager": dg.FilesystemIOManager(base_dir="./tmp_dg_storage")
    },
    jobs=[de_job, ml_job, fraud_detection],
    schedules=[era5_daily_schedule],
    asset_checks=[*all_de_checks, *all_ml_checks]
)
