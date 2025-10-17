import dagster as dg

from .de import assets as de_assets
from .ml import assets as ml_assets
from .ml.resources import (
    Era5RequestConfig,
    PromotionConfig,
    TuningConfig,
)
from .ml_fraud import assets as fraud_assets

all_de_assets = dg.load_assets_from_modules([de_assets])
all_de_checks = dg.load_asset_checks_from_modules([de_assets])
all_ml_assets = dg.load_assets_from_modules([ml_assets])
all_ml_checks = dg.load_asset_checks_from_modules([ml_assets])

all_fraud_assets = dg.load_assets_from_modules([fraud_assets])


@dg.failure_hook(required_resource_keys={"mlflow_tracking"})
def mlflow_failure_hook(context):
    mlflow_client = context.resources.mlflow_tracking
    error_message = f"Dagster job failed: {context.failure_event.message}"
    mlflow_client.set_tag("dagster_job_status", "failed")
    mlflow_client.set_tag("dagster_error_message", error_message)
    mlflow_client.log_param("dagster_failed_step", context.step_key)
    mlflow_client.log_param("dagster_run_id", context.run_id)


de_job = dg.define_asset_job(
    name="simple_data_engineering_example",
    selection=dg.AssetSelection.groups("de_ingest", "de_transform"),
)

fraud_job = dg.define_asset_job(
    name="our_fraud_detection_mlflow",
    selection=dg.AssetSelection.groups('ml_fraud_data_ingest', 'ml_fraud_data_transform',
    'ml_fraud_model', 'ml_fraud_slack_msg')
)
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

era5_daily_schedule = dg.ScheduleDefinition(
    job=ml_job,
    cron_schedule="0 7 * * *",  # Every day at 7:00 AM
    name="era5_daily_schedule"
)

# Define all assets and resources for Dagster to discover
defs = dg.Definitions(
    assets=[*all_ml_assets, *all_de_assets, *all_fraud_assets],
    resources={
        "io_manager": dg.FilesystemIOManager(base_dir="./tmp_dg_storage"),
    },
    jobs=[de_job, ml_job, fraud_job],
    schedules=[era5_daily_schedule],
    asset_checks=[*all_de_checks, *all_ml_checks]
)
