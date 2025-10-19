
import dagster as dg
import dagster_slack

from .de import assets as de_assets
from .ml import assets as ml_assets
from .ml.resources import (
    Era5RequestConfig,
    PromotionConfig,
    TuningConfig,
)
from .ml_fraud import assets as ml_fraud_assets
from .ml_fraud.resources import (
    data_config,
    mlflow_client,
    mlflow_resource,
    model_config,
    model_promotion_config,
)

# _ = load_dotenv(find_dotenv())

SLACK_TOKEN = dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN").get_value() or ""
print("Slack token: ", SLACK_TOKEN)

all_de_assets = dg.load_assets_from_modules([de_assets])
all_de_checks = dg.load_asset_checks_from_modules([de_assets])
all_ml_assets = dg.load_assets_from_modules([ml_assets])
all_ml_checks = dg.load_asset_checks_from_modules([ml_assets])
all_ml_fraud_assets = dg.load_assets_from_modules([ml_fraud_assets])
all_ml_fraud_asset_checks = dg.load_asset_checks_from_modules([ml_fraud_assets])


@dg.failure_hook(required_resource_keys={"mlflow_tracking"})
def mlflow_failure_hook(context):
    print("-------------- MLflow Failure Hook Triggered -----------\n" * 10)
    mlflow_client = context.resources.mlflow_tracking

    # TODO: check how to get this failure event (dagster mlflow docs)
    # error_message = f"Dagster job failed: {context.failure_event.message}"

    mlflow_client.log_metric("dagster_job_failed", 1)
    mlflow_client.set_tag("dagster_job_status", "failed")
    mlflow_client.end_run(status="FAILED")  # Updated for updating the mlflow run status
    # mlflow_client.set_tag("dagster_error_message", error_message)
    mlflow_client.log_param("dagster_failed_step", context.step_key)
    mlflow_client.log_param("dagster_run_id", context.run_id)


# Jobs
de_job = dg.define_asset_job(
    name="simple_data_engineering_example",
    selection=dg.AssetSelection.groups("de_ingest", "de_transform"),
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

ml_fraud_job = dg.define_asset_job(
    name="fraud_detection",
    selection=dg.AssetSelection.groups(
        "ml_fraud_ingesttion",
        "ml_fraud_transformation",
        "ml_fraud_data_split",
        "ml_fraud_main_model",
        "ml_fraud_evaluate_model",
        "ml_fraud_promote_model"
    )
)

# Schedules
era5_daily_schedule = dg.ScheduleDefinition(
    job=ml_job,
    cron_schedule="0 7 * * *",  # Every day at 7:00 AM
    name="era5_daily_schedule"
)

# Define all assets and resources for Dagster to discover
defs = dg.Definitions(
    assets=[
        *all_ml_assets,
        *all_de_assets,
        *all_ml_fraud_assets
    ],
    resources={
        "io_manager": dg.FilesystemIOManager(base_dir="./tmp_dg_storage"),
        "data_config": data_config,
        "model_config": model_config,
        "model_promotion_config": model_promotion_config,
        "mlflow": mlflow_resource,
        "mlflow_api_client": mlflow_client,
        "slack": dagster_slack.SlackResource(token=SLACK_TOKEN)
    },
    jobs=[
        de_job,
        ml_job,
        ml_fraud_job
    ],
    schedules=[era5_daily_schedule],
    asset_checks=[
        *all_de_checks,
        *all_ml_checks,
        *all_ml_fraud_asset_checks
    ]
)
