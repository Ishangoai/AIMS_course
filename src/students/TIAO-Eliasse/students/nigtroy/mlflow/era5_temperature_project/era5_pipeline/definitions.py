import dagster as dg

from . import assets
from .resources import mlflow_resource

my_assets = dg.load_assets_from_modules([assets])


@dg.failure_hook(required_resource_keys={"mlflow_tracking"})
def mlflow_failure_hook(context):
    mlflow_client = context.resources.mlflow_tracking
    error_message = f"Dagster job failed: {context.failure_event.message}"
    mlflow_client.set_tag("dagster_job_status", "failed")
    mlflow_client.set_tag("dagster_error_message", error_message)
    mlflow_client.log_param("dagster_failed_step", context.step_key)
    mlflow_client.log_param("dagster_run_id", context.run_id)


era5_full_pipeline_job = dg.define_asset_job(
    name="era5_temperature_pipeline_job",
    selection=dg.AssetSelection.all(),
    hooks={mlflow_failure_hook},
)

era5_daily_schedule = dg.ScheduleDefinition(
    job=era5_full_pipeline_job,
    cron_schedule="0 7 * * *",  # Every day at 7:00 AM
    name="era5_daily_schedule"
)

# Define all assets and resources for Dagster to discover
defs = dg.Definitions(
    assets=[*my_assets],
    resources={
        "mlflow_tracking": mlflow_resource,  # Ensure this points to your configured MLflow resource
    },
    jobs=[era5_full_pipeline_job],
    schedules=[era5_daily_schedule]
)
