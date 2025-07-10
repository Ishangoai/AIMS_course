import dagster as dg
from . import assets
from .resources import mlflow_resource, mlflow_client, CDSAPI

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
    config={
        "ops": {
            "raw_netcdf_dataset": {
                "config": {
                    "product_type": "reanalysis",
                    "variable": "2m_temperature",
                    "year": "2023",
                    "month": "01",
                    "day": [f"{i:02d}" for i in range(1, 16)],
                    "time": ["00:00", "06:00", "12:00", "18:00"],
                    "area": [50.0, -5.0, 45.0, 5.0],
                    "format": "netcdf",
                }
            },
            "promote_model_to_production": {
                "config": {
                    "staging_mse_threshold": 1.5,
                    "staging_mae_threshold": 1.5
                }
            },
            "tune_ridge_hyperparameters": {
                "config": {
                    "max_hyperopt_evals": 20
                }
            }
        }
    }
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
        "io_manager": dg.FilesystemIOManager(base_dir="./tmp_dg_storage"),
        "mlflow_tracking": mlflow_resource,
        "mlflow_client": mlflow_client,
        "cds_api": CDSAPI(),
    },
    jobs=[era5_full_pipeline_job],
    schedules=[era5_daily_schedule]
)
