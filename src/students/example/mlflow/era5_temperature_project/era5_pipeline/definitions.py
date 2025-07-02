import dagster as dg
import assets

my_assets = dg.load_assets_from_modules([assets])

# Define all assets and resources for Dagster to discover
defs = dg.Definitions(
    assets=[*my_assets],
    resources={
        "mlflow_tracking": mlflow_resource,  # Ensure this points to your configured MLflow resource
    },
    jobs=[era5_full_pipeline_job],
    schedules=[era5_daily_schedule]
)