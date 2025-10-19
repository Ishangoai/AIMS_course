import dagster as dg

from .de import assets as de_assets
from .ml import assets as ml_assets
from .ml.resources import Era5RequestConfig
from .ml_fraud import assets as ml_fraud_assets
from .ml_fraud.resources import (
    PromotionConfig,
    TuningConfig,
)
from .ml_fraud.resources import (
    data_api_resource as fraud_data_api_resource,
)
from .ml_fraud.resources import (
    mlflow_client as fraud_mlflow_client,
)

# Import updated fraud-specific resources and configs
from .ml_fraud.resources import (
    mlflow_resource as fraud_mlflow_resource,
)

all_de_assets = dg.load_assets_from_modules([de_assets])
all_de_checks = dg.load_asset_checks_from_modules([de_assets])
all_ml_assets = dg.load_assets_from_modules([ml_assets])
all_ml_checks = dg.load_asset_checks_from_modules([ml_assets])
all_ml_fraud_assets = dg.load_assets_from_modules([ml_fraud_assets])


de_job = dg.define_asset_job(
    name="simple_data_engineering_example",
    selection=dg.AssetSelection.groups("de_ingest", "de_transform"),
)

ml_job = dg.define_asset_job(
    name="era5_machine_learning_with_mlflow",
    selection=dg.AssetSelection.groups("ml_ingest", "ml_transform", "ml_model", "ml_evaluate", "ml_promote"),
    config={
        "ops": {
            "raw_xarray_dataset": {"config": Era5RequestConfig().model_dump()},
            "promote_model_to_production": {"config": PromotionConfig().model_dump()},
            "tune_ridge_hyperparameters": {"config": TuningConfig().model_dump()},
        }
    },
)

ml_fraud_job = dg.define_asset_job(
    name="fraud_detection_job",
    selection=dg.AssetSelection.groups(
        "ml_fraud_ingest",
        "ml_fraud_preprocess_split",
        "ml_fraud_tuning",
        "ml_fraud_training",
        "ml_fraud_evaluation",
        "ml_fraud_notification",
        "ml_fraud_registry",
    ),
)

era5_daily_schedule = dg.ScheduleDefinition(job=ml_job, cron_schedule="0 7 * * *", name="era5_daily_schedule")

# Register resources. Use unique keys for fraud mlflow and mlflow client to avoid colliding
# with other ml resources in the repo.
defs = dg.Definitions(
    assets=[*all_ml_assets, *all_de_assets,
    *all_ml_fraud_assets
    ],
    resources={
        "io_manager": dg.FilesystemIOManager(base_dir="./tmp_dg_storage"),
        # Updated fraud resources from ml_fraud.resources
        "fraud_mlflow": fraud_mlflow_resource,
        "fraud_mlflow_client": fraud_mlflow_client,
         "fraud_data_api": fraud_data_api_resource,
    },
    jobs=[
        de_job,
        ml_job,
        ml_fraud_job,
    ],
    schedules=[era5_daily_schedule],
    asset_checks=[*all_de_checks, *all_ml_checks],
)
