from dagster import define_asset_job

# Only include assets up to the splitting step
data_preparation_job = define_asset_job(
    name="data_preparation_job",
    selection=["ingest_dataset", "preprocess_data", "splitting_data"]
)

dump_data_artifacts_job = define_asset_job(
    name="dump_data_artifacts_job",
    selection=["ingest_dataset", "preprocess_data", "splitting_data", "save_data_artifacts"]
)

model_training_job = define_asset_job(
    name="model_training_job",
    selection=["train_model", "notify_modelling_results"]
)
