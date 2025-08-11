import os
from collections import abc

import dagster as dg
import hyperopt
import mlflow
import mlflow.sklearn as ms
import pandas as pd
import xarray as xr
from sklearn.linear_model import Ridge  # Changed from LinearRegression for tuning
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split  # For robust splitting

from .resources import CDSAPIResource, Era5RequestConfig, PromotionConfig, TuningConfig, mlflow_client, mlflow_resource


@dg.asset(
    description="Fetches raw ERA5 2m temperature data from the CDS.",
    resource_defs={"mlflow_tracking": mlflow_resource, "cds_api": CDSAPIResource()},
    compute_kind="python",
    group_name="ml_ingest"
)
def raw_xarray_dataset(
    context: dg.AssetExecutionContext,
    config: Era5RequestConfig,
) -> dg.MaterializeResult:
    """
    Fetches ERA5 temperature data and logs parameters to MLflow.
    Returns the path to the downloaded NetCDF file.
    """
    # Initialize mlflow tracking
    mlflow_client = context.resources.mlflow_tracking

    OUTPUT_FILENAME = os.getcwd() + "/era5_temperature_data.nc"

    # Convert the Config object to a dictionary for the CDS API
    era5_request_params_dict = config.model_dump()

    # Log parameters to MLflow
    # MLflow prefers flat dictionaries for parameters
    flat_params = {}
    for key, value in era5_request_params_dict.items():
        if isinstance(value, list):
            # Convert lists to comma-separated strings
            flat_params[f"{key}"] = ",".join(map(str, value))
        else:
            flat_params[key] = str(value)
    mlflow_client.log_params(flat_params)
    context.log.info(f"Logged parameters to MLflow: {flat_params}")

    # Assumes env var CDS_API_KEY is set as an environment variable
    # this can be confimed in bash terminal with "echo $CDS_API_KEY"
    c = context.resources.cds_api.client
    context.log.info("Download data from CSD API")

    c.retrieve(
        'reanalysis-era5-single-levels',
        era5_request_params_dict,
        OUTPUT_FILENAME
    )
    context.log.info(f"Successfully downloaded data to {OUTPUT_FILENAME}")

    # log file size to dagster
    size = os.path.getsize(OUTPUT_FILENAME)
    context.log.info(f"Logged {OUTPUT_FILENAME} as artifact to MLflow. Size: {size} bytes")

    # convert netcdf file to xarray Dataset
    ds = xr.open_dataset(OUTPUT_FILENAME)
    os.remove(OUTPUT_FILENAME)  # Remove the file after loading

    return dg.MaterializeResult(
        value=ds,
        metadata={
            "source": dg.MetadataValue.text(str(OUTPUT_FILENAME)),
            "size_bytes": dg.MetadataValue.int(size),
            "preview": dg.MetadataValue.md(ds.to_dataframe().head().to_markdown() or ""),
            "parameters": dg.MetadataValue.json(flat_params),
            "description": dg.MetadataValue.text(
                "Raw ERA5 2m temperature data for January 2023, fetched from the CDS."
            )
        }
    )


@dg.asset(
    description="Loads the raw xarray data into a pandas DataFrame and logs some metrics.",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_transform"
)
def raw_pandas_df(
    context: dg.AssetExecutionContext,
    raw_xarray_dataset: xr.Dataset
) -> dg.MaterializeResult:

    mlflow_client = context.resources.mlflow_tracking
    context.log.info(f"Processing file:\n {raw_xarray_dataset}")

    # Convert to pandas DataFrame
    # t2m is 2m air temperature
    df: pd.DataFrame = raw_xarray_dataset['t2m'].to_dataframe().reset_index()
    context.log.info(f"Pandas DataFrame:\n {df}")

    # Select and order columns
    df = pd.DataFrame(df[['valid_time', 'latitude', 'longitude', 't2m']])
    df = df.rename(columns={'valid_time': 'time'})

    dataset = mlflow_client.data.from_pandas(df, name="era5_raw_temperature_data")
    mlflow_client.log_input(dataset=dataset, context="training")

    columns = [dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]

    return dg.MaterializeResult(
        value=df,
        metadata={
            "preview": dg.MetadataValue.md(df.head().to_markdown() or ""),
            "dagster/row_count": len(df),
            "dagster/column_schema": dg.TableSchema(columns=columns)
        }
    )


@dg.asset(
    description="Takes spatial mean. Converts Kelvin to Celsius. Cleans columns",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_transform"
)
def clean_df(
    context: dg.AssetExecutionContext,
    raw_pandas_df: pd.DataFrame
) -> dg.MaterializeResult:
    mlflow_client = context.resources.mlflow_tracking
    context.log.info("Starting data cleaning.")

    # ERA5 data for a region will have multiple lat/lon points per time.
    # For a very simple time series model, we need a single value per timestamp.
    # Take a spatial mean across all lat/lon points for each timestamp.
    df_spatial_mean: pd.DataFrame = raw_pandas_df.groupby('time')['t2m'].mean().reset_index()
    context.log.info("Aggregated multiple lat/lon points by averaging 't2m' per timestamp.")

    num_time_steps = len(df_spatial_mean)
    mean_temp_kelvin = df_spatial_mean['t2m'].mean()

    mlflow_client.log_metric("processed_num_time_steps", num_time_steps)
    mlflow_client.log_metric("processed_mean_temperature_k", mean_temp_kelvin)
    context.log.info(f"Pandas DataFrame created. Time steps: {num_time_steps}, Mean Temp (K): {mean_temp_kelvin:.2f}")
    context.log.info("Logged metrics to MLflow.")

    # 1. Convert temperature from Kelvin to Celsius
    df_spatial_mean["t2m_celsius"] = df_spatial_mean["t2m"] - 273.15

    # 2. Ensure 'time' column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df_spatial_mean["time"]):
        df_spatial_mean["time"] = pd.to_datetime(df_spatial_mean["time"])

    # 3. Sort by time
    # Reset index after sort
    df_spatial_mean = df_spatial_mean.sort_values("time").reset_index(drop=True)

    # Log the cleaned data to MLflow
    spatial_mean_dataset = mlflow_client.data.from_pandas(df_spatial_mean, name="cleaned_spatial_mean_temperature")
    mlflow_client.log_input(dataset=spatial_mean_dataset, context="training")

    columns = [dg.TableColumn(k, str(v)) for k, v in df_spatial_mean.dtypes.to_dict().items()]

    return dg.MaterializeResult(
        value=df_spatial_mean,
        metadata={
            "dagster/row_count": len(df_spatial_mean),
            "dagster/column_schema": dg.TableSchema(columns=columns),
            "preview": dg.MetadataValue.md(df_spatial_mean.head().to_markdown() or "")
        }
    )


@dg.asset(
    description="Tunes Ridge regression hyperparameters using Hyperopt and prepares data splits.",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_model",
)
def tune_ridge_hyperparameters(  # noqa: C901
    context: dg.AssetExecutionContext,
    config: TuningConfig,
    clean_df: pd.DataFrame
) -> dict:
    mlflow_client = context.resources.mlflow_tracking
    context.log.info("Starting hyperparameter tuning for Ridge model.")

    if len(clean_df) < 20:  # Increased threshold for meaningful splits
        msg = "Not enough data points for hyperparameter tuning, training, and testing. Need at least 20."
        context.log.error(msg)
        raise ValueError(msg)

    # Feature Engineering: Create a lagged temperature feature
    lag_period = 1
    df_with_lag = clean_df.copy()
    feature_name = f"t2m_celsius_lag{lag_period}"
    df_with_lag[feature_name] = df_with_lag["t2m_celsius"].shift(lag_period)
    df_with_lag = df_with_lag.dropna().reset_index(drop=True)
    feature_names = [feature_name]

    if len(df_with_lag) < 10:  # Check after lag
        msg = f"Not enough data points after creating lag{lag_period} feature. Need at least 10."
        context.log.error(msg)
        raise ValueError(msg)

    X = df_with_lag[feature_names].values
    y = df_with_lag["t2m_celsius"].values

    # Split data: 70% for training + hyperopt validation, 30% for final test
    X_train_val, X_test, y_train_val, y_test = train_test_split(X,
                                                                y,
                                                                test_size=0.3,
                                                                random_state=42,
                                                                shuffle=False)  # Time series, so shuffle=False

    if len(X_train_val) < 5 or len(X_test) < 1:  # Need enough for hyperopt val and at least one test sample
        msg = "Train/validation or test set is too small after initial split."
        context.log.error(msg)
        raise ValueError(msg)

    context.log.info(f"Data split: X_train_val: {X_train_val.shape}, X_test: {X_test.shape}")  # type: ignore

    # Define Hyperopt search space for Ridge alpha
    search_space = {
        'alpha': hyperopt.hp.loguniform('alpha', -5, 2)  # Alpha between ~0.0067 and ~7.39
    }

    # MLflow experiment context for nested runs
    # Ensure the experiment exists or is created
    try:
        experiment = mlflow_client.get_experiment_by_name("era5_temperature_analysis")
        if experiment is None:
            experiment = mlflow_client.create_experiment("era5_temperature_analysis")
            experiment_id = experiment.experiment_id
        else:
            experiment_id = experiment.experiment_id
    except Exception:  # Handle cases where get_experiment_by_name might raise error if not found
        experiment_id = mlflow_client.create_experiment("era5_temperature_analysis")

    trials = hyperopt.Trials()

    # Objective function for Hyperopt
    def objective(params):
        trial_num = len(trials.trials)
        context.log.info(f"Starting Hyperopt Trial {trial_num} with params: {params}")
        try:
            alpha = params['alpha']
            # Split train_val further into training_for_hyperopt and validation_for_hyperopt
            X_train_h, X_val_h, y_train_h, y_val_h = train_test_split(X_train_val,
                                                                  y_train_val,
                                                                  test_size=0.25,
                                                                  random_state=42,
                                                                  shuffle=False)

            if len(X_train_h) == 0 or len(X_val_h) == 0:
                context.log.warning(f"Trial {trial_num}: Skipped due to empty train/val split for params: {params}")
                # This case should be rare given prior checks but good to have
                return {'loss': float('inf'), 'status': hyperopt.STATUS_OK, 'params': params}  # Penalize if split fails
            run_name = f"hyperopt_trial_{trial_num}_alpha_{alpha:.4f}"

            with mlflow_client.start_run(experiment_id=experiment_id,
                                        run_name=run_name,
                                        nested=True):
                mlflow_client.log_params(params)
                model = Ridge(alpha=alpha)
                model.fit(X_train_h, y_train_h)
                preds = model.predict(X_val_h)
                rmse = mean_squared_error(y_val_h, preds)
                mlflow_client.log_metric("validation_mse", rmse)
                context.log.info(f"Trial {trial_num} successful: params={params}, mse={rmse:.4f}")
            return {'loss': rmse, 'status': hyperopt.STATUS_OK, 'params': params}
        except Exception as e:
            context.log.error(f"Trial {trial_num}: Exception in objective function for params {params}: {e}",
                              exc_info=True)
            return {'loss': float('inf'), 'status': hyperopt.STATUS_FAIL, 'params': params, 'error_message': str(e)}

    best_hyperparams = hyperopt.fmin(
        fn=objective,
        space=search_space,
        algo=hyperopt.tpe.suggest,
        max_evals=config.max_hyperopt_evals,  # Number of iterations
        trials=trials
    )
    context.log.info(f"fmin completed. Returned best_hyperparams: {best_hyperparams}")
    best_trial_info = trials.best_trial
    best_alpha_to_log = float('nan')  # Initialize
    best_loss_to_log = float('inf')  # Initialize

    if best_trial_info is None:
        context.log.error("trials.best_trial is None.")
        if best_hyperparams:
            context.log.warning(f"Using fmin's output: {best_hyperparams}")
            best_alpha_to_log = best_hyperparams['alpha']
            # best_loss_to_log remains float('inf')
        else:
            raise ValueError("Hyperopt tuning failed: fmin returned no parameters and no successful trials found.")
    else:
        best_loss_to_log = best_trial_info['result']['loss']
        alpha_from_best_trial = best_trial_info['misc']['vals']['alpha'][0]
        best_alpha_to_log = best_hyperparams.get('alpha', alpha_from_best_trial)  # type: ignore
        context.log.info(
            f"Hyperopt best alpha (fmin): {best_hyperparams.get('alpha', 'N/A')}, "  # type: ignore
            f"Best alpha (trials.best_trial): {alpha_from_best_trial:.4f}, "
            f"Best validation_mse (trials.best_trial): {best_loss_to_log:.4f}"
        )

    context.log.info(f"Final best alpha: {best_alpha_to_log:.4f}, Corresponding mse: {best_loss_to_log:.4f}")
    mlflow_client.log_param("best_ridge_alpha", best_alpha_to_log)
    if best_loss_to_log != float('inf'):
        mlflow_client.log_metric("best_hyperopt_validation_mse", best_loss_to_log)
    else:
        # log something to indicate the metric wasn't reliably found
        mlflow_client.log_metric("best_hyperopt_validation_mse_unavailable", 1.0)

    final_best_params_output = {'alpha': best_alpha_to_log}

    # Return a single dictionary
    return {
        "best_params": final_best_params_output,
        "X_train_val": X_train_val,
        "y_train_val": y_train_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
    }


@dg.asset(
    description="Trains a Ridge model using the best hyperparameters found by Hyperopt.",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_model"
)
def train_tuned_model(
    context: dg.AssetExecutionContext,
    tune_ridge_hyperparameters
) -> dict:
    mlflow_client = context.resources.mlflow_tracking
    context.log.info("Starting final model training with tuned hyperparameters.")

    best_params = tune_ridge_hyperparameters["best_params"]
    X_train_val = tune_ridge_hyperparameters["X_train_val"]
    y_train_val = tune_ridge_hyperparameters["y_train_val"]
    X_test = tune_ridge_hyperparameters["X_test"]
    y_test = tune_ridge_hyperparameters["y_test"]
    feature_names = tune_ridge_hyperparameters["feature_names"]

    context.log.info(f"Training Ridge model with parameters: {best_params}")
    context.log.info(f"Training on {len(X_train_val)} samples.")

    final_model = Ridge(alpha=best_params['alpha'])
    final_model.fit(X_train_val, y_train_val)
    context.log.info("Final Ridge model trained.")

    train_params_log = {
        "model_type": "Ridge",
        "alpha": best_params['alpha'],
        "feature_used": ", ".join(feature_names),
        "lag_period": 1,  # Assuming lag_period is 1 as per feature eng.
        "final_train_samples": len(X_train_val),
        "final_test_samples": len(X_test)
    }
    mlflow_client.log_params(train_params_log)
    context.log.info(f"Logged final training parameters to MLflow: {train_params_log}")

    return {
        "model": final_model,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names
    }


@dg.asset(
    description="Evaluates the tuned model and logs model and metrics to MLflow.",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_evaluate"
)
def evaluate_model(
    context: dg.AssetExecutionContext,
    train_tuned_model: dict
) -> dg.MaterializeResult:
    mlflow_client = context.resources.mlflow_tracking
    context.log.info("Starting final model evaluation.")

    model = train_tuned_model["model"]
    X_test = train_tuned_model["X_test"]
    y_test = train_tuned_model["y_test"]
    feature_names = train_tuned_model.get("feature_names", ["feature"])

    if len(X_test) == 0:
        context.log.warning("Test set is empty. Skipping evaluation and model logging.")
        return dg.MaterializeResult(
            value={
                "status": "skipped_evaluation",
                "reason": "Test set empty, evaluation skipped.",
                "eval_metrics": {"test_mse": float('nan'), "test_mae": float('nan')},
                "model_version_info": None
            },
            metadata={
                "status": "skipped_evaluation",
                "reason": dg.MetadataValue.text("Test set was empty, no evaluation performed."),
                "test_mse": dg.MetadataValue.float(float('nan')),
                "test_mae": dg.MetadataValue.float(float('nan')),
                "test_r2": dg.MetadataValue.float(float('nan')),
            }
        )

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    context.log.info(f"Final Model Evaluation Metrics on Test Set: MSE={mse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

    eval_metrics = {"test_mse": mse, "test_mae": mae, "test_r2": r2}
    mlflow_client.log_metrics(eval_metrics)
    context.log.info(f"Logged final evaluation metrics to MLflow: {eval_metrics}")

    registered_model_name = "tuned-temp-forecaster"
    model_version_info = None

    with mlflow.start_run(nested=True) as current_run:
        context.log.info(f"Starting nested MLflow run for model logging: {current_run.info.run_id}")

        log_model_info = ms.log_model(
            sk_model=model,
            artifact_path="tuned_temperature_forecaster",
            input_example=pd.DataFrame(X_test[:min(5, len(X_test))], columns=feature_names),
            registered_model_name=registered_model_name
        )
        context.log.info(f"Model logged to MLflow Run ID: {current_run.info.run_id}")
        context.log.info(f"Logged model artifact URI: {log_model_info.model_uri}")

        # use search_model_versions with a proper filter string
        model_versions = mlflow_client.search_model_versions(
            filter_string=f"name='{registered_model_name}'"
        )

        # Find the model version registered in this run
        matching_versions = [
            mv for mv in model_versions if mv.run_id == current_run.info.run_id
        ]

        if matching_versions:
            registered_model_version = matching_versions[0]
            model_version_info = {
                "name": registered_model_version.name,
                "version": registered_model_version.version,
                "status": registered_model_version.status,
                "stage": registered_model_version.current_stage,
                "model_uri": f"models:/{registered_model_version.name}/{registered_model_version.version}"
            }
            context.log.info("Successfully retrieved registered model version info from registry.")
        else:
            context.log.error(
                f"Could not find registered model version for run ID {current_run.info.run_id} "
                f"and name '{registered_model_name}'."
            )
            raise Exception("Failed to retrieve registered model version details after logging.")

    output_value_for_downstream = {
        "eval_metrics": eval_metrics,
        "model_version_info": model_version_info,
        "status": "evaluated_successfully"
    }

    return dg.MaterializeResult(
        value=output_value_for_downstream,
        metadata={
            "test_mse": dg.MetadataValue.float(float(mse)),
            "test_mae": dg.MetadataValue.float(float(mae)),
            "test_r2": dg.MetadataValue.float(float(r2)),
            "model_name": dg.MetadataValue.text(model_version_info["name"]),
            "model_version": dg.MetadataValue.text(str(model_version_info["version"])),
            "mlflow_run_id": dg.MetadataValue.text(current_run.info.run_id),
            "mlflow_model_uri": dg.MetadataValue.text(model_version_info["model_uri"]),
            "mlflow_stage": dg.MetadataValue.text(model_version_info["stage"])
        }
    )


# Promoting a model to Staging means it passed your quality checks and is ready
# for more thorough testing or limited release
@dg.asset(
    description="Promotes the newly trained model to Staging if it meets performance criteria.",
    resource_defs={"mlflow_tracking": mlflow_resource, "mlflow_client": mlflow_client},
    compute_kind="python",
    group_name="ml_promote"
)
def promote_model_to_staging(
    context: dg.AssetExecutionContext,
    config: PromotionConfig,
    evaluate_model: dict
) -> dg.MaterializeResult:
    # Get the MLflow client from the context to interact with the model registry
    mlflow_client = context.resources.mlflow_client
    context.log.info("Starting model promotion to Staging.")

    # If the evaluation step was skipped, we also skip promotion
    if evaluate_model.get("status") == "skipped_evaluation":
        context.log.info("Evaluation was skipped in the previous step. Skipping staging promotion.")
        return dg.MaterializeResult(
            value={"status": "skipped_promotion", "reason": "evaluation_skipped_upstream"},
            metadata={"status": "skipped_promotion_due_to_upstream_skip"}
        )
    # Extract metrics and model version info from evaluation result
    eval_metrics = evaluate_model.get("eval_metrics", {})
    model_version_info = evaluate_model.get("model_version_info")

    # If no model version info was returned, skip promotion.
    # model_version_info might be None due to an upstream failure
    if not model_version_info:
        context.log.warning("No valid model version info found from evaluation. Skipping staging promotion.")
        return dg.MaterializeResult(
            value={"status": "skipped_promotion", "reason": "no_model_version_info_from_evaluation"},
            metadata={"status": "skipped_promotion_no_model_info"}
        )
    # Get performance metrics (default to infinity if missing)
    current_mse = eval_metrics.get("test_mse", float('inf'))
    current_mae = eval_metrics.get("test_mae", float('inf'))
    current_r2 = eval_metrics.get("test_r2", float('inf'))

    # STAGING_MSE_THRESHOLD = config.staging_mse_threshold
    # STAGING_MAE_THRESHOLD = config.staging_mae_threshold
    STAGING_R2_THRESHOLD = config.staging_r2_threshold
    # Log the evaluation metrics and threshold criteria
    context.log.info(f"Model evaluated with MSE: {current_mse:.4f}, MAE: {current_mae:.4f}")
    context.log.info(f"Staging promotion thresholds: R2 > {STAGING_R2_THRESHOLD}")

    # Check if model meets promotion criteria
    if current_r2 >= STAGING_R2_THRESHOLD:
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
                    "mse_at_promotion": dg.MetadataValue.float(current_mse),
                    "mae_at_promotion": dg.MetadataValue.float(current_mae),
                    "r2_at_promotion": dg.MetadataValue.float(current_r2)
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
                "mse": dg.MetadataValue.float(current_mse),
                "mae": dg.MetadataValue.float(current_mae),
                "r2": dg.MetadataValue.float(current_r2)
            }
        )


@dg.asset(
    description="Promotes the best model from Staging to Production, usually with manual approval.",
    resource_defs={"mlflow_tracking": mlflow_resource, "mlflow_client": mlflow_client},
    compute_kind="python",
    group_name="ml_promote"
)
def promote_model_to_production(
    context: dg.AssetExecutionContext,
    promote_model_to_staging: dict
) -> dg.MaterializeResult:
    # Get the MLflow client to interact with the model registry
    mlflow_client = context.resources.mlflow_client
    context.log.info("Starting model promotion to Production.")

    # Step 1: Check if a model was promoted to Staging previously
    if promote_model_to_staging.get("status") != "promoted_to_staging":
        # If no model was promoted to staging in the last step, skip production promotion
        context.log.info("No model was promoted to Staging in the previous step. Skipping production promotion.")
        return dg.MaterializeResult(
            value={"status": "skipped_production_promotion", "reason": "no_model_in_staging_from_previous_step"},
            metadata={"status": "skipped_production_promotion"}
        )
    # Get the model name from the previous promotion step
    model_name = promote_model_to_staging.get("model_name", "tuned-temp-forecaster")

    # Simulate manual approval
    # In a real scenario, this would involve a manual review/approval process.
    manual_approval_granted = True

    # Proceed with promotion only if manual approval is granted
    if manual_approval_granted:
        try:
            # Find the latest model version in Staging for the given model_name
            latest_staging_version = None
            for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
                if mv.current_stage == "Staging":
                    # Ensure we promote the highest version currently in Staging
                    if latest_staging_version is None or mv.version > latest_staging_version.version:
                        latest_staging_version = mv

            # If no model is found in staging, log a warning and skip promotion
            if not latest_staging_version:
                context.log.warning(f"No model found in Staging stage for '{model_name}'. Skipping prod promotion.")
                return dg.MaterializeResult(
                    value={"status": "skipped_production_promotion", "reason": "no_staging_model_found_for_prod"},
                    metadata={"status": "skipped_production_promotion_no_staging_model"}
                )

            # Extract the model name and version to promote
            prod_model_name = latest_staging_version.name
            prod_model_version = latest_staging_version.version

            # Archive all existing models in Production
            for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
                if mv.current_stage == "Production":
                    context.log.info(f"Archiving previous Production model '{mv.name}' (version {mv.version})")
                    mlflow_client.transition_model_version_stage(
                        name=mv.name,
                        version=mv.version,
                        stage="Archived"
                    )

            # Promote the new version to Production
            context.log.info(f"Promoting model '{prod_model_name}' (version {prod_model_version}) to Production")
            mlflow_client.transition_model_version_stage(
                name=prod_model_name,
                version=prod_model_version,
                stage="Production"
            )

            # Return success with metadata about the promoted model
            context.log.info(f"Model '{prod_model_name}' (version {prod_model_version}) promoted to Production.")
            return dg.MaterializeResult(
                value={
                    "status": "promoted_to_production",
                    "model_name": prod_model_name,
                    "model_version": prod_model_version,
                    "previous_metrics": promote_model_to_staging.get("metrics")
                },
                metadata={
                    "status": "promoted_to_production",
                    "model_name": dg.MetadataValue.text(prod_model_name),
                    "model_version": dg.MetadataValue.text(str(prod_model_version))
                }
            )
        except Exception as e:
            # Catch and log any error that occurs during the promotion process
            context.log.error(f"Error promoting model to Production: {e}")
            return dg.MaterializeResult(
                value={"status": "failed_production_promotion", "error": str(e)},
                metadata={"status": "failed_production_promotion", "error_message": dg.MetadataValue.text(str(e))}
            )

    # If manual approval was denied, skip promotion and return reason
    else:
        context.log.info("Manual approval not granted. Skipping production promotion")
        return dg.MaterializeResult(
            value={"status": "not_promoted_to_production", "reason": "manual_approval_denied"},
            metadata={"status": "not_promoted_to_production"}
        )


# Define a multi-asset check to validate the data quality of the assets
# This check is specific to the raw_pandas_df and ensures that the raw data
# does not contain null or impossible values
# usually you'd use @dg.multi_asset_check to check multiple assets
# but here we only check one asset
@dg.multi_asset_check(
    # Map checks to targeted assets
    specs=[
        dg.AssetCheckSpec(name="no_nulls", asset="raw_pandas_df", blocking=False),
        dg.AssetCheckSpec(name="impossible_temperatures", asset="raw_pandas_df", blocking=False),
    ]
)
def dq_check_ml(raw_pandas_df) -> abc.Iterable[dg.AssetCheckResult]:
    # Check for null temperature values
    num_null = raw_pandas_df["t2m"].isna().sum()
    yield dg.AssetCheckResult(
        check_name="no_nulls",
        passed=bool(num_null == 0),
        asset_key="raw_pandas_df",
    )

    # Check for impossible temperature values
    num_impossible_temperatures = (raw_pandas_df["t2m"] < 0).sum()
    yield dg.AssetCheckResult(
        check_name="impossible_temperatures",
        passed=bool(num_impossible_temperatures == 0),
        asset_key="raw_pandas_df",
    )
