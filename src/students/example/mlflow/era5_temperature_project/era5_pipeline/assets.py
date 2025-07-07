import os
import cdsapi
import xarray as xr
import pandas as pd
from sklearn.linear_model import Ridge  # Changed from LinearRegression for tuning
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split  # For robust splitting

import dagster as dg
import mlflow
import mlflow.sklearn as ms

# Hyperopt imports
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials


# import mlflow # Ensure mlflow library is installed


# Configuration for the ERA5 data request
ERA5_REQUEST_PARAMS = {
    'product_type': 'reanalysis',
    'variable': '2m_temperature',
    'year': '2023',
    'month': '01',
    'day': [f"{i:02d}" for i in range(1, 16)],  # Fetch for 15 days for more data
    'time': ['00:00', '06:00', '12:00', '18:00'],
    'area': [50, -5, 45, 5],  # North, West, South, East (example: a small region in Europe)
    'format': 'netcdf',
}
OUTPUT_FILENAME = "era5_temperature_data.nc"  # Output file name for the downloaded data
MAX_HYPEROPT_EVALS = 20  # Max evaluations for Hyperopt

# Define promotion criteria
STAGING_MSE_THRESHOLD = 2
STAGING_MAE_THRESHOLD = 2


@dg.asset(
    description="Fetches raw ERA5 2m temperature data from the CDS.",
    required_resource_keys={"mlflow_tracking"},
    compute_kind="python",
    group_name="1_ingestion"  # Updated group name
)
def raw_netcdf_file(context: dg.AssetExecutionContext) -> str:
    """
    Fetches ERA5 temperature data and logs parameters to MLflow.
    Returns the path to the downloaded NetCDF file.
    """
    mlflow_client = context.resources.mlflow_tracking  # Use mlflow_client as the variable name for clarity

    # Log parameters to MLflow
    # MLflow prefers flat dictionaries for parameters
    flat_params = {}
    for key, value in ERA5_REQUEST_PARAMS.items():
        if isinstance(value, list):
            flat_params[f"{key}"] = ",".join(map(str, value))  # Convert lists to comma-separated strings
        else:
            flat_params[key] = str(value)
    mlflow_client.log_params(flat_params)
    context.log.info(f"Logged parameters to MLflow: {flat_params}")

    # Check if CDSAPI_KEY and CDSAPI_URL are set, otherwise cdsapi.Client() might fail silently or use defaults
    if not (os.getenv("CDSAPI_URL") and os.getenv("CDSAPI_KEY")):
        context.log.warning(
            "CDSAPI_URL and/or CDSAPI_KEY environment variables are not set. "
            "Ensure your CDS API credentials are configured, e.g., in a .cdsapirc file or environment variables."
        )
    c = cdsapi.Client()  # Assumes .cdsapirc is configured or env vars are set
    try:
        context.log.info(f"Requesting data with parameters: {ERA5_REQUEST_PARAMS}")
        c.retrieve(
            'reanalysis-era5-single-levels',
            ERA5_REQUEST_PARAMS,
            OUTPUT_FILENAME
        )
        context.log.info(f"Successfully downloaded data to {OUTPUT_FILENAME}")

        # Log an artifact (the downloaded file) to MLflow
        mlflow_client.log_artifact(OUTPUT_FILENAME, artifact_path="raw_data")
        context.log.info(f"Logged {OUTPUT_FILENAME} as an artifact to MLflow.")

        return OUTPUT_FILENAME
    except Exception as e:
        context.log.error(f"Error fetching ERA5 data: {e}")
        raise


@dg.asset(
    description="Loads the raw NetCDF data into a pandas DataFrame and logs some metrics.",
    deps=[raw_netcdf_file],
    required_resource_keys={"mlflow_tracking"},
    compute_kind="python",
    group_name="2_processing"
)
def create_pandas_df(
    context: dg.AssetExecutionContext,
    raw_netcdf_file: str) -> dg.MaterializeResult:

    mlflow_client = context.resources.mlflow_tracking
    context.log.info(f"Processing file: {raw_netcdf_file}")
    try:
        ds = xr.open_dataset(raw_netcdf_file)
    except FileNotFoundError:
        context.log.error(f"Input file {raw_netcdf_file} not found")
        raise

    # Convert to pandas DataFrame
    df: pd.DataFrame = ds['t2m'].to_dataframe().reset_index()  # t2m is 2m temperature
    context.log.info(df)
    df = pd.DataFrame(df[['valid_time', 'latitude', 'longitude', 't2m']])  # Select and order columns
    df = df.rename(columns={'valid_time': 'time'})

    # For simplicity, if multiple lat/lon, average them or pick one. Here, we average if multiple.
    # ERA5 data for a region will have multiple lat/lon points per time.
    # For a very simple time series model, we need a single value per timestamp.

    if 'latitude' in df.columns and 'longitude' in df.columns:
        df_agg: pd.DataFrame = df.groupby('time')['t2m'].mean().reset_index()
        context.log.info("Aggregated multiple lat/lon points by averaging 't2m' per timestamp.")
    else:
        df_agg: pd.DataFrame = df

    num_time_steps = len(df_agg)
    mean_temp_kelvin = float(df_agg['t2m'].mean()) if not df_agg['t2m'].empty else float('nan')

    mlflow_client.log_metric("processed_num_time_steps", num_time_steps)
    if pd.notna(mean_temp_kelvin):
        mlflow_client.log_metric("processed_mean_temperature_k", mean_temp_kelvin)
    context.log.info(f"Pandas DataFrame created. Time steps: {num_time_steps}, Mean Temp (K): {mean_temp_kelvin:.2f}")
    context.log.info("Logged metrics to MLflow.")
    return dg.MaterializeResult(
        value=df_agg,
        metadata={
            "number of rows": dg.MetadataValue.int(len(df_agg)),
            "preview": dg.MetadataValue.md(df_agg.head().to_markdown() or "")
        }
    )


@dg.asset(
    description="Cleans the temperature data. Converts Kelvin to Celsius.",
    deps=[create_pandas_df],
    required_resource_keys={"mlflow_tracking"},
    compute_kind="python",
    group_name="2_processing"
)
def clean_df(context: dg.AssetExecutionContext,
                                  create_pandas_df: pd.DataFrame) -> dg.MaterializeResult:
    mlflow_client = context.resources.mlflow_tracking
    context.log.info("Starting data cleaning.")
    df = create_pandas_df.copy()

    if df.empty:
        context.log.warning("Input DataFrame is empty. Skipping cleaning.")

    # 1. Convert temperature from Kelvin to Celsius
    df["t2m_celsius"] = df["t2m"] - 273.15

    # 2. Ensure 'time' column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])

    # 3. Sort by time
    df = df.sort_values("time").reset_index(drop=True)  # Reset index after sort

    # 4. Handle missing values (drop rows with nulls in 't2m_celsius')
    initial_rows = len(df)
    df = df.dropna(subset=["t2m_celsius"])
    rows_after_dropna = len(df)
    context.log.info(f"Dropped {initial_rows - rows_after_dropna} rows with null t2m_celsius.")

    cleaned_rows = len(df)
    mean_temp_celsius = df["t2m_celsius"].mean() if not df["t2m_celsius"].empty else float('nan')

    mlflow_client.log_metric("cleaned_num_rows", cleaned_rows)
    if bool(pd.notna(mean_temp_celsius)):
        mlflow_client.log_metric("cleaned_mean_temperature_c", mean_temp_celsius)
        context.log.info(f"Data cleaned. Rows: {cleaned_rows}, Mean Temp (C): {mean_temp_celsius:.2f}")
    else:
        context.log.warning("Mean temperature (C) could not be calculated.")
        mlflow_client.log_metric("cleaned_mean_temperature_c", float('nan'))

    # Log a sample of the cleaned data to MLflow as a CSV artifact (optional)
    if not df.empty:
        sample_csv_path = "cleaned_sample.csv"
        df.head().to_csv(sample_csv_path, index=False)
        mlflow_client.log_artifact(sample_csv_path, artifact_path="processed_data_samples")
        try:
            # Option 1: If you want to create the dataset from the in-memory pandas DataFrame sample
            # cleaned_sample_dataset = from_pandas(
            #     df=sample_df_for_artifact,
            #     source=f"file://{os.path.abspath(sample_csv_path)}", # Original source file
            #     name="cleaned_temperature_sample"
            # )
            # mlflow.log_input(dataset=cleaned_sample_dataset, context="cleaned_data_sample")

            # Option 2: Create the dataset from the CSV file you just wrote (simpler if you just want to mark the file)
            dataset_source_uri = f"file://{os.path.abspath(sample_csv_path)}"
            cleaned_csv_as_input = mlflow_client.data.from_files(path=dataset_source_uri,
                                                                 name="cleaned_temperature_sample_csv")
            mlflow_client.log_input(dataset=cleaned_csv_as_input, context="cleaned_data_sample")
        except Exception as e:
            context.log.warning(f"Could not log {sample_csv_path} as an MLflow Dataset: {e}")

    return dg.MaterializeResult(
        value=df,
        metadata={
            "number of rows": dg.MetadataValue.int(len(df)),
            "preview": dg.MetadataValue.md(df.head().to_markdown() or "")
        }
    )


@dg.asset(
    description="Tunes Ridge regression hyperparameters using Hyperopt and prepares data splits.",
    deps=[clean_df],
    required_resource_keys={"mlflow_tracking"},
    compute_kind="python",
    group_name="3_modeling",
)
def tune_ridge_hyperparameters(context: dg.AssetExecutionContext,  # noqa: C901
                               clean_df: pd.DataFrame):
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
        'alpha': hp.loguniform('alpha', -5, 2)  # Alpha between ~0.0067 and ~7.39
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

    trials = Trials()

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
                return {'loss': float('inf'), 'status': STATUS_OK, 'params': params}  # Penalize if split fails
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
            return {'loss': rmse, 'status': STATUS_OK, 'params': params}
        except Exception as e:
            context.log.error(f"Trial {trial_num}: Exception in objective function for params {params}: {e}",
                              exc_info=True)
            return {'loss': float('inf'), 'status': STATUS_FAIL, 'params': params, 'error_message': str(e)}

    best_hyperparams = fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=MAX_HYPEROPT_EVALS,  # Number of iterations
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
    deps=["tune_ridge_hyperparameters"],
    required_resource_keys={"mlflow_tracking"},
    compute_kind="python",
    group_name="3_modeling"
)
def train_tuned_model(context: dg.AssetExecutionContext, tune_ridge_hyperparameters) -> dict:
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
    deps=["train_tuned_model"],
    required_resource_keys={"mlflow_tracking"},
    compute_kind="python",
    group_name="3_evaluation"
)
def evaluate_model(context: dg.AssetExecutionContext, train_tuned_model: dict) -> dg.MaterializeResult:
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
                "test_mae": dg.MetadataValue.float(float('nan'))
            }
        )

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    context.log.info(f"Final Model Evaluation Metrics on Test Set: MSE={mse:.4f}, MAE={mae:.4f}")

    eval_metrics = {"test_mse": mse, "test_mae": mae}
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
                f"Could not find registered model version for run ID {current_run.info.run_id} and name '{registered_model_name}'."
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
            "test_mse": dg.MetadataValue.float(mse),
            "test_mae": dg.MetadataValue.float(mae),
            "model_name": dg.MetadataValue.text(model_version_info["name"]),
            "model_version": dg.MetadataValue.text(str(model_version_info["version"])),
            "mlflow_run_id": dg.MetadataValue.text(current_run.info.run_id),
            "mlflow_model_uri": dg.MetadataValue.text(model_version_info["model_uri"]),
            "mlflow_stage": dg.MetadataValue.text(model_version_info["stage"])
        }
    )


@dg.asset(
    description="Promotes the newly trained model to Staging if it meets performance criteria.",
    deps=["evaluate_model"],
    required_resource_keys={"mlflow_tracking", "mlflow_client"},
    compute_kind="python",
    group_name="4_promotion"
)
def promote_model_to_staging(context: dg.AssetExecutionContext, evaluate_model: dict) -> dg.MaterializeResult:
    # mlflow_tracking = context.resources.mlflow_tracking   # for logging
    mlflow_client = context.resources.mlflow_client  # for stage transition
    context.log.info("Starting model promotion to Staging.")

    # Access the value returned by evaluate_model
    # Check if the evaluation was skipped upstream
    if evaluate_model.get("status") == "skipped_evaluation":
        context.log.info("Evaluation was skipped in the previous step. Skipping staging promotion.")
        return dg.MaterializeResult(
            value={"status": "skipped_promotion", "reason": "evaluation_skipped_upstream"},
            metadata={"status": "skipped_promotion_due_to_upstream_skip"}
        )

    eval_metrics = evaluate_model.get("eval_metrics", {})
    model_version_info = evaluate_model.get("model_version_info")

    # This check now also catches cases where model_version_info might be None due to an upstream failure
    if not model_version_info:
        context.log.warning("No valid model version info found from evaluation. Skipping staging promotion.")
        return dg.MaterializeResult(
            value={"status": "skipped_promotion", "reason": "no_model_version_info_from_evaluation"},
            metadata={"status": "skipped_promotion_no_model_info"}
        )

    current_mse = eval_metrics.get("test_mse", float('inf'))
    current_mae = eval_metrics.get("test_mae", float('inf'))

    context.log.info(f"Model evaluated with MSE: {current_mse:.4f}, MAE: {current_mae:.4f}")
    context.log.info(f"Staging promotion thresholds: MSE < {STAGING_MSE_THRESHOLD}, MAE < {STAGING_MAE_THRESHOLD}")

    if current_mse <= STAGING_MSE_THRESHOLD and current_mae <= STAGING_MAE_THRESHOLD:
        try:
            model_name = model_version_info["name"]
            model_version = model_version_info["version"]

            context.log.info(f"Model '{model_name}' (version {model_version}) meets criteria. Promoting to Staging")
            mlflow_client.transition_model_version_stage(
                name=model_name,
                version=model_version,
                stage="Staging"
            )
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
                    "mae_at_promotion": dg.MetadataValue.float(current_mae)
                }
            )
        except Exception as e:
            context.log.error(f"Error promoting model to Staging: {e}")
            return dg.MaterializeResult(
                value={"status": "failed_promotion_to_staging", "error": str(e)},
                metadata={"status": "failed_promotion_to_staging", "error_message": dg.MetadataValue.text(str(e))}
            )
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
                "mae": dg.MetadataValue.float(current_mae)
            }
        )


@dg.asset(
    # name="promote_model_to_production",
    description="Promotes the best model from Staging to Production, usually with manual approval.",
    deps=["promote_model_to_staging"],
    required_resource_keys={"mlflow_tracking", "mlflow_client"},
    compute_kind="python",
    group_name="4_promotion"
)
def promote_model_to_production(context: dg.AssetExecutionContext, promote_model_to_staging: dict) -> dg.MaterializeResult:
    # mlflow_client = context.resources.mlflow_tracking
    mlflow_client = context.resources.mlflow_client
    context.log.info("Starting model promotion to Production.")

    # Access the value returned by promote_model_to_staging
    if promote_model_to_staging.get("status") != "promoted_to_staging":
        context.log.info("No model was promoted to Staging in the previous step. Skipping production promotion.")
        return dg.MaterializeResult(
            value={"status": "skipped_production_promotion", "reason": "no_model_in_staging_from_previous_step"},
            metadata={"status": "skipped_production_promotion"}
        )

    model_name = promote_model_to_staging.get("model_name", "tuned-temp-forecaster")

    # In a real scenario, this would involve a manual review/approval process.
    manual_approval_granted = True  # This would likely be an external input or a check against an approval system

    if manual_approval_granted:
        try:
            # Find the latest model version in Staging for the given model_name
            latest_staging_version = None
            for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
                if mv.current_stage == "Staging":
                    # Ensure we promote the highest version currently in Staging
                    if latest_staging_version is None or mv.version > latest_staging_version.version:
                        latest_staging_version = mv

            if not latest_staging_version:
                context.log.warning(f"No model found in Staging stage for '{model_name}'. Skipping prod promotion.")
                return dg.MaterializeResult(
                    value={"status": "skipped_production_promotion", "reason": "no_staging_model_found_for_prod"},
                    metadata={"status": "skipped_production_promotion_no_staging_model"}
                )

            prod_model_name = latest_staging_version.name
            prod_model_version = latest_staging_version.version

            context.log.info(f"Manual approval granted for model '{prod_model_name}' (version {prod_model_version})")
            mlflow_client.transition_model_version_stage(
                name=prod_model_name,
                version=prod_model_version,
                stage="Production"
            )
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
            context.log.error(f"Error promoting model to Production: {e}")
            return dg.MaterializeResult(
                value={"status": "failed_production_promotion", "error": str(e)},
                metadata={"status": "failed_production_promotion", "error_message": dg.MetadataValue.text(str(e))}
            )
    else:
        context.log.info("Manual approval not granted. Skipping production promotion")
        return dg.MaterializeResult(
            value={"status": "not_promoted_to_production", "reason": "manual_approval_denied"},
            metadata={"status": "not_promoted_to_production"}
        )
