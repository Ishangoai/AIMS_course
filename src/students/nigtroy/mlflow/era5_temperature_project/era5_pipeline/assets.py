import os

import cdsapi
import dagster as dg
import pandas as pd
import xarray as xr

# Hyperopt imports
from hyperopt import STATUS_FAIL, STATUS_OK, Trials, fmin, hp, tpe
from sklearn.linear_model import Ridge  # Changed from LinearRegression for tuning
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split  # For robust splitting

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


@dg.asset(
    name="raw_era5_temperature_data",
    description="Fetches raw ERA5 2m temperature data from the CDS.",
    required_resource_keys={"mlflow_tracking"},
    compute_kind="python",
    group_name="1_ingestion"  # Updated group name
)
def fetch_era5_data(context: dg.AssetExecutionContext) -> str:
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
    name="processed_temperature_dataframe",
    description="Loads the raw NetCDF data into a pandas DataFrame and logs some metrics.",
    deps=[fetch_era5_data],
    required_resource_keys={"mlflow_tracking"},
    compute_kind="python",
    group_name="2_processing"
)
def process_temperature_dataframe(
    context: dg.AssetExecutionContext,
    raw_era5_temperature_data: str) -> dg.MaterializeResult:

    mlflow_client = context.resources.mlflow_tracking
    context.log.info(f"Processing file: {raw_era5_temperature_data}")
    try:
        ds = xr.open_dataset(raw_era5_temperature_data)
    except FileNotFoundError:
        context.log.error(f"Input file {raw_era5_temperature_data} not found")
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
    name="cleaned_temperature_data_pandas",
    description="Cleans the temperature data. Converts Kelvin to Celsius.",
    deps=[process_temperature_dataframe],
    required_resource_keys={"mlflow_tracking"},
    compute_kind="python",
    group_name="2_processing"
)
def clean_temperature_data_pandas(context: dg.AssetExecutionContext,
                                  processed_temperature_dataframe: pd.DataFrame) -> dg.MaterializeResult:
    mlflow_client = context.resources.mlflow_tracking
    context.log.info("Starting data cleaning.")
    df = processed_temperature_dataframe.copy()

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
    name="tuned_hyperparameters_and_data_split",
    description="Tunes Ridge regression hyperparameters using Hyperopt and prepares data splits.",
    deps=[clean_temperature_data_pandas],
    required_resource_keys={"mlflow_tracking"},
    compute_kind="python",
    group_name="3_modeling",
)
def tune_ridge_hyperparameters(context: dg.AssetExecutionContext,  # noqa: C901
                               cleaned_temperature_data_pandas: pd.DataFrame):
    mlflow_client = context.resources.mlflow_tracking
    context.log.info("Starting hyperparameter tuning for Ridge model.")

    if len(cleaned_temperature_data_pandas) < 20:  # Increased threshold for meaningful splits
        msg = "Not enough data points for hyperparameter tuning, training, and testing. Need at least 20."
        context.log.error(msg)
        raise ValueError(msg)

    # Feature Engineering: Create a lagged temperature feature
    lag_period = 1
    df_with_lag = cleaned_temperature_data_pandas.copy()
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
    name="trained_tuned_model_data",
    description="Trains a Ridge model using the best hyperparameters found by Hyperopt.",
    deps=["tuned_hyperparameters_and_data_split"],
    required_resource_keys={"mlflow_tracking"},
    compute_kind="python",
    group_name="3_modeling"
)
def train_tuned_model(context: dg.AssetExecutionContext, tuned_hyperparameters_and_data_split) -> dict:
    mlflow_client = context.resources.mlflow_tracking
    context.log.info("Starting final model training with tuned hyperparameters.")

    best_params = tuned_hyperparameters_and_data_split["best_params"]
    X_train_val = tuned_hyperparameters_and_data_split["X_train_val"]
    y_train_val = tuned_hyperparameters_and_data_split["y_train_val"]
    X_test = tuned_hyperparameters_and_data_split["X_test"]
    y_test = tuned_hyperparameters_and_data_split["y_test"]
    feature_names = tuned_hyperparameters_and_data_split["feature_names"]

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
    name="evaluated_temperature_model",
    description="Evaluates the tuned model and logs model and metrics to MLflow.",
    deps=["trained_tuned_model_data"],
    required_resource_keys={"mlflow_tracking"},
    compute_kind="python",
    group_name="3_modeling"
)
def evaluate_model(context: dg.AssetExecutionContext, trained_tuned_model_data: dict):
    mlflow_client = context.resources.mlflow_tracking
    context.log.info("Starting final model evaluation.")

    model = trained_tuned_model_data["model"]
    X_test = trained_tuned_model_data["X_test"]
    y_test = trained_tuned_model_data["y_test"]
    feature_names = trained_tuned_model_data.get("feature_names", ["feature"])  # Get feature names if provided

    if len(X_test) == 0:
        context.log.warning("Test set is empty. Skipping evaluation and model logging.")
        return {"warning": "Test set empty, evaluation skipped."}

    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    context.log.info(f"Final Model Evaluation Metrics on Test Set: MSE={mse:.4f}, MAE={mae:.4f}")

    eval_metrics = {"test_mse": mse, "test_mae": mae}  # Prefixed with test_ for clarity
    mlflow_client.log_metrics(eval_metrics)
    context.log.info(f"Logged final evaluation metrics to MLflow: {eval_metrics}")

    # Log the model to MLflow
    # For sklearn, it's good to provide an input example for signature inference
    input_example_df = pd.DataFrame(X_test[:5], columns=feature_names)
    mlflow_client.sklearn.log_model(
        sk_model=model,
        artifact_path="tuned_temperature_forecaster",  # This is how the model will be named in MLflow artifacts
        input_example=input_example_df,  # Provide an input example
        registered_model_name="tuned-temp-forecaster"  # Optional: register the model
    )
    context.log.info("Logged tuned model to MLflow Model Registry as 'tuned-temp-forecaster'.")
    return eval_metrics
