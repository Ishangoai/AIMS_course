import os
import cdsapi
from dagster_mlflow import end_mlflow_on_run_finished
import xarray as xr
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from dagster import asset, Definitions, AssetExecutionContext, define_asset_job

from .resources import mlflow_resource # Import the configured resource

# Configuration for the ERA5 data request
ERA5_REQUEST_PARAMS = {
    'product_type': 'reanalysis',
    'variable': '2m_temperature',
    'year': '2023',
    'month': '01',
    'day': ['01', '02', '03', '04', '05'], # Fetch for 5 days
    'time': ['00:00', '06:00', '12:00', '18:00'],
    'area': [50, -5, 45, 5],  # North, West, South, East (example: a small region in Europe)
    'format': 'netcdf',
}
OUTPUT_FILENAME = "era5_temperature_data.nc"

@asset(
    name="raw_era5_temperature_data",
    description="Fetches raw ERA5 2m temperature data from the CDS.",
    required_resource_keys={"mlflow_tracking"}, # Specify the MLflow resource
    compute_kind="python",
    group_name="era5_ingestion"
)
def fetch_era5_data(context: AssetExecutionContext) -> str:
    """
    Fetches ERA5 temperature data and logs parameters to MLflow.
    Returns the path to the downloaded NetCDF file.
    """
    mlflow = context.resources.mlflow_tracking

    # Log parameters to MLflow
    # MLflow prefers flat dictionaries for parameters
    flat_params = {}
    for key, value in ERA5_REQUEST_PARAMS.items():
        if isinstance(value, list):
            flat_params[f"{key}"] = ",".join(map(str, value)) # Convert lists to comma-separated strings
        else:
            flat_params[key] = str(value)
    mlflow.log_params(flat_params)
    context.log.info(f"Logged parameters to MLflow: {flat_params}")

    c = cdsapi.Client() # Assumes .cdsapirc is configured
    try:
        context.log.info(f"Requesting data with parameters: {ERA5_REQUEST_PARAMS}")
        c.retrieve(
            'reanalysis-era5-single-levels',
            ERA5_REQUEST_PARAMS,
            OUTPUT_FILENAME
        )
        context.log.info(f"Successfully downloaded data to {OUTPUT_FILENAME}")

        # Log an artifact (the downloaded file) to MLflow
        mlflow.log_artifact(OUTPUT_FILENAME, artifact_path="raw_data")
        context.log.info(f"Logged {OUTPUT_FILENAME} as an artifact to MLflow.")

        return OUTPUT_FILENAME
    except Exception as e:
        context.log.error(f"Error fetching ERA5 data: {e}")
        raise


@asset(
    name="processed_temperature_dataframe",  # Renamed for clarity, now outputs pandas DataFrame
    description="Loads the raw NetCDF data into a pandas DataFrame and logs some metrics.",
    deps=[fetch_era5_data],
    required_resource_keys={"mlflow_tracking"},
    compute_kind="python",
    group_name="2_processing"
)
def process_temperature_dataframe(context: AssetExecutionContext, raw_era5_temperature_data: str) -> pd.DataFrame:
    mlflow = context.resources.mlflow_tracking
    context.log.info(f"Processing file: {raw_era5_temperature_data}")
    ds = xr.open_dataset(raw_era5_temperature_data)

    # Convert to pandas DataFrame
    df = ds['t2m'].to_dataframe().reset_index()  # t2m is 2m temperature
    context.log.info(df)
    df = df[['valid_time', 'latitude', 'longitude', 't2m']]  # Select and order columns
    df = df.rename(columns={'valid_time': 'time'})

    # For simplicity, if multiple lat/lon, average them or pick one. Here, we average if multiple.
    # ERA5 data for a region will have multiple lat/lon points per time.
    # For a very simple time series model, we need a single value per timestamp.
    if 'latitude' in df.columns and 'longitude' in df.columns:
        df_agg = df.groupby('time')['t2m'].mean().reset_index()
        context.log.info(f"Aggregated multiple lat/lon points by averaging 't2m' per timestamp.")
    else:
        df_agg = df

    num_time_steps = len(df_agg)
    mean_temp_kelvin = float(df_agg['t2m'].mean())

    mlflow.log_metric("processed_num_time_steps", num_time_steps)
    mlflow.log_metric("processed_mean_temperature_k", mean_temp_kelvin)
    context.log.info(f"Pandas DataFrame created. Time steps: {num_time_steps}, Mean Temp (K): {mean_temp_kelvin:.2f}")
    context.log.info("Logged metrics to MLflow.")
    return df_agg


@asset(
    name="cleaned_temperature_data_pandas",
    description="Cleans the temperature data using pandas. Converts Kelvin to Celsius.",
    deps=[process_temperature_dataframe],
    required_resource_keys={"mlflow_tracking"},
    compute_kind="python",
    group_name="2_processing"
)
def clean_temperature_data_pandas(context: AssetExecutionContext,
                                  processed_temperature_dataframe: pd.DataFrame) -> pd.DataFrame:
    mlflow = context.resources.mlflow_tracking
    context.log.info("Starting data cleaning with pandas.")

    df = processed_temperature_dataframe.copy()

    # 1. Convert temperature from Kelvin to Celsius
    df["t2m_celsius"] = df["t2m"] - 273.15

    # 2. Ensure 'time' column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df["time"]):
        df["time"] = pd.to_datetime(df["time"])

    # 3. Sort by time
    df = df.sort_values("time")

    # 4. Handle missing values (drop rows with nulls in 't2m_celsius')
    initial_rows = len(df)
    df = df.dropna(subset=["t2m_celsius"])
    rows_after_dropna = len(df)
    if initial_rows > rows_after_dropna:
        context.log.info(f"Dropped {initial_rows - rows_after_dropna} rows with null t2m_celsius.")

    cleaned_rows = len(df)
    mean_temp_celsius = df["t2m_celsius"].mean()

    mlflow.log_metric("cleaned_num_rows", cleaned_rows)
    if pd.notnull(mean_temp_celsius):
        mlflow.log_metric("cleaned_mean_temperature_c", mean_temp_celsius)
        context.log.info(f"Data cleaned. Rows: {cleaned_rows}, Mean Temp (C): {mean_temp_celsius:.2f}")
    else:
        context.log.warning("Mean temperature could not be calculated (all nulls or empty dataframe).")
        mlflow.log_metric("cleaned_mean_temperature_c", float('nan'))

    # Log a sample of the cleaned data to MLflow as a CSV artifact
    sample_csv_path = "cleaned_sample.csv"
    df.head(5).to_csv(sample_csv_path, index=False)
    mlflow.log_artifact(sample_csv_path, artifact_path="processed_data_samples")
    context.log.info(f"Logged a sample of cleaned data to MLflow as {sample_csv_path}.")
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
        cleaned_csv_as_input = mlflow.data.from_files(path=dataset_source_uri, name="cleaned_temperature_sample_csv")
        mlflow.log_input(dataset=cleaned_csv_as_input, context="cleaned_data_sample")

        context.log.info(f"Logged {sample_csv_path} as an MLflow Dataset input.")
    except Exception as e:
        context.log.warning(f"Could not log {sample_csv_path} as an MLflow Dataset: {e}")


    return df


@asset(
    name="trained_temperature_model_data",
    description="Trains a simple linear regression model to predict temperature.",
    deps=[clean_temperature_data_pandas],
    required_resource_keys={"mlflow_tracking"},
    compute_kind="python",
    group_name="3_modeling"
)
def train_simple_model(context: AssetExecutionContext, cleaned_temperature_data_pandas: pd.DataFrame) -> dict:
    mlflow = context.resources.mlflow_tracking
    context.log.info("Starting model training.")
    context.log.info(len(cleaned_temperature_data_pandas))

    if len(cleaned_temperature_data_pandas) < 10:
        context.log.warning("Not enough data points to train a meaningful model. Skipping training.")
        raise ValueError("Not enough data points after cleaning to proceed with model training.")

    # Feature Engineering: Create a lagged temperature feature
    lag_period = 1
    df_with_lag = cleaned_temperature_data_pandas.copy()
    df_with_lag[f"t2m_celsius_lag{lag_period}"] = df_with_lag["t2m_celsius"].shift(lag_period)
    df_with_lag = df_with_lag.dropna()

    if len(df_with_lag) < 5:
        raise ValueError(f"Not enough data points after creating lag{lag_period} feature.")

    X = df_with_lag[[f"t2m_celsius_lag{lag_period}"]].values
    y = df_with_lag["t2m_celsius"].values

    test_size_ratio = 0.2
    if len(X) * test_size_ratio < 1:
        split_idx = len(X) - 1
    else:
        split_idx = int(len(X) * (1 - test_size_ratio))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Training or testing set is empty after split. Adjust data or split.")

    model = LinearRegression()
    model.fit(X_train, y_train)
    context.log.info("Linear Regression model trained.")

    train_params = {
        "model_type": "LinearRegression",
        "feature_used": f"t2m_celsius_lag{lag_period}",
        "lag_period": lag_period,
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    }
    mlflow.log_params(train_params)
    context.log.info(f"Logged training parameters to MLflow: {train_params}")

    return {"model": model, "X_test": X_test, "y_test": y_test, "feature_names": [f"t2m_celsius_lag{lag_period}"]}


@asset(
    name="evaluated_temperature_model",
    description="Evaluates the trained model and logs model and metrics to MLflow.",
    deps=["trained_temperature_model_data"],
    required_resource_keys={"mlflow_tracking"},
    compute_kind="python",
    group_name="3_modeling"
)
def evaluate_model(context: AssetExecutionContext, trained_temperature_model_data: dict):
    mlflow = context.resources.mlflow_tracking
    context.log.info("Starting model evaluation.")

    model = trained_temperature_model_data["model"]
    X_test = trained_temperature_model_data["X_test"]
    y_test = trained_temperature_model_data["y_test"]
    feature_names = trained_temperature_model_data.get("feature_names", ["feature"])  # Get feature names if provided

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    context.log.info(f"Model Evaluation Metrics: MSE={mse:.4f}, MAE={mae:.4f}")

    # Log metrics to MLflow
    eval_metrics = {"mse": mse, "mae": mae}
    mlflow.log_metrics(eval_metrics)
    context.log.info(f"Logged evaluation metrics to MLflow: {eval_metrics}")

    # Log the model to MLflow
    # For sklearn, it's good to provide an input example for signature inference
    if X_test.shape[0] > 0:
        # Create a Pandas DataFrame for the input example with correct feature names
        input_example_df = pd.DataFrame(X_test[:5], columns=feature_names)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="temperature_forecaster",  # This is how the model will be named in MLflow artifacts
            input_example=input_example_df,  # Provide an input example
            registered_model_name="simple-temp-forecaster"  # Optional: register the model
        )
        context.log.info("Logged model to MLflow Model Registry as 'simple-temp-forecaster'.")
    else:
        context.log.warning("Test set was empty, model not logged with input_example.")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="temperature_forecaster",
            registered_model_name="simple-temp-forecaster"
        )
        context.log.info("Logged model to MLflow Model Registry (without input_example).")

    # This asset could output the metrics, but primary goal is logging to MLflow
    return eval_metrics


era5_full_pipeline_job = define_asset_job(
    name="era5_temperature_pipeline_job", # This is your custom job name
    selection=[  # You can select specific assets or use "*" for all assets in the current Definitions
        fetch_era5_data,
        process_temperature_dataframe,
        clean_temperature_data_pandas,
        train_simple_model,
        evaluate_model,
    ],
    # You can also provide tags directly to the job definition.
    # These tags will be applied to all runs of this job by default,
    # unless overridden at launch time.
    # tags={"owner": "data_team", "project": "era5_analysis"}
    # hooks={end_mlflow_on_run_finished},
    # let key mlflow is mlflow_tracking

)

# Define all assets and resources for Dagster to discover
defs = Definitions(
    assets=[
        fetch_era5_data,
        process_temperature_dataframe,
        clean_temperature_data_pandas,
        train_simple_model,
        evaluate_model,
    ],
    resources={
        "mlflow_tracking": mlflow_resource,
    },
    jobs=[era5_full_pipeline_job],
)