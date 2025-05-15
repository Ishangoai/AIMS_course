import cdsapi
import xarray as xr
import pandas as pd
import polars as pl
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from dagster import asset, Definitions, AssetExecutionContext
from dagster_mlflow.resources import MlFlow

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
    mlflow: MlFlow = context.resources.mlflow_tracking

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
    mlflow: MlFlow = context.resources.mlflow_tracking
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
    name="cleaned_temperature_data_polars",
    description="Cleans the temperature data using Polars. Converts Kelvin to Celsius.",
    deps=[process_temperature_dataframe],
    required_resource_keys={"mlflow_tracking"},
    compute_kind="python",
    group_name="2_processing"
)
def clean_temperature_data_polars(context: AssetExecutionContext,
                                  processed_temperature_dataframe: pd.DataFrame) -> pl.DataFrame:
    mlflow: MlFlow = context.resources.mlflow_tracking
    context.log.info("Starting data cleaning with Polars.")

    # Convert pandas DataFrame to Polars DataFrame
    polars_df = pl.from_pandas(processed_temperature_dataframe)

    # Simple Cleaning Steps:
    # 1. Convert temperature from Kelvin to Celsius
    polars_df = polars_df.with_columns(
        (pl.col("t2m") - 273.15).alias("t2m_celsius")
    )

    # 2. Ensure 'time' column is datetime type (Polars often infers this well from pandas)
    # If not, explicit conversion might be needed:
    # polars_df = polars_df.with_columns(pl.col("time").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"))

    # 3. Sort by time (important for time series)
    polars_df = polars_df.sort("time")

    # 4. Handle missing values (if any) - for ERA5 2m temp, this is less common
    # Example: forward fill, or drop rows with nulls in 't2m_celsius'
    initial_rows = len(polars_df)
    polars_df = polars_df.drop_nulls(subset=["t2m_celsius"])
    rows_after_dropna = len(polars_df)
    if initial_rows > rows_after_dropna:
        context.log.info(f"Dropped {initial_rows - rows_after_dropna} rows with null t2m_celsius.")

    cleaned_rows = len(polars_df)
    mean_temp_celsius = polars_df["t2m_celsius"].mean()

    mlflow.log_metric("cleaned_num_rows", cleaned_rows)
    if mean_temp_celsius is not None:  # mean() can return None if all values are null
        mlflow.log_metric("cleaned_mean_temperature_c", mean_temp_celsius)
        context.log.info(f"Data cleaned. Rows: {cleaned_rows}, Mean Temp (C): {mean_temp_celsius:.2f}")
    else:
        context.log.warning("Mean temperature could not be calculated (all nulls or empty dataframe).")
        mlflow.log_metric("cleaned_mean_temperature_c", float('nan'))  # Log NaN if no data

    # Log a sample of the cleaned data to MLflow as a CSV artifact
    sample_df_pd = polars_df.head(5).to_pandas()
    sample_csv_path = "cleaned_sample.csv"
    sample_df_pd.to_csv(sample_csv_path, index=False)
    mlflow.log_artifact(sample_csv_path, artifact_path="processed_data_samples")
    context.log.info(f"Logged a sample of cleaned data to MLflow as {sample_csv_path}.")

    return polars_df


@asset(
    name="trained_temperature_model_data",  # Output includes model and test data
    description="Trains a simple linear regression model to predict temperature.",
    deps=[clean_temperature_data_polars],
    required_resource_keys={"mlflow_tracking"},
    compute_kind="python",
    group_name="3_modeling"
)
def train_simple_model(context: AssetExecutionContext, cleaned_temperature_data_polars: pl.DataFrame) -> dict:
    mlflow: MlFlow = context.resources.mlflow_tracking
    context.log.info("Starting model training.")
    context.log.info(cleaned_temperature_data_polars.height)

    if cleaned_temperature_data_polars.height < 10:  # Need enough data to lag and split
        context.log.warning("Not enough data points to train a meaningful model. Skipping training.")
        # This will cause downstream evaluation to fail if it strictly expects a model.
        # Consider how to handle this. For now, we raise an error or return a specific signal.
        raise ValueError("Not enough data points after cleaning to proceed with model training.")

    # Feature Engineering: Create a lagged temperature feature
    lag_period = 1  # Predict next based on current (adjust as needed)
    df_with_lag = cleaned_temperature_data_polars.with_columns(
        pl.col("t2m_celsius").shift(lag_period).alias(f"t2m_celsius_lag{lag_period}")
    ).drop_nulls()  # Drop rows where lag feature is null

    if df_with_lag.height < 5:  # Need at least a few points for train/test
        raise ValueError(f"Not enough data points after creating lag{lag_period} feature.")

    # Define features (X) and target (y)
    # We predict the current temperature based on the lagged temperature
    X_pl = df_with_lag.select([f"t2m_celsius_lag{lag_period}"])
    y_pl = df_with_lag.select("t2m_celsius")

    # Convert to pandas/numpy for scikit-learn (as of now, sklearn direct Polars support is limited)
    X = X_pl.to_pandas().values
    y = y_pl.to_pandas().values.ravel()  # .ravel() to make it a 1D array

    # Simple time-based split (e.g., last 20% for testing)
    # For more robust time series, use scikit-learn's TimeSeriesSplit or manual slicing
    test_size_ratio = 0.2
    if len(X) * test_size_ratio < 1:
        split_idx = len(X) - 1  # At least one sample for testing if possible
    else:
        split_idx = int(len(X) * (1 - test_size_ratio))

    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    if len(X_train) == 0 or len(X_test) == 0:
        raise ValueError("Training or testing set is empty after split. Adjust data or split.")

    # Train a Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)
    context.log.info("Linear Regression model trained.")

    # Log training parameters to MLflow
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
    mlflow: MlFlow = context.resources.mlflow_tracking
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


# Define all assets and resources for Dagster to discover
defs = Definitions(
    assets=[
        fetch_era5_data,
        process_temperature_dataframe,
        clean_temperature_data_polars,
        train_simple_model,
        evaluate_model,
    ],
    resources={
        "mlflow_tracking": mlflow_resource,
    },
)