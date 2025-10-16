
import dagster as dg
import numpy as np
import pandas as pd
from pandas._typing import ArrayLike

from ..ml.resources import mlflow_resource

URL = 'https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv'
@dg.asset(
    description="Download data for fraud detection task.",
    compute_kind="python",
    group_name="ml_fraud_ingest"
)

def load_fraud_data(
    context: dg.AssetExecutionContext
) -> dg.MaterializeResult:
    # try:
    #     df = pd.read_csv(context.op_config["data_url"])
    # except Exception as e:
    #     context.log.error(f"Failed to download data: {e}")
    #     raise
    df = pd.read_csv(URL)
    columns = [dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]

    data = dg.MaterializeResult(
     value=df,
        metadata={
            "preview": dg.MetadataValue.md(df.head().to_markdown() or ""),
            "dagster/row_count": len(df),
            "dagster/column_schema": dg.TableSchema(columns=columns)
        }
    )

    return data


@dg.asset(
    description="Prepocess fraud data for modeling.",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_transform"
)
def preprocess_fraud_data(
    context: dg.AssetExecutionContext,
    load_fraud_data: pd.DataFrame
) -> dg.MaterializeResult:
    raw_pandas_df = load_fraud_data
    mlflow_client = context.resources.mlflow_tracking
    context.log.info("Starting data preprocessing.")

    def check_imbalance(counts: list[int]) -> bool:
        """Simple imbalance check. Returns True if imbalanced."""

        min_class = min(counts)
        max_class = max(counts)
        ratio = min_class / max_class

        is_imbalanced = ratio < 0.5  # Standard threshold

        print(f"Imbalance ratio: {ratio:.2f}")
        print(f"Imbalanced: {is_imbalanced}")

        return is_imbalanced

    # ERA5 data for a region will have multiple lat/lon points per time.
    # For a very simple time series model, we need a single value per timestamp.
    # Take a spatial mean across all lat/lon points for each timestamp.
    # Count the occurrences of each class in the original dataframe
    # Index of zero is non-fraudulent transactions, index of one is fraudulent transactions
    context.log.info("Count for each class( fraudulent(1) or not(0)) before undersampling")
    df_fraud_vs_nonfraud_count = list(raw_pandas_df['Class'].value_counts())

    mlflow_client.log_metric("Number of fraudulent transactions", df_fraud_vs_nonfraud_count[1])
    mlflow_client.log_metric("Number of non-fraudulent transactions", df_fraud_vs_nonfraud_count[0])
    context.log.info(f"Count of fraudulent transactions: {df_fraud_vs_nonfraud_count[0]}")
    context.log.info(f"Count of non-fraudulent transactions: {df_fraud_vs_nonfraud_count[1]}")
    context.log.info("Logged metrics to MLflow.")

    # 1. Convert temperature from Kelvin to Celsius
    # df_spatial_mean["t2m_celsius"] = df_spatial_mean["t2m"] - 273.15

    # 2. Ensure 'time' column is datetime type
    if check_imbalance(df_fraud_vs_nonfraud_count):
        context.log.warning("The dataset is imbalanced.")
        mlflow_client.set_tag("data_imbalance", "true")
        sample_value = df_fraud_vs_nonfraud_count[0] if df_fraud_vs_nonfraud_count[1] > df_fraud_vs_nonfraud_count[0] else df_fraud_vs_nonfraud_count[1]
        mlflow_client.log_metric("minority_class_count", sample_value)
        y = raw_pandas_df['Class']
        non_fraud_indices = y[y == 0].index
        fraud_indices = y[y == 1].index

# Undersample non-fraudulent transactions
        undersampled_non_fraud_indices = pd.Index(non_fraud_indices).to_series().sample(
            n=sample_value, 
            random_state=42
        ).values
        # Concatenate the DataFrames of fraudulent and undersampled non-fraudulent transactions
        balanced_df = raw_pandas_df.loc[fraud_indices.union(undersampled_non_fraud_indices)] # type: ignore

        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        raw_pandas_df = balanced_df

    # 3. Sort by time
    # Reset index after sort
    # df_spatial_mean = df_spatial_mean.sort_values("time").reset_index(drop=True)

    # Log the cleaned data to MLflow
    # spatial_mean_dataset = mlflow_client.data.from_pandas(df_spatial_mean, name="cleaned_spatial_mean_temperature")
    # mlflow_client.log_input(dataset=spatial_mean_dataset, context="training")

    columns = [dg.TableColumn(k, str(v)) for k, v in raw_pandas_df.dtypes.to_dict().items()]

    context.log.info("Count for each class( fraudulent(1) or not(0)) before undersampling")
    context.log.info(f"Count of fraudulent transactions: {df_fraud_vs_nonfraud_count[0]}")
    context.log.info(f"Count of non-fraudulent transactions: {df_fraud_vs_nonfraud_count[1]}")
    context.log.info("Logged metrics to MLflow.")
    return dg.MaterializeResult(
        value=raw_pandas_df,
        metadata={
            "dagster/row_count": len(raw_pandas_df),
            "dagster/column_schema": dg.TableSchema(columns=columns),
            # "preview": dg.MetadataValue.md(df_spatial_mean.head().to_markdown() or "")
        }
    )
