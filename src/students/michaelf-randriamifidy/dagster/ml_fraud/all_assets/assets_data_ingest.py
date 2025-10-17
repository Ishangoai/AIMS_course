import os

import dagster as dg
import pandas as pd

from ...ml.resources import mlflow_resource
from ..resources import FraudResourceConfig


@dg.asset(
    description="Fetches raw fraud classification data from Kaggle.",
    resource_defs={"mlflow_tracking": mlflow_resource, "downloader": FraudResourceConfig()},
    compute_kind="python",
    group_name="data_ingest"
)
def pandas_data_df(
    context: dg.AssetExecutionContext
) -> dg.MaterializeResult:

    mlflow_client = context.resources.mlflow_tracking

    client = context.resources.downloader.client
    context.log.info("Download data from Kaggle")

    OUTPUT_FILENAME = os.getcwd() + "/fraud_data.csv"

    client.download_and_save(OUTPUT_FILENAME)

    context.log.info(f"Successfully download data to {OUTPUT_FILENAME}")

    df = pd.read_csv(OUTPUT_FILENAME)
    context.log.info(f"Pandas DataFrame:\n {df}")

    os.remove(OUTPUT_FILENAME)

    dataset = mlflow_client.data.from_pandas(df, name="fraud_data_classification")
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
