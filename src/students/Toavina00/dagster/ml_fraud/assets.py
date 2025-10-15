import dagster as dg
import pandas as pd

from ..ml.resources import mlflow_client, mlflow_resource

@dg.asset(
    description="Download credit card fraud data",
    compute_kind="python",
    group_name="ml_fraud_ingest"
)
def get_data(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:

    df = pd.read_csv("https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv")

    columns = [dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]

    return dg.MaterializeResult(
        value=df,
        metadata={
            "preview": dg.MetadataValue.md(df.head().to_markdown() or ""),
            "dagster/row_count": len(df),
            "dagster/column_schema": dg.TableSchema(columns=columns)
        }
    )

