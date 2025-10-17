import dagster as dg
import pandas as pd
from .resources import fraud_data_source


@dg.asset(
    description="Download data for fraud detection.",
    compute_kind="python",
    group_name="ml_fraud_ingest"
)
def fraud_data(
    context : dg.AssetExecutionContext,
    #resource_defs={"fraud_data": FraudDataConfig} 
    ) -> dg.MaterializeResult:
    
    df = pd.read_csv(fraud_data_source.data_url)
    columns = [dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]

    return dg.MaterializeResult(
        value=df,
        metadata={
            "preview": dg.MetadataValue.md(df.head().to_markdown() or ""),
            "dagster/row_count": len(df),
            "dagster/column_schema": dg.TableSchema(columns=columns)
        }
    )