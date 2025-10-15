import dagster as dg
URL = 'https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv'
@dg.asset(
    description="Download data for fraud detection task.",
    compute_kind="python",
    group_name="ml_fraud_ingest"
)

def load_fraud_data(
    context: dg.AssetExecutionContext
) -> dg.MaterializeResult:
    import pandas as pd
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
    