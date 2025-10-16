import dagster as dg


@dg.asset(
    description="Download data for fraud detection",
    compute_kind="python",
)
def download_fraud_data(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:
    import pandas as pd

    url = "http://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"
    df = pd.read_csv(url)
    return dg.MaterializeResult(
        value=df,
        metadata={
            "source": "Kaggle",
            "preview": dg.MetadataValue.md(df.head().to_markdown() or ""),
            "dagster/row_count": len(df),
            "dagster/column_schema": dg.TableSchema(
                columns=[dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]
            ),
            "dagster/column_count": len(df.columns),
            "features": list(df.columns),
        },
    )
