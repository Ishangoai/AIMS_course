import dagster as dg
import pandas as pd


@dg.asset(
    description="Import data for fraud detection",
    compute_kind="python",
    group_name="ml_fraud_group1"
)
def fraud_data(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:

    # Load fraud detection data from a CSV file.
    csv_path = "https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        context.log.error(f"Could not find the file at {csv_path}.")
        raise

    # Optionally, preview and attach simple metadata
    row_count = len(df)
    context.log.info(f"Raw data ingested with {row_count} rows.")
    column_schema = [dg.TableColumn(name, str(dtype)) for name, dtype in df.dtypes.items()]

    return dg.MaterializeResult(
        value=df,
        metadata={
            "preview": dg.MetadataValue.md(df.head().to_markdown() or ""),
            "dagster/row_count": row_count,
            "dagster/column_schema": dg.TableSchema(columns=column_schema)
        }
    )


@dg.asset_check(
    asset="fraud_data",
    description="Checks for null values and negative amounts in fraud_data"
)
def check_fraud_data(
    context: dg.AssetCheckExecutionContext,
    fraud_data: pd.DataFrame
    ) -> dg.AssetCheckResult:
    # Check for nulls
    num_nulls = fraud_data.isnull().sum().sum()

    # Check for negative amounts if 'Amount' column exists
    if 'Amount' in fraud_data.columns:
        negative_amounts = (fraud_data['Amount'] < 0).sum()
    else:
        negative_amounts = 0  # Consider 0 if column doesn't exist

    num_nulls = int(num_nulls)
    negative_amounts = int(negative_amounts)

    passed = (num_nulls == 0) and (negative_amounts == 0)
    metadata = {
        "num_nulls": dg.MetadataValue.int(num_nulls),
        "num_negative_amounts": dg.MetadataValue.int(negative_amounts)
    }

    return dg.AssetCheckResult(
        passed=passed,
        metadata=metadata,
        description=(
            "Passed" if passed else
            f"{'Nulls present. ' if num_nulls > 0 else ''}{'Negative Amounts found.' if negative_amounts > 0 else ''}"
        ),
        asset_key="fraud_data",
    )
