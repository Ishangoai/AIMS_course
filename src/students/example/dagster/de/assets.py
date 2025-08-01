import dagster as dg
import pandas as pd


@dg.asset(
    description="Ingests raw data from a source (in this example, we create a dummy DataFrame).",
    compute_kind="python",
    group_name="de_ingest"
)
def raw_data(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:
    # Example raw data simulating ingestion from a source
    # In a real scenario, this could be reading from a file, database, or API
    data = {
    'Date': ['2023-01-15', 'Feb 3, 2023', '30.02.2023', 'Feb 4, 2023'],
    'FoodItem': ['Applle', 'Aubergine', 'Eggplant', 'Apple'],
    'nItems': ['200', '400.0', 'eighty', 'five']
}

    # Create the DataFrame
    df = pd.DataFrame(data)

    # log info that can be view in real time in the dagster UI
    context.log.info(f"Raw data ingested with {len(df)} rows.")
    columns = [dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]

    return dg.MaterializeResult(
        value=df,
        metadata={
            "preview": dg.MetadataValue.md(df.head().to_markdown() or ""),
            "dagster/row_count": len(df),
            "dagster/column_schema": dg.TableSchema(columns=columns)
        }
    )


@dg.asset(
    description="Cleans the raw data by converting date formats and correcting data types.",
    compute_kind="python",
    group_name="de_transform"
)
def clean_data(
    context: dg.AssetExecutionContext,
    raw_data: pd.DataFrame
) -> dg.MaterializeResult:

    clean_data = raw_data.copy()
    clean_data['Date'] = pd.to_datetime(clean_data['Date'], errors='coerce', format="mixed")
    clean_data['FoodItem'] = clean_data['FoodItem'].replace({'Applle': 'Apple', 'Aubergine': 'Eggplant'})
    clean_data['nItems'] = clean_data['nItems'].replace({'eighty': '80', 'five': '5'})
    clean_data['nItems'] = pd.to_numeric(clean_data['nItems'], errors='coerce').astype('Int64')

    # log info that can be view in real time in the dagster UI
    context.log.info(f"Cleaned data with {len(clean_data)} rows after cleaning.")

    columns = [dg.TableColumn(k, str(v)) for k, v in clean_data.dtypes.to_dict().items()]

    return dg.MaterializeResult(
        value=clean_data,
        metadata={
            "preview": dg.MetadataValue.md(clean_data.head().to_markdown() or ""),
            "dagster/row_count": len(clean_data),
            "dagster/column_schema": dg.TableSchema(columns=columns)
        }
    )


@dg.asset(
    description="Aggregates the data by grouping by FoodItem and summing nItems.",
    compute_kind="python",
    group_name="de_transform"
)
def agg_data(
    context: dg.AssetExecutionContext,
    clean_data: pd.DataFrame
) -> dg.MaterializeResult:

    clean_data = clean_data[clean_data['Date'].notnull()]
    agg_data = clean_data.groupby('FoodItem').agg({'nItems': 'sum'})

    # log info that can be view in real time in the dagster UI
    context.log.info(f"Aggregated data with {len(agg_data)} rows.")

    columns = [dg.TableColumn(k, str(v)) for k, v in agg_data.dtypes.to_dict().items()]

    return dg.MaterializeResult(
        value=agg_data,
        metadata={
            "preview": dg.MetadataValue.md(agg_data.head().to_markdown() or ""),
            "dagster/row_count": len(agg_data),
            "dagster/column_schema": dg.TableSchema(columns=columns)
        }
    )
