import os
from collections import abc

import dagster as dg
import dagster_slack
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
    clean_data['nItems'] = pd.to_numeric(clean_data['nItems'], errors='coerce')

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
    resource_defs={"slack": dagster_slack.SlackResource(token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN"))},
    compute_kind="python",
    group_name="de_transform"
)
def agg_data(
    context: dg.AssetExecutionContext,
    clean_data: pd.DataFrame
) -> dg.MaterializeResult:

    slack: dagster_slack.SlackResource = context.resources.slack
    slack.get_client().chat_postMessage(
        channel='aims_course', text=f":wave: hey there, from {os.environ.get("GITHUB_USER", "default")}!"
    )

    clean_data_no_null: pd.DataFrame = clean_data.loc[clean_data['Date'].notnull()]
    agg_data: pd.DataFrame = clean_data_no_null.groupby('FoodItem').agg({'nItems': 'sum'}).reset_index()

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


# Define a multi-asset check to validate the data quality of the assets
@dg.multi_asset_check(
    # Map checks to targeted assets
    specs=[
        # blocking=False means that the check will not block the downstream asset materialization
        # if the check fails, the next downstream asset will still be materialized
        dg.AssetCheckSpec(name="all_alpha", asset="raw_data", blocking=False),
        dg.AssetCheckSpec(name="no_null_dates", asset="clean_data", blocking=False),
        dg.AssetCheckSpec(name="impossible_nitems", asset="clean_data", blocking=False),
    ]
)
def dq_check_de(raw_data, clean_data) -> abc.Iterable[dg.AssetCheckResult]:
    num_not_alpha = (~raw_data['FoodItem'].str.isalpha()).sum()
    yield dg.AssetCheckResult(
            check_name="all_alpha",
            passed=bool(num_not_alpha == 0),
            asset_key="raw_data",
        )

    # Check for null Date values in clean_data
    num_date_null = clean_data["Date"].isna().sum()
    yield dg.AssetCheckResult(
        check_name="no_null_dates",
        passed=bool(num_date_null == 0),
        asset_key="clean_data",
    )

    # Check for impossibe nItem values in clean_data
    num_impossible_nitems = (clean_data["nItems"] < 0).sum()
    yield dg.AssetCheckResult(
        check_name="impossible_nitems",
        passed=bool(num_impossible_nitems == 0),
        asset_key="clean_data",
    )
