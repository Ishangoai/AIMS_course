import dagster as dg
import pandas as pd


@dg.asset(
    description="Fetches raw ERA5 2m temperature data from the CDS.",
    required_resource_keys={"mlflow_tracking", "cds_api"},
    compute_kind="python",
    group_name="1_ingestion"
)
def raw_data(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:
    data = {
    'Date': ['2023-01-15', 'Feb 3, 2023', '30.02.2023', 'Feb 4, 2023'],
    'Food Item': ['Applle', 'Aubergine', 'Eggplant', 'Apple'],
    'nItems': ['200', '400.0', 'eighty', 'five']
}

    # Create the DataFrame
    df = pd.DataFrame(data)

    return dg.MaterializeResult(
        value=df
    )
