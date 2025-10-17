import dagster as dg


@dg.asset(
    description="Download data for fraud detection",
    compute_kind="python",
    group_name="ml_fraud_ingest"
)
def fraud_data(
    context: dg.AssetExecutionContext
) -> dg.MaterializeResult:
    # Simulate data ingestion
    data = dg.MaterializeResult(value={
        "TransactionID": [1, 2, 3, 4, 5],
        "Amount": [100.0, 250.5, 75.5, 300.0, 150.0],
        "IsFraud": [0, 1, 0, 1, 0]
    })
    return data
