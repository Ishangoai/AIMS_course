import dagster as dg

@dg.asset(
    description="Download data for fraud detection",
    compute_kind="Python"
)

def fraud_data(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:

    #Simulated data ingestion
    data= dg.MaterializeResult(value={
        "TransactionID": [1,2,3,4,5],
        "Amount": [100.0, 250.0, 75.0, 300.0, 150.0],
        "IsFraud":[0, 1, 0, 1, 0]
    })
    return data
