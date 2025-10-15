import dagster as dg

@dg.asset(
    description="Download data for fraud detection.",
    compute_kind="python"
)
def fraud_data(
    context : dg.AssetExecutionContext,
    resource_defs={"fraud_data": FraudDataConfig} 
    ) -> dg.MaterializeResult:
    
    #simulated data ingestion
    fraud_data_config = context.resources.fraud_data
    data = dg.MaterializeResult(value={
        "Transaction ID": [1,2,3,4,5],
        "Amount": [100.0, 250.5, 75.0, 300.0, 150.0],
        "IsFraud": [0, 1, 0, 1, 0]
    })
    return data