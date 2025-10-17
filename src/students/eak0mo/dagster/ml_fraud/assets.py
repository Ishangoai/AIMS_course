import dagster as dg

#import mlflow
import mlflow.sklearn as ms
import pandas as pd
import sklearn as sk
from sklearn.model_selection import train_test_split

from ..ml.resources import mlflow_client, mlflow_resource

# from sklearn.preprocessing.model_selection import train_test_split


@dg.asset(
    description="Loads data for fraud detection",
    compute_kind="python",
    group_name="Extract",
    resource_defs={"mlflow_tracking": mlflow_resource}
)
def fraud_data(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:

    # simulated data injestion
    """data = dg.MaterializeResult(value={
        "TransactionID": [1, 2, 3, 4, 5],
        "Amount": [100.0, 250.5, 75.0, 300.0, 150.0],
        "IsFraud": [0, 1, 0, 1, 0]
    })"""
    # importing and showing the data imported from the url
    mlflow = context.resources.mlflow_tracking
    fraud_df = pd.read_csv('https://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv')
    fraud_dataset = mlflow.data.from_pandas(fraud_df, name="Fraud Data")
    mlflow.log_input(dataset=fraud_dataset, context="training")

    context.log.info(f"Fraud detection data ingested with {len(fraud_df)} rows.")
    columns = [dg.TableColumn(k, str(v)) for k, v in fraud_df.dtypes.to_dict().items()]

    return dg.MaterializeResult(
        value=fraud_df,
        metadata={
            "preview": dg.MetadataValue.md(fraud_df.head().to_markdown() or ""),
            "dagster/row_count": len(fraud_df),
            "dagster/column_schema": dg.TableSchema(columns=columns)
        }
    )


@dg.asset(
    description="To eventually do some processing...",
    compute_kind="python",
    group_name="Transform",
)
def process_data(
    context: dg.AssetExecutionContext,
    fraud_data: pd.DataFrame
) -> dg.MaterializeResult:
    # to eventually do a process step on the data
    processed_data = fraud_data.copy()

    return dg.MaterializeResult(
        value=processed_data,
        metadata={
            "preview": dg.MetadataValue.md(processed_data.head().to_markdown() or "")
        }
    )


@dg.asset(
    description="Splits the data into training and testing sets",
    compute_kind="python",
    group_name="Transform",
)
def multi_split(
    context: dg.AssetExecutionContext,
    process_data: pd.DataFrame
) -> dict:
    # to split into multiple outputs
    work_data = process_data
    x = work_data[work_data.columns[:-1]]
    y = work_data[work_data.columns[-1]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    split_output_data = {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test
    }

    return dg.MaterializeResult(
        value=split_output_data,
        metadata={"x_train": dg.MetadataValue.md(x_train.head().to_markdown() or ""),
                    "y_train": dg.MetadataValue.md(y_train.head().to_markdown() or ""),
                    "x_test": dg.MetadataValue.md(x_test.head().to_markdown() or ""),
                    "y_test": dg.MetadataValue.md(y_test.head().to_markdown() or "")
        }
    )
