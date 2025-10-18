import dagster as dg
import pandas as pd
from sklearn.model_selection import train_test_split

from ..ml.resources import mlflow_resource
from .resources import FraudDataConfig


@dg.asset(
    description="Load raw fraud detection data from CSV URL",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_ingest"
)
def raw_fraud_data(
    context: dg.AssetExecutionContext,
    config: FraudDataConfig
) -> dg.MaterializeResult:
    """
    Downloads credit card fraud dataset from URL and logs to MLflow.
    Returns a pandas DataFrame.
    """
    mlflow_client = context.resources.mlflow_tracking

    context.log.info(f"Loading fraud detection data from: {config.data_url}")

    # Loading the CSV data
    df = pd.read_csv(config.data_url)

    context.log.info(f"Successfully loaded {len(df)} rows and {len(df.columns)} columns")

    # Log dataset info to MLflow and convert all to python types
    mlflow_client.log_param("data_source", config.data_url)
    mlflow_client.log_param("total_rows", len(df))
    mlflow_client.log_param("total_columns", len(df.columns))
    mlflow_client.log_param("fraud_cases", int(df['Class'].sum()))
    mlflow_client.log_param("normal_cases", int((df['Class'] == 0).sum()))

    fraud_percentage = float((df['Class'].sum() / len(df)) * 100)
    mlflow_client.log_metric("fraud_percentage", fraud_percentage)

    context.log.info(f"Fraud cases: {int(df['Class'].sum())}, Normal cases: {int((df['Class'] == 0).sum())}")
    context.log.info(f"Fraud percentage: {fraud_percentage:.2f}%")

    # Log dataset to MLflow
    dataset = mlflow_client.data.from_pandas(df, name="creditcard_fraud_raw_data")
    mlflow_client.log_input(dataset=dataset, context="training")

    columns = [dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]

    return dg.MaterializeResult(
        value=df,
        metadata={
            "dagster/row_count": len(df),
            "dagster/column_schema": dg.TableSchema(columns=columns),
            "preview": dg.MetadataValue.md(df.head(10).to_markdown() or ""),
            "fraud_count": dg.MetadataValue.int(int(df['Class'].sum())),
            "fraud_percentage": dg.MetadataValue.float(float(fraud_percentage)),
            "description": dg.MetadataValue.text(
                "Raw credit card fraud detection dataset"
            )
        }
    )


@dg.asset(
    description="Split data into 80% training and 20% test",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_transform"
)
def split_train_test(
    context: dg.AssetExecutionContext,
    config: FraudDataConfig,
    raw_fraud_data: pd.DataFrame
) -> dict:
    """
    Splits the fraud dataset into training (80%) and test (20%) sets.
    Uses stratified split to maintain fraud/normal ratio in both sets.
    """
    mlflow_client = context.resources.mlflow_tracking

    context.log.info("Splitting data into train and test sets")

    # Separate features and target
    X = raw_fraud_data.drop('Class', axis=1)
    y = raw_fraud_data['Class']

    context.log.info(f"Features shape: {X.shape}, Target shape: {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y  # Maintain fraud/normal ratio in both sets
    )

    context.log.info(f"Training set: {len(X_train)} samples")
    context.log.info(f"Test set: {len(X_test)} samples")

    # Calculate fraud counts in each set
    train_fraud = int(y_train.sum())
    test_fraud = int(y_test.sum())
    train_normal = int((y_train == 0).sum())
    test_normal = int((y_test == 0).sum())

    train_fraud_pct = (train_fraud / len(y_train)) * 100
    test_fraud_pct = (test_fraud / len(y_test)) * 100

    context.log.info(f"Training set - Fraud: {train_fraud} ({train_fraud_pct:.2f}%), Normal: {train_normal}")
    context.log.info(f"Test set - Fraud: {test_fraud} ({test_fraud_pct:.2f}%), Normal: {test_normal}")

    # Log split info to MLflow
    mlflow_client.log_param("train_size", len(X_train))
    mlflow_client.log_param("test_size", len(X_test))
    mlflow_client.log_param("test_split_ratio", config.test_size)
    mlflow_client.log_param("random_state", config.random_state)
    mlflow_client.log_param("stratified_split", True)

    mlflow_client.log_metric("train_fraud_count", train_fraud)
    mlflow_client.log_metric("test_fraud_count", test_fraud)
    mlflow_client.log_metric("train_fraud_percentage", train_fraud_pct)
    mlflow_client.log_metric("test_fraud_percentage", test_fraud_pct)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
