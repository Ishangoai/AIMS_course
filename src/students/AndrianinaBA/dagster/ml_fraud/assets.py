import dagster as dg
import pandas as pd

from .resources import fraud_data_source


@dg.asset(
    description="Download data for fraud detection.",
    compute_kind="python",
    group_name="ml_fraud_ingest"
)
def fraud_data(
    context: dg.AssetExecutionContext,
    # resource_defs={"fraud_data": FraudDataConfig}
    ) -> dg.MaterializeResult:

    df = pd.read_csv(fraud_data_source.data_url)
    columns = [dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]

    return dg.MaterializeResult(
        value=df,
        metadata={
            "preview": dg.MetadataValue.md(df.head().to_markdown() or ""),
            "dagster/row_count": len(df),
            "dagster/column_schema": dg.TableSchema(columns=columns)
        }
    )


# @dg.asset(
#     description="Just a depenemcy test",
#     deps=["fraud_data"],
#     compute_kind="python",
#     group_name="ml_fraud_ingest"
# )
# def dependence(
#     context: dg.AssetExecutionContext,
#     fraud_data: pd.DataFrame
#     ) -> None:
#     context.log.info(f"Fraud data has {len(fraud_data)} rows.")

@dg.asset(
    description="The model training asset for fraud detection.",
    compute_kind="python",
    group_name="ml_fraud_model",
    deps=["fraud_data"]
)
def split_fraud_model(
    context: dg.AssetExecutionContext,
    fraud_data: pd.DataFrame
) -> dg.MaterializeResult:

    from sklearn.model_selection import train_test_split

    X = fraud_data.drop(columns=['Class'])
    y = fraud_data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return dg.MaterializeResult(
        value=[X_train, X_test, y_train, y_test],
        metadata={
            "model_type": "RandomForestClassifier",
            "n_estimators": 100
        }
    )


@dg.asset(
    description="Training the model",
    compute_kind="python",
    group_name="ml_fraud_model",
    deps=["split_fraud_model"]
)
def train_fraud_model(
    context: dg.AssetExecutionContext,
    split_fraud_model: list
) -> dg.MaterializeResult:

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report

    X_train, X_test, y_train, y_test = split_fraud_model

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)

    context.log.info(f"Classification Report:\n{report}")

    return dg.MaterializeResult(
        value=model
    )
