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

    import numpy as np
    from sklearn.datasets import load_iris
    from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
    from sklearn.svm import SVC

    # 1. Load data
    X, y = load_iris(return_X_y=True)

    # 2. Define model and parameter grid
    model = SVC()
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf']
    }

    # 3. Inner cross-validation (for hyperparameter tuning)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_cv)

    # 4. Outer cross-validation (for performance estimation)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # 5. Nested cross-validation: use cross_val_score with GridSearchCV as the estimator
    nested_scores = cross_val_score(grid_search, X, y, cv=outer_cv)

    # 6. Print results
    print("Nested CV scores:", nested_scores)
    print("Mean score:", np.mean(nested_scores))
    return dg.MaterializeResult(
        value="Model trained successfully",
        metadata={
            "nested_cv_scores": nested_scores.tolist(),
            "mean_score": float(np.mean(nested_scores))
        }
    )
