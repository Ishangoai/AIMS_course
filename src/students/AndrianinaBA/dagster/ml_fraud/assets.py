import os

import dagster as dg
import dagster_slack
import mlflow
import pandas as pd
import pydantic as pyd

from ..de.assets import slack_resource
from .resources import fraud_data_source


class PromotionConfig(dg.Config):
    """Configuration for model promotion thresholds"""
    staging_accuracy_theshold: float = pyd.Field(
        default=0.95,
        description="Maximum acceptable MSE for promoting a model to Staging."
    )
    staging_recall_threshold: float = pyd.Field(
        default=0.8,
        description="Maximum acceptable MAE for promoting a model to Staging."
    )
    staging_f1_score: float = pyd.Field(
        default=0.8,
        description="Minimum acceptable R2 for promoting a model to Staging."
    )


@dg.asset(
    description="Download data for fraud detection.",
    compute_kind="python",
    group_name="ml_fraud_ingest"
)
def fraud_data(
    context: dg.AssetExecutionContext,
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


@dg.asset(
    description="The model training asset for fraud detection.",
    compute_kind="python",
    group_name="ml_fraud_train",
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
    splitted_data = [X_train, X_test, y_train, y_test]
    return dg.MaterializeResult(
        value=splitted_data,
        metadata={
            "model_type": "RandomForestClassifier",
            "n_estimators": 100
        }
    )


@dg.asset(
    description="Training the RandomForestClassifier model",
    compute_kind="python",
    group_name="ml_fraud_train",
    deps=["split_fraud_model"]
)
def train_fraud_model(
    context: dg.AssetExecutionContext,
    split_fraud_model: list,
) -> dg.MaterializeResult:

    from sklearn.ensemble import RandomForestClassifier

    X_train, _, y_train, _ = split_fraud_model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    context.log.info("Fraud detection model trained successfully.")

    return dg.MaterializeResult(
        value=model,
        metadata={
            "model_type": "RandomForestClassifier",
            "n_estimators": 100
        }
    )


@dg.asset(
    description="The REAL data traning with the RandomForestClassifier with MLFlow",
    compute_kind="python",
    group_name="ml_fraud_train",
    deps=["split_fraud_model"]
)
def train_real_fraud_model(
    context: dg.AssetExecutionContext,
    split_fraud_model: list,
) -> dg.MaterializeResult:
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import GridSearchCV, KFold

    X_train, _, y_train, _ = split_fraud_model
    X = X_train.to_numpy()
    y = y_train.to_numpy()

    param_grid = {'n_estimators': [50, 100, 200]}
    outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    outer_scores = []

    with mlflow.start_run(run_name="nested_cv_rf"):
        fold = 1
        for train_idx, test_idx in outer_cv.split(X, y):
            Xtr, Xval = X[train_idx], X[test_idx]
            ytr, yval = y[train_idx], y[test_idx]

            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42),
                param_grid,
                cv=3,
                scoring="accuracy"
            )

            with mlflow.start_run(run_name=f"fold_{fold}", nested=True):
                grid_search.fit(Xtr, ytr)
                best_model = grid_search.best_estimator_
                acc = accuracy_score(yval, best_model.predict(Xval))
                mlflow.log_metric("val_accuracy", float(acc))
                mlflow.log_params(grid_search.best_params_)
                outer_scores.append(acc)

            fold += 1

    mean_acc = np.mean(outer_scores)
    context.log.info(f"Mean outer 3-fold accuracy: {mean_acc:.3f}")
    mlflow.log_metric("mean_outer_accuracy", float(mean_acc))
    output = {"model_name": best_model}

    return dg.MaterializeResult(
        value=output,
        metadata={"mean_accuracy": float(mean_acc)}
)


@dg.asset(
    description="Evaluate the trained fraud detection model.",
    resource_defs={"slack": slack_resource},
    compute_kind="python",
    group_name="ml_fraud_evaluate",
    deps=["train_fraud_model", "split_fraud_model"]
)
def evaluate_fraud_model(
    context: dg.AssetExecutionContext,
    train_fraud_model,
    split_fraud_model: list
) -> dg.MaterializeResult:

    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    _, X_test, _, y_test = split_fraud_model
    y_pred = train_fraud_model.predict(X_test)

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    context.log.info("Fraud detection model evaluation completed.")
    context.log.info(f"Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1-Score: {f1:.3f}")
#     slack: dagster_slack.SlackResource = context.resources.slack
#     slack.get_client().chat_postMessage(
#         channel='aims_course_october2025',
#         text=f"""
# {os.environ.get("GITHUB_USER", "default")}'s fraud detection model evaluation results:
# - Accuracy: {acc:.3f} \t :pray:
# - Precision: {prec:.3f} \t :pray:
# - Recall: {rec:.3f} \t :pray:
# - F1-Score: {f1:.3f} \t :pray:
#         """
#     )

    return dg.MaterializeResult(
        value=report,
        metadata={
            "classification_report": dg.MetadataValue.md(pd.DataFrame(report).to_markdown() or ""),
            "confusion_matrix": dg.MetadataValue.md(pd.DataFrame(cm).to_markdown() or "")
        }
    )


# Promoting the model
@dg.asset(
    description="Promote the fraud detection model to Production if it meets the criteria.",
    compute_kind="python",
    group_name="ml_fraud_promote",
    deps=["evaluate_fraud_model", "train_real_fraud_model"]
)
def promote_fraud_model_staging(
    context: dg.AssetExecutionContext,
    evaluate_fraud_model,
    train_fraud_model,
    config: PromotionConfig
) -> dg.MaterializeResult:

    current_accuracy = evaluate_fraud_model['accuracy']
    current_recall = evaluate_fraud_model['1']['recall']
    current_f1_score = evaluate_fraud_model['1']['f1-score']

    STAGING_ACCURACY_THRESHOLD = config.staging_accuracy_theshold
    STAGING_RECALL_THRESHOLD = config.staging_recall_threshold
    STAGING_F1_SCORE = config.staging_f1_score

    # Log the evaluation metrics and threshold criteria
    context.log.info(f"""
                    Model evaluated with Accuracy: {current_accuracy:.4f}, Recall: {current_recall:.4f},
                    F1-Score: {current_f1_score:.4f}""")
    context.log.info(f"""
                     Staging promotion thresholds: Accuracy > {STAGING_ACCURACY_THRESHOLD},
                     Recall > {STAGING_RECALL_THRESHOLD},F1-Score > {STAGING_F1_SCORE}""")

    if (current_accuracy >= STAGING_ACCURACY_THRESHOLD and
        current_recall >= STAGING_RECALL_THRESHOLD and
        current_f1_score >= STAGING_F1_SCORE):
        context.log.info("Model meets the criteria for promotion to Production.")
        # Here you would add code to promote the model in MLflow Model Registry
        promotion_status = "Promoted to Production"
    else:
        context.log.info("Model does not meet the criteria for promotion to Production.")
        promotion_status = "Not Promoted"

    return dg.MaterializeResult(
        value=promotion_status,
        metadata={
            "current_accuracy": current_accuracy,
            "current_recall": current_recall,
            "current_f1_score": current_f1_score
        }
    )


@dg.asset(
    description="Production model for fraud detection.",
    compute_kind="python",
    group_name="ml_fraud_promote",
    deps=["promote_fraud_model_staging", "train_real_fraud_model"]
)
def fraud_production_model(
    context: dg.AssetExecutionContext,
    promote_fraud_model_staging,
    train_real_fraud_model,
) -> dg.MaterializeResult:

    if promote_fraud_model_staging == "Promoted to Production":
        model = train_real_fraud_model['model_name']
        context.log.info("Using the promoted Production model.")
    else:
        model = None
        context.log.info("No Production model available.")

    return dg.MaterializeResult(
        value=model,
        metadata={
            "model_status": promote_fraud_model_staging
        }
    )
