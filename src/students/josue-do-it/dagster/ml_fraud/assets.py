import os

import dagster as dg
import dagster_slack
import matplotlib.pyplot as plt

# import mlflow
import mlflow
import pandas as pd
from mlflow.sklearn import log_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from .resources import fraud_data_source

# ,ModelConfig,ModelPromotionConfig


@dg.asset(
    description="Download data for fraud dectection",
    compute_kind="python",
    group_name="ml_fraud_ingest",
    required_resource_keys={"mlflow"}
)
def fraud_data(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:
    df = pd.read_csv(fraud_data_source.data_source)
    columns = [dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]
    mlflow_client = context.resources.mlflow
    n_rows, n_columns = df.shape   # pyright: ignore[reportAttributeAccessIssue]
    mlflow_client.log_metric("num_rows", n_rows)
    mlflow_client.log_metric("num_columns", n_columns)
    return dg.MaterializeResult(
        value=df,
        metadata={
            "preview": dg.MetadataValue.md(df.head().to_markdown() or ""),
            "dagster/row_count": len(df),
            "dagster/column_schema": dg.TableSchema(columns=columns)
        }
    )


@dg.asset(
    description="Cleans fraud data(remove duplicates, dropp missing values, reset the index)",
    compute_kind="python",
    group_name="ml_fraud_transform",
    required_resource_keys={"mlflow"}
)
def clean_fraud(
    context: dg.AssetExecutionContext,
    fraud_data: pd.DataFrame
) -> dg.MaterializeResult:
    context.log.info("Starting data cleaning.")

    initial_rows = len(fraud_data)
    df = fraud_data.copy()
    df.drop_duplicates(inplace=True)
    removed_duplicates = initial_rows - len(df)
    context.log.info(f"Removed {removed_duplicates} duplicate rows.")

    before_dropna = len(df)
    df.dropna()
    removed_na = before_dropna - len(df)
    context.log.info(f"Removed {removed_na} rows with missing values.")

    df.reset_index(drop=True)
    context.log.info("Reset DataFrame index.")

    if 'Time' in df.columns:
        df.drop(['Time'], axis=1, inplace=True)

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
    description="Split dataset (features (X) , target (y) , 80/20 train-test split)",
    compute_kind="python",
    group_name="model_ml_fraud",
    required_resource_keys={"mlflow"}
)
def split_fraud_data(
    context: dg.AssetExecutionContext,
    clean_fraud: pd.DataFrame
) -> dict:
    mlflow_client = context.resources.mlflow
    X = clean_fraud.drop('Class', axis=1)
    y = clean_fraud['Class']
    context.log.info(f"Features X shape: {X.shape}, Target y shape: {y.shape}")   # pyright: ignore[reportAttributeAccessIssue]

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y
    )

    context.log.info(f"Training set: X_train {X_train.shape}, y_train {y_train.shape}")  # pyright: ignore[reportAttributeAccessIssue]
    context.log.info(f"Test set: X_test {X_test.shape}, y_test {y_test.shape}")   # pyright: ignore[reportAttributeAccessIssue]
    features = list(clean_fraud.columns)
    mlflow_client.log_metric("train_set_size", len(X_train))
    mlflow_client.log_metric("test_set_size", len(X_test))

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": features[:-1]
    }


@dg.asset(
    description="Tune RandomForest hyperparameters (GridSearchCV 3-fold) ",
    compute_kind="python",
    group_name="model_ml_fraud",
    required_resource_keys={"mlflow"}
    )
def tune_fraud_model(
    context: dg.AssetExecutionContext,
    split_fraud_data: dict
) -> dg.MaterializeResult:
    mlflow_client = context.resources.mlflow
    X_train = split_fraud_data["X_train"]
    y_train = split_fraud_data["y_train"]

    context.log.info("Starting GridSearchCV 3-fold cross-validation for RandomForest...")

    # Define hyperparameter grid
    param_grid = {
        "n_estimators": [50, 100, 150]
        # "max_depth": [None, 5, 10],
        # "min_samples_split": [2, 5, 10],
        # "min_samples_leaf": [1, 2, 4],
    }

    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=cv,
            scoring="accuracy",
            n_jobs=-1
        )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    acc = grid_search.best_score_
    cv_n = 3
    mlflow_client.log_param("cv_folds", cv_n)
    mlflow_client.log_params(param_grid)
    mlflow_client.log_params({f"best_{k}": v for k, v in best_params.items()})

    context.log.info(
            f"accuracy = {acc:.4f} | Best Params = {best_params}"
        )
    return dg.MaterializeResult(
        value=best_model,
        metadata={
            "best_params": dg.MetadataValue.text(str(best_params)),
        },
    )


@dg.asset(
    description="Evaluate the tuned RandomForest model on the test dataset",
    compute_kind="python",
    group_name="ml_fraud_evaluate",
    required_resource_keys={"mlflow"}
)
def evaluate_fraud_model(
    context: dg.AssetExecutionContext,
    tune_fraud_model,
    split_fraud_data: dict,
    # feature_names: list
) -> dg.MaterializeResult:

    mlflow_client = context.resources.mlflow
    registered_model_name = "tuned-fraud-model"
    best_model = tune_fraud_model
    X_test = split_fraud_data["X_test"]
    y_test = split_fraud_data["y_test"]
    feature_names = split_fraud_data["feature_names"]

    # context.log.info("Testing of the test set....")
    # if len(X_test) == 0:
    #     context.log.warning("⚠️ Test set is empty. Skipping final evaluation.")
    #     return dg.MaterializeResult(
    #         value={"status": "skipped_evaluation"},
    #         metadata={"reason": dg.MetadataValue.text("Test set empty.")}
    #     )

    context.log.info("sucess....")
    context.log.info("Starting model evaluation on test data...")
    # mlflow.set_tracking_uri("file:./mlruns")

    # Predictions
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]
    context.log.info("Final model trained successfully.")
    mlflow_client.sklearn.log_model(best_model, "random_forest_model")

    # Metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    context.log.info(f"Model test accuracy: {accuracy:.4f}")
    roc_auc = float(roc_auc_score(y_test, y_proba))
    precision = float(precision_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    f1 = float(f1_score(y_test, y_pred))

    context.log.info(
        f"✅ Model performance:\n"
        f"  Accuracy: {accuracy:.4f}\n"
        f"  Precision: {precision:.4f}\n"
        f"  Recall: {recall:.4f}\n"
        f"  F1-score: {f1:.4f}\n"
        f"  ROC-AUC: {roc_auc:.4f}"
    )

    # Classification report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    # report_df = pd.DataFrame(report_dict).transpose()
    context.log.info("Generated classification report.")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix - Fraud Detection Model")

    # Save plot as temporary image
    plot_path = "/tmp/confusion_matrix.png"
    plt.savefig(plot_path)
    plt.close()

    context.log.info(f"Confusion matrix saved at {plot_path}")

    # Log metrics and artifact to MLflow
    context.log.info("logging... to MLflow....")

    # MLflow logging
    # mlflow.set_experiment("ml_fraud_detection")
    # model_version_info = None
    # mlflow_client = mlflow.MlflowClient()

    with mlflow.start_run(run_name="log_fraud_model", nested=True):
        # Log metrics
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        mlflow.log_metric("test_f1", f1)
        if roc_auc:
            mlflow_client.log_metric("test_roc_auc", roc_auc)

        # Log confusion matrix
        mlflow_client.log_artifact(plot_path, artifact_path="plots")
        mlflow_client.log_param("model_type", "RandomForest")
        mlflow.log_artifact(plot_path, artifact_path="plots")
        mlflow.log_param("model_type", "RandomForest")

        # Log model
        log_model(
            sk_model=best_model,
            artifact_path="tuned_fraud_model",
            registered_model_name=registered_model_name,
            input_example=pd.DataFrame(X_test[:min(5, len(X_test))], columns=feature_names)
        )
        context.log.info(f"Model logged and registered as '{registered_model_name}'.")

        # Retrieve registered model version info

        # Register model in MLflow (preserves previous behavior)
        MODEL_ARTIFACT_NAME = "random_forest_model"
        try:
            run_id = mlflow_client.get_run(mlflow_client.active_run().info.run_id).info.run_uuid
        except Exception:
            run_id = None
        model_uri = f"runs:/{run_id}/{MODEL_ARTIFACT_NAME}" if run_id else MODEL_ARTIFACT_NAME
        try:
            model_info = mlflow_client.register_model(model_uri, registered_model_name,
)
            context.log.info(f"Registered model '{model_info.name}' version {model_info.version}")
        except Exception as e:
            context.log.warning(f"Model registration skipped/failed: {e}")
            model_info = None

        # model_versions = mlflow_client.search_model_versions(run_id)
        # matching_versions = [mv for mv in model_versions if mv.run_id == run.info.run_id]

        # if matching_versions:
        #     mv = matching_versions[0]
        #     model_version_info = {
        #         "name": mv.name,
        #         "version": mv.version,
        #         "status": mv.status,
        #         "stage": mv.current_stage,
        #         "model_uri": f"models:/{mv.name}/{mv.version} | runs:/{run_id}/{MODEL_ARTIFACT_NAME}"
        #     }
        #     context.log.info(f"Registered model version info retrieved: {model_version_info}")
        # else:
        #     raise Exception(f"No registered model version found for run {run.info.run_id}")

    return dg.MaterializeResult(
        value={
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "report": report_dict,
            "confusion_matrix_path": plot_path,
            # "model_version_info": model_version_info
        },
        metadata={
            "accuracy": dg.MetadataValue.float(accuracy),
            "precision": dg.MetadataValue.float(precision),
            "recall": dg.MetadataValue.float(recall),
            "f1_score": dg.MetadataValue.float(f1),
            "roc_auc": dg.MetadataValue.float(roc_auc) if roc_auc else dg.MetadataValue.text("N/A"),
            "confusion_matrix": dg.MetadataValue.path(plot_path),
            # "preview": dg.MetadataValue.md(report_df.head().to_markdown())
            # "registered_model": dg.MetadataValue.json(model_version_info),
            # "model_name": dg.MetadataValue.text(model_version_info["name"]),
            # "model_version": dg.MetadataValue.text(str(model_version_info["version"])),
            # "mlflow_run_id": dg.MetadataValue.text(run.info.run_id),
            # "mlflow_model_uri": dg.MetadataValue.text(model_version_info["model_uri"]),
            # "mlflow_stage": dg.MetadataValue.text(model_version_info["stage"])
        }
    )


@dg.asset(
    description="Send ML performance to Slack",
    compute_kind="python",
    group_name="slack_message",
    ins={"ml_fraud_evaluate": dg.AssetIn("evaluate_fraud_model")},
    resource_defs={"slack_notif": dagster_slack.SlackResource(token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN"))},
)
def fraud_message_slack(
    context: dg.AssetExecutionContext,
    ml_fraud_evaluate: dict
    ) -> None:
    recall = float(ml_fraud_evaluate.get("recall", 0.0))
    accuracy = float(ml_fraud_evaluate.get("accuracy", 0.0))
    precision = float(ml_fraud_evaluate.get("precision", 0.0))
    f1 = float(ml_fraud_evaluate.get("f1_score", 0.0))
    roc_auc = float(ml_fraud_evaluate.get("roc_auc", 0.0))

    model_name = " RandomForest Classifier "
    experiment = "📊 Fraud detection ml"
    trained_by = os.environ.get("GITHUB_USER")

    msg = (
        f"Fraud Detection Model Training\n\n"
        f" Trained by: Josue ({trained_by}) 😄 & Rojoniaina 👧\n"
        f"📊 *Performance Metrics:*\n"
        f"•  Accuracy: {accuracy:.4f} 🎯 \n"
        f"•  Precision: {precision:.4f}🚀\n"
        f"•  F1 Score: {f1:.4f}\n"
        f"•  Recall: {recall:.4f}\n"
        f"•  ROC-AUC: {roc_auc:.4f}\n\n"
        f" Model: {model_name}\n"
        f" Experiment: {experiment}\n\n"
        f" *GOOD!* Model. Ready for deployment! 🚀"
    )

    slacks = context.resources.slack_notif
    slacks.get_client().chat_postMessage(
        channel='aims_course_october2025',
        text=msg
    )
    context.log.info("Slack notification sent successfully.\n"
        f"Fraud Detection Model Training\n\n"
        f" Trained by: Josue{trained_by} 😄 & Rojoniaina 👧\n"
        f"📊 *Performance Metrics:*\n"
        f"•  Accuracy: {accuracy:.4f} 🎯 \n"
        f"•  Precision: {precision:.4f}🚀\n"
        f"•  F1 Score: {f1:.4f}\n"
        f"•  Recall: {recall:.4f}\n"
        f"•  ROC-AUC: {roc_auc:.4f}\n\n"
        f" Model: {model_name}\n"
        f" Experiment: {experiment}\n\n"
        # f" *GOOD!* Model. Ready for deployment! 🚀"")
    )
