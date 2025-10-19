import dagster as dg
import dagster_slack
import pandas as pd

from ..ml.resources import PromotionConfig, mlflow_resource

# Global variables for model and experiment configuration
MODEL_NAME = "FraudDetectionModel"
EXPERIMENT_NAME = "ml_fraud_detection"
TRAIN_PARENT_RUN_NAME = "train_parent"
REGISTER_MODEL_RUN_NAME = "register_model"

slack_instance = dagster_slack.SlackResource(token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN"))

resource_defs = {
    "mlflow_tracking": mlflow_resource,
    "slack_ml": slack_instance,
}


@dg.asset(
    description="Download data for fraud detection",
    group_name="ml_fraud_data",
    compute_kind="python",
    resource_defs=resource_defs,
)
def download_fraud_data(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:
    url = "http://raw.githubusercontent.com/aduuna/Kaggle-Data-Credit-Card-Fraud-Detection/master/samplecreditcard.csv"
    df = pd.read_csv(url)
    return dg.MaterializeResult(
        value=df,
        metadata={
            "source": "Kaggle",
            "preview": dg.MetadataValue.md(df.head().to_markdown() or ""),
            "dagster/row_count": len(df),
            "dagster/column_schema": dg.TableSchema(
                columns=[dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]
            ),
            "dagster/column_count": len(df.columns),
            "features": list(df.columns),
        },
    )


@dg.asset(
    description="Perform undersampling", group_name="ml_fraud_data", compute_kind="python", resource_defs=resource_defs
)
def undersample_fraud_data(
    context: dg.AssetExecutionContext,
    download_fraud_data: pd.DataFrame,
) -> dg.MaterializeResult:
    from typing import cast

    # Use the DataFrame directly (no need to load it again)
    df: pd.DataFrame = download_fraud_data

    df = df.sample(frac=1, random_state=42)  # Shuffle the data
    fraud_df: pd.DataFrame = cast(pd.DataFrame, df[df["Class"] == 1])
    non_fraud_df: pd.DataFrame = cast(pd.DataFrame, df[df["Class"] == 0][: len(fraud_df)])

    normal_distributed_df: pd.DataFrame = pd.concat([fraud_df, non_fraud_df], ignore_index=True)
    new_df: pd.DataFrame = normal_distributed_df.sample(frac=1, random_state=42)

    return dg.MaterializeResult(
        value=new_df,
        metadata={
            "preview": dg.MetadataValue.md(new_df.head().to_markdown() or ""),
            "dagster/row_count": len(new_df),
            "dagster/column_schema": dg.TableSchema(
                columns=[dg.TableColumn(k, str(v)) for k, v in new_df.dtypes.to_dict().items()]
            ),
            "dagster/column_count": len(new_df.columns),
            "fraud_count": len(new_df[new_df["Class"] == 1]),
            "non_fraud_count": len(new_df[new_df["Class"] == 0]),
        },
    )


@dg.asset(
    description="Standardization of features",
    group_name="ml_fraud_data",
    compute_kind="python",
    resource_defs=resource_defs,
)
def standardize_fraud_data(
    context: dg.AssetExecutionContext,
    undersample_fraud_data: pd.DataFrame,
) -> dg.MaterializeResult:
    from sklearn.preprocessing import StandardScaler

    df = undersample_fraud_data

    # Features to be standardized
    features = [col for col in df.columns if col in ["Amount", "Time"]]

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit and transform the features
    df[features] = scaler.fit_transform(df[features])

    return dg.MaterializeResult(
        value=df,
        metadata={
            "preview": dg.MetadataValue.md(df.head().to_markdown() or ""),
            "dagster/row_count": len(df),
            "dagster/column_schema": dg.TableSchema(
                columns=[dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]
            ),
            "dagster/column_count": len(df.columns),
            "features": features,
        },
    )


@dg.asset(
    description="Split data into train and test sets",
    group_name="ml_fraud_data",
    compute_kind="python",
    resource_defs=resource_defs,
)
def split_fraud_data(
    context: dg.AssetExecutionContext,
    standardize_fraud_data: pd.DataFrame,
) -> dg.MaterializeResult:
    from typing import Tuple, cast

    from sklearn.model_selection import train_test_split

    df: pd.DataFrame = standardize_fraud_data

    train_df: pd.DataFrame
    test_df: pd.DataFrame
    train_df, test_df = cast(Tuple[pd.DataFrame, pd.DataFrame], train_test_split(df, test_size=0.2, random_state=42))

    return dg.MaterializeResult(
        value={"train": train_df, "test": test_df},
        metadata={
            "preview_train": dg.MetadataValue.md(train_df.head().to_markdown() or "No train data preview"),
            "preview_test": dg.MetadataValue.md(test_df.head().to_markdown() or "No test data preview"),
            "dagster/train_row_count": len(train_df),
            "dagster/test_row_count": len(test_df),
            "dagster/column_schema": dg.TableSchema(
                columns=[dg.TableColumn(k, str(v)) for k, v in df.dtypes.to_dict().items()]
            ),
            "dagster/column_count": len(df.columns),
        },
    )


@dg.asset(
    description="Train a random forest classifer model",
    group_name="ml_fraud_model",
    compute_kind="python",
    resource_defs=resource_defs,
)
def train_fraud_model(
    context: dg.AssetExecutionContext,
    split_fraud_data: dict,
) -> dg.MaterializeResult:
    import mlflow
    import numpy as np
    from mlflow import sklearn as mlflow_sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    data = split_fraud_data
    train_df = data["train"]

    X_train = train_df.drop(columns=["Class"]).to_numpy()
    y_train = train_df["Class"].to_numpy()

    # Define the hyperparameter grid
    param_grid = {"n_estimators": [25, 50, 75, 100]}

    mlflow_client = context.resources.mlflow_tracking

    # Use a unique experiment name to avoid conflicts
    import datetime

    experiment_name = EXPERIMENT_NAME
    # Check if experiment exists, if not create it
    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    if experiment is None:
        try:
            experiment_id = mlflow_client.create_experiment(experiment_name)
            context.log.info(f"Created new experiment: {experiment_name}")
        except Exception as e:
            # If experiment creation fails (e.g., already exists), try to get it again
            context.log.warning(f"Failed to create experiment: {e}. Trying to get existing experiment.")
            experiment = mlflow_client.get_experiment_by_name(experiment_name)
            if experiment is not None:
                experiment_id = experiment.experiment_id
                context.log.info(f"Using existing experiment: {experiment_name}")
            else:
                # As a fallback, try to get by ID or use default experiment
                context.log.warning("Creating a new experiment with timestamp to avoid conflicts")
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                experiment_name = f"fraud_detection_{timestamp}"
                experiment_id = mlflow_client.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
        context.log.info(f"Using existing experiment: {experiment_name}")

    if mlflow.active_run() is not None:
        mlflow.end_run()

    mlflow.set_experiment(experiment_id=experiment_id)

    with mlflow.start_run(run_name=TRAIN_PARENT_RUN_NAME, experiment_id=experiment_id):
        best_score = -np.inf
        best_params = None
        best_model = None

        for n_estimators in param_grid["n_estimators"]:
            with mlflow.start_run(run_name=f"n_estimators_{n_estimators}", nested=True, experiment_id=experiment_id):
                context.log.info(f"Training with n_estimators={n_estimators}")
                model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)

                cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring="accuracy")
                mean_cv_score = np.mean(cv_scores)
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_metric("mean_cv_accuracy", float(mean_cv_score))

                model.fit(X_train, y_train)

                if mean_cv_score > best_score:
                    best_score = mean_cv_score
                    best_params = n_estimators
                    best_model = model

        context.log.info(f"Best n_estimators: {best_params} with CV accuracy: {best_score}")

    mlflow_sklearn.log_model(
        best_model,
        artifact_path="ml_fraud_model",
        input_example=pd.DataFrame(X_train[:5], columns=train_df.drop(columns=["Class"]).columns),
        registered_model_name=MODEL_NAME,
    )
    return dg.MaterializeResult(
        value=best_model,
        metadata={
            "best_n_estimators": int(best_params) if best_params is not None else 0,
            "best_cv_accuracy": float(best_score),
        },
    )


@dg.asset(
    description="Evaluate the trained model on the test set",
    group_name="ml_fraud_model",
    compute_kind="python",
    resource_defs=resource_defs,
)
def evaluate_fraud_model(
    context: dg.AssetExecutionContext,
    train_fraud_model: object,
    split_fraud_data: dict,
) -> dg.MaterializeResult:
    import io
    from typing import Any, Dict

    import matplotlib.pyplot as plt
    import mlflow
    from mlflow.tracking import MlflowClient
    from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve

    data = split_fraud_data
    test_df = data["test"]

    X_test = test_df.drop(columns=["Class"]).to_numpy()
    y_test = test_df["Class"].to_numpy()

    model = train_fraud_model
    y_pred = model.predict(X_test)  # type: ignore

    class_report: Dict[str, Any] = classification_report(y_test, y_pred, output_dict=True)  # type: ignore
    conf_matrix = confusion_matrix(y_test, y_pred)

    context.log.info("Classification Report:")
    context.log.info(classification_report(y_test, y_pred))
    context.log.info("Confusion Matrix:")
    context.log.info(conf_matrix)

    # Type-safe access to classification report
    class_0_metrics = class_report.get("0", {})
    class_1_metrics = class_report.get("1", {})

    if not isinstance(class_0_metrics, dict) or not isinstance(class_1_metrics, dict):
        raise ValueError("Unexpected classification report format")

    mlflow_client = MlflowClient()

    # Get the experiment (it should exist from training step)
    experiment = mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        # If the exact experiment doesn't exist, try to find any fraud detection experiment
        all_experiments = mlflow_client.search_experiments()
        fraud_experiments = [exp for exp in all_experiments if EXPERIMENT_NAME in exp.name.lower()]
        if fraud_experiments:
            experiment_id = fraud_experiments[0].experiment_id
            context.log.info(f"Using experiment: {fraud_experiments[0].name}")
        else:
            # Create new experiment if none found
            experiment_id = mlflow_client.create_experiment(EXPERIMENT_NAME)
            context.log.info(f"Created new experiment: {EXPERIMENT_NAME}")
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_id=experiment_id)

    # log the confusion matrix as an image in mlflow
    _, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(conf_matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    # Add text annotations
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], "d"), ha="center", va="center", color="black")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    from PIL import Image

    image = Image.open(buf)
    mlflow.log_image(image, "confusion_matrix.png")
    buf.close()

    # Log ROC curve
    y_prob = model.predict_proba(X_test)[:, 1]  # type: ignore
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    context.log.info(f"ROC AUC: {roc_auc}")
    _, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fpr, tpr)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    from PIL import Image

    image = Image.open(buf)
    mlflow.log_image(image, "roc_curve.png")
    buf.close()

    return dg.MaterializeResult(
        value={
            "model": model,
            "classification_report": class_report,
            "roc_auc": roc_auc,
        },
        metadata={
            "classification_report": dg.MetadataValue.json(class_report),
            "confusion_matrix": dg.MetadataValue.md(pd.DataFrame(conf_matrix).to_markdown()
                or "No confusion matrix data"),
            "accuracy": float(class_report.get("accuracy", 0.0)),
            "precision_class_0": float(class_0_metrics.get("precision", 0.0)),
            "recall_class_0": float(class_0_metrics.get("recall", 0.0)),
            "f1_score_class_0": float(class_0_metrics.get("f1-score", 0.0)),
            "precision_class_1": float(class_1_metrics.get("precision", 0.0)),
            "recall_class_1": float(class_1_metrics.get("recall", 0.0)),
            "f1_score_class_1": float(class_1_metrics.get("f1-score", 0.0)),
            "roc_auc": float(roc_auc),
        },
    )


@dg.asset(
    description="Register the trained model in MLflow Model Registry",
    group_name="ml_fraud_model",
    compute_kind="python",
    resource_defs=resource_defs,
)
def register_fraud_model(
    context: dg.AssetExecutionContext,
    evaluate_fraud_model: dict,
) -> dg.MaterializeResult:
    import mlflow
    from mlflow.tracking import MlflowClient

    model = evaluate_fraud_model["model"]

    mlflow_client = MlflowClient()

    # Get the experiment (it should exist from training step)
    experiment = mlflow_client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        # If the exact experiment doesn't exist, try to find any fraud detection experiment
        all_experiments = mlflow_client.search_experiments()
        fraud_experiments = [exp for exp in all_experiments if EXPERIMENT_NAME in exp.name.lower()]
        if fraud_experiments:
            experiment_id = fraud_experiments[0].experiment_id
            context.log.info(f"Using experiment: {fraud_experiments[0].name}")
        else:
            # Create new experiment if none found
            experiment_id = mlflow_client.create_experiment(EXPERIMENT_NAME)
            context.log.info(f"Created new experiment: {EXPERIMENT_NAME}")
    else:
        experiment_id = experiment.experiment_id

    with mlflow.start_run(run_name=REGISTER_MODEL_RUN_NAME, experiment_id=experiment_id, nested=True):
        current_run = mlflow.active_run()
        if current_run is None:
            raise ValueError("No active MLflow run found")

        model_uri = f"runs:/{current_run.info.run_id}/model"
        model_name = MODEL_NAME

        try:
            mlflow_client.create_registered_model(model_name)
            context.log.info(f"Created new registered model: {model_name}")
        except Exception as e:
            context.log.info(f"Registered model {model_name} already exists. {e}")

        mv = mlflow_client.create_model_version(
            name=model_name,
            source=model_uri,
            run_id=current_run.info.run_id,
        )
        context.log.info(f"Registered model version: {mv.version}")

        # Transition the model to "Staging"
        mlflow_client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=False,
        )
        context.log.info(f"Transitioned model version {mv.version} to Staging")

    return dg.MaterializeResult(
        value=model,
        metadata={
            "model_name": model_name,
            "model_version": mv.version,
            "model_stage": "Staging",
        },
    )


@dg.asset(
    description="Send Slack notification with model evaluation results",
    group_name="ml_fraud_notifications",
    compute_kind="python",
    resource_defs=resource_defs,
)
def send_slack_notification(
    context: dg.AssetExecutionContext,
    evaluate_fraud_model: dict,
) -> dg.MaterializeResult:
    import os
    from typing import Any, Dict

    # Extract evaluation results
    class_report: Dict[str, Any] = evaluate_fraud_model["classification_report"]
    roc_auc = evaluate_fraud_model["roc_auc"]

    # Type-safe access to classification report
    class_0_metrics = class_report.get("0", {})
    class_1_metrics = class_report.get("1", {})

    if not isinstance(class_0_metrics, dict) or not isinstance(class_1_metrics, dict):
        # Fallback values if the structure is unexpected
        class_0_metrics = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}
        class_1_metrics = {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}

    slack: dagster_slack.SlackResource = context.resources.slack_ml
    slack.get_client().chat_postMessage(
        channel="aims_course_october2025",
        text=f"""{os.environ.get("GITHUB_USER", "default")}: Fraud detection model evaluation completed.
        Accuracy: {class_report.get("accuracy", 0.0):.4f}
        Precision (Class 1): {class_1_metrics.get("precision", 0.0):.4f}
        Recall (Class 1): {class_1_metrics.get("recall", 0.0):.4f}
        F1-Score (Class 1): {class_1_metrics.get("f1-score", 0.0):.4f}
        Precision (Class 0): {class_0_metrics.get("precision", 0.0):.4f}
        Recall (Class 0): {class_0_metrics.get("recall", 0.0):.4f}
        F1-Score (Class 0): {class_0_metrics.get("f1-score", 0.0):.4f}
        ROC AUC: {roc_auc:.4f}
        """,
    )

    return dg.MaterializeResult(
        value="Slack notification sent successfully",
        metadata={
            "channel": "aims_course_october2025",
            "user": os.environ.get("GITHUB_USER", "default"),
            "message_sent": True,
        },
    )


@dg.asset(
    description="Promote model to staging in MLflow Model Registry",
    group_name="ml_fraud_promote",
    compute_kind="python",
    resource_defs=resource_defs,
)
def promote_fraud_model_to_staging(
    context: dg.AssetExecutionContext, config: PromotionConfig, evaluate_fraud_model: dict
) -> dg.MaterializeResult:
    # Get the MLflow client from the context to interact with the model registry
    mlflow_client = context.resources.mlflow_tracking
    context.log.info("Starting model promotion to Staging.")

    # If the evaluation step was skipped, we also skip promotion
    if evaluate_fraud_model.get("status") == "skipped_evaluation":
        context.log.info("Evaluation was skipped in the previous step. Skipping staging promotion.")
        return dg.MaterializeResult(
            value={"status": "skipped_promotion", "reason": "evaluation_skipped_upstream"},
            metadata={"status": "skipped_promotion_due_to_upstream_skip"},
        )
    # Extract metrics and model version info from evaluation result
    eval_metrics = evaluate_fraud_model.get("eval_metrics", {})
    model_version_info = evaluate_fraud_model.get("model_version_info")

    # If no model version info was returned, skip promotion.
    # model_version_info might be None due to an upstream failure
    if not model_version_info:
        context.log.warning("No valid model version info found from evaluation. Skipping staging promotion.")
        return dg.MaterializeResult(
            value={"status": "skipped_promotion", "reason": "no_model_version_info_from_evaluation"},
            metadata={"status": "skipped_promotion_no_model_info"},
        )

    try:
        # Extract the model name and version for promotion
        model_name = model_version_info["name"]
        model_version = model_version_info["version"]

        # Promote the model to the 'Staging' stage
        context.log.info(f"Model '{model_name}' (version {model_version}) meets criteria. Promoting to Staging")
        mlflow_client.transition_model_version_stage(name=model_name, version=model_version, stage="Staging")

        # Return successful result with status and relevant metadata
        context.log.info(f"Model '{model_name}' (version {model_version}) promoted to Staging.")
        return dg.MaterializeResult(
            value={
                "status": "promoted_to_staging",
                "model_name": model_name,
                "model_version": model_version,
                "metrics": eval_metrics,
            },
            metadata={
                "status": "promoted_to_staging",
                "model_name": dg.MetadataValue.text(model_name),
                "model_version": dg.MetadataValue.text(str(model_version)),
            },
        )
    except Exception as e:
        # Handle any exception during promotion and log the error
        context.log.error(f"Error promoting model to Staging: {e}")
        return dg.MaterializeResult(
            value={"status": "failed_promotion_to_staging", "error": str(e)},
            metadata={"status": "failed_promotion_to_staging", "error_message": dg.MetadataValue.text(str(e))},
        )


@dg.asset(
    description="Promotes the best model from Staging to Production, usually with manual approval.",
    resource_defs=resource_defs,
    compute_kind="python",
    group_name="ml_fraud_promote",
)
def promote_fraud_model_to_production(
    context: dg.AssetExecutionContext, promote_fraud_model_to_staging: dict
) -> dg.MaterializeResult:
    # Get the MLflow client to interact with the model registry
    mlflow_client = context.resources.mlflow_tracking
    context.log.info("Starting model promotion to Production.")

    # Step 1: Check if a model was promoted to Staging previously
    if promote_fraud_model_to_staging.get("status") != "promoted_to_staging":
        # If no model was promoted to staging in the last step, skip production promotion
        context.log.info("No model was promoted to Staging in the previous step. Skipping production promotion.")
        return dg.MaterializeResult(
            value={"status": "skipped_production_promotion", "reason": "no_model_in_staging_from_previous_step"},
            metadata={"status": "skipped_production_promotion"},
        )
    # Get the model name from the previous promotion step
    model_name = promote_fraud_model_to_staging.get("model_name", MODEL_NAME)

    # Simulate manual approval
    # In a real scenario, this would involve a manual review/approval process.
    manual_approval_granted = True

    # Proceed with promotion only if manual approval is granted
    if manual_approval_granted:
        try:
            # Find the latest model version in Staging for the given model_name
            latest_staging_version = None
            for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
                if mv.current_stage == "Staging":
                    # Ensure we promote the highest version currently in Staging
                    if latest_staging_version is None or mv.version > latest_staging_version.version:
                        latest_staging_version = mv

            # If no model is found in staging, log a warning and skip promotion
            if not latest_staging_version:
                context.log.warning(f"No model found in Staging stage for '{model_name}'. Skipping prod promotion.")
                return dg.MaterializeResult(
                    value={"status": "skipped_production_promotion", "reason": "no_staging_model_found_for_prod"},
                    metadata={"status": "skipped_production_promotion_no_staging_model"},
                )

            # Extract the model name and version to promote
            prod_model_name = latest_staging_version.name
            prod_model_version = latest_staging_version.version

            # Archive all existing models in Production
            for mv in mlflow_client.search_model_versions(f"name='{model_name}'"):
                if mv.current_stage == "Production":
                    context.log.info(f"Archiving previous Production model '{mv.name}' (version {mv.version})")
                    mlflow_client.transition_model_version_stage(name=mv.name, version=mv.version, stage="Archived")

            # Promote the new version to Production
            context.log.info(f"Promoting model '{prod_model_name}' (version {prod_model_version}) to Production")
            mlflow_client.transition_model_version_stage(
                name=prod_model_name, version=prod_model_version, stage="Production"
            )

            # Return success with metadata about the promoted model
            context.log.info(f"Model '{prod_model_name}' (version {prod_model_version}) promoted to Production.")
            return dg.MaterializeResult(
                value={
                    "status": "promoted_to_production",
                    "model_name": prod_model_name,
                    "model_version": prod_model_version,
                    "previous_metrics": promote_fraud_model_to_staging.get("metrics"),
                },
                metadata={
                    "status": "promoted_to_production",
                    "model_name": dg.MetadataValue.text(prod_model_name),
                    "model_version": dg.MetadataValue.text(str(prod_model_version)),
                },
            )
        except Exception as e:
            # Catch and log any error that occurs during the promotion process
            context.log.error(f"Error promoting model to Production: {e}")
            return dg.MaterializeResult(
                value={"status": "failed_production_promotion", "error": str(e)},
                metadata={"status": "failed_production_promotion", "error_message": dg.MetadataValue.text(str(e))},
            )

    # If manual approval was denied, skip promotion and return reason
    else:
        context.log.info("Manual approval not granted. Skipping production promotion")
        return dg.MaterializeResult(
            value={"status": "not_promoted_to_production", "reason": "manual_approval_denied"},
            metadata={"status": "not_promoted_to_production"},
        )
