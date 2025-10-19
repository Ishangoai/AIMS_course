import os
import pickle
import tempfile

import dagster as dg
import dagster_slack
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn as ms
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, KFold, train_test_split

from ..ml.resources import mlflow_resource
from .resources import FraudDataConfig


@dg.asset(
    description="Dowload data for fraud detection",
    compute_kind="python",
    group_name="ml_fraud_ingest",
    resource_defs={"fraud_data": FraudDataConfig(), "mlflow_tracking": mlflow_resource
    }
)
def fraud_detection(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:

    # Load data from csv
    fraud_data_config = context.resources.fraud_data.data_source
    context.log.info(f"Processing file:\n {fraud_data_config}")

    # Convert to pandas DataFrame
    df: pd.DataFrame = pd.read_csv(fraud_data_config)
    context.log.info(f"Pandas DataFrame shape: {df.shape}")

    mlflow_client = context.resources.mlflow_tracking

    dataset = mlflow_client.data.from_pandas(df, name="fraud_detection_data")
    mlflow_client.log_input(dataset=dataset, context="training")
    column_names = list(df.columns)
    target_variable = column_names[-1]

    # Drop target variable name
    column_names.pop()

    # Add a light materialization with some metadata so Dagster shows it
    try:
        context.log.info("Materializing fraud_detection dataset metadata.")
        context.log_event(dg.AssetMaterialization(asset_key=dg.AssetKey("fraud_detection"), metadata={
            "rows": df.shape[0],
            "columns": df.shape[1],
            "source": fraud_data_config
        }))
    except Exception:
        # Older/newer dagster APIs differ; fall back to logging
        context.log.info(f"fraud_detection metadata: rows={df.shape[0]}, cols={df.shape[1]}")

    return dg.MaterializeResult(
        value=df,
        metadata={
            "source": dg.MetadataValue.text(str(fraud_data_config)),
            "preview": dg.MetadataValue.md(df.head().to_markdown() or ""),
            "description": dg.MetadataValue.text(
                "Fraud Detection Data fetched from Kaggle."
            ),
            "rows": dg.MetadataValue.int(df.shape[0]),
            "columns": dg.MetadataValue.int(df.shape[1]),
            "features": dg.MetadataValue.text(", ".join(column_names)),
            "target": dg.MetadataValue.text(target_variable)
        }
    )


@dg.asset(
    description="Split data into train and test",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_model"
)
def train_test_split_data(context: dg.AssetExecutionContext, fraud_detection: pd.DataFrame) -> dict:
    """Split data into 80–20 train/test sets."""
    # Expect the dataset to use column name 'Class' for fraud label (as in common creditcard datasets)

    mlflow_client = context.resources.mlflow_tracking
    context.log.info("Performing train test split for the data.")

    feature_names = fraud_detection.columns[:-1]

    X = fraud_detection.drop(columns=["Class"])
    y = fraud_detection["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    try:
        mlflow_client.log_input(dataset=mlflow_client.data.from_pandas(X_train, name="X_train"), context="training")
        mlflow_client.log_input(dataset=mlflow_client.data.from_pandas(X_test, name="X_test"), context="training")

    except Exception:
        context.log.info("Could not log train/test splits to MLflow "
        "(client may not support log_input for these objects).")

    context.log.info(f"Train/Test split sizes: X_train={X_train.shape}, X_test={X_test.shape}")  # type: ignore

    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test, "feature_names": feature_names}


@dg.asset(
    description="Perform nested 3-fold cross-validation with "
    "GridSearch tuning using RandomForestClassifier and log metrics to MLflow",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_model",
)
def cross_validation_results(
    context: dg.AssetExecutionContext,
    train_test_split_data: dict,
) -> dict:
    """Nested 3-fold CV with inner GridSearch hyperparameter tuning."""
    X_train = train_test_split_data["X_train"]
    y_train = train_test_split_data["y_train"]

    # --- Base model ---
    base_model = RandomForestClassifier(random_state=42)

    # --- Hyperparameter grid ---
    param_grid = {
        "n_estimators": [10, 20, 50, 100]
    }

    outer_cv = KFold(n_splits=3, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

    mlflow_client = context.resources.mlflow_tracking

    accuracies, precisions, recalls, f1s, confusion_matrices = [], [], [], [], []
    best_models = []

    fold = 1
    for train_idx, test_idx in outer_cv.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[test_idx]

        # --- Inner GridSearchCV for tuning ---
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="accuracy",
            n_jobs=-1,
        )

        grid_search.fit(X_tr, y_tr)
        best_model = grid_search.best_estimator_
        best_models.append(best_model)

        # --- Evaluate on outer fold ---
        y_pred = best_model.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred)
        rec = recall_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        cm = confusion_matrix(y_val, y_pred)

        accuracies.append(acc)
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        confusion_matrices.append(cm)

        # Log fold-wise metrics
        try:
            mlflow_client.log_metric(f"fold_{fold}_accuracy", acc)
            mlflow_client.log_metric(f"fold_{fold}_precision", prec)
            mlflow_client.log_metric(f"fold_{fold}_recall", rec)
            mlflow_client.log_metric(f"fold_{fold}_f1", f1)
        except Exception as e:
            context.log.warning(f"Could not log fold {fold} metrics to MLflow: {e}")

        context.log.info(f"Fold {fold} best params: {grid_search.best_params_}")
        fold += 1

    # --- Aggregate final metrics ---
    mean_acc = float(np.mean(accuracies))
    mean_prec = float(np.mean(precisions))
    mean_rec = float(np.mean(recalls))
    mean_f1 = float(np.mean(f1s))
    mean_cm = np.mean(confusion_matrices, axis=0)

    # --- Log aggregate metrics ---
    try:
        mlflow_client.log_metric("cv_mean_accuracy", mean_acc)
        mlflow_client.log_metric("cv_mean_precision", mean_prec)
        mlflow_client.log_metric("cv_mean_recall", mean_rec)
        mlflow_client.log_metric("cv_mean_f1", mean_f1)

    except Exception as e:
        context.log.warning(f"Could not log aggregated CV metrics: {e}")

    # --- Log confusion matrix using Matplotlib ---
    try:
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(mean_cm, cmap="Blues")
        for i in range(mean_cm.shape[0]):
            for j in range(mean_cm.shape[1]):
                ax.text(j, i, int(mean_cm[i, j]), ha="center", va="center", color="black")  # type: ignore
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title("Mean Confusion Matrix (Nested 3-Fold CV)")
        fig.colorbar(im, ax=ax)

        with tempfile.TemporaryDirectory() as tmpdir:
            cm_path = os.path.join(tmpdir, "mean_confusion_matrix.png")
            plt.tight_layout()
            plt.savefig(cm_path)
            plt.close(fig)
            mlflow_client.log_artifact(cm_path)
    except Exception as e:
        context.log.warning(f"Could not log confusion matrix artifact: {e}")

    context.log.info(
        f"Nested 3-Fold CV Results — Accuracy: {mean_acc:.3f}, "
        f"Precision: {mean_prec:.3f}, Recall: {mean_rec:.3f}, F1: {mean_f1:.3f}"
    )

    # --- Select final best model (highest mean accuracy) ---
    best_model_idx = int(np.argmax(accuracies))
    final_best_model = best_models[best_model_idx]

    # ✅ Return only aggregate scores and final model
    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": train_test_split_data["X_test"],
        "y_test": train_test_split_data["y_test"],
        "feature_names": train_test_split_data["feature_names"],
        "mean_accuracy": mean_acc,
        "mean_precision": mean_prec,
        "mean_recall": mean_rec,
        "mean_f1": mean_f1,
        "best_model": final_best_model,
        "best_params": final_best_model.get_params()
    }


@dg.asset(
    description="Trains the random forest model using the best hyperparameters found.",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_model"
)
def train_random_forest_tuned_model(
    context: dg.AssetExecutionContext,
    cross_validation_results: dict
) -> dict:
    mlflow_client = context.resources.mlflow_tracking
    context.log.info("Starting final model training with tuned hyperparameters.")

    best_params = cross_validation_results["best_params"]
    X_train = cross_validation_results["X_train"]
    y_train = cross_validation_results["y_train"]
    X_test = cross_validation_results["X_test"]
    y_test = cross_validation_results["y_test"]
    feature_names = cross_validation_results["feature_names"]

    context.log.info(f"Training Ridge model with parameters: {best_params}")
    context.log.info(f"Training on {len(X_train)} samples.")

    final_model = RandomForestClassifier(n_estimators=best_params['n_estimators'])
    final_model.fit(X_train, y_train)
    context.log.info("Final Random Forest model trained.")

    train_params_log = {
        "model_type": "Random Forest Classifier",
        "n_estimators": best_params['n_estimators'],
        "feature_used": ", ".join(feature_names),
        "final_train_samples": len(X_train),
        "final_test_samples": len(X_test)
    }
    mlflow_client.log_params(train_params_log)
    context.log.info(f"Logged final training parameters to MLflow: {train_params_log}")

    return {
        "final_model": final_model,
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
        "feature_names": feature_names
    }


@dg.asset(
    description="Computes and logs feature importances of the trained Random Forest model.",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_model"
)
def feature_importance_analysis(
    context: dg.AssetExecutionContext,
    train_random_forest_tuned_model: dict
) -> dict:
    """
    Computes the feature importances from the trained Random Forest model
    and logs them to MLflow for interpretability analysis.
    """

    mlflow_client = context.resources.mlflow_tracking
    final_model: RandomForestClassifier = train_random_forest_tuned_model["final_model"]
    feature_names = train_random_forest_tuned_model["feature_names"]
    X_train = train_random_forest_tuned_model["X_train"]
    y_train = train_random_forest_tuned_model["y_train"]
    X_test = train_random_forest_tuned_model["X_test"]
    y_test = train_random_forest_tuned_model["y_test"]

    context.log.info("Computing feature importances from the trained model.")

    # Compute importances
    importances = final_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    }).sort_values(by="importance", ascending=False)

    context.log.info("Feature importance computation complete.")
    context.log.info(f"Features Ranking:\n{feature_importance_df}")

    # Log to MLflow
    mlflow_client.log_table(feature_importance_df, artifact_file="feature_importance.parquet")
    mlflow_client.log_figure(
        feature_importance_df.plot(
            kind="barh", x="feature", y="importance", legend=False, title="Feature Importance"
        ).get_figure(),
        artifact_file="feature_importance/feature_importance_plot.png"
    )

    context.log.info("Logged feature importance table and plot to MLflow.")

    testing_values = {
        "final_model": final_model,
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
        "feature_names": list(feature_importance_df["feature"].head(9))
        }

    return testing_values


@dg.asset(
    description="Retrains Random Forest using top 10 most important features, plus Time and.",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_model"
)
def retrain_top_features_model(
    context: dg.AssetExecutionContext,
    feature_importance_analysis: dict
) -> dict:
    mlflow_client = context.resources.mlflow_tracking
    top_features = feature_importance_analysis["feature_names"]
    top_features.append("Time")
    top_features.append("Amount")
    context.log.info(f"Training features: {top_features}")

    X_train = feature_importance_analysis["X_train"]
    y_train = feature_importance_analysis["y_train"]
    X_test = feature_importance_analysis["X_test"]
    y_test = feature_importance_analysis["y_test"]
    final_model = feature_importance_analysis["final_model"]
    X_train_top = X_train[top_features]

    model = RandomForestClassifier(
        n_estimators=final_model.n_estimators
    )
    model.fit(X_train_top, y_train)

    mlflow_client.log_param("Number of training features", len(top_features))
    mlflow_client.log_param("selected_features", ", ".join(top_features))

    # return {"model_top10": model, "top_features": top_features}
    testing_values = {
        "final_model": model,
        "X_test": X_test[top_features],
        "y_test": y_test,
        "feature_names": top_features,
        }

    return testing_values


@dg.asset(description="Perform test on the test set",
    resource_defs={"mlflow_tracking": mlflow_resource},
    compute_kind="python",
    group_name="ml_fraud_evaluate",)
def test_fraud_detection_model(
    context: dg.AssetExecutionContext,
    retrain_top_features_model: dict) -> dg.MaterializeResult:

    final_model = retrain_top_features_model["final_model"]
    X_test = retrain_top_features_model["X_test"]
    y_test = retrain_top_features_model["y_test"]
    feature_names = retrain_top_features_model["feature_names"]

    y_pred = final_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow_client = context.resources.mlflow_tracking

    context.log.info(f"Final Model Evaluation Metrics on Test Set: "
        f"\nAccuracy={acc:.4f}, \nPrecision={prec:.4f}, \nRecall={rec:.4f}, \nF1 Score={f1:.4f}")

    eval_metrics = {"test_accuracy": acc, "test_precision": prec, "test_recall": rec, "test_f1": f1}
    mlflow_client.log_metrics(eval_metrics)
    context.log.info(f"Logged final evaluation metrics to MLflow: {eval_metrics}")

    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot confusion matrix with matplotlib only
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap=plt.cm.Blues)  # type: ignore Use a colormap like 'Blues'
    fig.colorbar(cax, ax=ax)  # Correct way to attach color bar to axis

    # Set titles and labels
    ax.set_title('Confusion Matrix', pad=20)
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')

    # Set tick marks and labels
    ax.set_xticks(np.arange(len(np.unique(y_test))))
    ax.set_yticks(np.arange(len(np.unique(y_test))))
    ax.xaxis.set_ticks_position('bottom')

    # Annotate each cell with numeric value
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha='center', va='center', color='black')

    plt.tight_layout()

    # Save to temporary file and log artifact

    cm_path = os.getcwd() + "/src/students/Magguire/dagster/ml_fraud/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close(fig)

    mlflow_client.log_artifact(cm_path, artifact_path="confusion_matrix")
    context.log.info(f"Confusion matrix saved and logged to MLflow: {cm_path}")

    # Register model
    registered_model_name = "tuned-random-forest"
    model_version_info = None

    with mlflow.start_run(nested=True) as current_run:
        context.log.info(f"Starting nested MLflow run for model logging: {current_run.info.run_id}")

        log_model_info = ms.log_model(
            sk_model=final_model,
            artifact_path="tuned_random_forest",
            input_example=pd.DataFrame(X_test[:min(5, len(X_test))], columns=feature_names),
            registered_model_name=registered_model_name
        )
        context.log.info(f"Model logged to MLflow Run ID: {current_run.info.run_id}")
        context.log.info(f"Logged model artifact URI: {log_model_info.model_uri}")

        # use search_model_versions with a proper filter string
        model_versions = mlflow_client.search_model_versions(
            filter_string=f"name='{registered_model_name}'"
        )

        # Find the model version registered in this run
        matching_versions = [
            mv for mv in model_versions if mv.run_id == current_run.info.run_id
        ]

        if matching_versions:
            registered_model_version = matching_versions[0]
            model_version_info = {
                "name": registered_model_version.name,
                "version": registered_model_version.version,
                "status": registered_model_version.status,
                "stage": registered_model_version.current_stage,
                "model_uri": f"models:/{registered_model_version.name}/{registered_model_version.version}"
            }
            context.log.info("Successfully retrieved registered model version info from registry.")

        else:
            context.log.error(
                f"Could not find registered model version for run ID {current_run.info.run_id} "
                f"and name '{registered_model_name}'."
            )
            raise Exception("Failed to retrieve registered model version details after logging.")

    output_value_for_downstream = {
        "eval_metrics": eval_metrics,
        "best_model": final_model,
        "model_version_info": model_version_info,
        "status": "evaluated_successfully"
    }

    return dg.MaterializeResult(
        value=output_value_for_downstream,
        metadata={
            "test_accuracy": dg.MetadataValue.float(float(acc)),
            "test_precision": dg.MetadataValue.float(float(prec)),
            "test_recall": dg.MetadataValue.float(float(rec)),
            "test_f1": dg.MetadataValue.float(float(f1)),
            "model_name": dg.MetadataValue.text(model_version_info["name"]),
            "model_version": dg.MetadataValue.text(str(model_version_info["version"])),
            "mlflow_run_id": dg.MetadataValue.text(current_run.info.run_id),
            "mlflow_model_uri": dg.MetadataValue.text(model_version_info["model_uri"]),
            "mlflow_stage": dg.MetadataValue.text(model_version_info["stage"])
        }
    )


@dg.asset(
    description="Save the tuned RandomForest model as a pickle file for Gradio use.",
    compute_kind="python",
    group_name="ml_fraud_store",
    resource_defs={"mlflow_tracking": mlflow_resource,
    "slack_resource": dagster_slack.SlackResource(token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN"))}
)
def save_tuned_model(context: dg.AssetExecutionContext, test_fraud_detection_model: dict) -> str:
    """
    Saves the tuned RandomForest model to a local pickle file
    and returns the absolute file path.
    """
    best_model = test_fraud_detection_model["best_model"]

    # Make sure models directory exists relative to Dagster project root
    save_dir = os.getcwd() + "/src/students/Magguire/gradioapp/utils/"

    model_path = os.path.join(save_dir, "fraud_detection_model_test.pkl")

    context.log.info(f"Saving model to: {model_path}")

    # Send message on slack
    slack: dagster_slack.SlackResource = context.resources.slack_resource
    slack.get_client().chat_postMessage(
        channel='aims_course_october2025',
        text=f"🇰🇪 {os.environ.get("USER_1", "default")} & 🇧🇯 {os.environ.get("USER_2", "default")}'s "
        "dagster pipeline has run sucessfully \U0001FAE1  "
        "\n 📈 *Model metrics:*"
        f"\n• Accuracy: {round(test_fraud_detection_model['eval_metrics']['test_accuracy'] * 100, 2)}%"
        f"\n• Recall: {round(test_fraud_detection_model['eval_metrics']['test_recall'] * 100, 2)}%"
        f"\n• Precision: {round(test_fraud_detection_model['eval_metrics']['test_precision'] * 100, 2)}%"
        f"\n• F1 Score: {round(test_fraud_detection_model['eval_metrics']['test_f1'] * 100, 2)}%"
        "\n      *Asante sana*"
        "\n      *Au revoir*"
    )

    try:
        with open(model_path, "wb") as f:
            pickle.dump(best_model, f)
        context.log.info(f"Model successfully saved at {model_path}")
    except Exception as e:
        context.log.error(f"Failed to save model: {e}")
        raise

    # Optional: Log file artifact to MLflow
    try:
        mlflow_client = context.resources.mlflow_tracking
        mlflow_client.log_artifact(model_path)
    except Exception as e:
        context.log.warning(f"Could not log model to MLflow: {e}")

    # Emit a materialization event with the path so Dagster UI records it
    try:
        context.log_event(dg.AssetMaterialization(asset_key=dg.AssetKey("tuned_random_forest_artifact"), metadata={
            "model_path": model_path
        }))
    except Exception:
        context.log.info(f"Saved model at {model_path}")

    # Return file path so you can use it downstream (e.g. in Gradio)
    return model_path
