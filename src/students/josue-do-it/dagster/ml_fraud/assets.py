import os

import dagster as dg

# import dagster_slack
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split, KFold

from .resources import fraud_data_source


@dg.asset(
    description="Download data for fraud dectection",
    compute_kind="python",
    group_name="ml_fraud_ingest"
)
def fraud_data(
    context: dg.AssetExecutionContext,
) -> dg.MaterializeResult:
    df = pd.read_csv(fraud_data_source.data_source)
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
    description="Cleans fraud data(remove duplicates, dropp missing values, reset the index)",
    compute_kind="python",
    group_name="ml_fraud_transform"
)
def clean_fraud(
    context: dg.AssetExecutionContext,
    fraud_data: pd.DataFrame
) -> dg.MaterializeResult:
    context.log.info("Starting data cleaning.")

    initial_rows = len(fraud_data)
    df = fraud_data.drop_duplicates()
    removed_duplicates = initial_rows - len(df)
    context.log.info(f"Removed {removed_duplicates} duplicate rows.")

    before_dropna = len(df)
    df = df.dropna()
    removed_na = before_dropna - len(df)
    context.log.info(f"Removed {removed_na} rows with missing values.")

    df = df.reset_index(drop=True)
    context.log.info("Reset DataFrame index.")
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
    group_name="ml_fraud_detection"
)
def split_fraud_data(
    context: dg.AssetExecutionContext,
    clean_fraud: pd.DataFrame
) -> dict:

    X = clean_fraud.drop('Class', axis=1)
    y = clean_fraud['Class']
    context.log.info(f"Features X shape: {X.shape}, Target y shape: {y.shape}")

    # Train-test split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=y
    )

    context.log.info(f"Training set: X_train {X_train.shape}, y_train {y_train.shape}")
    context.log.info(f"Test set: X_test {X_test.shape}, y_test {y_test.shape}")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }


@dg.asset(
    description="Tune RandomForest hyperparameters (GridSearchCV) ",
    compute_kind="python",
    group_name="ml_fraud_detection"
)
def tune_fraud_model(
    context: dg.AssetExecutionContext,
    split_fraud_data: dict
) -> dg.MaterializeResult:
    X_train = split_fraud_data["X_train"]
    y_train = split_fraud_data["y_train"]

    context.log.info("Starting RandomForest hyperparameter tuning...")

    # Define outer cross-validation (model evaluation)
    outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
    # outer_scores = []

    # RandomForest model
    rf = RandomForestClassifier(random_state=42)

    # Define hyperparameter grid
    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [None, 5, 100],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # Outer loop: performance estimation
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_train, y_train)):
        X_tr, X_te = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_tr, y_te = y_train.iloc[train_idx], y_train.iloc[test_idx]
        # X_tr, X_te = X_train[train_idx], X_train[test_idx]
        # y_tr, y_te = y_train[train_idx], y_train[test_idx]

        # Inner loop: hyperparameter tuning
        inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)
        rf = RandomForestClassifier(random_state=42)

        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=inner_cv,
            scoring="roc_auc",
            n_jobs=-1
        )

        grid_search.fit(X_tr, y_tr)

        # Best model on outer fold
        # # y_pred = best_rf.predict_proba(X_te)[:, 1]
        # auc_score = roc_auc_score(y_te, y_pred)

        # outer_scores.append(auc_score) Fold {fold_idx + 1} | AUC: {auc_score:.4f} | 
        context.log.info(
            f"Best params: {grid_search.best_params_}"
        )

    # final_auc = np.mean(outer_scores)
    # context.log.info(f"Final nested CV AUC: {final_auc:.4f}")


    # # 3-fold GridSearchCV
    # grid_search = GridSearchCV(
    #     estimator=rf,
    #     param_grid=param_grid,
    #     cv=3,
    #     n_jobs=-1,  
    #     scoring="accuracy",
    # )

    # # Fit model to training data
    # grid_search.fit(X_train, y_train)

    # # Retrieve best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    context.log.info(f"Best hyperparameters found: {best_params}")

    return dg.MaterializeResult(
        value=best_model,
        metadata={
            "best_params": str(best_params)
        }
    )


@dg.asset(
    description="Evaluate the tuned RandomForest model on the test dataset",
    compute_kind="python",
    group_name="ml_fraud_evaluate"
)
def evaluate_fraud_model(
    context: dg.AssetExecutionContext,
    tune_fraud_model,
    split_fraud_data: dict
) -> dg.MaterializeResult:

    best_model = tune_fraud_model
    X_test = split_fraud_data["X_test"]
    y_test = split_fraud_data["y_test"]

    context.log.info("Starting model evaluation on test data...")

    # Predictions
    y_pred = best_model.predict(X_test)

    # Accuracy
    accuracy = float(accuracy_score(y_test, y_pred))
    context.log.info(f"Model test accuracy: {accuracy:.4f}")

    # Classification report
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
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
    mlflow.set_experiment("ml_fraud_detection")
    with mlflow.start_run(run_name="evaluate_fraud_model"):
        mlflow.log_metric("test_accuracy", accuracy)
        mlflow.log_artifact(plot_path, artifact_path="plots")
        mlflow.log_param("model_type", "RandomForest")
        context.log.info("Metrics and confusion matrix logged to MLflow.")

    return dg.MaterializeResult(
        value=report_df,
        metadata={
            "test_accuracy": dg.MetadataValue.float(accuracy),
            "confusion_matrix": dg.MetadataValue.path(plot_path),
            "preview": dg.MetadataValue.md(report_df.head().to_markdown() or "")
        }
    )


@dg.asset(
    description="Send ML performance to Slack",
    compute_kind="notification",
    group_name="ml_fraud_notify",
    ins={"ml_fraud_evaluate": dg.AssetIn("evaluate_fraud_model")}
)
def fraud_notify_slack(context: dg.AssetExecutionContext, ml_fraud_evaluate) -> None:
    # accuracy = float(ml_fraud_evaluate["accuracy"])  # ensure plain float
    accuracy = float(
    ml_fraud_evaluate.get("test_accuracy", ml_fraud_evaluate.get("accuracy", 0.0))
    )

    metric = "Accuracy"

    msg = (
        f"{os.environ.get("GITHUB_USER", "default")}'s dagster pipeline sucessfully run!!!!! 🚀\n"
        f"*Fraud Detection Model Update*\n"
        f"{metric}: `{accuracy:.4f}` 🎯\n"
    )
    slack = context.resources.slack
    slack.get_client().chat_postMessage(
        channel='aims_course_october2025',
        text=msg
    )
    context.log.info("Slack notification sent successfully.")
