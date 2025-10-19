import os
import tempfile

import dagster as dg
import dagster_slack
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn as ms
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    train_test_split,  # For robust splitting
)

from .resources import fraud_data_source


@dg.asset(
    description="Download data for fraud detection from a fixed URL resource.",
    compute_kind="python",
    group_name="ml_fraud"
)
def fraud_data(
    context: dg.AssetExecutionContext,
)  -> dg.MaterializeResult:
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
    description="Cleans the raw data by converting date formats and correcting data types.",
    compute_kind="python",
    group_name="ml_fraud_clean"
)
def clean_data_fraud(
    context: dg.AssetExecutionContext,
    fraud_data: pd.DataFrame
) -> dg.MaterializeResult:

    clean_data = fraud_data.copy()
    clean_data['Time'] = pd.to_datetime(clean_data['Time'], errors='coerce', format="mixed")
    # clean_data['nItems'] = pd.to_numeric(clean_data['nItems'], errors='coerce')

    # log info that can be view in real time in the dagster UI
    context.log.info(f"Cleaned data with {len(clean_data)} rows after cleaning.")

    columns = [dg.TableColumn(k, str(v)) for k, v in clean_data.dtypes.to_dict().items()]

    return dg.MaterializeResult(
        value=clean_data,
        metadata={
            "preview": dg.MetadataValue.md(clean_data.head().to_markdown() or ""),
            "dagster/row_count": len(clean_data),
            "dagster/column_schema": dg.TableSchema(columns=columns)
        }
    )


@dg.asset(
    description="Split data for fraud detection from a fixed URL resource.",
    compute_kind="python",
    group_name="ml_fraud_split"
)
def split_fraud_data(context: dg.AssetExecutionContext, clean_data_fraud: pd.DataFrame) -> dict:
    # Copie pour éviter les effets de bord
    clean_data = clean_data_fraud.copy()
    clean_data = clean_data.dropna()

    # Split features / target
    y = clean_data["Class"]
    X = clean_data.drop("Class", axis=1)

    # Train/Test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    # Retourner les splits en tant que dictionnaire
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }


@dg.asset(
    description="Train Random Forest for fraud detection using pre-split data and log results to MLflow.",
    compute_kind="python",
    group_name="ml_fraud_train"
)
def train_fraud_rf(
    context: dg.AssetExecutionContext,
    split_fraud_data
) -> dg.MaterializeResult:
    """Train Random Forest on training data and log results to MLflow."""

    # --- Extraction des données ---
    X_train = split_fraud_data["X_train"]
    y_train = split_fraud_data["y_train"]

    # --- Logging des infos ---
    context.log.info(f"X_train shape: {X_train.shape}")
    context.log.info(f"y_train shape: {y_train.shape}")
    context.log.info(f"Colonnes X_train: {list(X_train.columns)}")

    # --- Nettoyage / préparation ---
    # Garder uniquement les colonnes numériques
    X_train = X_train.select_dtypes(include=["number"]).copy()

    # Encoder y_train si nécessaire
    if y_train.dtype == "object":
        y_train = y_train.astype("category").cat.codes

    # Vérifier que le dataset est valide
    if X_train.shape[0] < 5 or len(y_train.unique()) < 2:
        raise ValueError("Le jeu d'entraînement est trop petit ou ne contient pas assez de classes distinctes.")

    # --- Fonction de cross-validation ---
    def cross_validation(X_train, y_train):
        rf = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [50, 100, 200]}
        cv = KFold(n_splits=3, shuffle=True, random_state=42)
        grid_search = GridSearchCV(
            rf,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            # error_score='raise'  # pour afficher les erreurs internes
        )
        grid_search.fit(X_train, y_train)
        return grid_search

    # --- MLflow Logging ---
    with mlflow.start_run(run_name="RF_GridSearch", nested=True):
        grid_search = cross_validation(X_train, y_train)

        # Log des essais
        for i, params in enumerate(grid_search.cv_results_['params']):
            mean_score = grid_search.cv_results_['mean_test_score'][i]
            with mlflow.start_run(run_name=f"trial_{i}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("mean_cv_accuracy", mean_score)

        # Meilleur modèle
        best_model = grid_search.best_estimator_

        # Sauvegarde dans MLflow
        ms.log_model(best_model, "best_rf_model")
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("best_cv_score", grid_search.best_score_)

        # Logs Dagster
        context.log.info(f"✅ Best params: {grid_search.best_params_}")
        context.log.info(f"✅ Best CV score: {grid_search.best_score_:.4f}")

    # --- Retour Dagster ---
    return dg.MaterializeResult(
    value=best_model,
    metadata={
        "best_params": dg.MetadataValue.json({
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in grid_search.best_params_.items()
        }),
        "best_cv_score": dg.MetadataValue.float(float(grid_search.best_score_)),
    }
)


@dg.asset(
    description="Generate confusion matrix plot on test data and log it as an MLflow artifact.",
    compute_kind="python",
    group_name="ml_fraud_eval"
)
def fraud_rf_confusion_matrix(
    context: dg.AssetExecutionContext,
    split_fraud_data,
    train_fraud_rf
) -> dg.MaterializeResult:
    """Generate a confusion matrix plot for the test set and log it to MLflow."""

    # --- Extract test data ---
    X_test = split_fraud_data["X_test"]
    y_test = split_fraud_data["y_test"]
    model = train_fraud_rf

    # --- Cleaning: numeric-only, fill NAs ---
    X_test = X_test.select_dtypes(include=["number"]).fillna(0)
    y_test = y_test.fillna(0)

    # --- Predict ---
    y_pred = model.predict(X_test)

    # --- Metrics ---
    acc = float(accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)

    context.log.info(f"✅ Test accuracy: {acc:.4f}")
    context.log.info(f"Confusion matrix:\n{cm}")

    # --- Plot Confusion Matrix ---
    fig, ax = plt.subplots(figsize=(6, 6))
    cax = ax.matshow(cm, cmap="Blues")
    for (i, j), val in np.ndenumerate(cm):
        ax.text(j, i, str(val), ha='center', va='center', color='black')
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"Confusion Matrix (Accuracy={acc:.2%})")
    fig.colorbar(cax)

    # --- Save plot to temp file ---
    tmpdir = tempfile.mkdtemp()
    plot_path = os.path.join(tmpdir, "confusion_matrix.png")
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    # --- Log to MLflow ---
    with mlflow.start_run(run_name="fraud_rf_confusion_matrix", nested=True):
        mlflow.log_artifact(plot_path, artifact_path="plots")
        mlflow.log_metric("test_accuracy", acc)

    # --- Return result with VALUE ---
    result_dict = {
        "accuracy": acc,
        "confusion_matrix": cm.tolist(),
        "plot_path": plot_path
    }

    return dg.MaterializeResult(
        value=result_dict,  # ✅ Now we have a value to pass downstream
        metadata={
            "test_accuracy": dg.MetadataValue.float(acc),
            "confusion_matrix_image": dg.MetadataValue.path(plot_path),
        })


@dg.asset(
    description="Message into the slack channel with the ML prediction performance on the test set.",
    resource_defs={"myslack": dagster_slack.SlackResource(token=dg.EnvVar("SLACK_AIMS_COURSE_BOT_TOKEN"))},
    compute_kind="python",
    group_name="ml_fraud_eval"
)
def send_message(
    context: dg.AssetExecutionContext,
    fraud_rf_confusion_matrix: dict  # ✅ Now receives the dict
) -> dg.MaterializeResult:
    """
    Envoie un message Slack avec la performance du modèle ML (accuracy) sur le test set.
    """

    # --- Extraire métriques from the VALUE, not metadata ---
    accuracy = float(fraud_rf_confusion_matrix["accuracy"])  # ✅ Access from dict value
    user = os.environ.get("GITHUB_USER", "default")
    emoji_success = ":white_check_mark:"
    emoji_fun = ":rocket:"

    # --- Envoyer le message sur Slack ---
    slack: dagster_slack.SlackResource = context.resources.myslack
    slack.get_client().chat_postMessage(
        channel='aims_course_october2025',
        text=(
            f"{emoji_success} Model evaluated: Fraud Detection {emoji_success}\n"
            f"👤 Pipeline run by: {user}\n"
            f"📊 Performance Summary:\n"
            f"    - Accuracy: {accuracy:.2%} -> {'Passed' if accuracy > 0.5 else 'Failed'}\n"
            f"{emoji_fun} Test succesful !"
        )
    )

    context.log.info(f"Slack message sent with accuracy: {accuracy:.2%}")

    # --- Retourner MaterializeResult avec métadonnées ---
    return dg.MaterializeResult(
        value=fraud_rf_confusion_matrix,
        metadata={
            "preview": dg.MetadataValue.md(f"Accuracy: {accuracy:.2%}"),
            "test_accuracy": dg.MetadataValue.float(accuracy)
        })
