
import dagster as dg
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split

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


@dg.multi_asset(
    outs={
        "train_data": dg.AssetOut(description="Training dataset (80% split)"),
        "test_data": dg.AssetOut(description="Test dataset (20% split)")
    },
    description="Split fraud data into train (80%) and test (20%) sets with stratification",
    compute_kind="python",
    group_name="data_fraud_split"
)
def train_test_split_data(
    context: dg.AssetExecutionContext,
    fraud_data: pd.DataFrame
):
    """
    Split the fraud data into training and test sets in ONE operation.
    Uses stratified split to maintain fraud ratio in both sets.
    """
    if len(fraud_data) < 100:
        msg = "Not enough data points for splitting. Need at least 100 samples."
        context.log.error(msg)
        raise ValueError(msg)

    X = fraud_data.drop(columns=["Class"])
    y = fraud_data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    train_df = X_train.copy()
    train_df["Class"] = y_train

    test_df = X_test.copy()
    test_df["Class"] = y_test

    context.log.info(f"Training set size: {len(train_df)} ({len(train_df) / len(fraud_data) * 100:.1f}%)")
    context.log.info(f"Test set size: {len(test_df)} ({len(test_df) / len(fraud_data) * 100:.1f}%)")
    context.log.info(f"Train fraud ratio: {y_train.mean():.4f}")
    context.log.info(f"Test fraud ratio: {y_test.mean():.4f}")

    # ✅ Dagster expects yields, not return dict
    yield dg.Output(train_df, output_name="train_data")
    yield dg.Output(test_df, output_name="test_data")


import dagster as dg
import pandas as pd


@dg.asset(
    description="Train a RandomForest with 3-fold cross-validation and log each trial in MLflow.",
    compute_kind="python",
    group_name="ml_fraud_training"
)
def train_random_forest_model(
    context: dg.AssetExecutionContext,
    train_data: pd.DataFrame,
):
    """
    Perform 3-fold cross-validation to tune one hyperparameter (n_estimators)
    of a RandomForestClassifier. Each trial is logged as an MLflow nested run.
    """

    # 🧩 Setup MLflow experiment (direct configuration)
    mlflow.set_tracking_uri("file:./mlruns")   # dossier local
    mlflow.set_experiment("fraud_detection")   # nom d'expérience

    X = train_data.drop(columns=["Class"])
    y = train_data["Class"]

    # Paramètre à tuner
    param_grid = {"n_estimators": [50, 100, 200]}

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    context.log.info("Starting 3-fold cross-validation with RandomForestClassifier")

    with mlflow.start_run(run_name="RandomForest_CV_Tuning") as parent_run:
        mlflow.log_param("cv_folds", 3)
        mlflow.log_param("tuned_hyperparameter", "n_estimators")

        results = []
        for n_est in param_grid["n_estimators"]:
            with mlflow.start_run(run_name=f"trial_n_estimators_{n_est}", nested=True):
                scores = []
                for train_idx, val_idx in cv.split(X, y):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                    clf = RandomForestClassifier(n_estimators=n_est, random_state=42)
                    clf.fit(X_train, y_train)
                    preds = clf.predict(X_val)
                    acc = accuracy_score(y_val, preds)
                    scores.append(acc)

                mean_acc = sum(scores) / len(scores)
                mlflow.log_param("n_estimators", n_est)
                mlflow.log_metric("mean_accuracy", mean_acc)
                results.append((n_est, mean_acc))
                context.log.info(f"n_estimators={n_est} → mean CV accuracy={mean_acc:.4f}")

        # Sélection du meilleur modèle
        best_n, best_score = max(results, key=lambda x: x[1])
        mlflow.log_param("best_n_estimators", best_n)
        mlflow.log_metric("best_mean_accuracy", best_score)
        context.log.info(f"Best hyperparameter: n_estimators={best_n} (accuracy={best_score:.4f})")

        # Entraîner modèle final
        best_model = RandomForestClassifier(n_estimators=best_n, random_state=42)
        best_model.fit(X, y)

        # Loguer le modèle final
        mlflow.sklearn.log_model(best_model, "model")

    return best_model


@dg.asset(
    description="Evaluate the trained RandomForest model on test data and log confusion matrix to MLflow.",
    compute_kind="python",
    group_name="ml_fraud_evaluation"
)
def evaluate_model_on_test_data(
    context: dg.AssetExecutionContext,
    train_random_forest_model,
    test_data: pd.DataFrame,
):
    """
    Evaluate the trained RandomForest model on the 20% test set,
    generate a confusion matrix plot, and log it as an image artifact in MLflow.
    """

    import mlflow
    from sklearn.metrics import classification_report

    # 🧩 Setup MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("fraud_detection")

    X_test = test_data.drop(columns=["Class"])
    y_test = test_data["Class"]

    # Prédictions
    y_pred = train_random_forest_model.predict(X_test)

    # Calcul des métriques
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    context.log.info(f"Test accuracy: {report['accuracy']:.4f}")
    context.log.info(f"Confusion matrix:\n{cm}")

    # Log dans MLflow
    with mlflow.start_run(run_name="Model_Evaluation", nested=True):
        mlflow.log_metric("test_accuracy", report["accuracy"])
        mlflow.log_metric("precision_0", report["0"]["precision"])
        mlflow.log_metric("recall_0", report["0"]["recall"])
        mlflow.log_metric("precision_1", report["1"]["precision"])
        mlflow.log_metric("recall_1", report["1"]["recall"])

        # 🔹 Générer et enregistrer le graphique
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(ax=ax, cmap="Blues", colorbar=False)
        plt.title("Confusion Matrix - Test Set")

        # Sauvegarde temporaire
        image_path = "confusion_matrix.png"
        plt.savefig(image_path, bbox_inches="tight")
        plt.close(fig)

        # Log de l’image dans MLflow
        mlflow.log_artifact(image_path, artifact_path="plots")

    return {"test_accuracy": report["accuracy"], "confusion_matrix": cm.tolist()}
