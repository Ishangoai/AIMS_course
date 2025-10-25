from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str) -> None:
    "Save the confusion matrix to a specified output path"
    cm = confusion_matrix(y_true, y_pred)
    np.savetxt(output_path, cm, delimiter=',')


def save_confusion_matrix_img(y_true: np.ndarray, y_pred: np.ndarray, output_path: str) -> None:
    "Save the confusion matrix as an image to a specified output path"

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)  # type: ignore
    plt.savefig(output_path)
    plt.close()


def model_training_testing(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    param_dist: dict, random_state: int
    ) -> Dict:
    "Train a RandomForestClassifier model and return the ROC AUC score on the test set"
    # Define the model
    rf = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'  # adjusts weights inversely proportional to class frequencies
    )

    # Setup RandomizedSearchCV
    n_iter_search = 20
    random_search = RandomizedSearchCV(rf, param_distributions=param_dist,
                                       n_iter=n_iter_search, cv=5, scoring='recall',
                                       n_jobs=-1, random_state=random_state)

    # Fit the model
    random_search.fit(X_train, y_train)

    # Best model
    best_rf = random_search.best_estimator_

    # Predictions
    y_pred = best_rf.predict(X_test)  # type: ignore
    y_proba = best_rf.predict_proba(X_test)[:, 1]  # type: ignore

    # Evaluation
    roc_auc = float(roc_auc_score(y_test, y_proba))
    accuracy = float(accuracy_score(y_test, y_pred))
    recall = float(recall_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("ROC AUC Score:", roc_auc)

    return {
        "accuracy": accuracy,
        "recall": recall,
        "roc_auc": roc_auc,
        "cm": cm,
        "model": best_rf
    }
