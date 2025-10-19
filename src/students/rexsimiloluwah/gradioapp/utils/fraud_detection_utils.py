import os
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

MODEL_FILENAME = "fraud_detection_model.pkl"
MODEL_PATH = os.path.join(os.path.dirname(__file__), MODEL_FILENAME)

EXPECTED_FEATURES = [
    "V14", "V17", "V10", "V12", "V11", "V16",
    "V4", "V9", "V18", "V7", "V3", "Amount"
]

# lazy-loaded model handle
_model = None


def _ensure_model():
    """Lazy-load and return the model. Raises FileNotFoundError if missing."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
        _model = joblib.load(MODEL_PATH)
    return _model


def _validate_features(features: List[Any]) -> np.ndarray:
    """
    Validate and convert feature list to numpy array.
    Raises ValueError if length mismatches or conversion fails.
    """
    if not isinstance(features, (list, tuple, np.ndarray)):
        raise ValueError("features must be a list-like of numeric values.")
    if len(features) != len(EXPECTED_FEATURES):
        raise ValueError(
            f"Expected {len(EXPECTED_FEATURES)} features in order {EXPECTED_FEATURES}, "
            f"got {len(features)}."
        )
    try:
        arr = np.asarray(features, dtype=float).reshape(1, -1)
    except Exception as e:
        raise ValueError(f"Failed to convert features to numeric array: {e}")
    return arr


def _predict_and_probs(model, features_input) -> Tuple[int, Optional[float], Optional[float]]:
    """
    Run model.predict and (if available) model.predict_proba to compute:
      - prediction (int; 0 or 1 typically)
      - prob_pos: P(class==1) or None if unavailable
      - predicted_probability: probability of the predicted class or None if unavailable

    Accepts either a pandas.DataFrame with proper column names or a numpy array.
    """
    # prediction (may raise)
    prediction = int(model.predict(features_input)[0])

    prob_pos: Optional[float] = None
    predicted_probability: Optional[float] = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features_input)[0]  # e.g., [p_class0, p_class1]
        classes = list(model.classes_) if hasattr(model, "classes_") else None

        # probability of positive class (label 1) if present in classes
        if classes is not None and 1 in classes:
            idx_pos = classes.index(1)
            prob_pos = float(probs[idx_pos])

        # probability for the predicted class
        if classes is not None:
            try:
                pred_idx = classes.index(prediction)
                predicted_probability = float(probs[pred_idx])
            except ValueError:
                predicted_probability = None
        else:
            # fallback assumption for binary
            if len(probs) == 2:
                predicted_probability = float(probs[prediction])
                if prediction == 1:
                    prob_pos = float(probs[1])
            else:
                predicted_probability = None

    return prediction, prob_pos, predicted_probability


def _build_message(prediction: int,
                   prob_pos: Optional[float],
                   predicted_probability: Optional[float]) -> str:
    """
    Build a human-friendly message string from prediction and probability info.
    """
    if prediction == 1:
        if prob_pos is not None:
            return f"High risk of credit card fraud ({prob_pos * 100:.2f}% probability)"
        elif predicted_probability is not None:
            return f"High risk of credit card fraud ({predicted_probability * 100:.2f}% probability)"
        else:
            return "High risk of credit card fraud (probability unavailable)"
    else:
        if prob_pos is not None:
            prob_nonfraud = (1.0 - prob_pos) * 100.0
            return f"Low risk of credit card fraud ({prob_nonfraud:.2f}% probability)"
        elif predicted_probability is not None:
            return f"Predicted non-fraud (probability: {predicted_probability * 100.0:.2f}%)"
        else:
            return "Predicted non-fraud (probability unavailable)"


def predict_credit_card_fraud(features: List[float]) -> Dict[str, Any]:
    """
    Predict credit card fraud using the lazily-loaded model and helper utilities.

    Returns a dict with:
      - status: bool
      - prediction: 0 or 1 (or None on error)
      - probability_pos: probability of class=1 if available else None
      - predicted_probability: probability for the returned prediction if available else None
      - message: friendly message
      - error: error string or None
    """
    try:
        # Validate and convert to numpy array for safety/validation
        _ = _validate_features(features)
        model = _ensure_model()

        # Build a DataFrame with the same column names the model was trained on
        features_df = pd.DataFrame([features], columns=EXPECTED_FEATURES)  # type: ignore[call-overload]

        # Pass the DataFrame to predictions to preserve feature names and avoid sklearn warning
        prediction, prob_pos, predicted_probability = _predict_and_probs(model, features_df)

        msg = _build_message(prediction, prob_pos, predicted_probability)

        return {
            "status": True,
            "prediction": prediction,
            "probability_pos": prob_pos,
            "predicted_probability": predicted_probability,
            "message": msg,
            "error": None
        }

    except Exception as exc:
        return {
            "status": False,
            "prediction": None,
            "probability_pos": None,
            "predicted_probability": None,
            "message": None,
            "error": f"Error during prediction: {str(exc)}"
        }


if __name__ == "__main__":
    sample_normal = [0.35237478, -0.010299228, -0.156840906, 0.538299087,
                     0.847755891, 0.906489408, -0.146817811, -0.49952308,
                     0.576837916, -0.166862547, 0.220932254, 8.99]

    sample_fraud = [-10.5, -15.0, -5.0, -7.5,
                    5.0, -8.0, 5.0, -3.0,
                    -4.0, -1.0, -2.0, 500.0]

    print("Normal sample:", predict_credit_card_fraud(sample_normal))
    print("Fraud sample:  ", predict_credit_card_fraud(sample_fraud))
