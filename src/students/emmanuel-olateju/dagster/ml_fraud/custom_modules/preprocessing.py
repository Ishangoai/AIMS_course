from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the input DataFrame by handling missing values and duplicates."""
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def split_features_labels(df: pd.DataFrame, label_column: str) -> Tuple[np.ndarray, np.ndarray]:
    """Splits the DataFrame into features and labels."""
    X = df.drop(columns=[label_column]).to_numpy()
    y = df[label_column].to_numpy()
    return (X, y)


def data_splitting(X: np.ndarray, y: np.ndarray, test_size: float, random_state: int) -> Dict:
    """Splits the data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y
        )
    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test
    }
