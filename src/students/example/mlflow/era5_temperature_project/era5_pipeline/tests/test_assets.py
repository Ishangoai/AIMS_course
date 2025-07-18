import typing
from unittest import mock

import dagster as dg
import numpy as np
import pandas as pd
import pytest
import sklearn.linear_model as sk

from ..assets import (
    clean_df,
    train_tuned_model,
    # tune_ridge_hyperparameters,
)

# from ..resources import TuningConfig


# Dummy pandas Dataset for testing clean_df
@pytest.fixture
def dummy_raw_pandas_df():
    # Create a simple pandas DataFrame directly
    df = pd.DataFrame({
        "t2m": [280.0, 281.5, 279.8, 282.1],
        "latitude": [0.0, 1.0, -1.0, 0.5],
        "longitude": [0.0, 2.0, -2.0, 1.5],
        "time": [
            pd.Timestamp("2023-01-01 00:00"),
            pd.Timestamp("2023-01-01 00:00"),
            pd.Timestamp("2023-01-01 00:00"),
            pd.Timestamp("2023-01-01 00:00"),
        ]
    })
    return df


def test_create_pandas_df(dummy_raw_pandas_df):

    basic_context = dg.build_asset_context(resources={"mlflow_tracking": mock.Mock()})
    df_actual: typing.Any = clean_df(basic_context, dummy_raw_pandas_df)

    df_expected = pd.DataFrame({
        "time": [pd.Timestamp("2023-01-01 00:00")],
        "t2m": [280.85],
        "t2m_celsius": [7.7],
    })

    pd.testing.assert_frame_equal(df_actual.value, df_expected)


def test_train_tuned_model():
    mock_mlflow = mock.MagicMock()
    context = dg.build_asset_context(resources={"mlflow_tracking": mock_mlflow})

    input_data = {
        "best_params": {"alpha": 1.0},
        "X_train_val": np.array([[1], [2], [3], [4]]),
        "y_train_val": np.array([2, 4, 6, 8]),
        "X_test": np.array([[5], [6]]),
        "y_test": np.array([10, 12]),
        "feature_names": ["x"]
    }

    expected_result_type = dict
    expected_model_type = sk.Ridge
    expected_test_size = input_data["X_test"].shape
    expected_feature_name = input_data["feature_names"]

    actual_result: dict[str, typing.Any] = train_tuned_model(context, input_data)  # type: ignore
    actual_model = actual_result["model"]

    assert isinstance(actual_result, expected_result_type)
    assert isinstance(actual_model, expected_model_type)
    assert actual_result["X_test"].shape == expected_test_size
    assert actual_result["feature_names"] == expected_feature_name
