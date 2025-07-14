import pytest
import pandas as pd
import numpy as np
import dagster as dg
import typing
from unittest import mock
from ..assets import (
    clean_df,
    tune_ridge_hyperparameters,
)
from ..resources import TuningConfig


# Dummy pandas Dataset for testing clean_df
@pytest.fixture
def dummy_df_input():
    # Create a simple pandas DataFrame directly
    df_input = pd.DataFrame({
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
    return df_input


def test_create_pandas_df(dummy_df_input):

    basic_context = dg.build_asset_context(resources={"mlflow_tracking": mock.Mock()})
    df_actual: typing.Any = clean_df(basic_context, dummy_df_input)

    df_expected = pd.DataFrame({
        "time": [pd.Timestamp("2023-01-01 00:00")],
        "t2m": [280.85],
        "t2m_celsius": [7.7],
    })

    pd.testing.assert_frame_equal(df_actual.value, df_expected)


# Simple clean DataFrame with enough rows for tuning tests
@pytest.fixture
def dummy_clean_df():
    return pd.DataFrame({
        "time": pd.date_range("2023-01-01", periods=25, freq="h"),
        "t2m_celsius": np.linspace(0, 24, 25)
    })


# Patch hyperopt.fmin and hyperopt.Trials for controlling tuning behavior
@pytest.fixture
def patch_hyperopt():
    with mock.patch("era5_pipeline.assets.hyperopt.fmin") as mock_fmin, \
         mock.patch("era5_pipeline.assets.hyperopt.Trials") as mock_trials:

        mock_fmin.return_value = {"alpha": 1.23}
        mock_trials_instance = mock_trials.return_value
        mock_trials_instance.best_trial = {
            "result": {"loss": 0.42},
            "misc": {"vals": {"alpha": [1.23]}}
        }
        yield mock_fmin, mock_trials


def test_tune_ridge_hyperparameters(dummy_clean_df, patch_hyperopt):
    dummy_tuning_config = TuningConfig(max_hyperopt_evals=1)
    basic_context = dg.build_asset_context(resources={"mlflow_tracking": mock.Mock()})
    result: typing.Any = tune_ridge_hyperparameters(basic_context, dummy_tuning_config, dummy_clean_df)

    # Check that the output dictionary contains the expected keys
    expected_keys = {
        "best_params", "X_train_val", "y_train_val", "X_test", "y_test", "feature_names"
    }
    assert expected_keys.issubset(result.keys()), "Missing keys in tuning result"

    # Confirm that the best alpha hyperparameter is as mocked
    assert result["best_params"]["alpha"] == 1.23, "Best alpha value mismatch"
