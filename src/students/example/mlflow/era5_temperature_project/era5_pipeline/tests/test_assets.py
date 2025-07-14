import pytest
import pandas as pd
import numpy as np
import xarray as xr
import dagster as dg
import typing
from unittest import mock
from ..assets import (
    raw_pandas_df,
    tune_ridge_hyperparameters,
)
from ..resources import TuningConfig


# Dummy xarray Dataset for testing create_pandas_df
@pytest.fixture
def dummy_xr_dataset():
    return xr.Dataset({
        "t2m": (("valid_time", "latitude", "longitude"), [[[280.0]]]),
        "latitude": (("latitude",), [0.0]),
        "longitude": (("longitude",), [0.0]),
        "valid_time": (("valid_time",), pd.date_range("2023-01-01", periods=1))
    })


def test_create_pandas_df(dummy_xr_dataset):

    basic_context = dg.build_asset_context(resources={"mlflow_tracking": mock.Mock()})
    result: typing.Any = raw_pandas_df(basic_context, dummy_xr_dataset)

    # Verify the output is a DataFrame
    assert isinstance(result.value, pd.DataFrame), "Output is not a DataFrame"
    # Check important columns exist in the DataFrame
    assert "t2m" in result.value.columns, "t2m column missing in DataFrame"
    assert "time" in result.value.columns, "time column missing in DataFrame"


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
