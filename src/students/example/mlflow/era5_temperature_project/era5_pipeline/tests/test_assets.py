import pytest
import pandas as pd
import numpy as np
import xarray as xr
from unittest import mock
from dagster import build_op_context
from typing import Any
from era5_pipeline.assets import (
    create_pandas_df,
    tune_ridge_hyperparameters,
    evaluate_model
)
from era5_pipeline.resources import TuningConfig


# Mock mlflow_tracking resource for reuse in all tests
@pytest.fixture
def mlflow_mock():
    return mock.Mock()


# Provide a minimal Dagster context with mlflow mocked resource
@pytest.fixture
def basic_context(mlflow_mock):
    return build_op_context(resources={"mlflow_tracking": mlflow_mock})


# Dummy xarray Dataset for testing create_pandas_df
@pytest.fixture
def dummy_xr_dataset():
    return xr.Dataset({
        "t2m": (("valid_time", "latitude", "longitude"), [[[280.0]]]),
        "latitude": (("latitude",), [0.0]),
        "longitude": (("longitude",), [0.0]),
        "valid_time": (("valid_time",), pd.date_range("2023-01-01", periods=1))
    })


def test_create_pandas_df(basic_context, dummy_xr_dataset):

    result: Any = create_pandas_df(basic_context, dummy_xr_dataset)

    # Verify the output is a DataFrame
    assert isinstance(result.value, pd.DataFrame), "Output is not a DataFrame"
    # Check important columns exist in the DataFrame
    assert "t2m" in result.value.columns, "t2m column missing in DataFrame"
    assert "time" in result.value.columns, "time column missing in DataFrame"


# Simple clean DataFrame with enough rows for tuning tests
@pytest.fixture
def dummy_clean_df():
    """
    Generate a DataFrame with hourly temperature readings over 25 hours,
    enough for feature lags and train/test splits
    """
    return pd.DataFrame({
        "time": pd.date_range("2023-01-01", periods=25, freq="h"),
        "t2m_celsius": np.linspace(0, 24, 25)
    })


# tuning config limiting hyperopt search evaluations
@pytest.fixture
def dummy_tuning_config():
    return TuningConfig(max_hyperopt_evals=1)


# Patch hyperopt.fmin and hyperopt.Trials for controlling tuning behavior
@pytest.fixture
def patch_hyperopt():
    """
    Patch hyperopt fmin and Trials methods to fake hyperparameter tuning output,
    avoiding the need to run actual optimization during tests
    """
    with mock.patch("era5_pipeline.assets.hyperopt.fmin") as mock_fmin, \
         mock.patch("era5_pipeline.assets.hyperopt.Trials") as mock_trials:

        mock_fmin.return_value = {"alpha": 1.23}
        mock_trials_instance = mock_trials.return_value
        mock_trials_instance.best_trial = {
            "result": {"loss": 0.42},
            "misc": {"vals": {"alpha": [1.23]}}
        }
        yield mock_fmin, mock_trials


def test_tune_ridge_hyperparameters(
    basic_context, dummy_clean_df, dummy_tuning_config, patch_hyperopt
):
    result: Any = tune_ridge_hyperparameters(basic_context, dummy_tuning_config, dummy_clean_df)

    # Check that the output dictionary contains the expected keys
    expected_keys = {
        "best_params", "X_train_val", "y_train_val", "X_test", "y_test", "feature_names"
    }
    assert expected_keys.issubset(result.keys()), "Missing keys in tuning result"

    # Confirm that the best alpha hyperparameter is as mocked
    assert result["best_params"]["alpha"] == 1.23, "Best alpha value mismatch"


# Provides dummy trained model dict with mocked predict method
@pytest.fixture
def dummy_train_tuned_model():

    mock_model = mock.Mock()
    mock_model.predict.return_value = np.array([2.5, 3.5, 4.5])

    return {
        "model": mock_model,
        "X_test": np.array([[1], [2], [3]]),
        "y_test": np.array([2.0, 3.0, 5.0]),
        "feature_names": ["lag_feature"]
    }


# Patch MLflow related methods to avoid real external calls during evaluation
@pytest.fixture
def patch_mlflow_methods():

    with mock.patch("era5_pipeline.assets.mlflow.start_run") as mock_start_run, \
         mock.patch("era5_pipeline.assets.ms.log_model") as mock_log_model:

        # Mock context manager for mlflow.start_run
        run_mock = mock.MagicMock()
        run_mock.__enter__.return_value.info.run_id = "test_run_123"
        mock_start_run.return_value = run_mock

        # Mock the log_model return object with model_uri
        mock_log_model.return_value.model_uri = "runs:/test_run_123/model"

        yield mock_start_run, mock_log_model


def test_evaluate_model(
    basic_context, dummy_train_tuned_model, patch_mlflow_methods, mlflow_mock
):
    """
    Test that evaluate_model runs properly, interacts with MLflow mocks,
    and returns results containing expected evaluation metrics and metadata.
    """
    # Setup a fake model version for search_model_versions call
    mock_model_version = mock.Mock()
    mock_model_version.run_id = "test_run_123"
    mock_model_version.name = "tuned-temp-forecaster"
    mock_model_version.version = "1"
    mock_model_version.status = "READY"
    mock_model_version.current_stage = "None"

    mlflow_mock.search_model_versions.return_value = [mock_model_version]

    result: Any = evaluate_model(basic_context, dummy_train_tuned_model)

    # Assert the evaluation status succeeded
    assert result.value["status"] == "evaluated_successfully", "Evaluation did not succeed"

    # Assert presence of key evaluation metrics
    assert "test_mse" in result.value["eval_metrics"], "test_mse metric missing"
    assert "test_mae" in result.value["eval_metrics"], "test_mae metric missing"

    # Assert metadata values are of expected types and format
    assert isinstance(result.metadata["test_mse"].value, float), "test_mse metadata not float"
    assert isinstance(result.metadata["model_version"].value, str), "model_version metadata not string"
    assert result.metadata["mlflow_model_uri"].value.startswith("models:/"), "mlflow_model_uri format incorrect"
