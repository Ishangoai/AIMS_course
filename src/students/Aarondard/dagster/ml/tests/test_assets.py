import typing
from unittest import mock

import dagster as dg
import pandas as pd
import pytest

from ..assets import clean_df


@pytest.fixture
def dummy_raw_pandas_df():
    # Create a simple pandas DataFrame directly
    df = pd.DataFrame(
        {
            "t2m": [280.0, 281.5, 279.8, 282.1],
            "latitude": [0.0, 1.0, -1.0, 0.5],
            "longitude": [0.0, 2.0, -2.0, 1.5],
            "time": [
                pd.Timestamp("2023-01-01 00:00"),
                pd.Timestamp("2023-01-01 00:00"),
                pd.Timestamp("2023-01-01 00:00"),
                pd.Timestamp("2023-01-01 00:00"),
            ],
        }
    )
    return df


@mock.patch("dagster_mlflow.mlflow_tracking", dg.ResourceDefinition.mock_resource())
def test_create_pandas_df(dummy_raw_pandas_df):
    clean_df.resource_defs["mlflow_tracking"]._resource_fn = dg.ResourceDefinition.mock_resource().resource_fn

    ctx = dg.build_asset_context()
    df_actual: typing.Any = clean_df(ctx, dummy_raw_pandas_df)

    df_expected = pd.DataFrame(
        {
            "time": [pd.Timestamp("2023-01-01 00:00")],
            "t2m": [280.85],
            "t2m_celsius": [7.7],
        }
    )

    pd.testing.assert_frame_equal(df_actual.value, df_expected)
