import typing
from unittest import mock

import dagster as dg
import pandas as pd
import pytest

from ..assets import agg_data


# Dummy pandas Dataset for testing clean_df
@pytest.fixture
def dummy_clean_data():
    # Create a simple pandas DataFrame directly
    df = pd.DataFrame({
        "Date": [
            pd.Timestamp("2023-01-15 00:00"),
            pd.Timestamp("2023-02-03 00:00"),
            None,  # Simulating a missing date
            pd.Timestamp("2023-02-04 00:00"),
        ],
        "FoodItem": ['Apple', 'Eggplant', 'Eggplant', 'Apple'],
        "nItems": [200, 400, 80, 5]
    })
    return df


def test_agg_data(dummy_clean_data):

    basic_context = dg.build_asset_context()
    df_actual: typing.Any = agg_data(basic_context, dummy_clean_data)

    df_expected = pd.DataFrame({
        "FoodItem": ['Apple', 'Eggplant'],
        "nItems": [205, 400]
    })

    pd.testing.assert_frame_equal(df_actual.value, df_expected)
