import os
import typing
from unittest.mock import MagicMock, patch

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


@patch.dict(os.environ, {"SLACK_AIMS_COURSE_BOT_TOKEN": "dummy-token"})
@patch("dagster_slack.SlackResource.get_client")
def test_agg_data(mock_get_client, dummy_clean_data):
    mock_client = MagicMock()
    mock_get_client.return_value = mock_client
    mock_client.chat_postMessage.return_value = None

    basic_context = dg.build_asset_context()
    df_actual: typing.Any = agg_data(basic_context, dummy_clean_data)

    df_expected = pd.DataFrame({
        "FoodItem": ['Apple', 'Eggplant'],
        "nItems": [205, 400]
    })

    pd.testing.assert_frame_equal(df_actual.value, df_expected)

    mock_client.chat_postMessage.assert_called_once()
