import os
from unittest.mock import patch

from ..agent import get_llm


@patch('src.students.joshsalako.essay_writer.agent.ChatGoogleGenerativeAI')
@patch('src.students.joshsalako.essay_writer.agent.getpass')
def test_get_llm_api_key_exists(mock_getpass_module, mock_chat_model, monkeypatch):
    """
    Tests that get_llm uses the existing environment variable
    and does not prompt for input if the API key is already set.
    """
    monkeypatch.setenv("GOOGLE_API_KEY", "test-api-key-from-env")

    get_llm()

    mock_getpass_module.getpass.assert_not_called()
    mock_chat_model.assert_called_once_with(
        model="gemini-2.5-flash-lite", temperature=0.5
    )


@patch('src.students.joshsalako.essay_writer.agent.ChatGoogleGenerativeAI')
@patch('src.students.joshsalako.essay_writer.agent.getpass')
def test_get_llm_api_key_not_set(mock_getpass_module, mock_chat_model, monkeypatch):
    """
    Tests that get_llm prompts for user input
    if the API key environment variable is not set.
    """
    mock_getpass_module.getpass.return_value = "test-api-key-from-input"
    if "GOOGLE_API_KEY" in os.environ:
        monkeypatch.delenv("GOOGLE_API_KEY")

    get_llm()

    mock_getpass_module.getpass.assert_called_once_with("Enter your Google API Key: ")
    assert os.environ["GOOGLE_API_KEY"] == "test-api-key-from-input"
    mock_chat_model.assert_called_once()
