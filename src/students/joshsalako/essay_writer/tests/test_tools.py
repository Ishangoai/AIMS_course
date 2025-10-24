from unittest.mock import MagicMock, patch

import pytest
import requests

from ..tools import WikipediaSearchTool, count_words, get_wikipedia_tool


@pytest.mark.parametrize("text, expected", [
    ("hello world", 2),
    ("one", 1),
    ("", 0),
    ("   ", 0),
    ("  leading and trailing spaces  ", 4),
])
def test_count_words(text, expected):
    """Tests the count_words function with various inputs."""
    assert count_words(text) == expected


def test_get_wikipedia_tool():
    """Tests if get_wikipedia_tool returns a valid tool instance."""
    tool = get_wikipedia_tool()
    assert isinstance(tool, WikipediaSearchTool)
    assert tool.name == "wikipedia"
    assert tool.description.startswith("A wrapper around Wikipedia.")


@patch('src.students.joshsalako.essay_writer.tools.requests.Session', autospec=True)
def test_wikipedia_search_tool_success(mock_session):
    """Tests the success case for the Wikipedia search tool."""
    mock_instance = mock_session.return_value
    mock_instance.headers = MagicMock()
    mock_search_response = MagicMock()
    mock_search_response.json.return_value = ["query", ["Test Page"], "", ["url"]]
    mock_content_response = MagicMock()
    page_content = "This is the content of the test page."
    mock_content_response.json.return_value = {
        "query": {"pages": {"12345": {"extract": page_content}}}
    }
    mock_instance.get.side_effect = [mock_search_response, mock_content_response]

    tool = WikipediaSearchTool()
    result = tool._run("test query")

    assert result == page_content
    assert mock_instance.get.call_count == 2


@patch('src.students.joshsalako.essay_writer.tools.requests.Session', autospec=True)
def test_wikipedia_search_tool_no_search_result(mock_session):
    """Tests the case where Wikipedia search returns no results."""
    mock_instance = mock_session.return_value
    mock_instance.headers = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = ["query", [], "", []]
    mock_instance.get.return_value = mock_response

    tool = WikipediaSearchTool()
    result = tool._run("unknown query")

    assert result == "No good Wikipedia Search Result was found"
    mock_instance.get.assert_called_once()


@patch('src.students.joshsalako.essay_writer.tools.requests.Session', autospec=True)
def test_wikipedia_search_tool_page_does_not_exist(mock_session):
    """Tests the case where a page title is found, but the page does not exist."""
    mock_instance = mock_session.return_value
    mock_instance.headers = MagicMock()
    mock_search_response = MagicMock()
    mock_search_response.json.return_value = ["query", ["Nonexistent Page"], "", []]
    mock_content_response = MagicMock()
    mock_content_response.json.return_value = {
        "query": {"pages": {"-1": {"title": "Nonexistent Page"}}}
    }
    mock_instance.get.side_effect = [mock_search_response, mock_content_response]

    tool = WikipediaSearchTool()
    result = tool._run("nonexistent page")

    assert result == "Page titled 'Nonexistent Page' does not exist on Wikipedia."


@patch('src.students.joshsalako.essay_writer.tools.requests.Session', autospec=True)
def test_wikipedia_search_tool_request_exception(mock_session):
    """Tests the handling of a requests exception."""
    mock_instance = mock_session.return_value
    mock_instance.headers = MagicMock()
    mock_instance.get.side_effect = requests.exceptions.RequestException("API error")

    tool = WikipediaSearchTool()
    result = tool._run("any query")

    assert "An error occurred with the Wikipedia API: API error" in result


@patch('src.students.joshsalako.essay_writer.tools.requests.Session', autospec=True)
def test_wikipedia_search_tool_content_truncation(mock_session):
    """Tests that long content is correctly truncated."""
    mock_instance = mock_session.return_value
    mock_instance.headers = MagicMock()
    mock_search_response = MagicMock()
    mock_search_response.json.return_value = ["query", ["Long Page"], "", []]
    mock_content_response = MagicMock()
    long_content = "a" * 5000
    mock_content_response.json.return_value = {
        "query": {"pages": {"123": {"extract": long_content}}}
    }
    mock_instance.get.side_effect = [mock_search_response, mock_content_response]

    tool = WikipediaSearchTool(doc_content_chars_max=4000)
    result = tool._run("long page")

    assert len(result) == 4000
    assert result == "a" * 4000
