"""
This module contains the tools for the essay writer agent.
- A custom Wikipedia search tool that uses the 'requests' library
- A word counter utility function
"""
import requests
from langchain_core.tools import BaseTool


class WikipediaSearchTool(BaseTool):
    """
    A tool for searching Wikipedia that directly uses the Wikipedia API,
    avoiding the need for the 'wikipedia' library.
    """
    name: str = "wikipedia"
    description: str = (
        "A wrapper around Wikipedia. "
        "Useful for when you need to answer general questions about "
        "people, places, companies, facts, historical events, or other subjects. "
        "Input should be a search query."
    )
    doc_content_chars_max: int = 4000

    def _run(self, query: str) -> str:
        """Use the Wikipedia tool."""
        try:
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/58.0.3029.110 Safari/537.36"
            })
            url = "https://en.wikipedia.org/w/api.php"

            # Step 1: Search for the page title
            search_params = {
                "action": "opensearch", "search": query, "limit": "1",
                "namespace": "0", "format": "json",
            }
            search_response = session.get(url=url, params=search_params, timeout=5)
            search_response.raise_for_status()
            search_data = search_response.json()

            if not search_data[1]:
                return "No good Wikipedia Search Result was found"
            page_title = search_data[1][0]

            # Step 2: Fetch page content
            content_params = {
                "action": "query", "format": "json", "titles": page_title,
                "prop": "extracts", "explaintext": True,
            }
            content_response = session.get(url=url, params=content_params, timeout=5)
            content_response.raise_for_status()
            content_data = content_response.json()
            page_info = content_data["query"]["pages"]
            page_id = list(page_info.keys())[0]

            if page_id == "-1":
                return f"Page titled '{page_title}' does not exist on Wikipedia."

            extract = page_info[page_id].get("extract", "")
            return extract[:self.doc_content_chars_max] if extract else "No content found."

        except requests.exceptions.RequestException as e:
            return f"An error occurred with the Wikipedia API: {e}"


def get_wikipedia_tool():
    """
    Returns a configured Wikipedia search tool.
    This tool can be used to search for information on Wikipedia.
    """
    return WikipediaSearchTool()


def count_words(text: str) -> int:
    """
    Counts the number of words in a given string.
    """
    return len(text.split())
