from __future__ import annotations

import logging
import os

from langchain_core.tools import tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
# check me out

# Initialize the Gemini model, adjust temperature as needed, 0.0 is deterministic and 2.0 is more creative
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0, google_api_key=GOOGLE_API_KEY)

search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)


@tool
def search_google(query: str) -> str:
    """Search Google for latest information on topic."""
    logger.info(f"Searching Google for: {query}")
    return search.run(f"{query} ")


graph = create_react_agent(
    gemini_model,
    tools=[search_google],
    prompt="You are a helpful assistant. Always answer in a funny way",
)
