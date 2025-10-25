import os
import logging

from langchain.chat_models import init_chat_model
from langchain_google_community import GoogleSearchAPIWrapper
from langchain.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import AnyMessage, ToolMessage, HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from pydantic import BaseModel, Field

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set up provider
llm = init_chat_model(
    "google_genai:gemini-2.0-flash-lite",
    temperature=1.0,
)

search = GoogleSearchAPIWrapper()

# -------- Tools ---------
@tool
def search_google(query: str) -> str:
    """Search Google for relevant information on topic."""
    logger.info(f"Searching Google for: {query}")
    return search.run(f"{query} ")

@tool
def length_checker(report: str) -> int:
    """Compute the number of words a text"""
    logger.info("Computing text length")
    return len(report.split())

tools = [search_google, length_checker]
tools_by_name = {tool.name: tool for tool in tools}
llm_with_tools = llm.bind_tools(tools) 


# -------- Schema ---------
class GraphState(BaseModel):
    message: list[str]
    outline: str
    report: str
    approval: bool


# -------- Nodes ----------

PROMPT = """\
You are a helpful assistant. You are tasked to write report related to one of the following topics:
- MLOps
- CI/CD
- Agentic AI
- APIs

You must follow these rule:
- You must write factual content
- You must keep the report within 950 and 1050
- You must use the provided tool for content retrieval and length checking
- The report must cover theory and technical details
"""

agent = create_react_agent(
    llm,
    tools=tools,
    prompt=PROMPT
)
