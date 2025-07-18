"""LangGraph single-node graph template.

Returns a predefined response. Replace logic and configuration as needed.
"""

from __future__ import annotations

import logging
import typing
from dataclasses import dataclass

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

# from langgraph.graph import StateGraph
from langgraph.prebuilt import create_react_agent

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize the Gemini model
gemini_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.0)

search = GoogleSearchAPIWrapper()


@tool
def search_exchange_rate(query: str) -> str:
    """Search Google for the latest exchange rate information."""
    logger.info(f"Searching for exchange rate for: {query}")
    return search.run(f"{query} exchange rate")


graph = create_react_agent(
    gemini_model,
    tools=[search_exchange_rate],
    prompt="You are a helpful assistant in converting currencies using the latest exchange rate.",
)

# inputs = {"messages": [{"role": "user", "content": "how much is 90 AED in cedis at the moment?"}]}
# result = graph.invoke(inputs)
# logger.info(result["messages"][-1].content)


class Configuration(typing.TypedDict):
    """Configurable parameters for the agent.

    Set these when creating assistants OR when invoking the graph.
    See: https://langchain-ai.github.io/langgraph/cloud/how-tos/configuration_cloud/
    """

    my_configurable_param: str


@dataclass
class State:
    """Input state for the agent.

    Defines the initial structure of incoming data.
    See: https://langchain-ai.github.io/langgraph/concepts/low_level/#state
    """

    changeme: str = "example"


async def call_model(state: State, config: RunnableConfig) -> typing.Dict[str, typing.Any]:
    """Process input and returns output.

    Can use runtime configuration to alter behavior.
    """
    configuration = config.get("configurable", {})
    return {
        "changeme": "output from call_model. "
        f'Configured with {configuration.get("my_configurable_param")}'
    }



# Define the graph
# graph = (
#     StateGraph(State, config_schema=Configuration)
#     .add_node(call_model)
#     .add_edge("__start__", "call_model")
#     .compile(name="New Graph")
# )

# --- Gradio UI for chat ---
import gradio as gr

def chat_with_graph(message, history):
    """Pass the full chat history to the graph and return the assistant's reply."""
    # Convert Gradio's history to the format expected by the graph
    messages = []
    if history:
        for user_msg, assistant_msg in history:
            messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})
    # Add the latest user message
    messages.append({"role": "user", "content": message})
    inputs = {"messages": messages}
    result = graph.invoke(inputs)
    reply = result["messages"][-1].content
    return reply

with gr.Blocks() as demo:
    gr.Markdown("# Global Currency Assistant.\nYou can tell me which currency you prefer to convert")
    chatbot = gr.ChatInterface(
        fn=chat_with_graph,
        title="Currency Assistant",
        description="Ask about currency conversion and get the latest exchange rates.",
    )

if __name__ == "__main__":
    demo.launch()
