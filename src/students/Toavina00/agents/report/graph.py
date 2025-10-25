import json
import logging
from typing import List, Literal, TypedDict

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_community import GoogleSearchAPIWrapper
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field


def parse_json_to_pydantic(json_string: str, pydantic_model: BaseModel):
    """
    Parses a JSON string and attempts to load it into a Pydantic model.

    Args:
        json_string: The string containing JSON data.
        pydantic_model: The Pydantic model class to which the JSON should be mapped.

    Returns:
        An instance of the pydantic_model if successful, otherwise None.
    """
    try:
        # Strip common delimiters like "```json" and "```"
        if json_string.strip().startswith("```json"):
            json_string = json_string.strip()[len("```json"):].strip()
        if json_string.strip().endswith("```"):
            json_string = json_string.strip()[:-len("```")].strip()

        data = json.loads(json_string)
        return pydantic_model.model_validate(data)  # Use model_validate for Pydantic v2
    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        print(f"Problematic JSON string: {json_string}")
        return None
    except ValidationError as e:
        print(f"Pydantic validation error: {e}")
        print(f"Problematic JSON data: {data}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print(f"Problematic JSON string: {json_string}")
        return None


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


# -------- Schema ---------
class PlanSchema(BaseModel):
    """Result of the planning phase"""

    success: bool = Field(description="Status of the planning phase")
    content: str = Field(description="Detailed outline of the report or reason of failure")


class CriticSchema(BaseModel):
    """Review of the report"""

    approval: bool = Field(description="Decision of the critic for the approval of the report")
    review: str = Field(description="Review and suggestion of the critic in case of non-approval")


class ReportState(TypedDict):
    topic: str
    outline: PlanSchema
    search_contents: List[str]
    report: str
    critic_review: CriticSchema
    iterations: int


# -------- Agents ----------
PLANNING_PROMPT = """\
You are a helpful assistant. You are tasked of the planning part of writing report related to one of the following topics:
- MLOps
- CI/CD
- Agentic AI
- APIs

You must follow these rule:
- If the topic provided by the user is not amongst the specified topics, you should set `success` as `False` and detail the reason of the error in `content` 
- You must find relevant information regarding the topic provided by the user using the provided tool
- You must output an outline in `content` following the structure: Introduction, Body, Conclusion
- The outline must be detailed but each part should be concise and only focused on the essential
- The detailed outline should be written in markdown
- You must use the retrieved information to detail each part of the outline
- If the outline was successfuly written, you shoud set `success` as `True`
"""

planning_agent = create_react_agent(
    llm,
    tools=[search_google],
    prompt=PLANNING_PROMPT,
    response_format=PlanSchema
)

RESEARCH_PROMPT = """\
You are a helpful assistant. You are given an outline. You are tasked to search relevant contents in order to write a report.

You must follow these rule:
- YOU MUST NOT CREATE CONTENTS BUT USE THE PROVIDED TOOL TO GENERATE CONTENTS.
- You must gather enough information relevant to the outline.
- You must gather enough information to help writing the report.
- You must use the provided tool to search information and you can perform multiple tool call in order to gather the needed information.
"""

research_agent = create_react_agent(
    llm,
    tools=[search_google],
    prompt=RESEARCH_PROMPT,
)


WRITING_PROMPT = """\
You are a helpful assistant. You are given an outline, contents and instructions. You are tasked to write a report based on the provided information.

You must follow these rule:
- You must abide to the provided outline.
- You must abide to the given instructions.
- YOU MUST NOT CREATE CONTENTS BUT ONLY USE THE PROVIDED CONTENT WHEN WRITING THE REPORT.
- The length should be within 950-1050.
- The report must be technical but also include theory.
"""

writing_agent = create_react_agent(
    llm,
    tools=[length_checker],
    prompt=WRITING_PROMPT
)


CRITIC_PROMPT = """\
You are a helpful assistant. You are given an outline, contents, and a written report. You are tasked to review the report.

You must follow these rule:
- You must check if the report follow the outline
- You must check using the provided tool that the report has between 950 and 1050 words.
- You must check that the report only include facts from the provided contents.
- If the report passes the checks you shoud set `approval` to True otherwise set it to false and write a review on how to improve the report.
"""

critic_agent = create_react_agent(
    llm,
    tools=[length_checker],
    prompt=CRITIC_PROMPT,
    response_format=CriticSchema,
)


# ------------ Nodes -------------
def plan_node(state: ReportState) -> ReportState:
    logger.info("Executing plan_node")
    planning_output = planning_agent.invoke({"messages": [HumanMessage(content=state["topic"])]})
    outline = parse_json_to_pydantic(planning_output["messages"][-1].content, PlanSchema)

    if not outline.success:
        raise ValueError(f"Planning failed: {outline.content}")

    return {"outline": outline, "topic": state["topic"], "search_contents": [], "report": "", "critic_review": None, "iterations": state["iterations"]}


def research_node(state: ReportState) -> ReportState:
    logger.info("Executing research_node")
    outline_content = state["outline"].content
    research_output = research_agent.invoke({"messages": [HumanMessage(content=f"Outline:\n{outline_content}")]})

    search_contents = [research_output] if isinstance(research_output, str) else research_output

    return {"search_contents": search_contents, "outline": state["outline"], "topic": state["topic"], "report": "", "critic_review": None, "iterations": state["iterations"]}


def write_node(state: ReportState) -> ReportState:
    logger.info("Executing write_node")
    outline_content = state["outline"].content
    search_contents_str = "\n".join(state["search_contents"])
    instructions = ""
    if state["critic_review"] and not state["critic_review"].approval:
        instructions = f"Here is the critic's review for improvement: {state['critic_review'].review}\n"

    writing_prompt = f"""
    Outline:
    {outline_content}

    Contents:
    {search_contents_str}

    Instructions:
    {instructions}
    Write a report based on the provided outline, contents, and instructions.
    Ensure the report is between 950-1050 words, technical, and includes theory.
    """

    report_output = writing_agent.invoke({"messages": [HumanMessage(content=writing_prompt)]})
    report_content = report_output["messages"][-1].content

    return {"report": report_content, "outline": state["outline"], "topic": state["topic"], "search_contents": state["search_contents"], "critic_review": None, "iterations": state["iterations"]}


def critic_node(state: ReportState) -> ReportState:
    logger.info("Executing critic_node")
    outline_content = state["outline"].content
    search_contents_str = "\n".join(state["search_contents"])
    report_content = state["report"]

    critic_prompt = f"""
    Outline:
    {outline_content}

    Contents:
    {search_contents_str}

    Report:
    {report_content}

    Review the report based on the provided outline and contents.
    Check if the report follows the outline, has between 950 and 1050 words, and only uses facts from the provided contents.
    """

    critic_output = critic_agent.invoke({"messages": [HumanMessage(content=critic_prompt)]})
    critic_review = parse_json_to_pydantic(critic_output["messages"][-1].content, CriticSchema)

    return {"critic_review": critic_review, "report": state["report"], "outline": state["outline"], "topic": state["topic"], "search_contents": state["search_contents"], "iterations": state["iterations"] + 1}


# Define conditional edge
def decide_to_rewrite(state: ReportState) -> Literal["rewrite", "end"]:
    logger.info("Executing decide_to_rewrite")
    if state["critic_review"] and state["critic_review"].approval:
        return "end"
    else:
        if state["iterations"] >= 3:
            logger.warning("Max rewrite iterations reached. Ending with unapproved report.")
            return "end"
        return "rewrite"


# ---------- Graph --------------
workflow = StateGraph(ReportState)

workflow.add_node("plan", plan_node)
workflow.add_node("research", research_node)
workflow.add_node("write", write_node)
workflow.add_node("critic", critic_node)

workflow.set_entry_point("plan")

workflow.add_edge("plan", "research")
workflow.add_edge("research", "write")
workflow.add_edge("write", "critic")

workflow.add_conditional_edges(
    "critic",
    decide_to_rewrite,
    {
        "rewrite": "write",
        "end": END
    }
)

app = workflow.compile()


def generate_report(topic: str):
    initial_state = ReportState(
        topic=topic,
        outline=None,
        search_contents=[],
        report="",
        critic_review=None,
        iterations=0
    )
    result = app.invoke(initial_state)['report']

    return result
