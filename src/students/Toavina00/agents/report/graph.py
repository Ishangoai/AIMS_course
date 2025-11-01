import logging
from typing import Literal

from agents.report.model import CriticSchema, PlanSchema, ReportState
from agents.report.prompt import CRITIC_PROMPT, PLANNING_PROMPT, RESEARCH_PROMPT, WRITING_PROMPT
from agents.report.utils import parse_json_to_pydantic
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_google_community import GoogleSearchAPIWrapper
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import create_react_agent

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set up provider

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


def planning(topic: str, temperature):
    llm = init_chat_model(
    "google_genai:gemini-2.0-flash-lite",
    temperature=temperature,
    )

    planning_agent = create_react_agent(
        llm,
        tools=[search_google],
        prompt=PLANNING_PROMPT,
        response_format=PlanSchema
    )

    planning_output = planning_agent.invoke({"messages": [HumanMessage(content=topic)]})["structured_response"]
    return planning_output.content


def compile_graph(temperature: float, llm):

    # llm = init_chat_model(
    #     "google_genai:gemini-2.0-flash-lite",
    #     temperature=temperature,
    # )

    # -------- Agents ----------
    # planning_agent = create_react_agent(
    #     llm,
    #     tools=[search_google],
    #     prompt=PLANNING_PROMPT,
    #     response_format=PlanSchema
    # )

    research_agent = create_react_agent(
        llm,
        tools=[search_google],
        prompt=RESEARCH_PROMPT,
    )

    writing_agent = create_react_agent(
        llm,
        tools=[length_checker],
        prompt=WRITING_PROMPT
    )

    critic_agent = create_react_agent(
        llm,
        tools=[length_checker],
        prompt=CRITIC_PROMPT,
        response_format=CriticSchema,
    )

    # ------------ Nodes -------------
    # def plan_node(state: ReportState) -> ReportState:
    #     logger.info("Executing plan_node")
    #     planning_output = planning_agent.invoke({"messages": [HumanMessage(content=state["topic"])]})
    #     outline = parse_json_to_pydantic(planning_output["messages"][-1].content, PlanSchema)

    #     if not outline.success:
    #         raise ValueError(f"Planning failed: {outline.content}")

    #     return {
    #         "outline": outline,
    #         "topic": state["topic"],
    #         "search_contents": [],
    #         "report": "",
    #         "critic_review": None,
    #         "iterations": state["iterations"]
    #     }

    def research_node(state: ReportState) -> ReportState:
        logger.info("Executing research_node")
        outline_content = state["outline"].content
        research_output = research_agent.invoke({"messages": [HumanMessage(content=f"Outline:\n{outline_content}")]})

        search_contents = [research_output] if isinstance(research_output, str) else research_output

        return {
            "search_contents": search_contents,
            "outline": state["outline"],
            "topic": state["topic"],
            "report": "",
            "critic_review": None,
            "iterations": state["iterations"]
        }

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
        """

        report_output = writing_agent.invoke({"messages": [HumanMessage(content=writing_prompt)]})
        report_content = report_output["messages"][-1].content

        return {
            "report": report_content,
            "outline": state["outline"],
            "topic": state["topic"],
            "search_contents": state["search_contents"],
            "critic_review": None,
            "iterations": state["iterations"]
        }

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
        """

        critic_output = critic_agent.invoke({"messages": [HumanMessage(content=critic_prompt)]})
        critic_review = parse_json_to_pydantic(critic_output["messages"][-1].content, CriticSchema)

        return {
            "critic_review": critic_review,
            "report": state["report"],
            "outline": state["outline"],
            "topic": state["topic"],
            "search_contents": state["search_contents"],
            "iterations": state["iterations"] + 1
        }

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

    # workflow.add_node("plan", plan_node)
    workflow.add_node("research", research_node)
    workflow.add_node("write", write_node)
    workflow.add_node("critic", critic_node)

    workflow.set_entry_point("research")

    # workflow.add_edge("plan", "research")
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

    return app


def generate_report(topic: str, temperature: float, outline: str):
    initial_state = ReportState(
        topic=topic,
        outline=PlanSchema(content=outline, success=True),
        search_contents=[],
        report="",
        critic_review=None,
        iterations=0
    )

    llm = init_chat_model(
        "google_genai:gemini-2.0-flash-lite",
        temperature=temperature,
    )

    result = compile_graph(temperature, llm).invoke(initial_state)['report']

    return result
