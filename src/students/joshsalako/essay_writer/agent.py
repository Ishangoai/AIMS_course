"""
This module defines the agentic system for writing reports.
It includes the planner, writer, and reviewer agents, and orchestrates
their interactions using LangGraph.
"""

import getpass
import os
from typing import List, TypedDict

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from .tools import get_wikipedia_tool

# --- Model Initialization ---


def get_llm():
    """Initializes and returns the Google Gemini LLM."""
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API Key: ")
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.5)

# --- Agent State ---


class ReportState(TypedDict):
    """Represents the state of our process."""
    topic: str
    plan: dict
    draft_sections: list[str]
    review: str
    final_report: str

# --- Planner Agent ---


class SectionPlan(BaseModel):
    """Data model for a single section of the report."""
    title: str = Field(description="The title of the section")
    description: str = Field(description="A brief description of what the section should cover")
    word_count: int = Field(description="The target word count for this section")


class ReportPlan(BaseModel):
    """Data model for the report plan."""
    title: str = Field(description="The title of the report")
    sections: List[SectionPlan] = Field(description="List of sections in the report")


def create_planner(llm):
    """Creates the planner agent."""
    prompt_template = """You are an expert report planner.
    Your job is to create a detailed plan for a report on a given topic.
    Your final plan must produce a report with a total word count of approximately 1040 words.

    Create a plan with the following sections:
    1. An introduction.
    2. At least 3-4 main body sections that cover the topic in detail.
    3. A conclusion.

    For each section, you must provide:
    - A clear title.
    - A brief description of what the section should cover.
    - A specific target word count.

    **Crucially, the word counts for all sections must sum up to exactly 1040 words.
    ** Distribute the words logically, with the introduction and conclusion being shorter than the main body sections.

    Topic: "{topic}"
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt | llm.with_structured_output(ReportPlan)

# --- Writer Agent ---


def create_writer(llm):
    """Creates the writer agent."""
    tools = [get_wikipedia_tool()]
    prompt_template = """You are an expert technical writer.
    Your task is to write a section of a report based on a provided plan.
    You have access to a Wikipedia search tool to gather information. Use it to find relevant facts and information.
    Write the section in a technical and informative style. Do not hallucinate.

    You have access to the following tools:
    {tools}

    To get the information you need, use the following format:

    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action (if your first search fails, try a broader or more general query)
    Observation: the result of the action

    When you have enough information, respond with the final, complete section content in this format:

    Thought: I now know the final answer
    Final Answer: [The content for the section, written as plain text.
    Do NOT include or return back the section title in your answer.]

    Begin!

    Write a report section based on the following details, you MUST adhere strictly to the word count provided:
    {input}

    Thought:{agent_scratchpad}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


# --- Reviewer Agent ---


def create_reviewer(llm):
    """Creates the reviewer agent."""
    prompt_template = """You are an expert editor and reviewer. Your task is to review a draft of a report.
    Please check for factual accuracy, clarity, coherence, grammar, and technical tone.
    Provide your feedback as a list of comments and suggestions for improvement.
    If the report is good, state that it is well-written and ready.

    Report Draft:
    ---
    {draft}
    ---
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt | llm

# --- Graph Nodes ---


def planner_node(state: ReportState):
    """Node that runs the planner agent."""
    print("---PLANNING---")
    llm = get_llm()
    planner = create_planner(llm)
    plan = planner.invoke({"topic": state["topic"]})
    return {"plan": plan.dict()}


def writer_node(state: ReportState):
    """Node that runs the writer agent for each section."""
    print("---WRITING---")
    llm = get_llm()
    writer = create_writer(llm)
    sections = state["plan"]["sections"]
    draft_sections = []
    for i, section in enumerate(sections):
        print(f"---WRITING SECTION {i + 1}/{len(sections)}: {section['title']}---")

        # Format the input for the writer agent
        input_prompt = f"""
        Report Topic: {state["plan"]["title"]}
        Section to write:
        Title: {section['title']}
        Description: {section['description']}
        The section should be approximately {section['word_count']} words long.
        """
        response = writer.invoke({"input": input_prompt})
        draft_sections.append(response["output"])
    return {"draft_sections": draft_sections}


def combiner_node(state: ReportState):
    """Node that combines the written sections into a single report."""
    print("---COMBINING---")
    final_report = "\n\n".join(
        f"## {section['title']}\n\n{draft}"
        for section, draft in zip(state["plan"]["sections"], state["draft_sections"])
    )
    return {"final_report": final_report}


def reviewer_node(state: ReportState):
    """Node that runs the reviewer agent."""
    print("---REVIEWING---")
    llm = get_llm()
    reviewer = create_reviewer(llm)
    review = reviewer.invoke({"draft": state["final_report"]})
    return {"review": review.content}

# --- Graph Definition ---


def build_graph():
    """Builds and compiles the LangGraph for report generation."""
    workflow = StateGraph(ReportState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("combiner", combiner_node)
    workflow.add_node("reviewer", reviewer_node)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "writer")
    workflow.add_edge("writer", "combiner")
    workflow.add_edge("combiner", "reviewer")
    workflow.add_edge("reviewer", END)

    return workflow.compile()


def run_agent(topic: str):
    """Runs the agentic system for a given topic."""
    app = build_graph()
    final_state = app.invoke({"topic": topic})
    return final_state
