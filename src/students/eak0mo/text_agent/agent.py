"""
This module defines the agentic system with dual paths.

Path 1: Report writing using Wikipedia and context collector tools
Path 2: Warhammer 40K army list generation with validation (REVISED FORMATTING)
"""

import getpass
import os
import re
from typing import List, TypedDict

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from .tools import (
    get_context_collector_tool,
    get_context_comparison_tool,
    get_wh40k_army_tool,
    get_wh40k_rules_validator_tool,
    get_wikipedia_tool,
)

# --- Model Initialization ---


def get_llm():
    """Initialize and return the Google Gemini LLM."""
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API Key: ")
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0.4)


# --- Shared State ---


class AgentState(TypedDict):
    """Represent the state of our process."""

    topic: str
    is_warhammer: bool
    plan: dict
    draft_sections: list[str]
    review: str
    final_output: str
    formatted_list: str  # New field for formatted list


# --- Topic Classifier ---


class TopicClassification(BaseModel):
    """Data model for topic classification."""

    is_warhammer: bool = Field(
        description="True if topic is about Warhammer 40K army lists, False otherwise"
    )
    faction_name: str = Field(
        description="Normalized faction name if Warhammer topic, empty otherwise"
    )
    reasoning: str = Field(description="Explanation of classification decision")


def create_topic_classifier(llm):
    """Create the topic classifier agent."""
    prompt_template = """You are an expert classifier specializing in Warhammer 40K.
    Your job is to determine if a topic is requesting a Warhammer 40K army list or a general report on a subject area.

    Warhammer 40K Factions and Common Nicknames:
    - Space Marines (SM, Astartes, Marines, Ultramarines, Blood Angels, Dark Angels, Space Wolves)
    - Chaos Space Marines (CSM, Heretic Astartes, Chaos Marines, Traitor Marines)
    - Tyranids (Nids, Bugs, Hive Fleet)
    - Necrons (Crons, Robot Skeletons)
    - Orks (Greenskins, Boyz, Orkz)
    - Astra Militarum (AM, Imperial Guard, IG, Guard)
    - Tau Empire (Tau, T'au, Greater Good)
    - Aeldari (Eldar, Craftworlds, Asuryani)
    - Drukhari (Dark Eldar, DE)
    - Adeptus Mechanicus (AdMech, Mechanicus, Cult Mechanicus)
    - Thousand Sons (TS, Sons)
    - Death Guard (DG, Plague Marines)
    - World Eaters (WE, Khorne Marines)
    - Genestealer Cults (GSC, Cults)
    - Adeptus Custodes (Custodes, Golden Boys)
    - Imperial Knights (Knights, IK)
    - Chaos Knights (CK, Chaos Knights)
    - Adepta Sororitas (Sisters of Battle, SoB, Sisters)
    - Grey Knights (GK)
    - Leagues of Votann (LoV, Squats, Votann)
    - Chaos Daemons (Daemons, CD)

    If the topic mentions:
    - Building an army list
    - Any of the faction names or nicknames above
    - Points, units, detachments, competitive list
    - Warhammer 40K, 40k, WH40K
    Then classify as Warhammer (is_warhammer: True)

    Otherwise, classify as a general report topic (is_warhammer: False)

    Topic: "{topic}"
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt | llm.with_structured_output(TopicClassification)


# --- Report Path (Path 1) ---


class SectionPlan(BaseModel):
    """Data model for a single section of the report."""

    title: str = Field(description="The title of the section")
    description: str = Field(
        description="A brief description of what the section should cover"
    )
    word_count: int = Field(description="The target word count for this section")


class ReportPlan(BaseModel):
    """Data model for the report plan."""

    title: str = Field(description="The title of the report")
    sections: List[SectionPlan] = Field(description="List of sections in the report")


def create_report_planner(llm):
    """Create the report planner agent."""
    prompt_template = """You are an World class report planner.
    Your job is to create a detailed  and comprehensive plan for a report on a given topic.
    Your final plan must produce a report with a total word "
    "count of approximately 1000 words with an error range of 20 words.

    Create a plan with the following sections:
    1. An introduction.
    2. At least 3 main body sections that cover the topic in detail.
    3. A conclusion.

    For each section, you must provide:
    - A clear title.
    - A brief description of what the section should cover.
    - A specific target word count.

    **Crucially, the word counts for all sections must sum up to exactly 1040 words.**
    Distribute the words logically, with the introduction and conclusion being shorter
    than the main body sections.

    Topic: "{topic}"
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt | llm.with_structured_output(ReportPlan)


def create_report_writer(llm):
    """Create the report writer agent with full tool access."""
    tools = [
        get_wikipedia_tool(),
        get_context_collector_tool(),
        get_context_comparison_tool(),
    ]
    prompt_template = """You are an expert technical writer.
    Your task is to write a section of a report based on a provided plan.

    You have access to the following tools:
    {tools}

    Use these tools to gather information:
    - wikipedia: For general knowledge and facts
    - context_collector: For specific topics which are (ci/cd, data engineering, dag, mlops)
    - context_comparison: To verify your content aligns with gathered context

    Tool usage format:
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action

    When you have enough information, respond with:
    Thought: I now know enough on what to write
    Final Answer: [The content for the section, written as plain text.
    Do NOT include the section title in your answer.]

    Write a report section based on the following details.
    You MUST adhere strictly to the word count provided:
    {input}

    Thought:{agent_scratchpad}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )


# --- Warhammer Path (Path 2) ---


class UnitEntry(BaseModel):
    """Data model for a single unit in the army list."""

    name: str = Field(description="The name of the unit")
    quantity: int = Field(description="Number of models or units")
    points: int = Field(description="Point cost of the unit")
    role: str = Field(
        description="Battlefield role (HQ, Troops, Elites, Fast Attack, Heavy Support)"
    )
    equipment: str = Field(description="Wargear and equipment loadout")


class ArmyListPlan(BaseModel):
    """Data model for the Warhammer 40K army list plan."""

    faction: str = Field(description="The faction name")
    detachment: str = Field(description="The detachment type")
    total_points: int = Field(description="Total point limit (e.g., 1000, 2000)")
    units: List[UnitEntry] = Field(description="List of units in the army")
    strategy_notes: str = Field(description="Brief tactical overview")


def create_wh40k_planner(llm):
    """Create the Warhammer 40K army list planner."""
    prompt_template = """You are an expert Warhammer 40K army list builder.
    Your job is to create a competitive, legal and balanced army list for the specified faction.

    Faction: {faction}
    Point Limit: {points} (default to 2000 if not specified)

    Create an army list with:
    1. The best valid detachment option
    2. Appropriate HQ units (required) that is seen is many of the best lists from Goonhammer
    3. Use Troops that have High value for thier Cost or very good in the chosen or given detachment
    4. If a character/troop is requested in the prompt, YOU MUST include them in the list
    5. If a detachment is provided in the prompt, YOU MUST use that detachment
    6. Total points matching the limit
    7. Make use of the best value enhancements in the list.

    Provide a complete army list with unit names, quantities, roles, equipment,
    and point costs. Include a brief tactical strategy overview.

    Important: Base your list on Most used and best value units and current meta considerations.
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    return prompt | llm.with_structured_output(ArmyListPlan)


def create_wh40k_validator(llm):
    """Create the Warhammer 40K list validator agent."""
    tools = [get_wh40k_army_tool(), get_wh40k_rules_validator_tool()]
    prompt_template = """You are a Warhammer 40K rules expert and list validator.
    Your task is to validate an army list against official"
    " rules and current points cost, you must verify every troop can be used.

    You have access to the following tools:
    {tools}

    Use these tools to:
    - wh40k_army_data: Fetch faction stats and tournament data
    - wh40k_rules_validator: Check Munitorum Field Manual and Balance Dataslate

    Tool usage format:
    Thought: Do I need to use a tool? Yes
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action

    When validation is complete, respond with:
    Thought: I now know the final answer
    Final Answer: [Validation report with any issues found and recommendations]

    Validate the following army list:
    {input}

    Thought:{agent_scratchpad}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
    )


# --- Graph Nodes ---


def classifier_node(state: AgentState):
    """Node that classifies the topic."""
    print("---CLASSIFYING TOPIC---")
    llm = get_llm()
    classifier = create_topic_classifier(llm)
    classification = classifier.invoke({"topic": state["topic"]})
    print(f"Classification: {classification.reasoning}")
    return {
        "is_warhammer": classification.is_warhammer,
        "topic": (
            classification.faction_name
            if classification.is_warhammer
            else state["topic"]
        ),
    }


def report_planner_node(state: AgentState):
    """Node that runs the report planner agent."""
    print("---PLANNING REPORT---")
    llm = get_llm()
    planner = create_report_planner(llm)
    plan = planner.invoke({"topic": state["topic"]})
    return {"plan": plan.dict()}


def report_writer_node(state: AgentState):
    """Node that runs the report writer agent for each section."""
    print("---WRITING REPORT---")
    llm = get_llm()
    writer = create_report_writer(llm)
    sections = state["plan"]["sections"]
    draft_sections = []

    for i, section in enumerate(sections):
        # Ensure every section has a valid word count
        word_count = section.get("word_count", 200)
        if not isinstance(word_count, int) or word_count <= 0:
            word_count = 200  # Fallback default

        print(f"---WRITING SECTION {i + 1}/{len(sections)}: {section['title']}---")
        input_prompt = f"""
        Report Topic: {state["plan"]["title"]}
        Section to write:
        Title: {section['title']}
        Description: {section['description']}
        The section should be approximately {word_count} words long.
        """
        response = writer.invoke({"input": input_prompt})
        draft_sections.append(response["output"])

    return {"draft_sections": draft_sections}


def wh40k_planner_node(state: AgentState):
    """Node that runs the Warhammer 40K army list planner."""
    print("---PLANNING WARHAMMER 40K ARMY LIST---")
    llm = get_llm()
    planner = create_wh40k_planner(llm)

    # Extract point limit from topic if specified, default to 2000
    points = 2000
    topic_lower = state["topic"].lower()
    if "1000" in topic_lower or "1k" in topic_lower:
        points = 1000
    elif "2000" in topic_lower or "2k" in topic_lower:
        points = 2000

    plan = planner.invoke({"faction": state["topic"], "points": points})
    return {"plan": plan.dict()}


def wh40k_validator_node(state: AgentState):
    """Node that runs the Warhammer 40K validator agent."""
    print("---VALIDATING WARHAMMER 40K ARMY LIST---")
    llm = get_llm()
    validator = create_wh40k_validator(llm)

    # Format the army list for validation
    plan = state["plan"]
    list_text = f"""
    Faction: {plan['faction']}
    Detachment: {plan['detachment']}
    Total Points: {plan['total_points']}

    Units:
    """
    for unit in plan["units"]:
        list_text += f"\n- {unit['quantity']}x {unit['name']} ({unit['role']}) - {unit['points']} pts"
        list_text += f"\n  Equipment: {unit['equipment']}"

    list_text += f"\n\nStrategy: {plan['strategy_notes']}"

    response = validator.invoke({"input": list_text})
    return {"draft_sections": [response["output"]]}


def wh40k_formatter(state: AgentState):  # noqa: C901
    """Node that formats the Warhammer 40K army list into competitive format."""
    print("---FORMATTING WARHAMMER 40K ARMY LIST---")
    plan = state["plan"]
    validation_report = state["draft_sections"][0]

    # Header Block
    header = "+++++++++++++++++++++++++++++++++++++++++++++++\n"
    header += "\n"
    header += f"+ FACTION KEYWORD: Imperium - {plan['faction']}\n"
    header += f"+ DETACHMENT: {plan['detachment']}\n"
    header += f"+ TOTAL ARMY POINTS: {plan['total_points']}pts\n"
    # Placeholder for WARLORD/ENHANCEMENT/SECONDARIES as this data is not strictly in ArmyListPlan
    header += "+ WARLORD: [TBD from units list]\n"
    header += "+ ENHANCEMENT: [TBD]\n"
    header += f"+ NUMBER OF UNITS: {len(plan['units'])}\n"
    header += "\n"
    header += "+++++++++++++++++++++++++++++++++++++++++++++++\n\n"

    # Group units by role
    unit_groups = {
        "CHARACTER": [],
        "BATTLELINE": [],
        "OTHER DATASHEETS": [],
        "DEDICATED TRANSPORT": [],
        "ELITES": [],
        "FAST ATTACK": [],
        "HEAVY SUPPORT": [],
    }

    for unit in plan["units"]:
        # Standardize unit roles for grouping
        role = unit["role"].upper().replace("HQ", "CHARACTER").replace("TROOPS", "BATTLELINE")

        # Determine the correct key for grouping
        if role in unit_groups:
            group_key = role
        elif role in ["ELITES", "FAST ATTACK", "HEAVY SUPPORT", "DEDICATED TRANSPORT"]:
            group_key = role
        else:
            group_key = "OTHER DATASHEETS"  # Fallback

        unit_groups[group_key].append(unit)

    # Body Block
    body = ""
    unit_count = 0
    for role_name in unit_groups.keys():
        if unit_groups[role_name]:
            body += f"{role_name.replace('DEDICATED TRANSPORT', 'TRANSPORT').upper()}\n\n"
            for unit in unit_groups[role_name]:
                unit_count += 1
                body += f"{unit['quantity']}x {unit['name']} ({unit['points']} pts)\n"
                # Equipment cleanup for better readability
                equipment_lines = [e.strip() for e in unit['equipment'].split(',')]
                for line in equipment_lines:
                    # Look for enhancements in equipment line
                    if 'enhancement' in line.lower() or '(warlord)' in line.lower() or 'warlord' in line.lower():
                        body += f"• {line}\n"
                    else:
                        body += f"• {line}\n"
                body += "\n"
            body += "\n"  # Add extra space between roles

    # Combine list and validation report
    formatted_list = f"{header}{body}\n\n## Validation Report\n\n{validation_report}"

    # Final check for Warlord/Enhancement placeholders
    # Attempt to use the first CHARACTER with an enhancement/warlord tag for the header
    first_char_with_special = next((
        u for u in unit_groups['CHARACTER']
        if 'warlord' in u['equipment'].lower() or 'enhancement' in u['equipment'].lower()
    ), None)

    if first_char_with_special:
        warlord_name = first_char_with_special['name']
        enhancement_match = re.search(r"enhancement:\s*(.+)", first_char_with_special['equipment'], re.IGNORECASE)
        enhancement_name = enhancement_match.group(1).strip().split('(')[0].strip() if enhancement_match else "[TBD]"

        # Replace placeholders in the header
        formatted_list = formatted_list.replace("[TBD from units list]", warlord_name)
        formatted_list = formatted_list.replace("[TBD]", enhancement_name)
    else:
        # Fallback for header if no tags found
        first_char = next((u for u in unit_groups['CHARACTER']), None)
        if first_char:
            formatted_list = formatted_list.replace("[TBD from units list]", first_char['name'])

        # If no character, use a generic warlord
        formatted_list = formatted_list.replace("[TBD from units list]", "Generic Commander")
        formatted_list = formatted_list.replace("[TBD]", "None")

    return {"formatted_list": formatted_list}


def combiner_node(state: AgentState):
    """Node that combines sections into final output."""
    print("---COMBINING OUTPUT---")

    if state["is_warhammer"]:
        # Use the pre-formatted list for Warhammer
        final_output = state["formatted_list"]
    else:
        # Format report
        final_output = "\n\n".join(
            f"## {section['title']}\n\n{draft}"
            for section, draft in zip(
                state["plan"]["sections"], state["draft_sections"]
            )
        )

    return {"final_output": final_output}


def reviewer_node(state: AgentState):
    """Node that runs the reviewer agent."""
    print("---REVIEWING OUTPUT---")
    llm = get_llm()

    if state["is_warhammer"]:
        prompt_template = """You are a Warhammer 40K competitive play expert.
        Review this army list for legality, competitiveness, and strategic coherence.
        Check for rule compliance, point accuracy, and tactical viability.

        Army List:
        ---
        {draft}
        ---
        """
    else:
        prompt_template = """You are an expert editor and reviewer.
        Review this report for factual accuracy, clarity, coherence, grammar,
        and technical tone. Provide feedback and suggestions for improvement.

        Report Draft:
        ---
        {draft}
        ---
        """

    prompt = ChatPromptTemplate.from_template(prompt_template)
    reviewer = prompt | llm
    review = reviewer.invoke({"draft": state["final_output"]})
    return {"review": review.content}


def route_after_classification(state: AgentState):
    """Route to appropriate path based on classification."""
    if state["is_warhammer"]:
        return "wh40k_planner"
    else:
        return "report_planner"


# --- Graph Definition ---


def build_graph():
    """Build and compile the dual-path LangGraph."""
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("classifier", classifier_node)
    workflow.add_node("report_planner", report_planner_node)
    workflow.add_node("report_writer", report_writer_node)
    workflow.add_node("wh40k_planner", wh40k_planner_node)
    workflow.add_node("wh40k_validator", wh40k_validator_node)
    workflow.add_node("wh40k_formatter", wh40k_formatter)  # NEW NODE
    workflow.add_node("combiner", combiner_node)
    workflow.add_node("reviewer", reviewer_node)

    # Set entry point
    workflow.set_entry_point("classifier")

    # Add conditional routing after classification
    workflow.add_conditional_edges(
        "classifier",
        route_after_classification,
        {"report_planner": "report_planner", "wh40k_planner": "wh40k_planner"},
    )

    # Report path
    workflow.add_edge("report_planner", "report_writer")
    workflow.add_edge("report_writer", "combiner")

    # Warhammer path (updated to include formatter)
    workflow.add_edge("wh40k_planner", "wh40k_validator")
    workflow.add_edge("wh40k_validator", "wh40k_formatter")  # Edge to NEW NODE
    workflow.add_edge("wh40k_formatter", "combiner")        # Edge from NEW NODE

    # Common path
    workflow.add_edge("combiner", "reviewer")
    workflow.add_edge("reviewer", END)

    return workflow.compile()


def run_agent(topic: str):
    """Run the agentic system for a given topic and return a standardized result dict."""
    app = build_graph()
    final_state = app.invoke({"topic": topic})

    # Extract the main output from the LangGraph state
    final_output = final_state.get("final_output", "")
    is_warhammer = final_state.get("is_warhammer", False)
    plan = final_state.get("plan", {})
    review = final_state.get("review", "")

    # Build a consistent return structure
    return {
        "final_report": final_output or "⚠️ No final output was produced.",
        "is_warhammer": is_warhammer,
        "plan": plan,
        "review": review,
    }
