"""
Writer Agent Module
Creates structured reports based on research notes and feedback
"""

from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


def writer_agent(
    topic: str,
    research_notes: str,
    llm: ChatGoogleGenerativeAI,
    feedback: str = None,
    previous_draft: str = None
) -> str:
    """
    Writer Agent: Creates a structured report draft based on research notes
    Can incorporate feedback from previous iterations
    Args:
        topic (str): The topic of the report
        research_notes (str): Research notes to base the report on
        llm (ChatGoogleGenerativeAI): The LLM instance
        feedback (str, optional): Feedback from QA or editor agents
        previous_draft (str, optional): Previous version of the report
    Returns:
        str: Complete report draft
    """

    # Base system message
    system_content = """You are a Technical Writer Agent specialized in creating well-structured reports.

Your task is to write a comprehensive technical report with the following requirements:
1. Length: Approximately 1000 words (±50 words)
2. Structure: Clear introduction, 2-3 main sections with descriptive titles, and a conclusion
3. Style: Professional, clear, and informative
4. Content: Based on the provided research notes

Format:
# [Report Title]

## Introduction
[Introduction content]

## [Section 1 Title]
[Section 1 content]

## [Section 2 Title]
[Section 2 content]

## [Section 3 Title] (if needed)
[Section 3 content]

## Conclusion
[Conclusion content]
"""

    # If this is a revision, add feedback instructions
    if feedback and previous_draft:
        system_content += """

**REVISION MODE**:
You are revising a previous draft based on feedback. Your task is to:
1. Address ALL issues mentioned in the feedback
2. Maintain the target word count (950-1050 words)
3. Improve clarity and structure where needed
4. Keep what works well in the previous draft
5. Make targeted improvements based on the feedback

Be thorough but efficient in your revisions."""

        human_content = f"""Topic: {topic}

Research Notes:
{research_notes}

Previous Draft:
{previous_draft}

Feedback to Address:
{feedback}

Write the COMPLETE REVISED REPORT addressing all feedback points. Target: 1000 words (±50 words)."""
    else:
        # First draft
        human_content = f"""Topic: {topic}

Research Notes:
{research_notes}

Write a complete 1000-word report (±50 words)."""

    writer_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_content),
        HumanMessage(content=human_content)
    ])

    chain = writer_prompt | llm | StrOutputParser()
    draft_report = chain.invoke({})

    return draft_report
