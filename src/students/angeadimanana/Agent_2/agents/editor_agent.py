"""
Editor Agent Module
Reviews and improves the draft for clarity, structure, and flow
"""

from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.helpers import count_words


def editor_agent(
    topic: str,
    draft_report: str,
    fact_check_results: str,
    llm: ChatGoogleGenerativeAI
) -> str:
    """
    Editor Agent: Reviews and improves the draft for clarity, structure, and flow
    Args:
        topic (str): The topic of the report
        draft_report (str): The draft report to edit
        fact_check_results (str): Feedback from fact-checker
        llm (ChatGoogleGenerativeAI): The LLM instance
    Returns:
        str: Edited report
    """
    current_word_count = count_words(draft_report)

    editor_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are an Editor Agent specialized in refining technical reports.

Your task is to:
1. Review the draft report for clarity, coherence, and flow
2. Ensure proper structure (clear introduction, well-organized sections, strong conclusion)
3. Incorporate feedback from the fact-checker
4. Fix any issues identified in the fact-check
5. Improve readability and professional tone
6. **CRITICAL**: Maintain the target word count (1000 ±50 words, so 950-1050 words)
7. Ensure all section titles are clear and descriptive

**WORD COUNT RULE**: If the draft is already close to or above 1000 words, you MUST:
- Make edits concise
- Remove unnecessary words while fixing issues
- Do NOT expand content - focus on improving what exists
- If adding clarifications, remove redundant phrases elsewhere

Output the COMPLETE EDITED REPORT with all improvements applied.
Do not provide commentary - just output the final edited version."""),
        HumanMessage(content=f"""Topic: {topic}

Draft Report (Current word count: {current_word_count} words):
{draft_report}

Fact-Checker Feedback:
{fact_check_results}

Edit and improve the report, incorporating all feedback.
**IMPORTANT**: Keep the final word count between 950-1050 words. The current draft has {current_word_count} words.
Output the complete edited report.""")
    ])

    chain = editor_prompt | llm | StrOutputParser()
    edited_report = chain.invoke({})

    return edited_report
