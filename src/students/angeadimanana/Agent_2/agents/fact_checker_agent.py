"""
Fact-Checker Agent Module
Verifies claims in the draft against research notes
"""

from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


def fact_checker_agent(
    topic: str,
    draft_report: str,
    research_notes: str,
    llm: ChatGoogleGenerativeAI
) -> str:
    """
    Fact-Checker Agent: Verifies claims in the draft against research notes
    Args:
        topic (str): The topic of the report
        draft_report (str): The draft report to verify
        research_notes (str): Research notes for verification
        llm (ChatGoogleGenerativeAI): The LLM instance
    Returns:
        str: Fact-check results with issues and recommendations
    """
    fact_checker_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a Fact-Checker Agent specialized in verifying technical content accuracy.

Your task is to:
1. Review the draft report carefully
2. Cross-reference claims with the research notes
3. Identify any unsupported, questionable, or inaccurate statements
4. Provide specific corrections or suggestions for improvement

Output format:
## Fact-Check Results

### Verified Claims
- [List claims that are well-supported by research]

### Issues Found
- [List any unsupported or questionable claims with specific line/section references]
- [Provide corrections or suggestions]

### Recommendations
- [Suggest improvements to strengthen factual accuracy]

If everything is accurate, state that clearly."""),
        HumanMessage(content=f"""Topic: {topic}

Research Notes:
{research_notes}

Draft Report to Verify:
{draft_report}

Perform a thorough fact-check and provide your analysis.""")
    ])

    chain = fact_checker_prompt | llm | StrOutputParser()
    fact_check_results = chain.invoke({})

    return fact_check_results
