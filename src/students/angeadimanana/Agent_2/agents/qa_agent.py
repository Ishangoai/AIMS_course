"""
QA Agent Module
Final quality check of the report with scoring
"""

from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.helpers import count_words


def qa_agent(
    topic: str,
    edited_report: str,
    research_notes: str,
    llm: ChatGoogleGenerativeAI
) -> str:
    """
    QA Agent: Final quality check of the report
    Args:
        topic (str): The topic of the report
        edited_report (str): The edited report to evaluate
        research_notes (str): Research notes for reference
        llm (ChatGoogleGenerativeAI): The LLM instance
    Returns:
        str: QA evaluation with scores and feedback
    """
    word_count = count_words(edited_report)

    qa_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a QA Agent specialized in evaluating technical reports.

Your task is to evaluate the report on these criteria:
1. **Structure** (0-1): Does it have clear introduction, main sections with titles, and conclusion?
2. **Word Count** (0-1): Is it between 950-1050 words? (1.0 if yes, proportional penalty if no)
3. **Factual Accuracy** (0-1): Are claims well-supported and accurate based on research?
4. **Clarity** (0-1): Is the writing clear, professional, and easy to understand?
5. **Completeness** (0-1): Does it comprehensively cover the topic?

Output format:
## QA Evaluation Report

### Scores
- Structure: [score]/1.0
- Word Count: [score]/1.0 (Actual: [count] words)
- Factual Accuracy: [score]/1.0
- Clarity: [score]/1.0
- Completeness: [score]/1.0

**Overall Score: [average]/1.0**

### Strengths
- [List what the report does well]

### Issues (if any)
- [List specific issues that need fixing]

### Recommendations for Revision (if score < 0.8)
- [Specific actionable feedback for the Writer Agent]

### Final Decision
[APPROVED] or [NEEDS REVISION]

Be strict but fair in your evaluation."""),
        HumanMessage(content=f"""Topic: {topic}

Report to Evaluate (Word count: {word_count}):
{edited_report}

Research Notes for Reference:
{research_notes}

Provide a comprehensive QA evaluation.""")
    ])

    chain = qa_prompt | llm | StrOutputParser()
    qa_results = chain.invoke({})

    return qa_results
