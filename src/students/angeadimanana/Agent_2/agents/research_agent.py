"""
Research Agent Module
Gathers comprehensive information about a given topic
"""

from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, SystemMessage
from langchain.schema.output_parser import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI


def research_agent(topic: str, llm: ChatGoogleGenerativeAI) -> str:
    """
    Research Agent: Gathers information about the given topic
    Args:
        topic (str): The topic to research
        llm (ChatGoogleGenerativeAI): The LLM instance
    Returns:
        str: Comprehensive research notes
    """
    research_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
        You are a Research Agent specialized in gathering comprehensive information about technical topics.
        Your task is to provide detailed research notes that will be used to write a technical report.

        For the given topic, provide:
        1. Key concepts and definitions
        2. Important features or characteristics
        3. Real-world applications and examples
        4. Current trends or best practices
        5. Common challenges or considerations

        Organize your research in a clear, structured format with bullet points."""),
        HumanMessage(content=f"Research topic: {topic}\n\nProvide comprehensive research notes.")
    ])

    chain = research_prompt | llm | StrOutputParser()
    research_notes = chain.invoke({})

    return research_notes
