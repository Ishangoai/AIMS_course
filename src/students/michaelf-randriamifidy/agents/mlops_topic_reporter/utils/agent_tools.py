import logging
import os
import re

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Get API key
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
GOOGLE_CSE_ID = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# Initialize the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-lite",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.1
)


# ========================
# CONFIG
# ========================
TARGET_WORDS = 1000
TOLERANCE = 50
EXPERT_FIELD = "Machine Learning Pipeline (MLOPS)"
ALLOWED_TOPICS = ["CI/CD",
                "RestFullAPI and Docker",
                "Data Engineering",
                "end to end ML Lifecycle",
                "best practices in Python",
                "Gradio",
                "Distributed and Single Thread Data manipulation"
                "MLFlow and Dagster",
                "AI Agents",
                "Machine Learning : Overview",
                "Deployment of ML Models"
                ]

logger = logging.getLogger(__name__)


# ========================
# Utility: Count words ignoring Markdown
# ========================
@tool
def count_words(text: str) -> int:
    """
    Count words in Markdown format string
    """
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`.*?`', '', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'[#>*\-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return len(text.strip().split())


# ========================
# Web Search Tool
# ========================
search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)


@tool
def search_google(query: str) -> str:
    """Search Google for latest information on a topic."""
    logger.info(f"Searching Google for: {query}")
    return search.run(query)


# ========================
# Function to create Agent 1
# ========================
def create_topic_extractor_agent():
    topic_schemas = [
        ResponseSchema(name="topic", description="Main topic inferred from the input text."),
        ResponseSchema(name="key_concepts", description="List of key concepts from input text or online search."),
        ResponseSchema(name="message", description="If outside expertise or scope, message explaining it")
    ]
    topic_parser = StructuredOutputParser.from_response_schemas(topic_schemas)

    topic_prompt = ChatPromptTemplate.from_template(f"""
    You are TopicExtractorAgent.
    You are an expert only in the field: {EXPERT_FIELD}.

    Instructions:
    1. Extract the main topic from the input text.
    2. Only forward tasks if the topic is in the allowed topics list: {ALLOWED_TOPICS}.
    3. Extract key concepts from input text or online search.
    4. Always return a JSON with keys: 'topic', 'key_concepts', 'message'.
    5. If the topic is outside scope, set 'message' to explain it; else, use empty string.

    Text: {{text}}

    {{format_instructions}}
    """)

    chain = LLMChain(
        llm=llm,
        prompt=topic_prompt,
        output_parser=topic_parser,
        output_key="topic_data"
    )
    return chain, topic_parser


# ========================
# Function to create Agent 2
# ========================
def create_research_agent():
    research_schemas = [
        ResponseSchema(name="facts", description="List of concise factual statements with sources")
    ]
    parser = StructuredOutputParser.from_response_schemas(research_schemas)
    prompt = ChatPromptTemplate.from_template("""
You are ResearchAgent.
You have access to the `search_google` tool.

Task:
1. Use Google search to find information about the topic and key concepts.
2. Summarize retrieved information into concise, factual statements.
3. Return JSON with key 'facts', each fact containing a statement and source.

Input topic_data: {topic_data}

{format_instructions}
""")
    return LLMChain(llm=llm, prompt=prompt, output_parser=parser, output_key="research_data"), parser


# ========================
# Function to create Agent 3
# ========================
def create_report_writer_agent():
    report_schemas = [
        ResponseSchema(name="report_text", description=f"A report in Markdown format (~{TARGET_WORDS} words)"),
        ResponseSchema(name="length", description="Word count of the report")
    ]
    parser = StructuredOutputParser.from_response_schemas(report_schemas)
    prompt = ChatPromptTemplate.from_template(f"""
You are ReportWriterAgent.
Write a Markdown report with structure:

# Title
## Introduction
## Main Body (with sub-titles for key concepts)
## Conclusion

Input: topic_data={{topic_data}}, research_data={{research_data}}

Target ~{TARGET_WORDS} words and make it really near that to within {TOLERANCE} words.
Do not excede the autorized threshold of words!!!


{{format_instructions}}
""")
    return LLMChain(llm=llm, prompt=prompt, output_parser=parser, output_key="report_data"), parser


# ========================
# Function to create Agent 4
# ========================
def create_fact_checker_agent():
    factcheck_schemas = [
        ResponseSchema(name="status", description="Pass/Fail if facts are correct"),
        ResponseSchema(name="comments", description="Notes on factual issues"),
        ResponseSchema(name="corrected_report_text", description="Report with corrections if needed"),
        ResponseSchema(name="issues_found", description="List of issues found")
    ]
    parser = StructuredOutputParser.from_response_schemas(factcheck_schemas)
    prompt = ChatPromptTemplate.from_template("""
You are FactCheckerAgent.
Review the report text and check all statements against provided facts.

Input:
Report: {report_data}
Facts: {research_data}

Return JSON keys: 'status', 'comments', 'corrected_report_text', 'issues_found'

{format_instructions}
""")
    return LLMChain(llm=llm, prompt=prompt, output_parser=parser, output_key="factcheck_data"), parser


# ========================
# Function to create Agent 5
# ========================
def create_report_reviewer_agent():
    review_schemas = [
        ResponseSchema(name="status", description="Pass/Fail if report meets criteria"),
        ResponseSchema(name="comments", description="Feedback on issues"),
        ResponseSchema(name="final_text", description="Final report text"),
        ResponseSchema(name="word_count", description="Word count of the report text")
    ]
    parser = StructuredOutputParser.from_response_schemas(review_schemas)
    prompt = ChatPromptTemplate.from_template(f"""
You are ReportReviewerAgent.
Review the corrected report text.

Criteria:

- Markdown structure (Title, Introduction, Main Body, Conclusion)
- Grammar, conciseness, clarity, coherence
- Word count ~{TARGET_WORDS} +/- {TOLERANCE}

Return JSON keys: 'status', 'comments', 'final_text', 'word_count'

Report text: {{corrected_text}}

{{format_instructions}}
""")
    return LLMChain(llm=llm, prompt=prompt, output_parser=parser, output_key="review_data"), parser


def do_report(text: str, max_retries: int = 2):
    # ========================
    # Create agents
    # ========================
    topic_chain, topic_parser = create_topic_extractor_agent()
    research_chain, research_parser = create_research_agent()
    report_chain, report_parser = create_report_writer_agent()
    factcheck_chain, factcheck_parser = create_fact_checker_agent()
    review_chain, review_parser = create_report_reviewer_agent()

    # ========================
    # Step 1: Agent 1 - Topic Extraction
    # ========================
    topic_result = topic_chain.run(text=text, format_instructions=topic_parser.get_format_instructions())

    if topic_result.get("message"):
        # Topic is outside scope or not allowed
        return {
            "topic_data": topic_result,
            "research_data": None,
            "report_data": None,
            "factcheck_data": None,
            "review_data": {
                "status": "Skipped",
                "comments": topic_result["message"],
                "final_text": "I am not able to generate a report on this topic as it is outside my area of expertise.",
                "word_count": 0
            }
        }

    # ========================
    # Step 2: Agent 2 - Research
    # ========================
    research_result = research_chain.run(
        topic_data=topic_result, format_instructions=research_parser.get_format_instructions()
    )

    # ========================
    # Step 3: Agent 3 - Report Writing
    # ========================
    report_result = report_chain.run(
        topic_data=topic_result,
        research_data=research_result,
        format_instructions=report_parser.get_format_instructions()
    )

    # ========================
    # Step 4: Agent 4 - Fact Checking
    # ========================
    factcheck_result = factcheck_chain.run(
        report_data=report_result,
        research_data=research_result,
        format_instructions=factcheck_parser.get_format_instructions()
    )

    corrected_report = factcheck_result.get("corrected_report_text", report_result.get("report_text", ""))

    # ========================
    # Step 5: Agent 5 - Review with retry
    # ========================
    retries = 0
    while retries <= max_retries:
        word_count = count_words(corrected_report)
        review_result = review_chain.run(
            corrected_text=corrected_report,
            format_instructions=review_parser.get_format_instructions()
        )
        report_result["length"] = word_count
        review_result["word_count"] = word_count

        if abs(word_count - TARGET_WORDS) <= TOLERANCE and (
            review_result["status"] == "Pass" or retries == max_retries
            ):
            break  # Exit if passed or max retries reached

        # Retry: regenerate report and re-check
        retries += 1
        report_result = report_chain.run(
            topic_data=topic_result,
            research_data=research_result,
            format_instructions=report_parser.get_format_instructions()
        )
        factcheck_result = factcheck_chain.run(
            report_data=report_result,
            research_data=research_result,
            format_instructions=factcheck_parser.get_format_instructions()
        )
        corrected_report = factcheck_result.get("corrected_report_text", report_result.get("report_text", ""))

    return {
        "topic_data": topic_result,
        "research_data": research_result,
        "report_data": report_result,
        "factcheck_data": factcheck_result,
        "review_data": review_result
    }


if __name__ == "__main__":
    text_input = "I need info about Gradio"
    pipeline_result = do_report(text_input)

    print("\n=== Agent 1: TopicExtractorAgent ===")
    print(pipeline_result["topic_data"])
    print("\n=== Agent 2: ResearchAgent ===")
    print(pipeline_result["research_data"])
    print("\n=== Agent 3: ReportWriterAgent ===")
    print(pipeline_result["report_data"])
    print("\n=== Agent 4: FactCheckerAgent ===")
    print(pipeline_result["factcheck_data"])
    print("\n=== Agent 5: ReportReviewerAgent ===")
    print(pipeline_result["review_data"])
