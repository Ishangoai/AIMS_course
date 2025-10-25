import os
import logging
import re
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.chains import LLMChain
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import tool
from langchain.utilities import GoogleSearchAPIWrapper

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
ALLOWED_TOPICS = [
    "CI/CD",
    "RestFullAPI and Docker",
    "Data Engineering",
    "end to end ML Lifecycle",
    "best practices in Python",
    "Gradio",
    "Distributed and Single Thread Data manipulation",
    "MLFlow and Dagster",
    "AI Agents",
]

logger = logging.getLogger(__name__)

search = GoogleSearchAPIWrapper(google_api_key=GOOGLE_API_KEY, google_cse_id=GOOGLE_CSE_ID)

# ========================
# TOOLS
# ========================
@tool
def count_words(text: str) -> int:
    """Count words in Markdown text, ignoring code blocks and markup."""
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`.*?`', '', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'[#>*\-]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return len(text.strip().split())

@tool
def search_google(query: str) -> str:
    """Search Google for the latest information on a topic."""
    logger.info(f"Searching Google for: {query}")
    return search.run(query)


# ========================
# AGENT 1: TopicExtractor
# ========================
def create_topic_extractor_agent():
    topic_schemas = [
        ResponseSchema(name="topic", description="Main topic inferred from the input text."),
        ResponseSchema(name="key_concepts", description="List of key concepts from input text or online search."),
        ResponseSchema(name="message", description="If outside expertise or scope, message explaining it"),
    ]
    topic_parser = StructuredOutputParser.from_response_schemas(topic_schemas)

    topic_prompt = ChatPromptTemplate.from_template(f"""
You are Agent 1: TopicExtractorAgent.
You are an expert only in the field: {EXPERT_FIELD}.

Instructions:
1. Extract the main topic from the input text.
2. Only forward tasks if the topic is in the allowed topics list: {ALLOWED_TOPICS}.
3. Extract key concepts from input text or online search.
4. Always return a JSON with keys: 'topic', 'key_concepts', 'message'.
5. If the topic is outside scope, set 'message' to explain it; else, empty string.

Text: {{text}}

{{{{format_instructions}}}}
""")
    chain = LLMChain(llm=llm, prompt=topic_prompt, output_parser=topic_parser, output_key="topic_data")
    return chain, topic_parser


# ========================
# AGENT 2: ResearchAgent
# ========================
def create_research_agent():
    research_schemas = [
        ResponseSchema(name="facts", description="List of concise factual statements with sources")
    ]
    parser = StructuredOutputParser.from_response_schemas(research_schemas)
    tools = [search_google]

    prompt = ChatPromptTemplate.from_template("""
You are Agent 2: ResearchAgent.
You have access to these tools:
{tools}

Instructions:
1. Use `search_google` to find information about the topic and key concepts.
2. Summarize each fact (20–50 words) with its source.
3. Return JSON with key 'facts'.

Input topic data:
{input}

{{format_instructions}}

Available Tools: {tool_names}

{agent_scratchpad}
""")

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=tools,
                             verbose=False,
                             handle_parsing_errors=True,
                             max_iterations=3)
    return executor, parser


# ========================
# AGENT 3: ReportWriterAgent
# ========================
def create_report_writer_agent():
    report_schemas = [
        ResponseSchema(name="report_text", description=f"A Markdown report (~{TARGET_WORDS} words)"),
        ResponseSchema(name="length", description="Word count of the report"),
    ]
    parser = StructuredOutputParser.from_response_schemas(report_schemas)
    tools = [count_words]

    prompt = ChatPromptTemplate.from_template(f"""
You are Agent 3: ReportWriterAgent.
You have access to these tools:
{{tools}}

Task:
1. Write a structured Markdown report.
2. Use the word counter to estimate report length.
3. Include: # Title, ## Introduction, ## Main Body, ## Conclusion.
4. Target ≈ {TARGET_WORDS} words.

Input:
{{input}}

{{{{format_instructions}}}}

Available Tools: {{tool_names}}

{{agent_scratchpad}}
""")

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=tools,
                             verbose=False,
                             handle_parsing_errors=True,
                             max_iterations=1)
    return executor, parser


# ========================
# AGENT 4: FactCheckerAgent
# ========================
def create_fact_checker_agent():
    factcheck_schemas = [
        ResponseSchema(name="status", description="Pass/Fail if all facts are correct"),
        ResponseSchema(name="comments", description="Notes about factual issues"),
        ResponseSchema(name="corrected_report_text", description="Corrected report if needed"),
        ResponseSchema(name="issues_found", description="List of factual issues"),
    ]
    parser = StructuredOutputParser.from_response_schemas(factcheck_schemas)
    tools = [search_google]

    prompt = ChatPromptTemplate.from_template("""
You are Agent 4: FactCheckerAgent.
You have access to fact-verification tools:
{tools}

Instructions:
1. Check the given report text against research facts.
2. Optionally use `search_google` to verify uncertain claims.
3. Output JSON: 'status', 'comments', 'corrected_report_text', 'issues_found'.

Input:
{input}

{{format_instructions}}

Available Tools: {tool_names}

{agent_scratchpad}
""")

    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=tools,
                             verbose=False,
                             handle_parsing_errors=True,
                             max_iterations=1)
    return executor, parser


# ========================
# AGENT 5: ReportReviewerAgent
# ========================
def create_report_reviewer_agent():
    review_schemas = [
        ResponseSchema(name="status", description="Pass/Fail if report meets criteria"),
        ResponseSchema(name="comments", description="Feedback on issues"),
        ResponseSchema(name="final_text", description="Final report text"),
        ResponseSchema(name="word_count", description="Word count of the report text"),
    ]
    parser = StructuredOutputParser.from_response_schemas(review_schemas)
    prompt = ChatPromptTemplate.from_template(f"""
You are Agent 5: ReportReviewerAgent.
Review the corrected report text.

Criteria:
- Markdown structure (Title, Introduction, Main Body, Conclusion)
- Grammar, conciseness, clarity, coherence
- Word count ~{TARGET_WORDS} +/- {TOLERANCE}

Return JSON keys: 'status', 'comments', 'final_text', 'word_count'

Report text: {{corrected_text}}

{{{{format_instructions}}}}
""")
    chain = LLMChain(llm=llm, prompt=prompt, output_parser=parser, output_key="review_data")
    return chain, parser


# ========================
# PIPELINE RUNNER
# ========================
def do_report(text: str, max_retries: int = 2):
    topic_chain, topic_parser = create_topic_extractor_agent()
    research_executor, research_parser = create_research_agent()
    report_executor, report_parser = create_report_writer_agent()
    factcheck_executor, factcheck_parser = create_fact_checker_agent()
    review_chain, review_parser = create_report_reviewer_agent()

    # Step 1: Topic Extraction
    topic_result = topic_chain.run(text=text, format_instructions=topic_parser.get_format_instructions())
    if topic_result.get("message"):
        return {
            "topic_data": topic_result,
            "research_data": None,
            "report_data": None,
            "factcheck_data": None,
            "review_data": {"status": "Skipped", "comments": topic_result["message"], "final_text": "", "word_count": 0},
        }

    # Step 2: Research
    research_result = research_executor.invoke({"input": topic_result})

    # Step 3: Report Writing
    report_result = report_executor.invoke({"input": {"topic_data": topic_result, "research_data": research_result}})

    # Step 4: Fact Checking
    factcheck_result = factcheck_executor.invoke({"input": {"report_data": report_result, "research_data": research_result}})
    corrected_report = factcheck_result.get("corrected_report_text", report_result.get("report_text", ""))

    # Step 5: Review with retry
    retries = 0
    while retries <= max_retries:
        word_count = count_words(corrected_report)
        review_result = review_chain.run(corrected_text=corrected_report, format_instructions=review_parser.get_format_instructions())
        if abs(word_count - TARGET_WORDS) <= TOLERANCE and (review_result["status"] == "Pass" or retries == max_retries):
            break
        retries += 1
        report_result = report_executor.invoke({"input": {"topic_data": topic_result, "research_data": research_result}})
        factcheck_result = factcheck_executor.invoke({"input": {"report_data": report_result, "research_data": research_result}})
        corrected_report = factcheck_result.get("corrected_report_text", report_result.get("report_text", ""))

    return {
        "iteration": retries,
        "topic_data": topic_result,
        "research_data": research_result,
        "report_data": report_result,
        "factcheck_data": factcheck_result,
        "review_data": review_result,
    }


# ========================
# MAIN
# ========================
if __name__ == "__main__":
    text_input = "I need info about Gradio"
    result = do_report(text_input)

    print("\n=== Agent 1: TopicExtractorAgent ===")
    print(result["topic_data"])
    print("\n=== Agent 2: ResearchAgent ===")
    print(result["research_data"])
    print("\n=== Agent 3: ReportWriterAgent ===")
    print(result["report_data"])
    print("\n=== Agent 4: FactCheckerAgent ===")
    print(result["factcheck_data"])
    print("\n=== Agent 5: ReportReviewerAgent ===")
    print(result["review_data"])
