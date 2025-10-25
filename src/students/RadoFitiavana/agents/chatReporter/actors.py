import json
from pathlib import Path

import langchain.prompts as prompts
import langchain.schema.runnable as runnable

from .ChatState import ChatState
from .utils import model_router as router
from .utils import tools

# Load topics once
HERE = Path(__file__).parent
with (HERE / "data" / "topics.json").open("r") as f:
    TOPICS = json.load(f)


def sanitize_output(output):
    """Convert list to string and remove JSON/Markdown fences."""
    if isinstance(output, list):
        output = "\n".join(output)
    return str(output).replace("```json", "").replace("```", "").strip()


def make_classifier() -> runnable.RunnableLambda:
    """
    Classifies the user query into a topic from TOPICS using an LLM.
    Returns None if no topic matches.
    """
    llm = router.get_model("classifier")
    prompt = prompts.ChatPromptTemplate.from_template("""
You are an MLOps expert assistant. Given a user query, determine which topic from the list below
is most relevant. Respond ONLY with the topic name exactly as in the list, or "None"..

Topics:
{topics}

User Query:
{query}

""")

    def classify(state: ChatState):
        inputs = {
            "topics": ", ".join([t["topic"] for t in TOPICS]),
            "query": state.query or ""
        }

        res = llm.invoke(prompt.format(**inputs))
        topic_name = sanitize_output(res.content).strip()
        print(f"Found topic: {topic_name}")

        # Find topic document if matched
        state.topic_doc = next((t for t in TOPICS if t["topic"] == topic_name), None)
        return state

    return runnable.RunnableLambda(classify)


def make_retriever() -> runnable.RunnableLambda:
    """
    Retrieves context for a topic. If no topic is matched, returns None context.
    """
    def retrieve(state: ChatState):
        td = state.topic_doc
        if td:
            state.context = (
                f"Topic: {td['topic']}\nDefinition: {td['definition']}\n"
                f"Use cases: {', '.join(td['use_cases'])}\n"
                f"Tools: {', '.join(td['examples_of_tools'])}"
            )
        else:
            state.context = None
        return state

    return runnable.RunnableLambda(retrieve)


def make_generator() -> runnable.RunnableLambda:
    """
    Generates a professional report of ~1000 words using topic context if available.
    Incorporates previous generator answer and checker feedback to iteratively improve.
    """
    llm = router.get_model("generator")

    prompt = prompts.ChatPromptTemplate.from_template("""
You are an expert MLOps assistant. Write a professional, coherent, and logically structured report
of approximately 800 words. Aim for 800–900 words.

Instructions:
- Use the provided CONTEXT if available.
- If CONTEXT is not available, then just answer the user query with your knowledge
- Answer USER_QUERY in detail.
- If PREVIOUS_REPORT is available, then improve it based on CHECKER_FEEDBACK and return it as your report.
- Ensure clarity, logical flow, and completeness.
- Include examples if relevant.
- Your report should not exceed 900 words
- Output text only, no markdown, no json

CONTEXT:
{context}

USER_QUERY:
{query}

CHECKER_FEEDBACK :
{checker_feedback}

PREVIOUS_REPORT:
{previous_report}
""")

    def generate(state: ChatState):
        inputs = {
            "context": state.context or "None",
            "query": state.query,
            "previous_report": state.last_generator_answer or "None",
            "checker_feedback": state.feedback or "None"
        }

        res = llm.invoke(prompt.format(**inputs))
        text = sanitize_output(res.content)

        state.last_generator_answer = text
        state.iteration_count += 1

        return state

    return runnable.RunnableLambda(generate)


def make_checker() -> runnable.RunnableLambda:
    """
    Checks the generated report for logic, coherence, and word count.
    Provides explicit feedback about how many words to add or remove.
    """
    llm = router.get_model("checker").bind_tools([tools.count_words])
    prompt = prompts.ChatPromptTemplate.from_template("""
You are a quality assurance expert. Review the following report carefully.

Check:
1. Logic and coherence: point out any inconsistencies or unclear sections.
2. Word count: target 950–1050 words. Use the count_words tool to get the exact count.

Instructions:
- If word count < 950: indicate "The following report is too short, increase the number of words."
- If word count > 1050: indicate "The following report is too long, reduce the number of words."
- If the text is mostly nonsense: indicate "The following report is almost nonsense, improve logic and coherence"
- If all checks pass, respond exactly with "PASS".
- Output text only, no markdown, no json

Report:
{report}
""")

    def check(state: ChatState):
        answer_text = state.last_generator_answer
        inputs = {"report": answer_text}
        res = llm.invoke(prompt.format(**inputs))
        state.feedback = sanitize_output(res.content)

        state.ok = False

        wc = tools.count_words(answer_text)
        print(f"Word count: {wc}")

        if "PASS" in state.feedback.upper():
            state.ok = True
            state.answer = state.last_generator_answer
            return state

        return state

    return runnable.RunnableLambda(check)
