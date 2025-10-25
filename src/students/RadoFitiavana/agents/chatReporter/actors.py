import langchain.prompts as prompts
import langchain.schema.runnable as runnable 
import .utils.model_router as router 
import .utils.tools as tools 
import json

# Load topics
with open("data/topics.json", "r") as f:
    TOPICS = json.load(f)

# Context Classifier
def make_classifier():
    def classify(inputs):
        query = inputs["query"].lower()
        for topic in TOPICS:
            if topic["topic"].lower() in query:
                return {"topic_doc": topic}
        return {"topic_doc": None}
    return runnable.RunnableLambda(classify)

# Topic Retriever
def make_retriever():
    def retrieve(inputs):
        topic_doc = inputs.get("topic_doc")
        if not topic_doc:
            return {"context": "No specific MLOps topic found in your question."}
        return {"context": f"Topic: {topic_doc['topic']}\nDefinition: {topic_doc['definition']}\nUse cases: {', '.join(topic_doc['use_cases'])}\nTools: {', '.join(topic_doc['examples_of_tools'])}"}
    return runnable.RunnableLambda(retrieve)

# Answer/Report Generator
def make_generator():
    llm = router.get_model("generator")
    prompt = prompts.ChatPromptTemplate.from_template("""
You are an expert MLOps assistant.
Write a detailed, analytical report (~1000 ±50 words) about the topic below.
Output JSON:
{{
  "topic": "<topic_name>",
  "report": "<~1000-word text>"
}}

Context:
{context}

User Query:
{query}
""")
    return prompt | llm

# Report Checker
def make_checker():
    llm = router.get_model("checker").bind_tools([tools.count_words])
    prompt = prompts.ChatPromptTemplate.from_template("""
You are a quality checker reviewing a report.
Check:
1. Grammar & coherence
2. Logical flow
3. Word count 950–1050 (use `count_words` tool)

Report:
{answer}

If all checks pass, respond with "PASS". Else, explain why.
""")

    def check(inputs):
        answer = inputs["answer"]
        msg = prompt.format(answer=answer)
        res = llm.invoke(msg)
        text = res.content.strip()

        if hasattr(res, "tool_calls") and res.tool_calls:
            for tool_call in res.tool_calls:
                if tool_call["name"] == "count_words":
                    wc = tools.count_words(tool_call["args"]["text"])
                    if wc < 950 or wc > 1050:
                        return {"ok": False, "feedback": f"Word count = {wc} words."}

        if "PASS" in text.upper():
            return {"ok": True}
        return {"ok": False, "feedback": text}

    return runnable.RunnableLambda(check)
