from typing import Dict, Tuple

def execute_graph(query: str, compiled_graph, memory_obj):
    """
    Run the LangGraph agent with memory and looping checker.
    Returns the final generator answer and metadata.
    """
    inputs = {"query": query, "memory": memory_obj}

    state = compiled_graph.invoke(inputs)

    answer = state.get("answer", "No response.")

    metadata = {
        "topic": state.get("topic_doc", {}).get("topic", "Unknown"),
        "ok": state.get("ok", False),
        "feedback": state.get("feedback", None)
    }

    if hasattr(memory_obj, "update"):
        memory_obj.update({"query": query, "answer": answer})

    return answer, metadata
