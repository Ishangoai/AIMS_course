
from agents.chatReporter.ChatState import ChatState


def execute_graph(query: str, compiled_graph, memory_obj):
    """
    Run the LangGraph agent end-to-end.
    Returns the final generator answer and structured metadata.
    """
    state = ChatState(query=query)

    result_state = compiled_graph.invoke(state)

    answer = result_state.get("last_generator_answer", "No response generated.")
    topic_doc = result_state.get("topic_doc", None)
    topic_name = topic_doc.get("topic") if topic_doc else "General"
    ok_status = result_state.get("ok", False)

    metadata = {
        "topic": topic_name,
        "ok": ok_status,
        "feedback": result_state.get("feedback", None)
    }

    if hasattr(memory_obj, "update"):
        try:
            memory_obj.update({"query": query, "answer": answer})
        except Exception as e:
            print(f"[Warning] Memory update failed: {e}")

    return answer, metadata
