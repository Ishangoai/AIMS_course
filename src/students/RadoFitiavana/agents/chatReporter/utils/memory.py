import langchain.memory as memory
from . import model_router as router

from typing import Dict

def create_memory(session_id: str, memory_store: Dict):
    """
    Creates per-session memory.
    If conversation grows too large, auto-summarize it.
    """
    if session_id in memory_store:
        return memory_store[session_id] 

    summarizer = router.get_model("summarizer")
    _memory = memory.ConversationSummaryBufferMemory(
        llm=summarizer,
        max_token_limit=1000,
        memory_key="history",
        return_messages=True
    )
    return _memory
