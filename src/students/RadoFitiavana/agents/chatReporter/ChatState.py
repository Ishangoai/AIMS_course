from pydantic import BaseModel
from typing import Optional

class ChatState(BaseModel):
    query: str
    iteration_count: int = 0
    topic_doc: Optional[dict] = None
    context: Optional[str] = None
    feedback: Optional[str] = None
    last_generator_answer: Optional[str] = None
    answer: Optional[str] = None
    ok: bool = False
