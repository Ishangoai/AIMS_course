from pydantic import BaseModel


class EvaluateRequest(BaseModel):
    expression: str


class UserRequest(BaseModel):
    username: str
    name: str | None = None
