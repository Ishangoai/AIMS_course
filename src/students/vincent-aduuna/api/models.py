from pydantic import BaseModel


class EvaluateRequest(BaseModel):
    expression: str


class UpdateUserRequest(BaseModel):
    username: str
