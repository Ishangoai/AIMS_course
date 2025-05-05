from pydantic import BaseModel


class EvaluateRequest(BaseModel):
    expression: str


class UpdateUserRequest(BaseModel):
    name: str


class UserRequest(BaseModel):
    username: str
    name: str | None = None
