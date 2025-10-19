from pydantic import BaseModel


class EvaluateRequest(BaseModel):
    expression: str


class UpdateUserRequest(BaseModel):
    username: str | None = None
    name: str


class UserRequest(BaseModel):
    username: str
    name: str | None = None
