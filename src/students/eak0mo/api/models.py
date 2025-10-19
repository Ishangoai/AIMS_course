from typing import List

from pydantic import BaseModel


class EvaluateRequest(BaseModel):
    expression: str


class UpdateUserRequest(BaseModel):
    username: str | None = None
    name: str


class UserRequest(BaseModel):
    username: str
    name: str | None = None


class FraudPredictionRequest(BaseModel):
    features: List[float]  # 28 features + Amount = 29 values


class FraudPredictionResponse(BaseModel):
    prediction: int
    probability: float
