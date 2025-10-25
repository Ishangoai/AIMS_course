from typing import List, TypedDict

from pydantic import BaseModel, Field


class PlanSchema(BaseModel):
    """Result of the planning phase"""

    success: bool = Field(description="Status of the planning phase")
    content: str = Field(description="Detailed outline of the report or reason of failure")


class CriticSchema(BaseModel):
    """Review of the report"""

    approval: bool = Field(description="Decision of the critic for the approval of the report")
    review: str = Field(description="Review and suggestion of the critic in case of non-approval")


class ReportState(TypedDict):
    topic: str
    outline: PlanSchema
    search_contents: List[str]
    report: str
    critic_review: CriticSchema
    iterations: int
