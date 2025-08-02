"""
Agents package for the essay composition system.

This package contains all the specialized agents that work together
to compose essays through a multi-agent workflow.
"""

from .planner import PlannerAgent, planner_agent
from .writer import WriterAgent, writer_agent

# Export agents for easy access
__all__ = [
    "PlannerAgent",
    "planner_agent",
    "WriterAgent",
    "writer_agent",
]
