"""
Agentic Report Generation System

A multi-agent system for automatically generating technical reports
with validation, revision, and quality checking.

Usage:
    from report_generator import run_agent

    result = run_agent("MLOps Best Practices")
    print(result['final_report'])
"""

from .agent import run_agent, visualize_graph
from .tools import count_words, validate_structure

__version__ = "1.0.0"
__all__ = ["run_agent", "visualize_graph", "count_words", "validate_structure"]
