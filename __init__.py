# report_agent_app/__init__.py
"""
MLOps ReportGen Pro - Agentic Report Generation System

A multi-agent system for generating high-quality technical reports
on MLOps topics using competitive refinement with specialized AI agents.

Authors: Fandresena & Vicent
Course: MLOps
Date: January 2025
"""

__version__ = "1.0.0"
__author__ = "Fandresena & Vicent"

# Import main components for easy access
from .fandresena_agent import (
    MAX_WORDS,
    MIN_WORDS,
    TARGET_WORDS,
    count_words,
    generate_essay,
    generate_essay_with_context,
)
from .gradioapp import launch, report_app
from .web_search import get_search_context, search_google

__all__ = [
    'generate_essay_with_context',
    'generate_essay',
    'count_words',
    'launch',
    'report_app',
    'get_search_context',
    'search_google',
    'MIN_WORDS',
    'MAX_WORDS',
    'TARGET_WORDS'
]
