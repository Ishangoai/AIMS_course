"""
Configuration file for the agentic report system
"""

# LLM Configuration
GOOGLE_API_KEY = "AIzaSyConloRzNoTi14lPOp9fr7G_95VUxu19EA"
MODEL_NAME = "gemini-2.0-flash-lite"
TEMPERATURE = 0.7

# Report Configuration
TARGET_WORD_COUNT = 1000
WORD_COUNT_TOLERANCE = 50
MIN_WORD_COUNT = TARGET_WORD_COUNT - WORD_COUNT_TOLERANCE
MAX_WORD_COUNT = TARGET_WORD_COUNT + WORD_COUNT_TOLERANCE

# QA Configuration
QA_THRESHOLD = 0.999  # Minimum score to approve report
MAX_REVISION_ITERATIONS = 5  # Maximum number of revision loops
