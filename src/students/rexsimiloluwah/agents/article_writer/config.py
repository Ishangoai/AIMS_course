import os
from dataclasses import dataclass

from dotenv import find_dotenv, load_dotenv

_ = load_dotenv(find_dotenv())


class Config:
    """Central configuration"""
    MODEL_FALLBACK = "gemini-1.5-flash"
    LLM_CALL_MAX_RETRIES = 3

    # Paths
    PROMPTS_PATH = "./prompts"

    @classmethod
    def get_api_key(cls) -> str:
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Set GEMINI_API_KEY or GOOGLE_API_KEY environment variable")
        return key


@dataclass
class RuntimeConfig:
    model_name: str = "gemini-2.0-flash-exp"
    writer_temperature: float = 0.45
    rewriter_temperature: float = 0.25
    validator_temperature: float = 0.20

    # word limits
    target_words: int = 1000
    tolerance: int = 50
    max_revisions: int = 5

    # optional caps
    writer_max_tokens: int | None = 2048
    rewriter_max_tokens: int | None = 2048
    validator_max_tokens: int | None = 2048


def merge_runtime(overrides: dict | None) -> RuntimeConfig:
    base = RuntimeConfig()
    if overrides:
        for k, v in overrides.items():
            if hasattr(base, k):
                setattr(base, k, v)
    return base
