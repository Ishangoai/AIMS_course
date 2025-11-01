from typing import Dict, List, TypedDict


# State definition
class ArticleState(TypedDict):
    """State object tracking the article generation workflow."""
    topic: str

    # Planning
    outline: str

    # Research
    research_sources: List[Dict[str, str]]

    # Writing
    article_draft: str
    revision_count: int
    word_count: int

    # Validation - Word Counter
    word_count_valid: bool
    word_count_feedback: str

    # Validation - Quality
    quality_valid: bool
    quality_feedback: str

    # Validation - combined
    validation_feedback: str
    is_valid: bool

    # runtime article word count limits
    target_words: int
    tolerance: int
    min_words: int
    max_words: int
    max_revisions: int
