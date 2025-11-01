from typing import Dict, List, TypedDict


# State definition
class ArticleState(TypedDict):
    """State object tracking the article generation workflow."""
    topic: str
    outline: str
    research_sources: List[Dict[str, str]]
    article_draft: str
    validation_feedback: str
    is_valid: bool
    revision_count: int
    word_count: int

    # runtime article word count limits
    target_words: int
    tolerance: int
    min_words: int
    max_words: int
    max_revisions: int
