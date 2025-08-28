import typing

from langgraph.graph import add_messages


class EssayState(typing.TypedDict):
    """
    The shared state for the multi-agent essay composition system.
    This state object is passed between all agents and contains all the
    information needed to track the essay creation process from initial
    topic to final output.
    """

    # Core essay information
    topic: str
    """The initial user-provided topic for the essay."""

    # Planning phase
    outline: typing.List[str]
    """A list of chapter or section titles created by the Planner agent."""

    # Writing phase
    draft_sections: typing.Dict[str, str]
    """
    A dictionary where keys are section titles and values are the written
    text for that section. Created and populated by the Writer agent.
    """

    # Research context
    section_research: typing.Dict[str, str]
    """
    Research context for each section, where keys are section titles
    and values are research findings to inform section writing.
    """

    # Review and critique phase
    critique: typing.List[str]
    """
    A list of feedback points, suggestions, or identified inconsistencies
    created by the Critic agent.
    """

    revision_notes: typing.List[str]
    """
    Specific instructions for the writer on what to change, provided by
    the Critic agent for targeted improvements.
    """

    # Metrics and tracking
    word_count: int
    """Current word count of the essay, calculated by utility tools."""

    target_word_count: int
    """Target word count for the essay (default: 2000+ words)."""

    # Final output
    final_essay: typing.Optional[str]
    """The compiled, complete essay text ready for delivery."""

    # Workflow control
    current_step: str
    """
    Tracks the current step in the workflow:
    - 'planning': Creating outline
    - 'writing': Drafting sections
    - 'reviewing': Critique and feedback
    - 'revising': Making improvements
    - 'finalizing': Compiling final essay
    - 'complete': Process finished
    """

    current_section_index: int
    """Index of the section currently being written (0-based)."""

    iteration_count: int
    """Number of revision iterations completed."""

    max_iterations: int
    """Maximum number of revision iterations allowed."""

    # Agent messages and communication
    messages: typing.Annotated[typing.List[typing.Dict], add_messages]
    """
    Messages exchanged between agents for coordination and status updates.
    Uses LangGraph's add_messages reducer for proper message handling.
    """

    # Quality control
    quality_score: typing.Optional[float]
    """
    Quality assessment score (0-10) assigned by the Critic agent.
    Used to determine if the essay meets quality standards.
    """

    minimum_quality_threshold: float
    """Minimum quality score required to complete the essay (default: 7.0)."""

    # Error handling and status
    errors: typing.List[str]
    """List of errors encountered during the process."""

    warnings: typing.List[str]
    """List of warnings or non-critical issues identified."""

    is_complete: bool
    """Flag indicating whether the essay composition process is complete."""


def create_initial_state(
    topic: str, target_word_count: int = 2000, max_iterations: int = 3, minimum_quality_threshold: float = 7.0
) -> EssayState:
    """
    Factory function to create an initial state for essay composition.

    Args:
        topic: The essay topic provided by the user
        target_word_count: Target word count for the essay
        max_iterations: Maximum number of revision iterations
        minimum_quality_threshold: Minimum quality score required

    Returns:
        EssayState: Initialized state object
    """
    return EssayState(
        topic=topic,
        outline=[],
        draft_sections={},
        section_research={},
        critique=[],
        revision_notes=[],
        word_count=0,
        target_word_count=target_word_count,
        final_essay=None,
        current_step="planning",
        current_section_index=0,
        iteration_count=0,
        max_iterations=max_iterations,
        messages=[],
        quality_score=None,
        minimum_quality_threshold=minimum_quality_threshold,
        errors=[],
        warnings=[],
        is_complete=False,
    )


def get_state_summary(state: EssayState) -> str:
    """
    Generate a human-readable summary of the current state.

    Args:
        state: Current essay state

    Returns:
        str: Formatted summary of the state
    """
    summary = f"""
Essay Composition State Summary:
================================
Topic: {state["topic"]}
Current Step: {state["current_step"]}
Word Count: {state["word_count"]}/{state["target_word_count"]}
Sections Completed: {len(state["draft_sections"])}/{len(state["outline"])}
Iteration: {state["iteration_count"]}/{state["max_iterations"]}
Quality Score: {state.get("quality_score", "Not assessed")}
Status: {"Complete" if state["is_complete"] else "In Progress"}

Outline Sections: {", ".join(state["outline"]) if state["outline"] else "Not created"}
Critiques: {len(state["critique"])} points
Revision Notes: {len(state["revision_notes"])} notes
Errors: {len(state["errors"])}
Warnings: {len(state["warnings"])}
"""
    return summary.strip()


# Export the main state type for use in agents
__all__ = ["EssayState", "create_initial_state", "get_state_summary"]
