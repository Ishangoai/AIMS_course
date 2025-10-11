import json
import logging
import os
import typing
from typing import Any, Dict

from agents.ai_agent.agents.planner import planner_agent
from agents.ai_agent.state import EssayState, create_initial_state
from agents.ai_agent.tools.counter import count_words_in_sections, word_counter_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

logger = logging.getLogger(__name__)


class EssayOrchestrator:
    """
    LangGraph orchestrator for the multi-agent essay writing system.

    This orchestrator coordinates the flow between:
    1. PlannerAgent - Creates research and outline
    2. WriterAgent - Writes the essay sections
    3. Counter tools - Validates word counts and completion
    """

    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initialize the essay writing orchestrator.

        Args:
            model_name: The LLM model to use for coordination
        """
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.0, google_api_key=os.getenv("GOOGLE_API_KEY"))
        self.graph = self._build_graph()
        self.app = self.graph.compile()

        logger.info("EssayOrchestrator initialized with LangGraph workflow")

    def _build_graph(self) -> StateGraph:
        """
        Build the LangGraph state graph for essay composition.

        Returns:
            StateGraph: Configured graph for essay writing workflow
        """
        # Create the state graph
        graph = StateGraph(EssayState)

        # Add agent nodes
        graph.add_node("planner", self._planner_node)

        # Break down writer into granular steps
        graph.add_node("writer_research", self._writer_research_node)
        graph.add_node("write_section", self._write_section_node)  # Individual section writing
        graph.add_node("writer_compile", self._writer_compile_node)

        graph.add_node("counter", self._counter_node)
        graph.add_node("coordinator", self._coordinator_node)

        # Define the workflow edges
        graph.add_edge("planner", "writer_research")
        graph.add_edge("writer_research", "write_section")

        # write_section will conditionally continue to itself or move to compile
        graph.add_conditional_edges(
            "write_section",
            self._should_continue_writing_sections,
            {
                "continue_section": "write_section",  # Write next section
                "compile": "writer_compile",  # All sections done, compile
            },
        )

        graph.add_edge("writer_compile", "counter")
        graph.add_edge("counter", "coordinator")

        # Coordinator decides whether to continue or end
        graph.add_conditional_edges(
            "coordinator",
            self._should_continue,
            {
                "continue": "writer_research",  # If word count too low, research and write more
                "end": END,  # If satisfied, end the process
            },
        )

        # Set entry point
        graph.set_entry_point("planner")

        return graph

    def _planner_node(self, state: EssayState) -> EssayState:
        """
        Execute the planner agent to create outline and conduct research.

        Args:
            state: Current essay state

        Returns:
            EssayState: Updated state with outline and research
        """
        logger.info("Executing planner node")
        try:
            updated_state = planner_agent(state)
            updated_state["messages"].append({"role": "assistant", "content": "Planning phase completed successfully"})
            return updated_state
        except Exception as e:
            logger.error(f"Planner node failed: {e}")
            state["errors"].append(f"Planner execution error: {str(e)}")
            return state

    def _writer_research_node(self, state: EssayState) -> EssayState:
        """
        Conduct research for each section that needs to be written.

        Args:
            state: Current essay state with outline

        Returns:
            EssayState: Updated state with research context
        """
        logger.info("Executing writer research node")
        try:
            from agents.ai_agent.agents.writer import WriterAgent

            updated_state = state.copy()
            writer = WriterAgent()

            # Get sections that need research
            outline = state.get("outline", [])
            if not outline:
                updated_state["warnings"].append("No outline available for research")
                return updated_state

            # Conduct research for each section
            research_results = {}
            for section_title in outline:
                research = writer.conduct_additional_research(state["topic"], section_title)
                research_results[section_title] = research
                logger.info(f"Research completed for section: {section_title}")

            # Store research in state (add this field to EssayState if needed)
            updated_state["section_research"] = research_results
            updated_state["current_step"] = "researching"
            updated_state["current_section_index"] = 0  # Reset section index for writing
            updated_state["messages"].append(
                {"role": "assistant", "content": f"Research completed for {len(outline)} sections"}
            )

            return updated_state

        except Exception as e:
            logger.error(f"Writer research node failed: {e}")
            state["errors"].append(f"Research execution error: {str(e)}")
            return state

    def _write_section_node(self, state: EssayState) -> EssayState:
        """
        Write a single section of the essay (called iteratively for each section).

        Args:
            state: Current essay state with outline and research

        Returns:
            EssayState: Updated state with the current section written
        """
        logger.info("Executing write_section node")
        try:
            from agents.ai_agent.agents.writer import WriterAgent

            updated_state = state.copy()
            writer = WriterAgent()

            outline = state.get("outline", [])
            research_results = state.get("section_research", {})
            existing_sections = state.get("draft_sections", {})
            current_section_index = state.get("current_section_index", 0)

            if not outline:
                updated_state["errors"].append("No outline available for writing")
                return updated_state

            # Check if we have more sections to write
            if current_section_index >= len(outline):
                logger.info("All sections have been written")
                updated_state["current_step"] = "sections_complete"
                return updated_state

            # Get the current section to write
            section_title = outline[current_section_index]

            logger.info(f"Writing section {current_section_index + 1}/{len(outline)}: {section_title}")

            # Get research context for this section
            research_context = research_results.get(section_title, "")

            # Write the section
            section_content = writer.write_section(
                state["topic"], section_title, outline, current_section_index, research_context
            )

            # Update draft sections
            draft_sections = existing_sections.copy()
            draft_sections[section_title] = section_content

            # Calculate word count for this section
            section_words = word_counter_tool.invoke({"text": section_content})

            # Update state
            updated_state["draft_sections"] = draft_sections
            updated_state["current_section_index"] = current_section_index + 1
            updated_state["current_step"] = "writing_sections"

            # Add progress message
            updated_state["messages"].append(
                {
                    "role": "assistant",
                    "content": f"Completed section {current_section_index + 1}/{len(outline)}: "
                    f"'{section_title}' ({section_words} words)",
                }
            )

            logger.info(f"Successfully wrote section '{section_title}': {section_words} words")
            return updated_state

        except Exception as e:
            logger.error(f"Write section node failed: {e}")
            state["errors"].append(f"Section writing error: {str(e)}")
            return state

    def _should_continue_writing_sections(self, state: EssayState) -> str:
        """
        Determine whether to continue writing more sections or move to compilation.

        Args:
            state: Current essay state

        Returns:
            str: "continue_section" or "compile"
        """
        outline = state.get("outline", [])
        current_section_index = state.get("current_section_index", 0)

        # Check if there are more sections to write
        if current_section_index < len(outline):
            logger.info(f"Continuing to write section {current_section_index + 1}/{len(outline)}")
            return "continue_section"
        else:
            logger.info("All sections written, moving to compilation")
            return "compile"

    def _writer_compile_node(self, state: EssayState) -> EssayState:
        """
        Compile all sections into the final essay format.

        Args:
            state: Current essay state with written sections

        Returns:
            EssayState: Updated state with compiled final essay
        """
        logger.info("Executing writer compile node")
        try:
            updated_state = state.copy()

            outline = state.get("outline", [])
            draft_sections = state.get("draft_sections", {})

            if not outline or not draft_sections:
                updated_state["warnings"].append("Missing outline or sections for compilation")
                return updated_state

            # Compile sections in order
            final_essay_parts = []
            compiled_sections = 0

            for section_title in outline:
                if section_title in draft_sections:
                    final_essay_parts.append(f"## {section_title}\n\n{draft_sections[section_title]}\n")
                    compiled_sections += 1
                else:
                    logger.warning(f"Section '{section_title}' not found in draft_sections")

            updated_state["final_essay"] = "\n".join(final_essay_parts)
            updated_state["current_step"] = "compiled"
            updated_state["messages"].append(
                {
                    "role": "assistant",
                    "content": f"Essay compilation completed. {compiled_sections}/{len(outline)} sections compiled.",
                }
            )

            return updated_state

        except Exception as e:
            logger.error(f"Writer compile node failed: {e}")
            state["errors"].append(f"Compilation error: {str(e)}")
            return state

    def _counter_node(self, state: EssayState) -> EssayState:
        """
        Execute word counting and validation using counter tools.

        Args:
            state: Current essay state with written content

        Returns:
            EssayState: Updated state with accurate word counts
        """
        logger.info("Executing counter node")
        try:
            state["current_step"] = "counting"
            # Use counter tools to get accurate statistics
            if state.get("draft_sections"):
                # Count words in all sections
                count_results = count_words_in_sections.invoke(json.dumps(state["draft_sections"]))
                state["word_count"] = count_results["total_words"]
                state["current_step"] = "counting_complete"

                # Add analysis message with section details
                state["messages"].append(
                    {
                        "role": "assistant",
                        "content": (
                            f"Word count analysis complete: {count_results['total_words']} total words "
                            f"across {count_results['sections_completed']} sections. "
                            f"Section breakdown: {count_results['section_counts']}"
                        ),
                    }
                )

                logger.info(f"Counter analysis: {count_results['total_words']} total words")
            else:
                state["warnings"].append("No draft sections found for counting")
                state["current_step"] = "counting_no_content"

        except Exception as e:
            logger.error(f"Counter node failed: {e}")
            state["errors"].append(f"Counter execution error: {str(e)}")
            state["current_step"] = "counting_failed"

        return state

    def _coordinator_node(self, state: EssayState) -> EssayState:
        """
        Coordinate the overall essay writing process and make decisions.

        Args:
            state: Current essay state

        Returns:
            EssayState: Updated state with coordination decisions
        """
        logger.info("Executing coordinator node")

        current_word_count = state.get("word_count", 0)
        target_word_count = state.get("target_word_count", 2000)
        completion_threshold = target_word_count * 0.9  # 90% of target

        updated_state = state.copy()

        # Determine if essay meets requirements
        if current_word_count >= completion_threshold:
            updated_state["is_complete"] = True
            updated_state["current_step"] = "complete"
            updated_state["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        f"Essay completion criteria met: {current_word_count} words (target: {target_word_count})"
                    ),
                }
            )
            logger.info(f"Essay completed with {current_word_count} words")
        else:
            updated_state["is_complete"] = False
            updated_state["current_step"] = "revision"
            updated_state["messages"].append(
                {
                    "role": "assistant",
                    "content": (
                        f"Essay needs more content: {current_word_count} words "
                        f"(target: {target_word_count}). Requesting additional writing."
                    ),
                }
            )
            logger.info(f"Essay needs more content: {current_word_count}/{target_word_count} words")

        return updated_state

    def _should_continue(self, state: EssayState) -> str:
        """
        Determine whether to continue writing or end the process.

        Args:
            state: Current essay state
        Returns:
            str: "continue" or "end"
        """
        is_complete = state.get("is_complete", False)

        # Check completion criteria
        if is_complete:
            logger.info("Essay meets completion criteria - ending workflow")
            return "end"

        if state["iteration_count"] >= state["max_iterations"]:
            logger.warning(f"Maximum iterations reached ({state['max_iterations']}) - ending workflow")
            state["warnings"].append(f"Reached maximum iterations {state['max_iterations']}")
            return "end"

        logger.info(f"Continuing essay writing (iteration {state['iteration_count'] + 1})")
        state["iteration_count"] += 1
        return "continue"

    def create_initial_state(
        self, topic: str, target_word_count: int = 2000, max_iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Create initial state for the orchestrator.

        Args:
            topic: Essay topic
            target_word_count: Target word count
            max_iterations: Maximum iterations

        Returns:
            Dict[str, Any]: Initial state dictionary
        """
        initial_state = create_initial_state(
            topic=topic, target_word_count=target_word_count, max_iterations=max_iterations
        )

        # Convert to dict for streaming
        return dict(initial_state)

    def create_essay_stream(self, topic: str, target_word_count: int = 2000, max_iterations: int = 3):
        logger.info(f"Starting streaming essay creation for topic: '{topic}'")
        initial_state = create_initial_state(
            topic=topic, target_word_count=target_word_count, max_iterations=max_iterations
        )
        initial_state["messages"].append({"role": "user", "content": f"Essay creation started for topic: '{topic}'"})

        return initial_state

    def get_essay_summary(self, state: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
        """
        Get a summary of the essay creation process and results.

        Args:
            state: Final essay state

        Returns:
            Dict: Summary information about the essay
        """
        return {
            "topic": state.get("topic", "Unknown"),
            "word_count": state.get("word_count", 0),
            "target_word_count": state.get("target_word_count", 2000),
            "sections_completed": len(state.get("draft_sections", {})),
            "outline": state.get("outline", []),
            "is_complete": state.get("is_complete", False),
            "iterations": state.get("iteration_count", 0),
            "errors": state.get("errors", []),
            "warnings": state.get("warnings", []),
            "has_final_essay": bool(state.get("final_essay")),
        }


# Factory function for easy instantiation
def create_essay_orchestrator(model_name: str = "gemini-2.5-flash") -> EssayOrchestrator:
    """
    Factory function to create an essay orchestrator.

    Args:
        model_name: The LLM model to use

    Returns:
        EssayOrchestrator: Configured orchestrator instance
    """
    return EssayOrchestrator(model_name=model_name)
