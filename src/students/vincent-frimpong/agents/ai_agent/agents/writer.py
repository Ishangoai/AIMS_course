import logging
import typing

from langchain_google_genai import ChatGoogleGenerativeAI

from agents.ai_agent.state import EssayState
from agents.ai_agent.tools import search_for_topic, word_counter_tool

logger = logging.getLogger(__name__)


class WriterAgent:
    """
    The Writer Agent transforms essay outlines into comprehensive written content.

    This agent takes the structured outline from the Planner Agent and writes
    detailed, well-researched sections for a 2000+ word essay.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.3):
        """
        Initialize the Writer Agent.

        Args:
            model_name: The LLM model to use for writing
            temperature: Temperature for creative but coherent writing
        """
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        self.name = "WriterAgent"

    def create_section_prompt(
        self,
        topic: str,
        section_title: str,
        outline: typing.List[str],
        current_section_index: int,
        research_context: str = "",
        target_words: int = 350,
    ) -> str:
        """
        Create a detailed prompt for writing a specific essay section.

        Args:
            topic: The main essay topic
            section_title: The title of the current section to write
            outline: The complete essay outline
            current_section_index: Index of the current section in the outline
            research_context: Research information to inform writing
            target_words: Target word count for this section

        Returns:
            str: Formatted prompt for the LLM
        """
        # Determine section type and context
        is_introduction = current_section_index == 0
        is_conclusion = current_section_index == len(outline) - 1
        previous_section = outline[current_section_index - 1] if current_section_index > 0 else None
        next_section = outline[current_section_index + 1] if current_section_index < len(outline) - 1 else None

        prompt = f"""You are an expert academic writer specializing in comprehensive essays.
Your task is to write a detailed, well-researched section for a {len(outline)}-part essay.

ESSAY TOPIC: {topic}

COMPLETE ESSAY OUTLINE:
        {chr(10).join([f"{i + 1}. {section}" for i, section in enumerate(outline)])}CURRENT SECTION TO WRITE:
{current_section_index + 1}. {section_title}

TARGET LENGTH: {target_words} words (aim for substantial, detailed content)

SECTION CONTEXT:
- Previous section: {previous_section if previous_section else "N/A (This is the introduction)"}
- Next section: {next_section if next_section else "N/A (This is the conclusion)"}

RESEARCH CONTEXT:
{research_context if research_context else "Use your knowledge to provide accurate, detailed information."}

WRITING GUIDELINES:
{"- Establish context, present your thesis, and provide a roadmap for the essay" if is_introduction else ""}
{"- Synthesize key arguments, provide final insights, and offer a compelling ending" if is_conclusion else ""}
{"- Develop the specific aspect covered in this section title" if not is_introduction and not is_conclusion else ""}
- Use clear, engaging academic prose
- Include specific examples, data, or evidence where relevant
- Ensure smooth transitions from the previous section
- Maintain logical flow toward the next section
- Write in a scholarly but accessible tone
- Include topic sentences and clear paragraph structure
- Aim for {target_words} words with substantive content

OUTPUT FORMAT:
Write ONLY the section content. Do not include the section title, numbering, or any meta-commentary.
Begin directly with the first paragraph of this section.

IMPORTANT:
Focus specifically on the aspect indicated by the section title "{section_title}" while maintaining coherence
with the overall essay structure.
"""

        return prompt

    def write_section(
        self,
        topic: str,
        section_title: str,
        outline: typing.List[str],
        current_section_index: int,
        research_context: str = "",
    ) -> str:
        """
        Write a specific section of the essay.

        Args:
            topic: The main essay topic
            section_title: The title of the section to write
            outline: The complete essay outline
            current_section_index: Index of current section
            research_context: Research context for informed writing

        Returns:
            str: The written section content
        """
        # Calculate target words based on total sections
        target_words = max(250, 2000 // len(outline)) if outline else 350
        try:
            # Create the writing prompt
            prompt = self.create_section_prompt(
                topic, section_title, outline, current_section_index, research_context, target_words
            )

            # Generate the section content
            response = self.llm.invoke(prompt)
            section_content = str(response.content) if hasattr(response, "content") else str(response)

            # Use counter tool for accurate word count
            actual_word_count = word_counter_tool.invoke({"text": section_content})
            logger.info(f"Successfully wrote section: {section_title} ({actual_word_count} words)")
            return section_content.strip()

        except Exception as e:
            logger.error(f"Failed to write section '{section_title}': {e}")
            # Return a fallback section
            return self._create_fallback_section(section_title, target_words)

    def _create_fallback_section(self, section_title: str, target_words: int = 300) -> str:
        """
        Create a basic fallback section if writing fails.

        Args:
            section_title: The section title
            target_words: Target word count

        Returns:
            str: Basic section content
        """
        return f"""This section focuses on {section_title.lower()}. Due to technical limitations,
this is a placeholder section that should be expanded with detailed analysis, examples,
and supporting evidence. The content should explore the key themes and arguments
related to this aspect of the topic, providing substantial insight and analysis
to meet the target length of approximately {target_words} words."""

    def conduct_additional_research(self, topic: str, section_title: str) -> str:
        """
        Conduct targeted research for a specific section.

        Args:
            topic: Main essay topic
            section_title: Specific section being written

        Returns:
            str: Research context for the section
        """
        try:
            # Create a focused search query
            search_query = f"{topic} {section_title}"
            research_result = search_for_topic(search_query)
            return f"SECTION-SPECIFIC RESEARCH:\n{research_result[:400]}..."
        except Exception as e:
            logger.warning(f"Additional research failed for {section_title}: {e}")
            return "No additional research available for this section."

    def write_essay(self, state: EssayState) -> EssayState:
        """
        Main method to write the complete essay from the outline.

        This is the core agent function that will be called by LangGraph.

        Args:
            state: Current essay state with outline from planner

        Returns:
            EssayState: Updated state with written sections
        """
        logger.info(f"WriterAgent starting to write essay for topic: {state['topic']}")

        if not state.get("outline"):
            logger.error("No outline found in state. Writer requires planner output.")
            updated_state = state.copy()
            updated_state["errors"].append("Writer error: No outline provided by planner")
            return updated_state

        try:
            updated_state = state.copy()
            outline = state["outline"]
            draft_sections = {}
            total_word_count = 0

            # Write each section
            for i, section_title in enumerate(outline):
                logger.info(f"Writing section {i + 1}/{len(outline)}: {section_title}")

                # Conduct additional research for this specific section
                section_research = self.conduct_additional_research(state["topic"], section_title)

                # Write the section
                section_content = self.write_section(state["topic"], section_title, outline, i, section_research)

                draft_sections[section_title] = section_content
                # Use counter tool for accurate word count
                word_count = word_counter_tool.invoke({"text": section_content})
                total_word_count += word_count

                logger.info(f"Completed section '{section_title}': {word_count} words")

            # Update state with written content
            updated_state["draft_sections"] = draft_sections
            updated_state["word_count"] = total_word_count
            updated_state["current_step"] = "review"
            updated_state["messages"].append(
                {
                    "role": "assistant",
                    "content": f"Completed essay draft with {len(draft_sections)} sections"
                    f" totaling {total_word_count} words",
                }
            )

            # Create final essay by combining sections
            final_essay_parts = []
            for section_title in outline:
                final_essay_parts.append(f"## {section_title}\n\n{draft_sections[section_title]}\n")

            updated_state["final_essay"] = "\n".join(final_essay_parts)

            logger.info(f"WriterAgent completed successfully. Total words: {total_word_count}")
            return updated_state

        except Exception as e:
            logger.error(f"WriterAgent failed: {e}")

            # Add error to state
            updated_state = state.copy()
            updated_state["errors"].append(f"Writing error: {str(e)}")
            updated_state["warnings"].append("Essay writing encountered errors")

            return updated_state


# Create the agent function for LangGraph integration
def writer_agent(state: EssayState) -> EssayState:
    """
    LangGraph node function for the Writer Agent.

    Args:
        state: Current essay state with outline from planner
    Returns:
        EssayState: Updated state with written essay content
    """
    agent = WriterAgent()
    return agent.write_essay(state)
