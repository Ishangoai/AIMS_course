import json
import logging
import re
import typing

from langchain_google_genai import ChatGoogleGenerativeAI

from agents.ai_agent.state import EssayState
from agents.ai_agent.tools import research_topic_tool, search_for_topic

logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    The Planner Agent creates comprehensive essay outlines.

    This agent takes a topic and creates a logical, well-structured outline
    for a 2000+ word essay using research and strategic planning.
    """

    def __init__(self, model_name: str = "gemini-2.5-flash", temperature: float = 0.0):
        """
        Initialize the Planner Agent.

        Args:
            model_name: The LLM model to use for planning
            temperature: Temperature for creative but structured planning
        """
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
        self.name = "PlannerAgent"

    def create_planning_prompt(self, topic: str, research_context: str = "") -> str:
        """
        Create a comprehensive prompt for essay planning.

        Args:
            topic: The essay topic
            research_context: Optional research context to inform planning

        Returns:
            str: Formatted prompt for the LLM
        """
        prompt = f"""You are an expert academic planner and essay strategist.
Your task is to create a comprehensive, logical, and detailed outline for a 2000+ word essay.

TOPIC: {topic}

REQUIREMENTS:
1. Create 5-7 main sections (including introduction and conclusion)
2. Each section should be substantial enough for 250-400 words
3. Ensure logical flow from introduction to conclusion
4. Include specific, descriptive section titles
5. Structure should demonstrate deep understanding of the topic

RESEARCH CONTEXT:
{research_context if research_context else "No additional research context provided."}

GUIDELINES:
- Introduction should establish context and thesis
- Body sections should cover different aspects/arguments
- Conclusion should synthesize and provide final insights
- Titles should be specific and engaging
- Structure should support a compelling narrative

OUTPUT FORMAT:
Your response MUST be a valid Python list of strings, where each string is a section title.
Format exactly like this:
["Section Title 1", "Section Title 2", "Section Title 3", ...]

EXAMPLE OUTPUT:
["Introduction: The Digital Revolution in Education", "Current Applications of AI in Learning",
 "Benefits and Transformative Potential", "Challenges and Ethical Considerations",
   "Future Implications and Trends", "Conclusion: Balancing Innovation with Human Values"]

Remember: Output ONLY the Python list, no additional text or formatting."""

        return prompt

    def extract_outline_from_response(self, response: str) -> typing.List[str]:
        """
        Extract the outline list from the LLM response.

        Args:
            response: Raw response from the LLM

        Returns:
            List[str]: Extracted outline sections
        """
        try:
            # Try to find a Python list in the response
            list_pattern = r"\[.*?\]"
            matches = re.findall(list_pattern, response, re.DOTALL)

            if matches:
                # Try to parse the first match as JSON
                list_str = matches[0]
                outline = json.loads(list_str)

                if isinstance(outline, list) and all(isinstance(item, str) for item in outline):
                    logger.info(f"Successfully extracted outline with {len(outline)} sections")
                    return outline

        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to parse outline from response: {e}")

        # Fallback: try to extract lines that look like section titles
        lines = response.strip().split("\n")
        outline = []

        for line in lines:
            line = line.strip()
            # Remove common prefixes and clean up
            line = re.sub(r"^\d+\.?\s*", "", line)  # Remove numbering
            line = re.sub(r"^[-•]\s*", "", line)  # Remove bullet points
            line = line.strip("\"'")  # Remove quotes

            if line and len(line) > 10:  # Reasonable length for a section title
                outline.append(line)

        # Ensure we have a reasonable number of sections
        if len(outline) < 3:
            logger.warning("Generated outline too short, using fallback")
            return self._create_fallback_outline(response.split("\n")[0] if response else "Essay Topic")

        logger.info(f"Extracted outline with {len(outline)} sections using fallback method")
        return outline[:7]  # Limit to 7 sections max

    def _create_fallback_outline(self, topic: str) -> typing.List[str]:
        """
        Create a basic fallback outline if LLM response parsing fails.

        Args:
            topic: The essay topic

        Returns:
            List[str]: Basic outline structure
        """
        return [
            f"Introduction: Understanding {topic}",
            f"Background and Context of {topic}",
            "Current State and Applications",
            "Benefits and Opportunities",
            "Challenges and Considerations",
            "Future Implications and Trends",
            "Conclusion: The Path Forward",
        ]

    def conduct_research(self, topic: str) -> str:
        """
        Conduct research on the topic to inform planning.

        Args:
            topic: The essay topic to research

        Returns:
            str: Research summary to include in planning context
        """
        try:
            # Use the research tool for comprehensive research
            research_results = research_topic_tool.invoke(
                {
                    "topic": topic,
                    "num_queries": 2,  # Limit to 2 queries for faster planning
                }
            )

            # Summarize research findings
            research_summary = "RESEARCH FINDINGS:\n"
            for query, results in research_results.items():
                if not results.startswith("Error"):
                    # Take first 200 characters of each result
                    summary = results[:200] + "..." if len(results) > 200 else results
                    research_summary += f"- {query}: {summary}\n"

            return research_summary

        except Exception as e:
            logger.warning(f"Research failed: {e}")
            # Fallback to simple search
            try:
                simple_result = search_for_topic(topic)
                return f"RESEARCH CONTEXT:\n{simple_result[:300]}..."
            except Exception as e2:
                logger.warning(f"Simple search also failed: {e2}")
                return "No research context available."

    def plan_essay(self, state: EssayState) -> EssayState:
        """
        Main method to plan the essay outline.

        This is the core agent function that will be called by LangGraph.

        Args:
            state: Current essay state

        Returns:
            EssayState: Updated state with outline
        """
        logger.info(f"PlannerAgent starting to plan essay for topic: {state['topic']}")

        try:
            # Step 1: Conduct research
            research_context = self.conduct_research(state["topic"])

            # Step 2: Create the planning prompt
            prompt = self.create_planning_prompt(state["topic"], research_context)

            # Step 3: Get outline from LLM
            response = self.llm.invoke(prompt)
            response_text = str(response.content) if hasattr(response, "content") else str(response)
            outline = self.extract_outline_from_response(response_text)

            # Step 4: Update state
            updated_state = state.copy()
            updated_state["outline"] = outline
            updated_state["current_step"] = "writing"
            updated_state["messages"].append(
                {"role": "assistant", "content": f"Created outline with {len(outline)} sections: {', '.join(outline)}"}
            )

            logger.info(f"PlannerAgent completed successfully. Outline: {outline}")
            return updated_state

        except Exception as e:
            logger.error(f"PlannerAgent failed: {e}")

            # Add error to state and create fallback outline
            updated_state = state.copy()
            updated_state["errors"].append(f"Planning error: {str(e)}")
            updated_state["outline"] = self._create_fallback_outline(state["topic"])
            updated_state["current_step"] = "writing"
            updated_state["warnings"].append("Used fallback outline due to planning error")

            return updated_state


# Create the agent function for LangGraph integration
def planner_agent(state: EssayState) -> EssayState:
    """
    LangGraph node function for the Planner Agent.

    Args:
        state: Current essay state
    Returns:
        EssayState: Updated state with essay outline
    """
    agent = PlannerAgent()
    return agent.plan_essay(state)
