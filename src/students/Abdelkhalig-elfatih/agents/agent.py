import operator
import os
import sys
from typing import Annotated, List, TypedDict

from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class ReportState(TypedDict):
    """State shared across all agents in the workflow"""
    topic: str
    research_data: Annotated[List[str], operator.add]
    outline: str
    sections: Annotated[List[dict], operator.add]
    draft_report: str
    final_report: str
    word_count: int
    feedback: str
    iteration: int


class AgenticReportWriter:
    def __init__(self, google_api_key: str, model: str = "gemini-2.0-flash-exp", temperature: float = 0.7):
        """Initialize the multi-agent system with Google Gemini"""
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            google_api_key=google_api_key,
            temperature=temperature,
            convert_system_message_to_human=True  # Gemini compatibility
        )
        self.graph = self._build_graph()

    @tool
    def count_words(text: str) -> int:
        """Count words in text accurately"""
        return len(text.strip().split())

    def research_agent(self, state: ReportState) -> ReportState:
        """Agent that conducts research on the topic"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research specialist. Your task is to identify key research areas
            for the given topic. Generate 3-5 specific search queries that would help gather
            comprehensive information about this topic.

            Focus on:
            - Core concepts and definitions
            - Current trends and best practices
            - Real-world applications
            - Challenges and solutions

            Return only the search queries, one per line."""),
            ("human", "Topic: {topic}")
        ])

        response = self.llm.invoke(prompt.format_messages(topic=state["topic"]))
        queries = response.content.strip().split('\n')

        # Store research queries
        state["research_data"] = [f"Research query: {q}" for q in queries if q.strip()]
        state["iteration"] = 0

        return state

    def outline_agent(self, state: ReportState) -> ReportState:
        """Agent that creates a structured outline"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert technical writer. Create a detailed outline for a 1000-word
            report on the given topic. The report should follow this structure:

            1. Introduction (150-200 words)
            2. Main Body (650-700 words, split into 3-4 subsections)
            3. Conclusion (150-200 words)

            For each section, provide:
            - Section title
            - Key points to cover
            - Approximate word count

            Format as:
            ## Section Title (word count)
            - Key point 1
            - Key point 2
            ..."""),
            ("human", """Topic: {topic}

            Research areas identified:
            {research}

            Create the outline.""")
        ])

        research_summary = '\n'.join(state["research_data"])
        response = self.llm.invoke(
            prompt.format_messages(
                topic=state["topic"],
                research=research_summary
            )
        )

        state["outline"] = response.content
        return state

    def writing_agent(self, state: ReportState) -> ReportState:
        """Agent that writes the report sections"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert technical writer specializing in {topic_area} topics.
            Write clear, informative, and engaging content.

            Guidelines:
            - Use professional but accessible language
            - Include specific examples and practical applications
            - Ensure technical accuracy
            - Write in a logical, flowing manner
            - Use transition sentences between paragraphs
            - Aim for the specified word count per section"""),
            ("human", """Write the following section of a report:

            Topic: {topic}

            Full Outline:
            {outline}

            Section to write now:
            {section_info}

            Write ONLY this section with its title. Be comprehensive and stay within the word count.""")
        ])

        # Parse outline into sections
        sections = self._parse_outline(state["outline"])

        # Write each section
        all_sections = []
        for i, section in enumerate(sections, 1):
            response = self.llm.invoke(
                prompt.format_messages(
                    topic_area=self._get_topic_area(state["topic"]),
                    topic=state["topic"],
                    outline=state["outline"],
                    section_info=section
                )
            )
            all_sections.append({
                "content": response.content,
                "section_name": section.split('\n')[0]
            })

        # Combine all sections
        state["sections"] = all_sections
        state["draft_report"] = "\n\n".join([s["content"] for s in all_sections])
        state["word_count"] = len(state["draft_report"].strip().split())

        return state

    def quality_agent(self, state: ReportState) -> ReportState:
        """Agent that reviews content for quality and accuracy"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a quality assurance technical specialist. Review the report for:

            1. Factual accuracy - flag any dubious claims
            2. Logical flow - ensure smooth transitions
            3. Completeness - check all outline points are covered
            4. Technical correctness - verify terminology and concepts
            5. Clarity - identify confusing sections

            Provide specific, actionable feedback. If the report is excellent, say "APPROVED"."""),
            ("human", """Topic: {topic}

            Outline:
            {outline}

            Draft Report:
            {draft}

            Word count: {word_count}

            Provide your review.""")
        ])

        response = self.llm.invoke(
            prompt.format_messages(
                topic=state["topic"],
                outline=state["outline"],
                draft=state["draft_report"],
                word_count=state["word_count"]
            )
        )

        state["feedback"] = response.content
        return state

    def editor_agent(self, state: ReportState) -> ReportState:
        """Agent that ensures proper length and formatting"""

        target_min = 950
        target_max = 1050
        current_count = state["word_count"]

        if target_min <= current_count <= target_max and "APPROVED" in state["feedback"]:
            state["final_report"] = state["draft_report"]
            return state

        # Need adjustment

        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert technical editor. Your task is to adjust the report to meet
            the word count requirement (700-900 words) while maintaining quality.

            Current word count: {current}
            Target range: 700-900 words

            If too long: Remove redundancy, tighten prose, get rid of some unnecessary details
            If too short: Add examples, expand explanations

            Quality feedback to address:
            {feedback}

            Return ONLY the final, polished report with proper formatting."""),
            ("human", """Topic: {topic}

            Draft report to edit:
            {draft}

            Make necessary adjustments.""")
        ])

        response = self.llm.invoke(
            prompt.format_messages(
                current=current_count,
                feedback=state["feedback"],
                topic=state["topic"],
                draft=state["draft_report"]
            )
        )

        edited_report = response.content
        new_word_count = len(edited_report.strip().split())

        state["final_report"] = edited_report
        state["word_count"] = new_word_count
        state["iteration"] += 1

        # If still not right and under 3 iterations, go back to quality check
        if not (target_min <= new_word_count <= target_max) and state["iteration"] < 10:
            state["draft_report"] = edited_report

        return state

    def should_continue(self, state: ReportState) -> str:
        """Determine if we need another editing iteration"""
        if state.get("final_report") and state["iteration"] < 10:
            target_min = 950
            target_max = 1050
            if target_min <= state["word_count"] <= target_max:
                return "end"
            else:
                return "continue"
        return "end"

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ReportState)

        # Add nodes
        workflow.add_node("research", self.research_agent)
        workflow.add_node("outline", self.outline_agent)
        workflow.add_node("writing", self.writing_agent)
        workflow.add_node("quality", self.quality_agent)
        workflow.add_node("editor", self.editor_agent)

        # Define edges
        workflow.set_entry_point("research")
        workflow.add_edge("research", "outline")
        workflow.add_edge("outline", "writing")
        workflow.add_edge("writing", "quality")
        workflow.add_edge("quality", "editor")

        # Conditional edge for iteration
        workflow.add_conditional_edges(
            "editor",
            self.should_continue,
            {
                "continue": "quality",
                "end": END
            }
        )

        return workflow.compile()

    def _parse_outline(self, outline: str) -> List[str]:
        """Parse outline into individual sections"""
        sections = []
        current_section = []

        for line in outline.split('\n'):
            if line.strip().startswith('##'):
                if current_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
            elif current_section:
                current_section.append(line)

        if current_section:
            sections.append('\n'.join(current_section))

        return sections

    def _get_topic_area(self, topic: str) -> str:
        """Extract general area from topic"""
        topic_lower = topic.lower()
        if any(word in topic_lower for word in ['ci/cd', 'continuous', 'deployment']):
            return "DevOps and Software Engineering"
        elif any(word in topic_lower for word in ['mlops', 'machine learning', 'ml']):
            return "Machine Learning Operations"
        elif any(word in topic_lower for word in ['api', 'rest', 'graphql']):
            return "API Development"
        elif 'gradio' in topic_lower:
            return "UI/UX and Application Development"
        else:
            return "Software Engineering and Data Science"

    def generate_report(self, topic: str) -> dict:
        """Generate a complete report on the given topic"""

        initial_state = {
            "topic": topic,
            "research_data": [],
            "outline": "",
            "sections": [],
            "draft_report": "",
            "final_report": "",
            "word_count": 0,
            "feedback": "",
            "iteration": 0
        }

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        return {
            "topic": final_state["topic"],
            "report": final_state["final_report"],
            "word_count": final_state["word_count"],
            "outline": final_state["outline"],
            "iterations": final_state["iteration"]
        }
