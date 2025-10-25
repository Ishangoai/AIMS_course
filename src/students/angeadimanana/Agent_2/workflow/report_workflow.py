"""
Report Workflow Module
Orchestrates the agentic workflow with feedback loop
"""

from typing import Any, Dict

from agents.editor_agent import editor_agent
from agents.fact_checker_agent import fact_checker_agent
from agents.qa_agent import qa_agent
from agents.research_agent import research_agent
from agents.writer_agent import writer_agent
from config import MAX_REVISION_ITERATIONS, QA_THRESHOLD
from langchain_google_genai import ChatGoogleGenerativeAI

from utils.helpers import count_words, extract_qa_score, is_report_approved


class ReportWorkflow:
    """
    Orchestrates the multi-agent workflow for report generation
    """

    def __init__(self, llm: ChatGoogleGenerativeAI):
        """
        Initialize the workflow with an LLM instance
        Args:
            llm (ChatGoogleGenerativeAI): The LLM instance to use
        """
        self.llm = llm

    def run(self, topic: str, verbose: bool = True) -> Dict[str, Any]:
        """
        Run the complete workflow with feedback loop
        Args:
            topic (str): The topic for the report
            verbose (bool): Whether to print progress information
        Returns:
            Dict[str, Any]: Results including final report, scores, and metadata
        """

        # Step 1: Research
        research_notes = research_agent(topic, self.llm)
        if verbose:
            print(f"✓ Research completed ({count_words(research_notes)} words)")

        # Step 2: Initial Draft
        current_draft = writer_agent(topic, research_notes, self.llm)
        if verbose:
            print(f"✓ Draft created ({count_words(current_draft)} words)")

        # Revision loop
        iteration = 0
        qa_score = 0.0
        qa_results = ""

        while iteration < MAX_REVISION_ITERATIONS:
            iteration += 1
            # Step 3: Fact-Checking
            fact_check_results = fact_checker_agent(
                topic,
                current_draft,
                research_notes,
                self.llm
            )

            # Step 4: Editing
            edited_report = editor_agent(
                topic,
                current_draft,
                fact_check_results,
                self.llm
            )

            # Step 5: QA Evaluation
            qa_results = qa_agent(
                topic,
                edited_report,
                research_notes,
                self.llm
            )
            qa_score = extract_qa_score(qa_results)

            # Check if approved
            if is_report_approved(qa_score, QA_THRESHOLD):
                return {
                    'final_report': edited_report,
                    'qa_score': qa_score,
                    'qa_results': qa_results,
                    'iterations': iteration,
                    'word_count': count_words(edited_report),
                    'research_notes': research_notes,
                    'status': 'APPROVED'
                }

            # If not approved and not last iteration, prepare for revision
            if iteration < MAX_REVISION_ITERATIONS:
                # Step 6: Revise based on feedback
                current_draft = writer_agent(
                    topic,
                    research_notes,
                    self.llm,
                    feedback=qa_results,
                    previous_draft=edited_report
                )
            else:
                if verbose:
                    print("\n⚠️  Maximum iterations reached")

        # If we exit the loop without approval
        if verbose:
            print("WORKFLOW COMPLETED")
            print(f"Final Score: {qa_score:.2f}/1.0")
            print(f"Status: {'APPROVED' if qa_score >= QA_THRESHOLD else 'NEEDS IMPROVEMENT'}")
            print(f"{'=' * 60}")

        return {
            'final_report': edited_report,
            'qa_score': qa_score,
            'qa_results': qa_results,
            'iterations': iteration,
            'word_count': count_words(edited_report),
            'research_notes': research_notes,
            'status': 'NEEDS_IMPROVEMENT' if qa_score < QA_THRESHOLD else 'APPROVED'
        }
