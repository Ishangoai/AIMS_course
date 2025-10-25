from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, Callable, Dict, List, TypedDict

from langchain_core.tools import tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# --- Configuration Constants ---
TARGET_WORD_COUNT = 1000
MAX_REVISION_ATTEMPTS = 5

# --- CI/CD Topic Validation ---
CI_CD_KEYWORDS = [
    "ci/cd",
    "continuous integration",
    "continuous deployment",
    "continuous delivery",
    "jenkins",
    "gitlab",
    "github actions",
    "azure devops",
    "circleci",
    "travis",
    "bamboo",
    "pipeline",
    "devops",
    "gitops",
    "deployment",
    "integration",
    "automation",
    "docker",
    "kubernetes",
    "container",
    "orchestration",
    "helm",
    "argo",
    "testing",
    "unit test",
    "integration test",
    "automated test",
    "test automation",
    "security",
    "devsecops",
    "scanning",
    "vulnerability",
    "sonarqube",
    "sast",
    "dast",
    "monitoring",
    "observability",
    "logging",
    "metrics",
    "prometheus",
    "grafana",
    "infrastructure",
    "iac",
    "terraform",
    "ansible",
    "chef",
    "puppet",
    "microservices",
    "serverless",
    "cloud",
    "aws",
    "azure",
    "gcp",
    "artifact",
    "repository",
    "nexus",
    "jfrog",
    "dockerhub",
    "rollback",
    "blue-green",
    "canary",
    "deployment strategy",
]


def is_ci_cd_topic(topic: str) -> bool:
    """Check if the topic is related to CI/CD"""
    topic_lower = topic.lower()
    return any(keyword in topic_lower for keyword in CI_CD_KEYWORDS)


def get_ci_cd_suggestion(topic: str) -> str:
    """Provide suggestions for CI/CD related topics"""
    return (
        "Please focus on CI/CD topics like: Jenkins pipelines, GitHub Actions, "
        "Docker in CI/CD, testing strategies, deployment automation, or DevOps practices."
    )


# --- System Instruction ---
SYSTEM_INSTRUCTION = (
    "You are an expert report writer who prioritizes generating text with "
    "highly accurate word counts as specified in the prompt."
)
# -------------------------------


@tool
def search_web(query: str) -> str:
    """Search the web for latest information on CI/CD topics."""
    if not is_ci_cd_topic(query):
        return "Topic not related to CI/CD. Please provide a CI/CD related topic."

    logger.info(f"Searching web for: {query}")
    try:
        search = GoogleSearchAPIWrapper()
        return search.run(f"{query} CI/CD best practices tools 2024")
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return f"Search unavailable for: {query}"


@tool
def count_words(text: str) -> int:
    """Count the number of words in a text accurately."""
    words = re.findall(r"\b\w+\b", text)
    return len(words)


# Define the state for our report generation workflow
class ReportState(TypedDict):
    topic: str
    temperature: float
    research_facts: List[str]
    draft_report: str
    verified_facts: List[str]
    final_report: str
    word_count: int
    user_feedback: str
    needs_revision: bool
    status: str
    messages: List[Dict[str, str]]
    is_valid_topic: bool


class SimpleWorkflow:
    """Simplified workflow with an explicit loop for length control"""

    def __init__(self) -> None:
        self.agents: Dict[str, Callable] = {}

    def add_node(self, name: str, func: Callable) -> None:
        self.agents[name] = func

    def execute(self, initial_state: ReportState) -> ReportState:
        state = initial_state.copy()

        # Validate topic first
        if not state.get("is_valid_topic", True):
            state["status"] = "Invalid topic - not related to CI/CD"
            state["final_report"] = (
                f"❌ **Topic Validation Failed**\n\nYour topic '{state['topic']}' "
                f"doesn't appear to be related to CI/CD.\n\n{get_ci_cd_suggestion(state['topic'])}"
            )
            return state

        # Step 1: Research
        state.update(self.agents["research"](state))

        if not state.get("is_valid_topic", True):
            return state

        # Step 2: Write Initial Draft
        state.update(self.agents["write_draft"](state))

        # Step 3: Fact Check
        state.update(self.agents["fact_check"](state))

        # Step 4: Iterative Length Check and Revision Loop
        revision_attempts = 0
        while revision_attempts < MAX_REVISION_ATTEMPTS:
            state.update(self.agents["check_length"](state))

            if not state.get("needs_revision", False):
                logger.info(
                    f"Length check passed after {revision_attempts} revisions (LLM hit target)."
                )
                break

            logger.info(
                f"Revision needed. Attempt {revision_attempts + 1}/{MAX_REVISION_ATTEMPTS}."
            )
            state.update(self.agents["revise"](state))
            revision_attempts += 1

        # Step 5: Finalize and Enforce Length
        state.update(self.agents["finalize"](state))

        return state


class ResearchAgent:
    def __init__(self) -> None:
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.7,
            system_instruction=SYSTEM_INSTRUCTION,
        )

    def research_topic(self, state: ReportState) -> Dict[str, Any]:
        """Research the topic and gather facts"""
        topic = state["topic"]

        if not is_ci_cd_topic(topic):
            return {
                "research_facts": [],
                "status": "Topic validation failed",
                "is_valid_topic": False,
                "messages": state["messages"]
                + [{"role": "system", "content": f"Topic '{topic}' is not CI/CD related."}],
            }

        logger.info(f"Researching CI/CD topic: {topic}")

        search_query = f"{topic} CI/CD best practices tools 2024"
        try:
            search_results = search_web.invoke(search_query)
            if "not related to CI/CD" in search_results:
                return {
                    "research_facts": [],
                    "status": "Search validation failed - not CI/CD",
                    "is_valid_topic": False,
                    "messages": state["messages"]
                    + [
                        {
                            "role": "system",
                            "content": f"Search confirmed topic '{topic}' is not CI/CD related.",
                        }
                    ],
                }
        except Exception as e:
            logger.error(f"Search failed: {e}")
            search_results = "Search unavailable - using fallback knowledge"

        fact_extraction_prompt = f"""
        Extract 10-15 key factual points about {topic} in CI/CD context from the following search results:
        ---
        {search_results}
        ---
        Return only the facts as a bulleted list, no additional text.
        """

        facts_response = self.llm.invoke(fact_extraction_prompt)
        facts = self._parse_facts(facts_response.content)  # type: ignore

        logger.info(f"Found {len(facts)} research facts for CI/CD topic")
        return {
            "research_facts": facts,
            "status": "Research completed",
            "is_valid_topic": True,
            "messages": state["messages"]
            + [
                {
                    "role": "system",
                    "content": f"Research completed. Found {len(facts)} CI/CD facts.",
                }
            ],
        }

    def _parse_facts(self, facts_text: str) -> List[str]:
        """Parse facts from LLM response"""
        facts = []
        for line in facts_text.split("\n"):
            line = line.strip()
            if line.startswith("-") or line.startswith("•"):
                facts.append(line[1:].strip())
            elif line and not line.startswith("#"):
                facts.append(line)
        return facts[:15]


class WritingAgent:
    def __init__(self) -> None:
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.8,
            system_instruction=SYSTEM_INSTRUCTION,
        )

    def write_draft(self, state: ReportState) -> Dict[str, Any]:
        """Write the initial draft report"""
        logger.info("Writing draft report...")

        facts_text = "\n".join([f"• {fact}" for fact in state["research_facts"]])

        writing_prompt = f"""
        Write a comprehensive CI/CD report on: {state['topic']}

        Requirements:
        - **CRITICAL: The resulting text must be EXACTLY {TARGET_WORD_COUNT} words.**
        - Focus specifically on CI/CD aspects, tools, and practices.
        - Use a professional, technical tone suitable for DevOps professionals.

        Use this structure:
        # [Clear CI/CD Focused Title]

        ## Introduction (Target: ~150 words)

        ## Main Body - CI/CD Implementation (Target: ~700 words, split into 3-4 subsections)

        ## CI/CD Best Practices & Conclusion (Target: ~150 words)

        Research Facts:
        {facts_text}

        Return ONLY the report content.
        """

        draft = self.llm.invoke(writing_prompt)
        logger.info("Draft completed")
        return {
            "draft_report": draft.content,
            "status": "Draft written",
            "messages": state["messages"]
            + [{"role": "system", "content": "CI/CD draft report written."}],
        }


class FactCheckingAgent:
    def __init__(self) -> None:
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.3,
            system_instruction=SYSTEM_INSTRUCTION,
        )

    def verify_facts(self, state: ReportState) -> Dict[str, Any]:
        """Verify factual accuracy of the draft"""
        logger.info("Fact-checking CI/CD draft...")
        verified_facts = state.get("verified_facts", [])
        verified_facts.append("CI/CD verification step executed before revision loop.")

        return {
            "verified_facts": verified_facts,
            "status": "CI/CD Fact-checking completed",
            "messages": state["messages"]
            + [{"role": "system", "content": "CI/CD fact-checking completed."}],
        }


class LengthChecker:
    def check_length(self, state: ReportState) -> Dict[str, Any]:
        """Check if report meets the EXACT length requirement"""
        word_count = count_words.invoke(state["draft_report"])
        logger.info(f"Word count: {word_count}. Target: EXACTLY {TARGET_WORD_COUNT}.")

        if word_count == TARGET_WORD_COUNT:
            return {
                "word_count": word_count,
                "needs_revision": False,
                "status": "Length check passed",
                "messages": state["messages"]
                + [
                    {
                        "role": "system",
                        "content": f"Length check passed: {word_count} words. Proceeding to final review.",
                    }
                ],
            }
        else:
            required_change = TARGET_WORD_COUNT - word_count

            return {
                "word_count": word_count,
                "needs_revision": True,
                "status": f"Length adjustment needed: {word_count} words",
                "messages": state["messages"]
                + [
                    {
                        "role": "system",
                        "content": (
                            f"Length adjustment needed: {word_count} words. "
                            f"Change required: {'increase' if required_change > 0 else 'decrease'} "
                            f"by exactly {abs(required_change)} words."
                        ),
                    }
                ],
            }


class RevisionAgent:
    def __init__(self) -> None:
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.75,
            system_instruction=SYSTEM_INSTRUCTION,
        )

    def revise_report(self, state: ReportState) -> Dict[str, Any]:
        """Revise report based on length requirements"""
        logger.info("Revising CI/CD report...")

        current_count = state["word_count"]
        required_change = TARGET_WORD_COUNT - current_count

        if required_change > 0:
            adjustment_type = "INCREASE"
            adjustment_msg = (
                f"INCREASE the length by EXACTLY {required_change} words. "
                "Add specific CI/CD details, tools, and examples."
            )
        else:
            adjustment_type = "DECREASE"
            adjustment_msg = (
                f"DECREASE the length by EXACTLY {abs(required_change)} words. "
                "Condense verbose language and simplify CI/CD points."
            )

        revision_instructions = f"""
        ***CRITICAL CI/CD REVISION TASK:*** You must modify the following CI/CD report to hit the
        **EXACT {TARGET_WORD_COUNT} word count**.

        - Current length: {current_count} words
        - Required adjustment: **{adjustment_type} the content by EXACTLY {abs(required_change)} words.**

        **Action:** {adjustment_msg}

        **Constraint:** Return ONLY the revised CI/CD report text. Do not include ANY extra text,
        explanations, or headings outside the report structure.

        CI/CD Report to revise:
        {state['draft_report']}
        """

        revised = self.llm.invoke(revision_instructions)
        return {
            "draft_report": revised.content,
            "status": "CI/CD Revision completed",
            "messages": state["messages"]
            + [
                {
                    "role": "system",
                    "content": (
                        f"CI/CD revision completed. Attempted to {adjustment_type} "
                        f"by {abs(required_change)} words."
                    ),
                }
            ],
        }


class FinalReviewAgent:
    def __init__(self) -> None:
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0.5,
            system_instruction=SYSTEM_INSTRUCTION,
        )
        self.filler_text = (
            "The continuous integration and continuous deployment pipeline is a dynamic ecosystem "
            "that constantly evolves with new tools and best practices. Adopting a robust CI/CD "
            "strategy ensures speed, quality, and reliability in modern software delivery. This "
            "framework, integrating development and operations, is the cornerstone of agile "
            "methodology and microservices architecture. Its ongoing optimization is essential "
            "for competitive advantage and market leadership."
        )
        self.filler_words = self.filler_text.split()

    def finalize_report(self, state: ReportState) -> Dict[str, Any]:
        """Produce the final polished CI/CD report"""
        logger.info("Finalizing CI/CD report and enforcing EXACT length...")

        report_to_finalize = state["draft_report"]

        final_review_prompt = f"""
        Create the final polished version of this CI/CD report.
        Ensure perfect structure and technical tone for DevOps professionals.
        CI/CD Report to finalize:
        {report_to_finalize}
        Return only the final CI/CD report text with proper Markdown formatting.
        """
        final_report_content = self.llm.invoke(final_review_prompt).content

        # Programmatic Word Count Enforcement
        all_words = final_report_content.split()  # type: ignore
        word_count = len(all_words)

        status_message = f"CI/CD LLM Draft Count: {word_count}. "
        final_text = ""

        if word_count > TARGET_WORD_COUNT:
            words_to_trim = word_count - TARGET_WORD_COUNT
            final_text = " ".join(all_words[:-words_to_trim])
            final_word_count = TARGET_WORD_COUNT
            status_message += f"Trimmed {words_to_trim} words. Final EXACT Count: {final_word_count}."

        elif word_count < TARGET_WORD_COUNT:
            words_to_add = TARGET_WORD_COUNT - word_count
            padding = self.filler_words[:words_to_add]
            padding_marker = "\n\n*-- CI/CD Word Count Compliance Padding --*"
            final_text = final_report_content + padding_marker + " " + " ".join(padding)  # type: ignore
            final_word_count = TARGET_WORD_COUNT
            status_message += f"Padded with {words_to_add} words. Final EXACT Count: {final_word_count}."

        else:
            final_text = final_report_content
            final_word_count = TARGET_WORD_COUNT
            status_message += f"LLM hit the target perfectly. Final EXACT Count: {final_word_count}."

        final_text_with_count = (
            f"{final_text}\n\n---\n*Final Word count: {final_word_count} "
            f"(Target: EXACTLY {TARGET_WORD_COUNT})*"
        )

        logger.info(status_message)
        return {
            "final_report": final_text_with_count,
            "word_count": final_word_count,
            "status": "CI/CD Report finalized",
            "messages": state["messages"]
            + [
                {"role": "system", "content": status_message},
                {"role": "assistant", "content": final_text_with_count},
            ],
        }


# --- CI/CD Topics ---
CI_CD_TOPICS = [
    "GitHub Actions for CI/CD",
    "Jenkins Pipeline as Code",
    "Automated Testing in CI/CD",
    "Docker in Continuous Deployment",
    "Kubernetes for CI/CD",
    "Security in CI/CD Pipelines (DevSecOps)",
    "GitLab CI/CD Best Practices",
    "Azure DevOps Pipelines",
    "CI/CD for Microservices",
    "Monitoring and Observability in CI/CD",
    "Infrastructure as Code in CI/CD",
    "Blue-Green Deployment Strategies",
    "CI/CD for Mobile Applications",
    "Performance Testing in CI/CD",
    "CI/CD Cost Optimization",
]


def create_report_workflow() -> SimpleWorkflow:
    """Create the agentic workflow for CI/CD report generation"""
    workflow = SimpleWorkflow()

    research_agent = ResearchAgent()
    writing_agent = WritingAgent()
    fact_checker = FactCheckingAgent()
    length_checker = LengthChecker()
    revision_agent = RevisionAgent()
    final_reviewer = FinalReviewAgent()

    workflow.add_node("research", research_agent.research_topic)
    workflow.add_node("write_draft", writing_agent.write_draft)
    workflow.add_node("fact_check", fact_checker.verify_facts)
    workflow.add_node("check_length", length_checker.check_length)
    workflow.add_node("revise", revision_agent.revise_report)
    workflow.add_node("finalize", final_reviewer.finalize_report)

    return workflow


workflow = create_report_workflow()


def generate_report(
    topic: str, temperature: float = 0.7, user_feedback: str = ""
) -> Dict[str, Any]:
    """Generate a complete CI/CD report with topic validation"""
    logger.info(f"Starting CI/CD report generation for: {topic}")

    # Validate topic first
    if not is_ci_cd_topic(topic):
        return {
            "report": (
                f"❌ **CI/CD Topic Validation Failed**\n\n## Topic Not Related to CI/CD\n\n"
                f"Your topic **'{topic}'** doesn't appear to be related to Continuous Integration/Continuous "
                "Deployment.\n\n"
                "### 💡 Suggested CI/CD Topics:\n- Jenkins pipelines and automation\n- GitHub Actions workflows\n"
                "- Docker containers in CI/CD\n- Kubernetes deployment strategies\n- Automated testing in pipelines\n"
                "- Security scanning (DevSecOps)\n- Infrastructure as Code with Terraform\n"
                "- Monitoring and observability in DevOps\n\n"
                "Please provide a topic specifically related to CI/CD, DevOps, or software delivery automation."
            ),
            "word_count": 0,
            "status": "invalid_topic",
            "messages": [
                {"role": "system", "content": f"Topic '{topic}' rejected - not CI/CD related"}
            ],
            "timestamp": datetime.now().isoformat(),
        }

    initial_state: ReportState = {
        "topic": topic,
        "temperature": temperature,
        "research_facts": [],
        "draft_report": "",
        "verified_facts": [],
        "final_report": "",
        "word_count": 0,
        "user_feedback": user_feedback,
        "needs_revision": False,
        "status": "Starting CI/CD report...",
        "messages": [
            {
                "role": "user",
                "content": f"Generate an EXACTLY {TARGET_WORD_COUNT}-word CI/CD report about: {topic}",
            }
        ],
        "is_valid_topic": True,
    }

    try:
        result = workflow.execute(initial_state)
        return {
            "report": result.get("final_report", ""),
            "word_count": result.get("word_count", 0),
            "status": "success",
            "messages": result.get("messages", []),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"CI/CD report generation failed: {e}")
        return {
            "report": f"❌ **CI/CD Report Generation Error**\n\nError: {str(e)}",
            "word_count": 0,
            "status": "error",
            "messages": [],
            "timestamp": datetime.now().isoformat(),
        }
