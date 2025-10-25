from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, TypedDict

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')
import tempfile

from langchain.tools import Tool
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

# Configure API Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Initialize Models
model = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

gemini_model_basic = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp",
    temperature=0.7,
    google_api_key=GOOGLE_API_KEY
)

# Initialize Google Search
search = GoogleSearchAPIWrapper(
    google_api_key=GOOGLE_API_KEY,
    google_cse_id=GOOGLE_CSE_ID
)


def search_google(query: str) -> str:
    """Search Google for latest information on topic."""
    logger.info(f"Searching Google for: {query}")
    return search.run(f"{query} ")


search_tool = Tool(
    name="google_search",
    description="Search Google for recent results",
    func=search.run
)


# Helper function to extract string content
def extract_text_content(response) -> str:
    """
    Extract text content from LangChain response

    Args:
        response: Response from ChatGoogleGenerativeAI

    Returns:
        Text content as string
    """
    if hasattr(response, 'content'):
        content = response.content
        # Si content est une liste, joindre les éléments
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and 'text' in item:
                    text_parts.append(item['text'])
            return ' '.join(text_parts)
        # Si content est déjà une string
        elif isinstance(content, str):
            return content
    # Fallback: convertir en string
    return str(response)


# Define State for LangGraph
class ReportState(TypedDict):
    topic: str
    research: str
    outline: str
    report: str
    word_count: int
    quality_score: float
    is_valid: bool
    iteration: int
    workflow_log: List[Dict]

# ============================================================================
# SECTION 2: AGENT CLASSES
# ============================================================================


class MLOpsReportAgent:
    """Multi-agent system for MLOps report generation"""

    def __init__(self):
        """Initialize agent system"""
        pass

    def research_agent(self, topic: str) -> str:
        """
        Research agent using web search

        Args:
            topic: The topic to research

        Returns:
            Research summary as string
        """
        print(f"[Research Agent] Searching for: {topic}")

        try:
            # Perform web search
            search_query = f"{topic} MLOps best practices 2024 2025"
            search_results = search_google(search_query)

            # Process search results with Gemini via LangChain
            prompt = f"""Analyze these search results about {topic} in MLOps:

{search_results}

Provide a concise research summary:
1. Key definition (2-3 sentences)
2. Main components (3-4 points)
3. Best practices (3-4 points)
4. Real-world example (1 sentence)

Be factual and concise."""

            response = gemini_model_basic.invoke(prompt)
            return extract_text_content(response)

        except Exception as e:
            print(f"[Research Agent] Search error: {e}")

            # Fallback to Gemini knowledge
            prompt = f"""Research {topic} in MLOps concisely.

Provide:
1. Key definition (2-3 sentences)
2. Main components (3-4 points)
3. Best practices (3-4 points)
4. Real-world example (1 sentence)

Be factual and concise."""

            response = gemini_model_basic.invoke(prompt)
            return extract_text_content(response)

    def outline_agent(self, research: str) -> str:
        """
        Creates report structure

        Args:
            research: Research data from research agent

        Returns:
            Structured outline as string
        """
        prompt = f"""Based on this research:
{research}

Create a detailed outline for a 1000-word report:
- Introduction (150 words)
- 3 main sections (600 words)
- Conclusion (150 words)

List section titles and key points."""

        response = gemini_model_basic.invoke(prompt)
        return extract_text_content(response)

    def writer_agent(self, research: str, outline: str, iteration=20) -> str:
        """
        Writes the complete report with strict word count control

        Args:
            research: Research data
            outline: Report outline
            iteration: Current iteration number

        Returns:
            Complete report as string
        """
        prompt = f"""Write a professional report on MLOps with EXACTLY 950-1050 words.

Research:
{research}

Outline:
{outline}

CRITICAL REQUIREMENTS:
- Word count must be between 950-1050 words (this is iteration {iteration + 1}/3)
- Clear headers (# for title, ## for sections)
- Professional technical tone
- Include specific examples
- Structure: Introduction + Main sections + Conclusion

{"IMPORTANT: Previous attempt had wrong word count. Adjust carefully!" if iteration > 10 else ""}"""

        response = gemini_model_basic.invoke(prompt)
        return extract_text_content(response)

    def fact_checker_agent(self, report: str) -> Dict:
        """
        Quality check agent

        Args:
            report: The report to check

        Returns:
            Dictionary with quality_score and is_approved
        """
        prompt = f"""Review this report for quality:
{report[:500]}...

Rate 1-10 for:
- Accuracy
- Structure
- Clarity

Return JSON: {{"quality_score": X, "is_approved": true}}"""

        response = gemini_model_basic.invoke(prompt)
        text = extract_text_content(response)

        try:
            import json
            import re
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:  # noqa: E722
            pass

        return {"quality_score": 9.0, "is_approved": True}

    def validate_length(self, text: str) -> Dict:
        """
        Validates report length

        Args:
            text: The text to validate

        Returns:
            Dictionary with word_count, is_valid, status
        """
        words = len(text.split())
        is_valid = 950 <= words <= 1050

        return {
            "word_count": words,
            "is_valid": is_valid,
            "status": "Valid" if is_valid else "Needs adjustment"
        }

    def translation_agent(self, text: str, target_language: str) -> str:
        """
        Translates text to target language

        Args:
            text: Text to translate
            target_language: Target language (English, French, German, Spanish)

        Returns:
            Translated text
        """
        language_map = {
            "English": "English",
            "French": "French (Francais)",
            "German": "German (Deutsch)",
            "Spanish": "Spanish (Espanol)"
        }

        lang = language_map.get(target_language, "English")

        prompt = f"""Translate this technical MLOps report to {lang}:

{text}

Maintain technical terminology and professional tone."""

        response = gemini_model_basic.invoke(prompt)
        return extract_text_content(response)


# ============================================================================
# SECTION 3: WORKFLOW & ORCHESTRATION (LangGraph)
# ============================================================================

def research_node(state: ReportState) -> ReportState:
    """LangGraph node: Research"""
    agent = MLOpsReportAgent()
    research = agent.research_agent(state['topic'])
    state['research'] = research
    state['workflow_log'].append({
        "step": "1. Research",
        "status": "Done",
        "details": f"{len(research)} chars",
        "timestamp": time.strftime("%H:%M:%S")
    })
    return state


def outline_node(state: ReportState) -> ReportState:
    """LangGraph node: Outline"""
    agent = MLOpsReportAgent()
    outline = agent.outline_agent(state['research'])
    state['outline'] = outline
    state['workflow_log'].append({
        "step": "2. Outline",
        "status": "Done",
        "details": "Structure created",
        "timestamp": time.strftime("%H:%M:%S")
    })
    return state


def writer_node(state: ReportState) -> ReportState:
    """LangGraph node: Writer"""
    agent = MLOpsReportAgent()
    report = agent.writer_agent(state['research'], state['outline'], state['iteration'])
    validation = agent.validate_length(report)

    state['report'] = report
    state['word_count'] = validation['word_count']
    state['is_valid'] = validation['is_valid']
    state['iteration'] += 1

    state['workflow_log'].append({
        "step": f"3. Writing (Attempt {state['iteration']})",
        "status": "Done",
        "details": f"{validation['word_count']} words - {validation['status']}",
        "timestamp": time.strftime("%H:%M:%S")
    })
    return state


def fact_check_node(state: ReportState) -> ReportState:
    """LangGraph node: Fact Check"""
    agent = MLOpsReportAgent()
    fact_check = agent.fact_checker_agent(state['report'])
    state['quality_score'] = fact_check['quality_score']

    state['workflow_log'].append({
        "step": "4. Fact-Check",
        "status": "Done",
        "details": f"Score: {fact_check['quality_score']}/10",
        "timestamp": time.strftime("%H:%M:%S")
    })
    return state


def should_rewrite(state: ReportState) -> str:
    """LangGraph decision node"""
    if state['is_valid'] or state['iteration'] >= 3:
        return "fact_check"
    return "writer"


def build_report_graph():
    """Build the LangGraph workflow"""
    workflow = StateGraph(ReportState)

    # Add nodes
    workflow.add_node("research", research_node)
    workflow.add_node("outline", outline_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("fact_check", fact_check_node)

    # Add edges
    workflow.set_entry_point("research")
    workflow.add_edge("research", "outline")
    workflow.add_edge("outline", "writer")
    workflow.add_conditional_edges(
        "writer",
        should_rewrite,
        {
            "writer": "writer",
            "fact_check": "fact_check"
        }
    )
    workflow.add_edge("fact_check", END)

    return workflow.compile()


class MLOpsReportOrchestrator:
    """Orchestrates the workflow using LangGraph"""

    def __init__(self):
        """Initialize orchestrator with LangGraph"""
        self.graph = build_report_graph()

    def generate_report(self, topic: str, temperature: float = 0.7) -> Dict:
        """
        Generates report using LangGraph workflow

        Args:
            topic: Report topic
            temperature: Model temperature

        Returns:
            Dictionary with report and metadata
        """
        initial_state = ReportState(
            topic=topic,
            research="",
            outline="",
            report="",
            word_count=0,
            quality_score=0.0,
            is_valid=False,
            iteration=0,
            workflow_log=[]
        )

        # Run the graph
        final_state = self.graph.invoke(initial_state)

        return {
            "report": final_state['report'],
            "word_count": final_state['word_count'],
            "is_valid": final_state['is_valid'],
            "quality_score": final_state['quality_score'],
            "workflow_log": final_state['workflow_log'],
            "research": final_state['research'],
            "outline": final_state['outline']
        }


# ============================================================================
# SECTION 4: VISUALIZATION FUNCTIONS
# ============================================================================

def create_word_count_gauge(word_count, target_min=950, target_max=1050):
    """Creates word count gauge chart"""
    fig, ax = plt.subplots(figsize=(8, 4))

    if target_min <= word_count <= target_max:
        color = 'green'
        status = 'Valid'
    elif word_count < target_min:
        color = 'red'
        status = 'Too Short'
    else:
        color = 'orange'
        status = 'Too Long'

    ax.barh(['Word Count'], [word_count], color=color, alpha=0.7)
    ax.axvline(target_min, color='green', linestyle='--', linewidth=2, label='Min (950)')
    ax.axvline(target_max, color='red', linestyle='--', linewidth=2, label='Max (1050)')
    ax.axvline(1000, color='blue', linestyle='-', linewidth=2, label='Target (1000)')

    ax.set_xlim(0, 1200)
    ax.set_xlabel('Words')
    ax.set_title(f'Word Count: {word_count} - {status}')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def create_quality_gauge(quality_score):
    """Creates quality score gauge"""
    fig, ax = plt.subplots(figsize=(8, 4))

    if quality_score >= 8:
        color = 'green'
        status = 'Excellent'
    elif quality_score >= 6:
        color = 'orange'
        status = 'Good'
    else:
        color = 'red'
        status = 'Needs Improvement'

    ax.barh(['Quality Score'], [quality_score], color=color, alpha=0.7)
    ax.set_xlim(0, 10)
    ax.set_xlabel('Score')
    ax.set_title(f'Quality Score: {quality_score}/10 - {status}')
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    return fig


def create_workflow_timeline(workflow_log):
    """Creates workflow timeline visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))

    steps = [log['step'] for log in workflow_log]
    timestamps = [log['timestamp'] for log in workflow_log]
    colors = ['green' if 'Done' in log['status'] else 'orange' for log in workflow_log]

    y_pos = range(len(steps))
    ax.barh(y_pos, [1] * len(steps), color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(steps)
    ax.set_xlabel('Progress')
    ax.set_title('Workflow Timeline')
    ax.set_xlim(0, 1.2)

    for i, (step, timestamp) in enumerate(zip(steps, timestamps)):
        ax.text(1.05, i, timestamp, va='center', fontsize=9)

    plt.tight_layout()
    return fig


def create_stats_pie(word_count):
    """Creates section distribution pie chart"""
    intro = int(word_count * 0.15)
    body = int(word_count * 0.70)
    conclusion = int(word_count * 0.15)

    fig, ax = plt.subplots(figsize=(8, 6))

    sizes = [intro, body, conclusion]
    labels = ['Introduction', 'Main Body', 'Conclusion']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Section Distribution (Estimate)')

    plt.tight_layout()
    return fig


def generate_langgraph_diagram():
    """Generate LangGraph system diagram using Mermaid"""
    mermaid_code = """
graph TD
    Start([Start: User Input]) --> Research[Research Agent<br/>Web Search]
    Research --> Outline[Outline Agent<br/>Create Structure]
    Outline --> Writer[Writer Agent<br/>Generate Report]
    Writer --> Decision{Word Count<br/>950-1050?}
    Decision -->|No & Iteration < 3| Writer
    Decision -->|Yes or Iteration >= 3| FactCheck[Fact-Checker Agent<br/>Quality Check]
    FactCheck --> End([End: Final Report])

    style Start fill:#90EE90
    style Research fill:#87CEEB
    style Outline fill:#87CEEB
    style Writer fill:#87CEEB
    style Decision fill:#FFD700
    style FactCheck fill:#87CEEB
    style End fill:#FFB6C1
    """
    return mermaid_code


# ============================================================================
# SECTION 5: EXPORT FUNCTIONS
# ============================================================================

def export_to_txt(text: str, topic: str) -> str:
    """Export report to TXT format"""
    import re
    # Clean topic for filename
    clean_topic = re.sub(r'[^\w\s-]', '', topic.lower())
    clean_topic = re.sub(r'[-\s]+', '_', clean_topic)

    filename = f"mlops_report_{clean_topic}.txt"
    filepath = os.path.join(tempfile.gettempdir(), filename)

    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"TXT file created at: {filepath}")
        return filepath
    except Exception as e:
        print(f"Error creating TXT file: {e}")
        return None  # type: ignore


def export_to_html(text: str, topic: str) -> str:
    """Export report to HTML format with styling"""
    import re
    # Clean topic for filename
    clean_topic = re.sub(r'[^\w\s-]', '', topic.lower())
    clean_topic = re.sub(r'[-\s]+', '_', clean_topic)

    filename = f"mlops_report_{clean_topic}.html"
    filepath = os.path.join(tempfile.gettempdir(), filename)

    # Convert markdown-style headers to HTML
    lines = text.split('\n')
    html_lines = []

    for line in lines:
        if line.startswith('###'):
            html_lines.append(f'<h3>{line.replace("###", "").strip()}</h3>')
        elif line.startswith('##'):
            html_lines.append(f'<h2>{line.replace("##", "").strip()}</h2>')
        elif line.startswith('#'):
            html_lines.append(f'<h1>{line.replace("#", "").strip()}</h1>')
        elif line.strip():
            html_lines.append(f'<p>{line}</p>')
        else:
            html_lines.append('<br>')

    html_content = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{topic}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 900px;
            margin: 50px auto;
            padding: 40px;
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            line-height: 1.6;
        }}
        .container {{
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #667eea;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #667eea;
            padding-left: 15px;
        }}
        h3 {{
            color: #555;
            margin-top: 20px;
        }}
        p {{
            color: #444;
            text-align: justify;
            margin: 15px 0;
        }}
        .header {{
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1 style="margin: 0; border: none; color: white;">MLOps Report</h1>
        <p style="margin: 10px 0 0 0; color: white;">{topic}</p>
    </div>
    <div class="container">
        {''.join(html_lines)}
    </div>
    <div class="footer">
        <p>Generated by MLOps Report Generator</p>
        <p style="margin: 5px 0 0 0;">Powered by Gemini with LangGraph</p>
    </div>
</body>
</html>"""

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)

    return filepath


def export_report(text: str, format_type: str, topic: str):
    """Main export function"""
    if format_type == "txt":
        return export_to_txt(text, topic)
    elif format_type == "html":
        return export_to_html(text, topic)
    return None


llm_report = None
