"""
MLOps Multi-Agent Pipeline - Parallel Competitive System
Writers A and B create concise reports in parallel
Proofreader selects the best one
If not acceptable, both refine again in a second iteration
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------------
# Configuration
# -------------------------
MIN_WORDS = 950
MAX_WORDS = 1050
TARGET_WORDS = 1000
QUALITY_THRESHOLD = 8.0
MAX_ITERATIONS = 2

DOMAIN_KNOWLEDGE = """
Expert technical knowledge from MLOps course for writing:
- CI/CD automation (GitHub Actions, GitLab CI)
- REST API design, Dockerization, microservices
- Testing, code quality, version control
- Streamlit or Gradio interfaces for ML apps
- Data pipelines, ETL, validation, monitoring
- Model lifecycle management (MLflow, Kubeflow, DVC)
- Cloud deployments on AWS, GCP, Azure

IMPORTANT:
- Write ONLY in short paragraphs (≤3 sentences).
- Avoid lists or code blocks.
- Concise, clear, and professional tone.
"""
api = os.getenv("GOOGLE_API_KEY")


def get_llm(temp=0.6):
    return ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=api,
        temperature=temp,
    )


# -------------------------
# Templates
# -------------------------
outline_template = ChatPromptTemplate.from_template("""
{domain_knowledge}
{web_search_context}

Create a **brief structured outline** for a technical report on: {topic}

REQUIREMENTS:
- Target: 1000 words total (strict)
- Structure:
  * Introduction: 1 paragraph (~150 words)
  * Three main body sections (~550 words total, distributed across sections)
  * Conclusion: 1 paragraph (~150 words)
- Clear section titles only, no full prose.
""")

writer_initial_template = ChatPromptTemplate.from_template("""
{domain_knowledge}
{web_search_context}

Write a comprehensive **technical report** on: {topic}

OUTLINE:
{outline}

RULES:
1. HARD LIMIT: 950–1050 words. Target exactly 1000 words. COUNT YOUR WORDS!
2. Structure (word distribution):
   - Introduction: 1 paragraph (~150 words) - Set context and importance
   - Body: EXACTLY 3 sections (~180 words each = 540 words total)
     * Each section: 3-4 sentences with technical depth
   - Conclusion: 1 paragraph (~150 words) - Synthesize key points
3. Markdown format:
   # Title
   ## Introduction
   ## Section 1
   ## Section 2
   ## Section 3
   ## Conclusion
4. Include specific technical details, not generic statements.
5. Use transitional phrases; keep flow smooth.
6. Tone: expert, factual, precise. Add depth without redundancy.
""")

writer_refine_template = ChatPromptTemplate.from_template("""
{domain_knowledge}
{web_search_context}

Refine the report on: {topic}

CURRENT REPORT:
{current_essay}

FEEDBACK:
{feedback}

IMPROVEMENT TASK:
- CRITICAL: Reach 950–1050 words total. Target exactly 1000 words.
- If currently under 950 words: EXPAND each section with more technical details, examples, or explanations
- If currently over 1050 words: CONDENSE without losing key information
- Structure: 1 intro paragraph (~150 words) + 3 body sections (~180 words each) + 1 conclusion paragraph (~150 words)
- Each body paragraph should have 3-4 sentences with sufficient technical depth
- Maintain markdown structure with 3 main sections
- Add clarity, natural flow, and technical precision
""")

proofreader_template = ChatPromptTemplate.from_template("""
You are a strict proofreader comparing TWO reports on {topic}.

REPORT A:
{essay_a}

REPORT B:
{essay_b}

Return JSON ONLY:
{{
  "better_report": "<A or B or TIE>",
  "total_score": <0-10>,
  "feedback": "<short improvement notes>",
  "is_acceptable": <true if total_score>=8.0>
}}
""")


# -------------------------
# Helpers
# -------------------------
def count_words(txt):
    txt = re.sub(r"[#*`\[\]_]", "", txt or "")
    return len(txt.split())


def evaluate(essay_a, essay_b, topic, llm):
    """Evaluate two essays and return the best one with metrics."""
    msg = proofreader_template.format_messages(topic=topic, essay_a=essay_a, essay_b=essay_b)

    try:
        res = llm.invoke(msg).content

        # Clean up response - remove markdown code blocks
        res = re.sub(r'```json\s*', '', res)
        res = re.sub(r'```\s*', '', res)
        res = res.strip()

        # Extract JSON - try multiple approaches
        json_str = None

        # Approach 1: Find first { to last }
        start = res.find('{')
        end = res.rfind('}')
        if start != -1 and end != -1 and end > start:
            json_str = res[start:end + 1]

        if not json_str:
            raise ValueError("No JSON object found in response")

        # Parse JSON
        data = json.loads(json_str)

        # Extract fields with defaults
        pick = str(data.get("better_report", "A")).upper().strip()
        if pick not in ["A", "B"]:
            pick = "A"

        best = essay_a if pick == "A" else essay_b
        wc = count_words(best)
        score = float(data.get("total_score", 5.0))
        fb = str(data.get("feedback", "No feedback provided"))
        ok = bool(data.get("is_acceptable", False))

        # Check word count constraint
        if not (MIN_WORDS <= wc <= MAX_WORDS):
            ok = False
            if wc < MIN_WORDS:
                shortage = MIN_WORDS - wc
                fb = f"TOO SHORT: {wc} words (need +{shortage} more). EXPAND sections with more technical details. " + fb  # noqa: E501
            else:
                excess = wc - MAX_WORDS
                fb = f"TOO LONG: {wc} words (remove {excess}). CONDENSE without losing key points. " + fb

        return best, score, fb, wc, ok, pick

    except Exception as e:
        # Fallback: pick essay with word count closest to target
        wc_a, wc_b = count_words(essay_a), count_words(essay_b)
        diff_a = abs(wc_a - TARGET_WORDS)
        diff_b = abs(wc_b - TARGET_WORDS)

        best = essay_a if diff_a <= diff_b else essay_b
        wc = count_words(best)
        pick = "A" if best == essay_a else "B"

        return best, 5.0, f"Auto-selected (eval error: {str(e)[:80]})", wc, False, pick


def generate(prompt, llm, **kw):
    """Generate text from a prompt template."""
    return llm.invoke(prompt.format_messages(**kw)).content


# -------------------------
# Pipeline
# -------------------------
def generate_essay_with_context(topic, web_search_context="", temperature=0.6, progress_callback=None):
    """Main pipeline: outline -> parallel writing -> evaluation -> optional refinement."""
    llm_a = get_llm(temperature)
    llm_b = get_llm(temperature + 0.1)

    def log(msg):
        if progress_callback:
            progress_callback(msg)
        print(msg)

    log("🎯 Generating report (Temperature: 0.6)")
    log("⏱️ This will take 2-3 minutes...\n")
    log("🧠 AI neurons firing up...")
    log("📋 Creating outline...")

    outline = llm_a.invoke(outline_template.format_messages(
        topic=topic,
        domain_knowledge=DOMAIN_KNOWLEDGE,
        web_search_context=web_search_context or "(No web context)"
    )).content

    log("✅ Outline created")
    log("✨ Channeling technical excellence...\n")

    best, feedback, wc = None, "Initial pass", 0

    for i in range(MAX_ITERATIONS):
        log("=" * 50)
        log(f"ITERATION {i + 1}/{MAX_ITERATIONS}")
        log("=" * 50)

        if i == 0:
            # First iteration: both writers create from scratch
            log("✍️ Writer A & B: Creating reports in parallel...")
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_a = ex.submit(
                    generate, writer_initial_template, llm_a,
                    domain_knowledge=DOMAIN_KNOWLEDGE,
                    web_search_context=web_search_context,
                    topic=topic,
                    outline=outline
                )
                fut_b = ex.submit(
                    generate, writer_initial_template, llm_b,
                    domain_knowledge=DOMAIN_KNOWLEDGE,
                    web_search_context=web_search_context,
                    topic=topic,
                    outline=outline
                )
                essay_a, essay_b = fut_a.result(), fut_b.result()
        else:
            # Subsequent iterations: both refine the best essay
            log("✍️ Writer A & B: Refining best report in parallel...")
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_a = ex.submit(
                    generate, writer_refine_template, llm_a,
                    domain_knowledge=DOMAIN_KNOWLEDGE,
                    web_search_context=web_search_context,
                    topic=topic,
                    current_essay=best,
                    feedback=feedback
                )
                fut_b = ex.submit(
                    generate, writer_refine_template, llm_b,
                    domain_knowledge=DOMAIN_KNOWLEDGE,
                    web_search_context=web_search_context,
                    topic=topic,
                    current_essay=best,
                    feedback=feedback
                )
                essay_a, essay_b = fut_a.result(), fut_b.result()

        # Evaluate both essays
        log("🔍 Proofreader: Comparing and evaluating...\n")
        best, score, feedback, wc, ok, winner = evaluate(essay_a, essay_b, topic, llm_a)

        log(f"📊 ITERATION {i + 1} RESULT")
        log(f"Winner: Report {winner}")
        log(f"Word Count: {wc}")
        log(f"Score: {score:.1f}/10")
        log(f"Feedback: {feedback[:130]}...\n")

        if ok:
            log("🎯 Acceptable report found!\n")
            break

    log("=" * 50)
    log("🎉 GENERATION COMPLETE!")
    log(f"📊 Final word count: {wc}")
    log(f"📄 Paragraphs: {best.count(chr(10))}")

    if wc < MIN_WORDS:
        log("⚠️ Status: Too Short")
    elif wc > MAX_WORDS:
        log("⚠️ Status: Too Long")
    else:
        log("✅ Status: Perfect")

    log("=" * 50)
    return best


def generate_essay(topic):
    """Simplified interface for generating an essay."""
    return generate_essay_with_context(topic, "", 0.6, None)
