"""
gradio_app.py
Enhanced Frontend UI for the Data Engineering Agentic System
Features: Real-time logging, statistics, readability analysis, and STRICT requirement enforcement
"""
import re
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
from gradio.themes import Soft

from .llms_workflow import generate_report_agentic_workflow

matplotlib.use('Agg')  # Non-interactive backend


# -------------------------------------------------------------
# 1. Create Reports Directory
# -------------------------------------------------------------
REPORTS_DIR = Path("generated_reports")
REPORTS_DIR.mkdir(exist_ok=True)


# -------------------------------------------------------------
# 2. Text Analysis Utilities
# -------------------------------------------------------------
def count_words(text):
    """Count words in text, excluding markdown syntax"""
    clean_text = re.sub(r'[#*_`\[\]]', '', text)  # Remove markdown symbols
    clean_text = re.sub(r'\s+', ' ', clean_text)
    words = [w for w in clean_text.strip().split() if w]
    return len(words)


def count_sentences(text):
    """Count sentences in text"""
    clean_text = re.sub(r'[#*_`]', '', text)
    sentences = re.split(r'[.!?]+(?=\s+|$)', clean_text)
    return len([s for s in sentences if s.strip()])


def count_paragraphs(text):
    """Count paragraphs in text"""
    paragraphs = text.split('\n\n')
    return len([p for p in paragraphs if p.strip()])


def count_syllables(word):
    """Simple syllable counter (approximation)"""
    word = word.lower()
    word = re.sub(r'(?:[^laeiouy]es|ed|[^laeiouy]e)$', '', word)
    word = re.sub(r'[aeiouy]{1,2}', 'a', word)
    syllable_count = len(word.split('a'))
    return max(1, syllable_count)


def calculate_readability_scores(text):
    """Calculate multiple readability metrics"""
    words = count_words(text)
    sentences = count_sentences(text)

    if sentences == 0 or words == 0:
        return {"error": "Insufficient text"}

    avg_sentence_length = words / sentences
    syllable_count = sum(count_syllables(word) for word in re.findall(r'\b\w+\b', text))
    avg_syllables_per_word = syllable_count / words if words > 0 else 0

    flesch_reading_ease = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word
    flesch_reading_ease = max(0, min(100, flesch_reading_ease))

    flesch_kincaid_grade = 0.39 * avg_sentence_length + 11.8 * avg_syllables_per_word - 15.59
    flesch_kincaid_grade = max(0, flesch_kincaid_grade)

    if flesch_reading_ease >= 90:
        difficulty = "Very Easy (5th grade)"
    elif flesch_reading_ease >= 80:
        difficulty = "Easy (6th grade)"
    elif flesch_reading_ease >= 70:
        difficulty = "Fairly Easy (7th grade)"
    elif flesch_reading_ease >= 60:
        difficulty = "Standard (8th-9th grade)"
    elif flesch_reading_ease >= 50:
        difficulty = "Fairly Difficult (10th-12th grade)"
    elif flesch_reading_ease >= 30:
        difficulty = "Difficult (College)"
    else:
        difficulty = "Very Difficult (College Graduate)"

    return {
        "flesch_reading_ease": round(flesch_reading_ease, 1),
        "flesch_kincaid_grade": round(flesch_kincaid_grade, 1),
        "difficulty": difficulty,
        "avg_sentence_length": round(avg_sentence_length, 1),
        "avg_syllables_per_word": round(avg_syllables_per_word, 2)
    }


def validate_report_structure(text):
    """Comprehensive validation of report structure and requirements"""
    issues = []

    # 1. Word count (exactly 1010)
    word_count = count_words(text)
    if word_count >= 950 and word_count <= 1050:
        issues.append(f"⚠️ Word count incorrect: {word_count} (need BETWEEN 950 and 1050)")

    # 2. Check for H1 title
    h1_match = re.search(r'^#\s+(.+)$', text, re.MULTILINE)
    if not h1_match:
        issues.append("⚠️ Missing H1 title (# Heading)")

    # 3. Check for exact "Introduction" heading
    has_intro = bool(re.search(r'^##\s+Introduction\s*$', text, re.MULTILINE))
    if not has_intro:
        issues.append("⚠️ Missing or incorrect 'Introduction' section (must be: ## Introduction)")

    # 4. Check for exact "Conclusion" heading
    has_conclusion = bool(re.search(r'^##\s+Conclusion\s*$', text, re.MULTILINE))
    if not has_conclusion:
        issues.append("⚠️ Missing or incorrect 'Conclusion' section (must be: ## Conclusion)")

    # 5. Count H2 sections (should have Introduction + 4 subtopics + Conclusion = 6 total)
    h2_sections = re.findall(r'^##\s+(.+)$', text, re.MULTILINE)
    if len(h2_sections) < 6:
        issues.append(f"⚠️ Not enough sections: {len(h2_sections)} (need 6: Intro + 4 subtopics + Conclusion)")
    elif len(h2_sections) > 6:
        issues.append(f"⚠️ Too many sections: {len(h2_sections)} (need exactly 6)")

    # 6. Check citations (exactly 3)
    citations = re.findall(r'\[([^\]]{3,}?)\](?!\()', text)  # [Text] but not [Text](url)
    num_citations = len(citations)
    if num_citations != 3:
        issues.append(f"⚠️ Citation count: {num_citations} (need exactly 3)")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "word_count": word_count,
        "has_intro": has_intro,
        "has_conclusion": has_conclusion,
        "num_sections": len(h2_sections),
        "num_citations": num_citations
    }


def generate_statistics_report(text):
    """Generate comprehensive text statistics"""
    words = count_words(text)
    sentences = count_sentences(text)
    paragraphs = count_paragraphs(text)
    characters = len(text)
    characters_no_spaces = len(text.replace(" ", ""))

    readability = calculate_readability_scores(text)

    stats_text = f"""## 📊 Text Statistics


### Basic Metrics
- **Word Count:** {words:,}
- **Sentence Count:** {sentences:,}
- **Paragraph Count:** {paragraphs:,}
- **Character Count:** {characters:,} (with spaces)
- **Character Count:** {characters_no_spaces:,} (without spaces)


### Readability Analysis
- **Flesch Reading Ease:** {readability.get('flesch_reading_ease', 'N/A')} / 100
- **Difficulty Level:** {readability.get('difficulty', 'N/A')}
- **Flesch-Kincaid Grade:** {readability.get('flesch_kincaid_grade', 'N/A')}
- **Avg. Sentence Length:** {readability.get('avg_sentence_length', 'N/A')} words
- **Avg. Syllables/Word:** {readability.get('avg_syllables_per_word', 'N/A')}


### Reading Time Estimates
- **Silent Reading:** ~{words // 250} minutes (250 wpm)
- **Oral Reading:** ~{words // 150} minutes (150 wpm)
"""

    return stats_text, readability


def generate_visualization(text, topic):
    """Generate matplotlib visualization of text statistics"""
    try:
        words = count_words(text)
        sentences = count_sentences(text)
        paragraphs = count_paragraphs(text)
        readability = calculate_readability_scores(text)

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Report Analysis: {topic[:50]}', fontsize=14, fontweight='bold')

        # 1. Basic Metrics Bar Chart
        metrics = ['Words', 'Sentences', 'Paragraphs']
        values = [words, sentences, paragraphs]
        colors = ['#3B82F6', '#8B5CF6', '#EC4899']
        ax1.bar(metrics, values, color=colors, alpha=0.7)
        ax1.set_title('Basic Text Metrics', fontweight='bold')
        ax1.set_ylabel('Count')
        for i, v in enumerate(values):
            ax1.text(i, v + max(values) * 0.02, str(v), ha='center', fontweight='bold')

        # 2. Readability Gauge
        flesch_score = readability.get('flesch_reading_ease', 50)
        ax2.barh(['Readability'], [flesch_score], color='#10B981', alpha=0.7)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel('Flesch Reading Ease Score')
        ax2.set_title('Readability Score (0-100)', fontweight='bold')
        ax2.text(flesch_score / 2, 0, f'{flesch_score:.1f}',  # type: ignore
                ha='center', va='center', fontsize=16, fontweight='bold', color='white')

        # 3. Word Count Target Visualization
        target = 1010
        ax3.barh(['Target'], [target], color='#10B981', alpha=0.3, label=f'Target ({target})')
        ax3.barh(['Actual'], [words], color='#3B82F6', alpha=0.7, label=f'Actual ({words})')
        ax3.set_xlabel('Word Count')
        ax3.set_title('Word Count vs Target', fontweight='bold')
        ax3.legend()
        ax3.set_xlim(min(900, target - 100), max(1100, target + 100, words + 50))

        # 4. Grade Level Comparison
        grade_level = readability.get('flesch_kincaid_grade', 12)
        grade_categories = ['Elementary\n(1-6)', 'Middle\n(7-9)', 'High School\n(10-12)', 'College\n(13+)']
        grade_ranges = [6, 9, 12, 16]
        colors_grade = ['#10B981' if grade_level <= r else '#E5E7EB' for r in grade_ranges]  # type: ignore

        ax4.bar(grade_categories, [6, 3, 3, 4], color=colors_grade, alpha=0.7)
        ax4.axhline(y=grade_level, color='#EF4444', linestyle='--', linewidth=2,
                    label=f'Report Level: {grade_level:.1f}')
        ax4.set_ylabel('Grade Level')
        ax4.set_title('Education Level', fontweight='bold')
        ax4.legend()

        plt.tight_layout()

        viz_path = REPORTS_DIR / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(viz_path, dpi=100, bbox_inches='tight')
        plt.close()

        return str(viz_path)

    except Exception as e:
        print(f"Visualization error: {e}")
        return None


# -------------------------------------------------------------
# 3. File Management
# -------------------------------------------------------------
def sanitize_filename(topic):
    """Convert topic to safe filename"""
    safe_name = re.sub(r'[^\w\s-]', '', topic)
    safe_name = re.sub(r'[\s]+', '_', safe_name)
    safe_name = safe_name[:50]
    return safe_name.lower()


def save_as_markdown(text, topic):
    """Save generated report as Markdown file (.md)"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_topic = sanitize_filename(topic)
    filename = REPORTS_DIR / f"{safe_topic}_{timestamp}.md"

    # Clean text - remove only the validation summary header, keep the report
    clean_text = re.sub(r'# 📄 Data Engineering Report.*?\n---\n\n', '', text, flags=re.DOTALL)

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("---\n")
        f.write(f"title: {topic}\n")
        f.write(f"generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("---\n\n")
        f.write(clean_text)

    return str(filename)


# -------------------------------------------------------------
# 4. Gradio Event Functions with Real-time Logging
# -------------------------------------------------------------
def run_agent(topic, temperature, tone, audience, progress=gr.Progress()):
    """Generate report using the agentic workflow with STRICT validation"""
    if not topic.strip():
        return (
            "⚠️ Please enter a valid topic.",
            "",
            "No statistics available.",
            None,
            "No logs available.",
            gr.update(interactive=False)
        )

    logs = "🚀 Starting workflow...\n"
    result = ""

    try:
        progress(0, desc="🚀 Initializing 8 specialized agents...")

        # Capture logs by redirecting stdout
        log_stream = StringIO()
        old_stdout = sys.stdout
        sys.stdout = log_stream

        try:
            progress(0.1, desc="🔍 Agent 1: Topic Validator analyzing...")

            # Call the workflow function
            report_output = generate_report_agentic_workflow(topic, temperature, tone, audience)

            # Restore stdout
            sys.stdout = old_stdout
            logs = log_stream.getvalue()

            if not logs.strip():
                logs = "✅ Workflow completed (no detailed logs captured)\n"

            # Check if workflow returned a rejection message
            if report_output.startswith("⚠️") or report_output.startswith("❌"):
                return (
                    report_output,
                    "",
                    "Topic rejected or workflow failed - no statistics available.",
                    None,
                    logs,
                    gr.update(interactive=False)
               )

            result = report_output

        except Exception as e:
            sys.stdout = old_stdout
            logs = log_stream.getvalue() + f"\n\n❌ CRITICAL WORKFLOW ERROR: {type(e).__name__}: {str(e)}"
            raise e

        progress(0.9, desc="📊 Validating strict requirements...")

        if not result.strip():
            return (
               "⚠️ Workflow finished but returned empty content.",
               "",
               "No statistics available.",
               None,
               logs,
               gr.update(interactive=False)
           )

        # STRICT validation of output
        validation = validate_report_structure(result)

        # Build validation report
        validation_status = []
        word_count = validation['word_count']
        validation_status.append(
            f"Word Count: {word_count} "
            f"({'✅ VALID' if word_count >= 950 and word_count <= 1050 else
            '❌ INVALID - MUST be BETWEEN 950 TO 1050 Words'})"
        )
        validation_status.append(
            f"Introduction Section: {'✅ Found' if validation['has_intro'] else
             '❌ Missing - MUST have ## Introduction'}"
        )
        validation_status.append(
            f"Conclusion Section: {'✅ Found' if validation['has_conclusion'] else
            '❌ Missing - MUST have ## Conclusion'}"
        )
        validation_status.append(
            f"Section Count: {validation['num_sections']} "
            f"({'✅ VALID' if validation['num_sections'] == 6 else '❌ INVALID - MUST have 6 sections'}"
        )
        validation_status.append(
            f"Citations: {validation['num_citations']} "
            f"({'✅ VALID' if validation['num_citations'] == 3 else '❌ INVALID - MUST have exactly 3'})"
        )

        validation_text = "\n".join(validation_status)

        # Add issues if any
        if validation['issues']:
            validation_text += "\n\n**Issues Found:**\n" + "\n".join(validation['issues'])

        # Generate statistics and visualization
        stats_text, _readability = generate_statistics_report(result)

        stats_text = f"""## ✅ Requirement Validation


{validation_text}


---


{stats_text}"""

        viz_path = generate_visualization(result, topic)

        # Format output with metadata header
        header = f"""# 📄 Data Engineering Report


**Topic:** {topic}
**Tone:** {tone} | **Audience:** {audience}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}


**Validation Summary:**
{validation_text}


---


"""
        display_output = header + result

        progress(1.0, desc="✅ Complete!")

        return (
            display_output,
            "",
            stats_text,
            viz_path,
            logs,
            gr.update(interactive=True)
        )

    except Exception as e:
        error_msg = (
            f"⚠️ Critical Error: {type(e).__name__}: {str(e)}\n\n"
            f"Please check your configuration and environment variables (.env file) and try again."
        )

        return (
            error_msg,
            "",
            "Error - no statistics available.",
            None,
            logs if logs else str(e),
            gr.update(interactive=False)
        )


def copy_to_editor(output_text):
    """Copy generated report to editable textbox"""
    if not output_text or output_text.startswith("⚠️") or not output_text.strip():
        gr.Warning("No valid content to copy!")
        return ""

    # Remove the metadata header
    clean_text = re.sub(r'# 📄 Data Engineering Report.*?\n---\n\n', '', output_text, flags=re.DOTALL)
    return clean_text.strip()


def handle_download(text, topic):
    """Handle Markdown download with topic-based filename"""
    if not text or text.startswith("⚠️") or not text.strip():
        gr.Warning("No valid content to download!")
        return None

    filename = save_as_markdown(text, topic)
    gr.Info(f"Report saved as Markdown: {filename}")
    return gr.File(value=filename, visible=True)


# -------------------------------------------------------------
# 5. Enhanced Gradio Layout
# -------------------------------------------------------------
with gr.Blocks(
   title="Data Engineering Report Generator",
   theme=Soft(),
   css="""
   .generate-btn {font-size: 16px !important; font-weight: bold !important;}
   .copy-btn {background: #4CAF50 !important;}
   .log-box {font-family: 'Courier New', monospace; font-size: 12px; height: 300px; overflow-y: scroll;}
   """
) as llm_app:

    gr.Markdown(
        """
        # 🔧 Data Engineering Report Generator
        ### AI-Powered Multi-Agent System for Comprehensive Technical Reports

        **Features:** 8 specialized agents | Real-time logging | Advanced analytics | **STRICT 1010 word enforcement**

        **Requirements:**
        - ✅ Topic must be Data Engineering related
        - ✅ Report length: **EXACTLY 1010 words** (strictly enforced)
        - ✅ Structure: **Captivating Title** → **Introduction** → **4 Subtopics** → **Conclusion**
        - ✅ Citations: **Exactly 3 references** in format [Source Name]
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            topic = gr.Textbox(
                label="📝 Enter Data Engineering Topic",
                placeholder="e.g., Real-time Data Pipeline Architecture, ETL vs ELT Strategies, "
                "Data Lake Design Patterns, Apache Kafka Streaming",
                lines=2
            )
        with gr.Column(scale=1):
            temperature = gr.Slider(
                minimum=0,
                maximum=1,
                value=0.3,
                step=0.1,
                label="🎨 Creativity Level",
                info="Lower = more focused, Higher = more creative"
            )

    with gr.Row():
        with gr.Column():
            tone = gr.Radio(
                choices=[
                    "Professional",
                    "Technical",
                    "Educational",
                    "Conversational",
                    "Academic"
                ],
                value="Professional",
                label="✍️ Tone",
                info="Select the writing style"
            )
        with gr.Column():
            audience = gr.Dropdown(
                choices=[
                    "Data Engineers",
                    "Software Engineers",
                    "Data Architects",
                    "Engineering Managers",
                    "Technical Leadership",
                    "Junior Engineers",
                    "Data Scientists",
                    "Students"
                ],
                value="Data Engineers",
                label="👥 Target Audience",
                info="Who is this report for?"
            )

    generate_btn = gr.Button(
        "🚀 Generate Report with AI Agents",
        variant="primary",
        size="lg",
        elem_classes="generate-btn"
    )

    with gr.Tabs():
        with gr.Tab("📄 Generated Report"):
            output = gr.Markdown(label="Final Report", show_label=False)

        with gr.Tab("📊 Statistics & Analysis"):
            with gr.Row():
                with gr.Column():
                    stats_output = gr.Markdown(label="Text Statistics", show_label=False)
                with gr.Column():
                    viz_output = gr.Image(label="Visual Analysis", show_label=False, width=450, height=450)

        with gr.Tab("🔍 Agent Activity Logs"):
            logs_output = gr.Textbox(
                label="Real-time Processing Logs",
                lines=20,
                max_lines=30,
                show_label=True,
                elem_classes="log-box",
                interactive=False
            )
            gr.Markdown("""
            **Agent Pipeline:**
            1. 🔍 **Topic Validator** - Validates Data Engineering relevance
            2. 📝 **Prompt Engineer** - Generates optimized prompts
            3. ✅ **Quality Assurance** - Reviews prompt quality
            4. 🎯 **Key Points Identifier** - Extracts essential topics
            5. 🔬 **Research Agent** - Generates comprehensive content
            6. 📊 **Grading & Refinement** - Fact-checks and improves
            7. 📐 **Structuring Agent** - Organizes to exact word count
            8. ✨ **Polish & Enforcement** - **STRICT requirement enforcement**
            """)

    with gr.Row():
        copy_btn = gr.Button(
            "📋 Copy to Editor",
            variant="secondary",
            size="sm",
            interactive=False,
            elem_classes="copy-btn"
        )
        gr.Markdown("← Click to copy report for editing")

    with gr.Accordion("📝 Edit Report (Optional)", open=False):
        edit_box = gr.Textbox(
            label="Make manual edits here",
            lines=15,
            placeholder="Click 'Copy to Editor' button above to load the generated report here for editing...",
            interactive=True
        )

    with gr.Row():
        download_btn = gr.Button("📥 Download as Markdown (.md)", variant="secondary")
        download_file = gr.File(label="Download File", visible=False)

    gr.Markdown(
        """
        ---
        ### 📊 Report Specifications (STRICTLY ENFORCED):

        **Structure (6 sections total):**
        1. **Captivating H1 Title** - Engaging, specific heading
        2. **## Introduction** - Overview and context
        3. **## Subtopic 1** - First key area (descriptive name)
        4. **## Subtopic 2** - Second key area (descriptive name)
        5. **## Subtopic 3** - Third key area (descriptive name)
        6. **## Subtopic 4** - Fourth key area (descriptive name)
        7. **## Conclusion** - Summary and final thoughts

        **Hard Requirements:**
        - **Word Count:** EXACTLY 1010 words (excluding title)
        - **Format:** Markdown with proper H1/H2 hierarchy
        - **Citations:** EXACTLY 3 references in format [Source Name]
        - **Section Titles:** "Introduction" and "Conclusion" must be exact

        ### 🎯 Strict Enforcement:
        - Agents will automatically trim or expand content to meet EXACTLY 1010 words
        - Section titles are validated for exact spelling
        - Citation count strictly enforced to exactly 3
        - Maximum 2 retry attempts if requirements not met

        ### 🤖 8-Agent Architecture:
        1. **Topic Validator** - Ensures Data Engineering relevance
        2. **Prompt Engineer** - Creates optimized instructions
        3. **Quality Assurance** - Reviews prompt quality
        4. **Key Points Identifier** - Extracts 4 essential subtopics
        5. **Research Agent** - Generates comprehensive content
        6. **Grading & Refinement** - Fact-checks and improves quality
        7. **Structuring Agent** - Organizes and adjusts to target length
        8. **Polish & Enforcement** - **STRICTLY enforces all requirements**

        ### ⚠️ Topic Requirements:
        Only **Data Engineering** topics accepted:
        - ✅ Data Pipelines, ETL/ELT, Stream Processing
        - ✅ Data Warehousing, Data Lakes, Lakehouses
        - ✅ Data Quality, Governance, Cataloging
        - ✅ Orchestration (Airflow, Dagster, Prefect)
        - ✅ Cloud Platforms (AWS, GCP, Azure data services)
        - ✅ Data Architecture, Modeling, Infrastructure
        - ❌ Pure software development (not data-focused)
        - ❌ Machine learning/AI (unless data infrastructure)
        - ❌ Frontend, mobile, or general business topics
        """
    )

    # Event Handlers
    generate_btn.click(
        fn=run_agent,
        inputs=[topic, temperature, tone, audience],
        outputs=[output, edit_box, stats_output, viz_output, logs_output, copy_btn]
    )

    copy_btn.click(
        fn=copy_to_editor,
        inputs=output,
        outputs=edit_box
    )

    download_btn.click(
        fn=handle_download,
        inputs=[edit_box, topic],
        outputs=download_file
    )

    gr.Examples(
        examples=[
            ["Stream Processing with Apache Kafka", 0.3, "Technical", "Data Engineers"],
            ["Introduction to Data Warehousing Concepts", 0.4, "Educational", "Students"],
            ["Implementing Data Quality Frameworks", 0.2, "Professional", "Data Architects"],
            ["Cost Optimization in Cloud Data Platforms", 0.3, "Professional", "Engineering Managers"],
            ["Change Data Capture (CDC) Patterns", 0.2, "Technical", "Software Engineers"],
            ["Data Mesh Architecture Principles", 0.3, "Academic", "Data Architects"],
        ],
       inputs=[topic, temperature, tone, audience],
       label="💡 Example Topics (All produce exactly 1010 word reports)"
   )


if __name__ == "__main__":
    llm_app.launch()
