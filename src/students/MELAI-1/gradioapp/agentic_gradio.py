import gradio as gr
import os
import requests
import json
from dotenv import load_dotenv
from datetime import datetime
import traceback

# --- Configuration ---
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")

# Configuration
TARGET_WORD_COUNT = 1000
WORD_COUNT_MIN = 950
WORD_COUNT_MAX = 1050

# Topics disponibles (liÃ©s au cours)
AVAILABLE_TOPICS = {
    "Agentic AI": "AI systems with autonomous reasoning and tool use",
    "CI/CD Pipelines": "Continuous Integration and Continuous Deployment practices",
    "MLOps": "Machine Learning Operations and model deployment",
    "REST APIs": "RESTful API design and implementation with FastAPI",
    "Gradio Interfaces": "Building interactive ML interfaces with Gradio"
}

# ============================================================================
# FONCTION PRINCIPALE DE GÃ‰NÃ‰RATION DE RAPPORT
# ============================================================================
def gradio_generate_report(topic: str, temperature: float, include_references: bool, 
                           save_to_file: bool, progress=gr.Progress()):
    """
    Generate comprehensive technical report with user-controlled parameters.
    
    Args:
        topic: Selected topic from course materials
        temperature: Model creativity (0.0-1.0)
        include_references: Include citation list
        save_to_file: Save report as .txt file
        progress: Gradio progress tracker
    """
    try:
        print(f"\n{'='*70}")
        print(f"ðŸš€ GENERATING REPORT")
        print(f"{'='*70}")
        print(f"Topic: {topic}")
        print(f"Temperature: {temperature}")
        print(f"{'='*70}\n")
        
        progress(0.1, desc="Initializing agents...")
        
        # Import local pour Ã©viter import circulaire
        from api.main import generate_report_with_params
        
        progress(0.2, desc="Researching topic...")
        
        # GÃ©nÃ©rer le rapport avec paramÃ¨tres personnalisÃ©s
        result = generate_report_with_params(
            topic=topic,
            temperature=temperature
        )
        
        progress(0.8, desc="Finalizing report...")
        
        # Format du rapport
        report_output = f"""# {topic}: A Comprehensive Technical Overview

**Generated:** {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}

**Word Count:** {result['metadata']['word_count']} words
**Model Temperature:** {temperature}
**Topic Category:** Course-Related Technical Report

---

{result['report']}
"""

        # Ajouter les rÃ©fÃ©rences si demandÃ©
        if include_references:
            report_output += "\n\n" + "="*70 + "\n"
            report_output += "## ðŸ“š Research References\n\n"
            for bullet in result['research']:
                report_output += f"**[{bullet['id']}]** {bullet['text']}\n"
                report_output += f"   - *Category:* {bullet.get('category', 'N/A')}\n"
                report_output += f"   - *Technical Depth:* {bullet.get('technical_depth', 'N/A')}\n\n"

        # Sauvegarder si demandÃ©
        save_message = ""
        if save_to_file:
            filename = f"report_{topic.replace(' ', '_').replace('/', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(report_output)
            save_message = f"\n\nâœ… **Report saved to:** `{filename}`"

        # Informations de qualitÃ©
        verified = sum(1 for fc in result['fact_checks']
                      if fc.get('status') in ['VERIFIED', 'SUPPORTED'])
        total = len(result['fact_checks'])
        accuracy = (verified / total * 100) if total > 0 else 0

        quality_output = f"""### ðŸ“Š Quality Metrics

**Factual Accuracy:** {accuracy:.0f}% ({verified}/{total} claims verified)
**Word Count:** {result['metadata']['word_count']} words (Target: 950-1050)
**Structure Valid:** {'âœ… Yes' if result['metadata']['structure_valid'] else 'âŒ No'}
**Revision Rounds:** {result['metadata']['revision_rounds']}
**Execution Time:** {result['metadata']['execution_time']}

**Status:** {'âœ… High Quality Report' if accuracy >= 80 else 'âœ… Verified Report' if accuracy >= 60 else 'âš ï¸ Review Recommended'}

{save_message}
"""

        progress(1.0, desc="Complete!")
        
        return report_output, quality_output, result

    except Exception as e:
        error_msg = f"""### âŒ Error Generating Report

**Error Type:** {type(e).__name__}
**Error Message:** {str(e)}

**Traceback:**
```
{traceback.format_exc()}
```

Please check your configuration and try again.
"""
        print(f"\nâŒ ERROR: {str(e)}")
        traceback.print_exc()
        return error_msg, "Error occurred", None


# ============================================================================
# HUMAN-IN-THE-LOOP: Ã‰DITION DU BROUILLON
# ============================================================================
def generate_draft_for_editing(topic: str, temperature: float):
    """Generate a draft report that users can edit before finalization."""
    try:
        from api.main import generate_report_with_params
        
        result = generate_report_with_params(topic=topic, temperature=temperature)
        
        draft_text = result['report']
        
        info = f"""### ðŸ“ Draft Generated Successfully

**Word Count:** {result['metadata']['word_count']}
**Status:** Ready for editing

**Instructions:**
1. Review the draft below
2. Make any edits you want in the text editor
3. Click "Finalize Edited Report" when done
4. The system will re-validate your edits

You can edit anything - structure, content, citations, etc.
"""
        
        return draft_text, info, result
        
    except Exception as e:
        error = f"âŒ Error generating draft: {str(e)}"
        return error, error, None


def finalize_edited_report(edited_text: str, include_refs: bool, save_file: bool, original_result):
    """Finalize user-edited report with validation."""
    try:
        if original_result is None:
            return "âŒ Please generate a draft first", ""
        
        # Compter les mots
        import re
        word_count = len(re.findall(r"\b\w+\b", edited_text))
        
        # VÃ©rifier la structure
        has_intro = bool(re.search(r'^##?\s+Introduction', edited_text, re.MULTILINE | re.IGNORECASE))
        has_body = bool(re.search(r'^##?\s+(Main|Body)', edited_text, re.MULTILINE | re.IGNORECASE))
        has_conclusion = bool(re.search(r'^##?\s+Conclusion', edited_text, re.MULTILINE | re.IGNORECASE))
        
        within_range = WORD_COUNT_MIN <= word_count <= WORD_COUNT_MAX
        structure_valid = has_intro and has_body and has_conclusion
        
        # Format final
        final_output = f"""# Edited Report (Human-in-the-Loop)

**Edited:** {datetime.now().strftime("%B %d, %Y at %H:%M:%S")}
**Word Count:** {word_count} words
**Edited by:** User

---

{edited_text}
"""

        if include_refs:
            final_output += "\n\n" + "="*70 + "\n"
            final_output += "## ðŸ“š Original Research References\n\n"
            for bullet in original_result['research']:
                final_output += f"**[{bullet['id']}]** {bullet['text']}\n\n"

        save_msg = ""
        if save_file:
            filename = f"edited_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(filename, "w", encoding="utf-8") as f:
                f.write(final_output)
            save_msg = f"\n\nâœ… **Saved to:** `{filename}`"

        validation = f"""### âœ… Report Finalized

**Word Count:** {word_count} ({'âœ… Within range' if within_range else 'âš ï¸ Outside range 950-1050'})
**Structure:** {'âœ… Valid (Intro, Body, Conclusion)' if structure_valid else 'âš ï¸ Missing sections'}

{save_msg}
"""
        
        return final_output, validation
        
    except Exception as e:
        return f"âŒ Error: {str(e)}", ""


# ============================================================================
# INTERFACE GRADIO COMPLÃˆTE
# ============================================================================
custom_css = """
.gradio-container { 
    max-width: 1400px !important; 
    margin: auto;
}
.header { 
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    padding: 40px; 
    border-radius: 12px; 
    color: white; 
    text-align: center;
    margin-bottom: 25px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}
.info-box {
    background: #f0f9ff;
    border-left: 4px solid #3b82f6;
    padding: 15px;
    border-radius: 8px;
    margin: 10px 0;
}
.generate-btn { 
    font-size: 1.1em !important; 
    padding: 18px 40px !important;
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
}
"""

with gr.Blocks(title="Agentic AI Report Generator - Assignment 3", css=custom_css) as report:
    
    # Variable pour stocker le rÃ©sultat (pour human-in-the-loop)
    stored_result = gr.State(None)
    
    # Header
    gr.HTML("""
    <div class="header">
        <h1 style="margin: 0; font-size: 2.5em;">ðŸ¤– Multi-Agent Report Generator</h1>
        <p style="margin-top: 15px; font-size: 1.2em;">Assignment 3 - Advanced Agentic System</p>
        <p style="margin-top: 5px; font-size: 0.9em; opacity: 0.9;">
            Automated research â€¢ Fact-checking â€¢ Multi-agent collaboration
        </p>
    </div>
    """)
    
    # Instructions
    with gr.Accordion("ðŸ“– How to Use This System", open=False):
        gr.Markdown("""
        ## System Overview
        
        This is a **multi-agent system** that generates technical reports using:
        - **Research Agent**: Gathers structured information
        - **Writer Agent**: Composes the report with proper citations
        - **Critic Agent**: Validates and revises for quality (word count, structure)
        - **Fact-Checker Agent**: Verifies claims against research
        
        ### Features:
        - âœ… **Automated Mode**: Generate complete reports automatically
        - âœ… **Human-in-the-Loop**: Edit drafts before finalization
        - âœ… **Customizable**: Control topic, temperature, references
        - âœ… **Quality Assurance**: Automatic fact-checking and validation
        
        ### Settings Explanation:
        - **Topic**: Choose from course-related subjects
        - **Temperature**: Higher = more creative, Lower = more factual (0.0-1.0)
        - **Include References**: Show all research sources with citations
        - **Save to File**: Automatically save report as .txt file
        """)
    
    # Main Tabs
    with gr.Tabs():
        
        # TAB 1: AUTOMATED GENERATION
        with gr.TabItem("ðŸ¤– Automated Generation", id=0):
            gr.Markdown("### Generate Complete Reports Automatically")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### âš™ï¸ Configuration")
                    
                    topic_dropdown = gr.Dropdown(
                        choices=list(AVAILABLE_TOPICS.keys()),
                        value="Agentic AI",
                        label="ðŸ“š Select Topic (Course-Related)",
                        info="All topics relate to course content"
                    )
                    
                    topic_description = gr.Markdown(
                        f"*{AVAILABLE_TOPICS['Agentic AI']}*"
                    )
                    
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="ðŸŒ¡ï¸ Model Temperature",
                        info="0.0 = Factual, 1.0 = Creative"
                    )
                    
                    include_refs_auto = gr.Checkbox(
                        label="ðŸ“š Include reference list",
                        value=True
                    )
                    
                    save_file_auto = gr.Checkbox(
                        label="ðŸ’¾ Save to .txt file",
                        value=True,
                        info="Saves with timestamp"
                    )
                    
                    generate_btn = gr.Button(
                        "ðŸš€ Generate Report",
                        variant="primary",
                        size="lg",
                        elem_classes="generate-btn"
                    )
                    
                    gr.Markdown("---")
                    
                    quality_display_auto = gr.Markdown("*Quality metrics will appear here*")
                
                with gr.Column(scale=2):
                    gr.Markdown("#### ðŸ“„ Generated Report")
                    report_display_auto = gr.Markdown(
                        value="*Your report will appear here after generation...*"
                    )
            
            # Update description when topic changes
            topic_dropdown.change(
                fn=lambda t: f"*{AVAILABLE_TOPICS[t]}*",
                inputs=[topic_dropdown],
                outputs=[topic_description]
            )
            
            # Generate report
            generate_btn.click(
                fn=gradio_generate_report,
                inputs=[topic_dropdown, temperature_slider, include_refs_auto, save_file_auto],
                outputs=[report_display_auto, quality_display_auto, stored_result]
            )
        
        # TAB 2: HUMAN-IN-THE-LOOP
        with gr.TabItem("âœï¸ Human-in-the-Loop (Bonus)", id=1):
            gr.Markdown("""
            ### Edit AI-Generated Drafts Before Finalization
            
            **Bonus Feature:** This mode lets you review and edit the AI draft before final submission.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Step 1: Generate Draft")
                    
                    topic_hitl = gr.Dropdown(
                        choices=list(AVAILABLE_TOPICS.keys()),
                        value="Agentic AI",
                        label="ðŸ“š Topic"
                    )
                    
                    temp_hitl = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.5,
                        step=0.1,
                        label="ðŸŒ¡ï¸ Temperature"
                    )
                    
                    draft_btn = gr.Button(
                        "ðŸ“ Generate Draft",
                        variant="secondary"
                    )
                    
                    draft_info = gr.Markdown("*Click to generate draft*")
                    
                    gr.Markdown("---")
                    gr.Markdown("#### Step 2: Finalize")
                    
                    include_refs_hitl = gr.Checkbox(
                        label="ðŸ“š Include references",
                        value=True
                    )
                    
                    save_hitl = gr.Checkbox(
                        label="ðŸ’¾ Save final version",
                        value=True
                    )
                    
                    finalize_btn = gr.Button(
                        "âœ… Finalize Edited Report",
                        variant="primary"
                    )
                    
                    final_status = gr.Markdown("")
                
                with gr.Column(scale=2):
                    gr.Markdown("#### âœï¸ Editable Draft")
                    draft_editor = gr.Textbox(
                        label="Edit the report below",
                        lines=20,
                        placeholder="Draft will appear here...",
                        interactive=True
                    )
                    
                    final_report_display = gr.Markdown("")
            
            # Generate draft
            draft_btn.click(
                fn=generate_draft_for_editing,
                inputs=[topic_hitl, temp_hitl],
                outputs=[draft_editor, draft_info, stored_result]
            )
            
            # Finalize edited report
            finalize_btn.click(
                fn=finalize_edited_report,
                inputs=[draft_editor, include_refs_hitl, save_hitl, stored_result],
                outputs=[final_report_display, final_status]
            )
        
        # TAB 3: SYSTEM DIAGRAM fdf 
        


# Export the interface
__all__ = ['report']