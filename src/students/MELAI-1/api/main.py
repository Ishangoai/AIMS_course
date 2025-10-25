# ============================================================================
# FICHIER CORRIGÉ: main.py
# MODIFICATION CRITIQUE: should_revise() et critic_node() pour arrêt immédiat
# ============================================================================

import os
import textwrap
import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.openapi.docs import get_swagger_ui_html

# Import des agents
from agentic.critic_agent import critic_agent
from agentic.fact_checker_agent import fact_checker 
from agentic.research_agent import research_agent
from agentic.writer_agent import writer_agent

from api.models import UpdateUserRequest, UserRequest
from api.safe_eval import safe_eval

from dotenv import load_dotenv
load_dotenv()

GOOGLE_API_KEY="AIzaSyBDQqvlBKGdvEAjSvX5yBdN5ObruaOzwfU"
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID", "")

TARGET_WORD_COUNT = 1000
WORD_COUNT_MIN = 950
WORD_COUNT_MAX = 1050
MAX_REVISION_ROUNDS = 5
REPORT_TOPIC = "Agentic AI"
NUM_RESEARCH_BULLETS = 10
NUM_FACT_CHECKS = 6
MODEL_NAME = "gemini-2.0-flash-exp"

print("✅ Configuration loaded")
print(f"   Topic: {REPORT_TOPIC}")
print(f"   Target: {WORD_COUNT_MIN}-{WORD_COUNT_MAX} words")

current_user = os.environ.get("GITHUB_USER", "default")
users = {}

# ============================================================================
# WORD COUNT ENFORCEMENT TOOL
# ============================================================================
import re

def count_words(text: str) -> int:
    """Efficient Python word count using regex."""
    return len(re.findall(r"\b\w+\b", text))


# ============================================================================
# STATE DEFINITION
# ============================================================================
from typing import TypedDict, List, Dict

class AgentState(TypedDict):
    """Shared state across all agents in the workflow."""
    topic: str
    research_bullets: List[Dict]
    report_draft: str
    word_count: int
    within_range: bool
    structure_valid: bool
    revision_count: int
    fact_checks: List[Dict]
    final_report: str
    metadata: Dict


# ============================================================================
# WORKFLOW CREATION - AVEC CORRECTION CRITIQUE
# ============================================================================
from langgraph.graph import StateGraph, END
from datetime import datetime

def create_workflow():
    """Create LangGraph workflow for multi-agent orchestration."""

    # Agent Nodes
    def research_node(state: AgentState) -> AgentState:
        """Research Agent Node"""
        print("\n🔬 RESEARCH AGENT EXECUTING...")
        bullets = research_agent.gather_research(state["topic"])
        return {**state, "research_bullets": bullets}

    def writer_node(state: AgentState) -> AgentState:
        """Writer Agent Node"""
        print("\n✏️ WRITER AGENT EXECUTING...")
        result = writer_agent.write_report(state["topic"], state["research_bullets"])
        
        # ✅ VÉRIFICATION IMMÉDIATE après écriture
        word_count = result["word_count"]
        within_range = WORD_COUNT_MIN <= word_count <= WORD_COUNT_MAX
        
        print(f"   📊 Draft: {word_count} words")
        print(f"   ✓ Within range: {within_range}")
        
        return {
            **state,
            "report_draft": result["draft"],
            "word_count": word_count,
            "within_range": within_range,
            "structure_valid": result["structure_valid"]
        }

    def critic_node(state: AgentState) -> AgentState:
        """
        Critic Agent Node - MODIFIÉ pour utiliser revise_until_valid
        Cette fonction gère TOUTE la boucle de révision en interne
        """
        print("\n🔍 CRITIC AGENT EXECUTING...")
        
        # ✅ UTILISE revise_until_valid qui s'arrête immédiatement
        result = critic_agent.revise_until_valid(state["report_draft"])
        
        return {
            **state,
            "report_draft": result["draft"],
            "revision_count": result["revision_count"],
            "word_count": result["word_count"],
            "within_range": result["within_range"],
            "structure_valid": result["structure_valid"]
        }

    def fact_check_node(state: AgentState) -> AgentState:
        """Fact-Checker Agent Node"""
        print("\n🎯 FACT-CHECKER AGENT EXECUTING...")
        claims = fact_checker.extract_claims(state["report_draft"])
        checks = fact_checker.verify_claims(claims, state["research_bullets"])
        return {
            **state,
            "fact_checks": checks,
            "final_report": state["report_draft"]
        }

    # ✅ ROUTING LOGIC CORRIGÉE - Arrêt immédiat si dans l'intervalle
    def should_revise(state: AgentState) -> str:
        """
        Conditional routing for revision loop.
        ARRÊT IMMÉDIAT si within_range ET structure_valid
        """
        word_count = state["word_count"]
        within_range = state["within_range"]
        structure_valid = state["structure_valid"]
        revision_count = state["revision_count"]
        
        print(f"\n🔄 ROUTING DECISION:")
        print(f"   Word count: {word_count} (target: {WORD_COUNT_MIN}-{WORD_COUNT_MAX})")
        print(f"   Within range: {within_range}")
        print(f"   Structure valid: {structure_valid}")
        print(f"   Revisions done: {revision_count}")
        
        # ✅ PREMIÈRE PRIORITÉ: Si dans l'intervalle ET structure valide → ARRÊT
        if within_range and structure_valid:
            print(f"   ✅ → VALIDATION RÉUSSIE - Proceeding to fact-check")
            return "fact_check"
        
        # ✅ DEUXIÈME PRIORITÉ: Si max révisions atteint → ARRÊT forcé
        if revision_count >= MAX_REVISION_ROUNDS:
            print(f"   ⚠️ → Max revisions ({MAX_REVISION_ROUNDS}) reached - Force fact-check")
            return "fact_check"
        
        # ❌ Sinon, besoin de révision
        print(f"   🔄 → Needs revision")
        return "critic"

    # Build Workflow
    workflow = StateGraph(AgentState)

    workflow.add_node("research", research_node)
    workflow.add_node("writer", writer_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("fact_check", fact_check_node)

    workflow.set_entry_point("research")
    workflow.add_edge("research", "writer")

    # ✅ EDGES CONDITIONNELS CORRIGÉS
    workflow.add_conditional_edges("writer", should_revise)
    
    # ✅ IMPORTANT: Après critic, on vérifie TOUJOURS si on doit continuer
    # Car critic_node utilise maintenant revise_until_valid qui peut tout résoudre
    workflow.add_conditional_edges("critic", should_revise)

    workflow.add_edge("fact_check", END)

    return workflow.compile()


# ============================================================================
# MAIN EXECUTION FUNCTION
# ============================================================================
def generate_report():
    """Main execution function - runs the complete multi-agent workflow."""
    print("\n" + "="*70)
    print("🚀 AGENTIC AI REPORT GENERATOR")
    print("="*70)
    print(f"Topic: {REPORT_TOPIC}")
    print(f"Target: {WORD_COUNT_MIN}-{WORD_COUNT_MAX} words")
    print(f"Model: {MODEL_NAME}")
    print("="*70)

    app = create_workflow()

    initial_state = {
        "topic": REPORT_TOPIC,
        "research_bullets": [],
        "report_draft": "",
        "word_count": 0,
        "within_range": False,
        "structure_valid": False,
        "revision_count": 0,
        "fact_checks": [],
        "final_report": "",
        "metadata": {}
    }

    start_time = datetime.now()

    try:
        final_state = app.invoke(initial_state)
        execution_time = (datetime.now() - start_time).total_seconds()

        # Python-based word count enforcement
        print("\n" + "="*70)
        print("📊 FINAL WORD COUNT VALIDATION")
        print("="*70)

        final_word_count = count_words(final_state["final_report"])
        final_state["word_count"] = final_word_count
        final_state["within_range"] = WORD_COUNT_MIN <= final_word_count <= WORD_COUNT_MAX

        print(f"   Current: {final_word_count}")
        print(f"   Target: {WORD_COUNT_MIN}-{WORD_COUNT_MAX}")
        print(f"   Status: {'✅ WITHIN RANGE' if final_state['within_range'] else '⚠️ OUT OF RANGE'}")

        # ✅ Cette partie ne devrait PAS être nécessaire avec le nouveau système
        if not final_state["within_range"]:
            print(f"\n⚠️ WARNING: Final report still out of range after all revisions")
            print(f"   This should not happen with the new revision system")

        # Compile metadata
        verified = sum(1 for fc in final_state["fact_checks"]
                      if fc.get("status") in ["VERIFIED", "SUPPORTED"])

        metadata = {
            "word_count": final_state["word_count"],
            "within_range": final_state["within_range"],
            "structure_valid": final_state["structure_valid"],
            "revision_rounds": final_state["revision_count"],
            "research_bullets": len(final_state["research_bullets"]),
            "verified_claims": f"{verified}/{len(final_state['fact_checks'])}",
            "execution_time": f"{execution_time:.2f}s",
            "timestamp": datetime.now().isoformat()
        }

        print("\n" + "="*70)
        print("✅ GENERATION COMPLETE")
        print("="*70)
        for k, v in metadata.items():
            print(f"   {k.replace('_', ' ').title()}: {v}")
        print("="*70)

        return {
            "report": final_state["final_report"],
            "research": final_state["research_bullets"],
            "fact_checks": final_state["fact_checks"],
            "metadata": metadata
        }

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# CUSTOM REPORT GENERATION WITH PARAMETERS
# ============================================================================
def generate_report_with_params(topic: str, temperature: float = 0.5):
    """
    Generate report with custom topic and temperature.
    UTILISE LE MÊME SYSTÈME D'ARRÊT IMMÉDIAT
    """
    print("\n" + "="*70)
    print("🚀 CUSTOM REPORT GENERATOR")
    print("="*70)
    print(f"Topic: {topic}")
    print(f"Temperature: {temperature}")
    print(f"Target: {WORD_COUNT_MIN}-{WORD_COUNT_MAX} words")
    print("="*70)

    from agentic.research_agent import ResearchAgent
    from agentic.writer_agent import WriterAgent
    from agentic.critic_agent import CriticAgent
    from agentic.fact_checker_agent import FactCheckerAgent
    
    custom_research = ResearchAgent(api_key=GOOGLE_API_KEY, model=MODEL_NAME)
    custom_research.llm.temperature = temperature
    
    custom_writer = WriterAgent(api_key=GOOGLE_API_KEY, model=MODEL_NAME)
    custom_writer.llm.temperature = temperature
    
    custom_critic = CriticAgent(api_key=GOOGLE_API_KEY, model=MODEL_NAME)
    custom_fact_checker = FactCheckerAgent(api_key=GOOGLE_API_KEY, model=MODEL_NAME)

    def create_custom_workflow():
        """Create workflow with custom agents and topic."""
        
        def research_node(state: AgentState) -> AgentState:
            print("\n🔬 RESEARCH AGENT EXECUTING...")
            bullets = custom_research.gather_research(topic, NUM_RESEARCH_BULLETS)
            return {**state, "research_bullets": bullets}

        def writer_node(state: AgentState) -> AgentState:
            print("\n✏️ WRITER AGENT EXECUTING...")
            result = custom_writer.write_report(topic, state["research_bullets"])
            
            word_count = result["word_count"]
            within_range = WORD_COUNT_MIN <= word_count <= WORD_COUNT_MAX
            
            print(f"   📊 Draft: {word_count} words")
            print(f"   ✓ Within range: {within_range}")
            
            return {
                **state,
                "report_draft": result["draft"],
                "word_count": word_count,
                "within_range": within_range,
                "structure_valid": result["structure_valid"]
            }

        def critic_node(state: AgentState) -> AgentState:
            """UTILISE revise_until_valid pour arrêt immédiat"""
            print("\n🔍 CRITIC AGENT EXECUTING...")
            result = custom_critic.revise_until_valid(state["report_draft"])
            
            return {
                **state,
                "report_draft": result["draft"],
                "revision_count": result["revision_count"],
                "word_count": result["word_count"],
                "within_range": result["within_range"],
                "structure_valid": result["structure_valid"]
            }

        def fact_check_node(state: AgentState) -> AgentState:
            print("\n🎯 FACT-CHECKER AGENT EXECUTING...")
            claims = custom_fact_checker.extract_claims(state["report_draft"])
            checks = custom_fact_checker.verify_claims(claims, state["research_bullets"])
            return {
                **state,
                "fact_checks": checks,
                "final_report": state["report_draft"]
            }

        def should_revise(state: AgentState) -> str:
            """MÊME LOGIQUE: Arrêt immédiat si dans l'intervalle"""
            word_count = state["word_count"]
            within_range = state["within_range"]
            structure_valid = state["structure_valid"]
            revision_count = state["revision_count"]
            
            print(f"\n🔄 ROUTING DECISION:")
            print(f"   Word count: {word_count}")
            print(f"   Within range: {within_range}")
            print(f"   Structure valid: {structure_valid}")
            
            # ✅ Arrêt immédiat si valide
            if within_range and structure_valid:
                print(f"   ✅ → VALIDATED - Proceeding to fact-check")
                return "fact_check"
            
            if revision_count >= MAX_REVISION_ROUNDS:
                print(f"   ⚠️ → Max revisions reached")
                return "fact_check"
            
            print(f"   🔄 → Needs revision")
            return "critic"

        workflow = StateGraph(AgentState)
        workflow.add_node("research", research_node)
        workflow.add_node("writer", writer_node)
        workflow.add_node("critic", critic_node)
        workflow.add_node("fact_check", fact_check_node)
        
        workflow.set_entry_point("research")
        workflow.add_edge("research", "writer")
        workflow.add_conditional_edges("writer", should_revise)
        workflow.add_conditional_edges("critic", should_revise)
        workflow.add_edge("fact_check", END)
        
        return workflow.compile()

    app = create_custom_workflow()

    initial_state = {
        "topic": topic,
        "research_bullets": [],
        "report_draft": "",
        "word_count": 0,
        "within_range": False,
        "structure_valid": False,
        "revision_count": 0,
        "fact_checks": [],
        "final_report": "",
        "metadata": {}
    }

    start_time = datetime.now()

    try:
        final_state = app.invoke(initial_state)
        execution_time = (datetime.now() - start_time).total_seconds()

        print("\n" + "="*70)
        print("📊 FINAL VALIDATION")
        print("="*70)

        final_word_count = count_words(final_state["final_report"])
        final_state["word_count"] = final_word_count
        final_state["within_range"] = WORD_COUNT_MIN <= final_word_count <= WORD_COUNT_MAX

        print(f"   Word Count: {final_word_count}")
        print(f"   Target: {WORD_COUNT_MIN}-{WORD_COUNT_MAX}")
        print(f"   Status: {'✅ OK' if final_state['within_range'] else '⚠️ OUT OF RANGE'}")

        verified = sum(1 for fc in final_state["fact_checks"]
                      if fc.get("status") in ["VERIFIED", "SUPPORTED"])

        metadata = {
            "word_count": final_state["word_count"],
            "within_range": final_state["within_range"],
            "structure_valid": final_state["structure_valid"],
            "revision_rounds": final_state["revision_count"],
            "research_bullets": len(final_state["research_bullets"]),
            "verified_claims": f"{verified}/{len(final_state['fact_checks'])}",
            "execution_time": f"{execution_time:.2f}s",
            "timestamp": datetime.now().isoformat(),
            "topic": topic,
            "temperature": temperature
        }

        print("\n" + "="*70)
        print("✅ GENERATION COMPLETE")
        print("="*70)
        for k, v in metadata.items():
            print(f"   {k.replace('_', ' ').title()}: {v}")
        print("="*70)

        return {
            "report": final_state["final_report"],
            "research": final_state["research_bullets"],
            "fact_checks": final_state["fact_checks"],
            "metadata": metadata
        }

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


# ============================================================================
# FASTAPI APP (reste inchangé)
# ============================================================================
app = FastAPI(
    title="AIMS Course API",
    description=textwrap.dedent("""
    ## Mounted Apps
    ----
    1. [**Heart Disease Prediction App**](/heart-disease/)
    2. [**Simple LLM Chatbot**](/llm-chat/)
    3. [**Report Generator**](/report/)
   
    -----
    """),
    version="1.0.0",
    contact={"name": "Support Team", "email": "vincent@ishango.ai"},
    redirect_slashes=False,
)

@app.get("/", include_in_schema=False)
def root():
    return get_swagger_ui_html(openapi_url="/openapi.json", title="AIMS Course API Docs")

@app.get("/hello")
def hello():
    return {"message": f"Hello from {current_user}!"}

@app.get("/evaluate")
def evaluate(expression: str):
    try:
        result = safe_eval(expression)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/register")
def register_user(request: UserRequest):
    username = request.username
    if not username:
        raise HTTPException(status_code=400, detail="Username is required")
    users[username] = request.model_dump().get("name", None)
    return {"message": f"User {username} registered successfully"}

@app.get("/register")
def get_registered_users():
    return {"users": users}

@app.get("/register/{username}")
def get_user_details(username: str):
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    return {"username": username, "name": users[username]}

@app.delete("/register/{username}/delete")
def delete_user(username: str):
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    del users[username]
    return {"message": f"User {username} deleted successfully"}

@app.put("/register/{username}")
def update_user_details(username: str, request: UpdateUserRequest):
    if username not in users:
        raise HTTPException(status_code=404, detail="User not found")
    users[username] = request.model_dump().get("name", None)
    return {"message": f"User {username} updated successfully"}

# Mount Gradio apps
from gradioapp.agentic_gradio import report
from gradioapp.heart_disease_app import heart_app
from agents.chatbot.llm_gradio import llm_chat

gr.mount_gradio_app(app, heart_app, path="/heart-disease")
gr.mount_gradio_app(app, llm_chat, path="/llm-chat")
gr.mount_gradio_app(app, report, path="/report")


if __name__ == "__main__":
    print("🚀 Testing report generation...")
    result = generate_report()
    print("\n📄 Preview:")
    print(result["report"][:500] + "...")