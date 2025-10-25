from langgraph.graph import END, StateGraph

from .agents import ReportState, editor_agent, fact_checker_agent, quality_control_agent, research_agent, writer_agent


def should_continue(state: ReportState) -> str:
    """
    Determine if the workflow should continue or end.

    Args:
        state: Current workflow state

    Returns:
        "continue" to revise, "end" to finish
    """
    if state["status"] == "COMPLETE":
        return "end"

    if state["iteration"] >= state["max_iterations"]:
        state["status"] = "MAX_ITERATIONS"
        state["final_report"] = state.get("edited_report", state.get("draft", ""))
        state["messages"].append("⚠️ Max iterations reached, using best available version")
        return "end"

    return "continue"


def create_workflow() -> StateGraph:
    """
    Create the LangGraph workflow for report generation.

    Returns:
        Compiled StateGraph workflow
    """
    # Create the graph
    workflow = StateGraph(ReportState)

    # Add nodes for each agent
    workflow.add_node("research", research_agent)
    workflow.add_node("writer", writer_agent)
    workflow.add_node("fact_checker", fact_checker_agent)
    workflow.add_node("editor", editor_agent)
    workflow.add_node("quality_control", quality_control_agent)

    # Define the workflow edges
    workflow.set_entry_point("research")
    workflow.add_edge("research", "writer")
    workflow.add_edge("writer", "fact_checker")
    workflow.add_edge("fact_checker", "editor")
    workflow.add_edge("editor", "quality_control")

    # Add conditional edge for iteration
    workflow.add_conditional_edges(
        "quality_control",
        should_continue,
        {
            "continue": "writer",  # Go back to writer for revision
            "end": END,
        },
    )

    # Compile the graph
    return workflow.compile()


def generate_report(
    topic: str, temperature: float = 0.7, max_iterations: int = 3, human_feedback: str = None, progress_callback=None
) -> dict:
    """
    Generate a report using the agentic workflow.

    Args:
        topic: The topic to write about
        temperature: LLM temperature (0.0-1.0)
        max_iterations: Maximum revision iterations
        human_feedback: Optional human feedback for revision
        progress_callback: Optional callback function for progress updates

    Returns:
        Dictionary with final report and metadata
    """
    # Initialize state
    initial_state: ReportState = {
        "topic": topic,
        "temperature": temperature,
        "research": None,
        "draft": None,
        "fact_check": None,
        "edited_report": None,
        "final_report": None,
        "word_count": 0,
        "feedback": None,
        "human_feedback": human_feedback,
        "iteration": 0,
        "max_iterations": max_iterations,
        "status": "IN_PROGRESS",
        "messages": [],
    }

    # Store progress callback in state for agents to use
    if progress_callback:
        initial_state["progress_callback"] = progress_callback

    # Create and run workflow
    workflow = create_workflow()
    final_state = workflow.invoke(initial_state)

    # Always return the best available report, even if it doesn't meet all requirements
    report = final_state.get("final_report") or final_state.get("edited_report") or final_state.get("draft") or ""

    return {
        "report": report,
        "word_count": final_state.get("word_count", 0),
        "status": final_state.get("status", "UNKNOWN"),
        "iterations": final_state.get("iteration", 0),
        "messages": final_state.get("messages", []),
    }


def get_workflow_diagram():
    """
    Get the workflow diagram for visualization.

    Returns:
        Mermaid diagram string
    """
    workflow = create_workflow()
    try:
        return workflow.get_graph().draw_mermaid_png()
    except Exception:
        return """
graph TD
    A[Start] --> B[Research Agent]
    B --> C[Writer Agent]
    C --> D[Fact Checker Agent]
    D --> E[Editor Agent]
    E --> F[Quality Control Agent]
    F -->|PASS| G[End]
    F -->|NEEDS_REVISION| C
    F -->|MAX_ITERATIONS| G
"""


def generate_diagram_png():
    """Generate and save the workflow diagram as PNG."""

    print("=" * 70)
    print("Generating Workflow Diagram")
    print("=" * 70)

    workflow = create_workflow()

    try:
        # Try to get PNG from LangGraph
        print("\nAttempting to generate PNG using LangGraph...")
        graph = workflow.get_graph()

        # Try to draw as PNG
        try:
            png_data = graph.draw_mermaid_png()
            with open("workflow_diagram.png", "wb") as f:
                f.write(png_data)
            print("✓ PNG diagram saved to: workflow_diagram.png")
            return True
        except Exception as e:
            print(f"PNG generation not available: {e}")

        # Fallback: Save mermaid and provide instructions
        print("\nGenerating Mermaid diagram (text format)...")
        diagram = graph.draw_mermaid()

        with open("workflow_diagram.mmd", "w") as f:
            f.write(diagram)
        print("✓ Mermaid diagram saved to: workflow_diagram.mmd")

        print("\n" + "=" * 70)
        print("To convert to PNG, you can:")
        print("=" * 70)
        print("1. Visit: https://mermaid.live/")
        print("2. Paste the content of workflow_diagram.mmd")
        print("3. Export as PNG")
        print("\nOr use the create_diagram_image.py script instead.")
        print("=" * 70)

        return False

    except Exception as e:
        print(f"\n✗ Could not generate diagram automatically: {e}")
        print("\nCreating manual diagram using matplotlib...")

        # Fallback to matplotlib
        try:
            import matplotlib.patches as mpatches  # noqa: F401
            import matplotlib.pyplot as plt
            from matplotlib.patches import FancyBboxPatch

            _, ax = plt.subplots(1, 1, figsize=(12, 10))
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 12)
            ax.axis("off")

            # Title
            ax.text(
                5,
                11.5,
                "Agentic Report Writing System",
                ha="center",
                va="top",
                fontsize=16,
                fontweight="bold",
            )

            # Colors
            colors = {
                "research": "#e1f5ff",
                "writer": "#fff4e1",
                "fact_checker": "#ffe1e1",
                "editor": "#e1ffe1",
                "qc": "#f0e1ff",
                "start_end": "#d0d0d0",
            }

            # Start
            start = FancyBboxPatch(
                (4, 10),
                2,
                0.6,
                boxstyle="round,pad=0.1",
                facecolor=colors["start_end"],
                edgecolor="black",
                linewidth=2,
            )
            ax.add_patch(start)
            ax.text(5, 10.3, "START", ha="center", va="center", fontweight="bold")

            # Research Agent
            research = FancyBboxPatch(
                (3.5, 8.5),
                3,
                0.8,
                boxstyle="round,pad=0.1",
                facecolor=colors["research"],
                edgecolor="black",
                linewidth=2,
            )
            ax.add_patch(research)
            ax.text(5, 8.9, "Research Agent", ha="center", va="center", fontweight="bold")

            # Writer Agent
            writer = FancyBboxPatch(
                (3.5, 7),
                3,
                0.8,
                boxstyle="round,pad=0.1",
                facecolor=colors["writer"],
                edgecolor="black",
                linewidth=2,
            )
            ax.add_patch(writer)
            ax.text(5, 7.4, "Writer Agent", ha="center", va="center", fontweight="bold")

            # Fact Checker Agent
            fact_checker = FancyBboxPatch(
                (3.5, 5.5),
                3,
                0.8,
                boxstyle="round,pad=0.1",
                facecolor=colors["fact_checker"],
                edgecolor="black",
                linewidth=2,
            )
            ax.add_patch(fact_checker)
            ax.text(
                5,
                5.9,
                "Fact Checker Agent",
                ha="center",
                va="center",
                fontweight="bold",
            )

            # Editor Agent
            editor = FancyBboxPatch(
                (3.5, 4),
                3,
                0.8,
                boxstyle="round,pad=0.1",
                facecolor=colors["editor"],
                edgecolor="black",
                linewidth=2,
            )
            ax.add_patch(editor)
            ax.text(5, 4.4, "Editor Agent", ha="center", va="center", fontweight="bold")

            # Quality Control Agent
            qc = FancyBboxPatch(
                (3.5, 2.5),
                3,
                0.8,
                boxstyle="round,pad=0.1",
                facecolor=colors["qc"],
                edgecolor="black",
                linewidth=2,
            )
            ax.add_patch(qc)
            ax.text(
                5,
                2.9,
                "Quality Control Agent",
                ha="center",
                va="center",
                fontweight="bold",
            )

            # End
            end = FancyBboxPatch(
                (4, 0.8),
                2,
                0.6,
                boxstyle="round,pad=0.1",
                facecolor=colors["start_end"],
                edgecolor="black",
                linewidth=2,
            )
            ax.add_patch(end)
            ax.text(5, 1.1, "END", ha="center", va="center", fontweight="bold")

            # Arrows
            arrow_props = dict(arrowstyle="->", lw=2, color="black")
            ax.annotate("", xy=(5, 9.3), xytext=(5, 10), arrowprops=arrow_props)
            ax.annotate("", xy=(5, 7.8), xytext=(5, 8.5), arrowprops=arrow_props)
            ax.annotate("", xy=(5, 6.3), xytext=(5, 7), arrowprops=arrow_props)
            ax.annotate("", xy=(5, 4.8), xytext=(5, 5.5), arrowprops=arrow_props)
            ax.annotate("", xy=(5, 3.3), xytext=(5, 4), arrowprops=arrow_props)
            ax.annotate("", xy=(5, 1.4), xytext=(5, 2.5), arrowprops=arrow_props)

            # Feedback loop
            feedback_props = dict(arrowstyle="->", lw=2, color="red", linestyle="--")
            ax.annotate("", xy=(6.5, 7.4), xytext=(6.5, 2.9), arrowprops=feedback_props)
            ax.annotate("", xy=(5, 7.4), xytext=(6.5, 7.4), arrowprops=feedback_props)
            ax.text(
                7,
                5,
                "NEEDS\nREVISION",
                ha="left",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="red",
            )

            plt.tight_layout()
            plt.savefig("workflow_diagram.png", dpi=300, bbox_inches="tight", facecolor="white")
            print("✓ PNG diagram saved to: workflow_diagram.png")
            plt.close()
            return True

        except ImportError:
            print("✗ matplotlib not available. Please install: pip install matplotlib")
            return False
        except Exception as e2:
            print(f"✗ Error creating matplotlib diagram: {e2}")
            return False


if __name__ == "__main__":
    generate_diagram_png()
