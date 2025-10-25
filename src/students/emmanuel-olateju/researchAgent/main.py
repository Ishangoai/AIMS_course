"""
Main script to run the agentic report generation system.
Uses existing environment variables.
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Verify API key is available in environment
if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError(
        "GOOGLE_API_KEY not found in environment variables.\n"
        "Please set it with: export GOOGLE_API_KEY='your-api-key'"
    )

from agent import run_agent


def main():
    """Run the report generation workflow."""
    print("=" * 70)
    print("🤖 AGENTIC REPORT GENERATION SYSTEM")
    print("=" * 70)

    # Generate a report
    result = run_agent(
        topic="Artificial Intelligence in Mechanical Engineering",
        temperature=0.7,
        max_iterations=3
    )

    print("\n" + "=" * 70)
    print("📄 FINAL REPORT")
    print("=" * 70)
    print(result["final_report"])

    print("\n" + "=" * 70)
    print("📊 QUALITY METRICS")
    print("=" * 70)
    for key, value in result["quality_metrics"].items():
        print(f"  {key}: {value}")

    print("\n" + "=" * 70)
    print("ℹ️  METADATA")
    print("=" * 70)
    for key, value in result["metadata"].items():
        if key != "review_feedback":  # Print review separately
            print(f"  {key}: {value}")

    if "review_feedback" in result["metadata"]:
        print("\n📝 Editorial Review:")
        print(result["metadata"]["review_feedback"])

    # Optional: Save report to file
    output_file = "generated_report.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"# {result['metadata']['title']}\n\n")
        f.write(result["final_report"])

    print(f"\n✅ Report saved to: {output_file}")


if __name__ == "__main__":
    main()
