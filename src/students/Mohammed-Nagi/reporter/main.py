"""
Entry point for the report generation system.
Provides CLI interface and result formatting.
"""

import sys
from pathlib import Path

from .agent import run_agent, visualize_graph


def print_separator(char="=", length=70):
    """Prints a visual separator."""
    print("\n" + char * length + "\n")


def format_report_output(result: dict) -> str:
    """
    Formats the final report with metadata for display/saving.
    """
    output = []

    # Add title
    output.append("=" * 70)
    output.append(f"GENERATED REPORT: {result['metadata']['title']}")
    output.append("=" * 70)
    output.append("")

    # Add metadata
    output.append("GENERATION STATISTICS:")
    output.append(f"  • Topic: {result['metadata']['topic']}")
    output.append(f"  • Word Count: {result['metadata']['final_word_count']}")
    output.append(f"  • Sections: {result['metadata']['total_sections']}")
    output.append(f"  • Iterations: {result['metadata']['iterations_needed']}")
    output.append(f"  • Validation: {'✓ Passed' if result['metadata']['validation_passed'] else '✗ Failed'}")
    output.append("")
    output.append("-" * 70)
    output.append("")

    # Add the report
    output.append(result['final_report'])
    output.append("")
    output.append("-" * 70)
    output.append("")

    # Add review
    output.append("EDITORIAL REVIEW:")
    output.append(result['review_feedback'])
    output.append("")
    output.append("=" * 70)

    return "\n".join(output)


def save_report(content: str, filename: str = "generated_report.md"):
    """Saves report to file."""
    filepath = Path(filename)
    filepath.write_text(content, encoding="utf-8")
    print(f"\n✓ Report saved to: {filepath.absolute()}")


def main(topic: str = None, save_to_file: bool = True, visualize: bool = False):
    """
    Main function to run the report generation system.

    Args:
        topic: Subject for the report. If None, will prompt user.
        save_to_file: Whether to save output to markdown file
        visualize: Whether to generate workflow diagram

    Usage:
        # Interactive mode
        python -m your_package.main

        # Direct mode
        python -m your_package.main "MLOps Best Practices"

        # With visualization
        python -m your_package.main "CI/CD" --visualize
    """

    # Handle visualization request
    if visualize:
        print("\nGenerating workflow diagram...")
        try:
            diagram = visualize_graph()
            print("✓ Workflow diagram generated!")
            print("\nTo view: save the output to a .png file or use a Mermaid viewer.")
            print(diagram if isinstance(diagram, str) else "PNG generated successfully")
            return
        except Exception as e:
            print(f"✗ Could not generate diagram: {e}")
            return

    # Get topic
    if topic is None:
        print("\n" + "=" * 70)
        print("AGENTIC REPORT GENERATION SYSTEM")
        print("=" * 70)
        print("\nSuggested topics:")
        print("  • MLOps Best Practices")
        print("  • CI/CD Pipelines for Machine Learning")
        print("  • API Design Patterns")
        print("  • Gradio for ML Interfaces")
        print()
        topic = input("Enter your report topic: ").strip()

        if not topic:
            print("✗ No topic provided. Exiting.")
            sys.exit(1)

    # Generate report
    print_separator()
    print(f"🤖 Generating report on '{topic}'")
    print("This will take 2-4 minutes...")
    print_separator()

    try:
        # Run the agent
        result = run_agent(topic)

        # Format output
        formatted_output = format_report_output(result)

        # Display to console
        print_separator("=")
        print(formatted_output)

        # Save to file
        if save_to_file:
            filename = f"report_{topic.replace(' ', '_').lower()}.md"
            save_report(formatted_output, filename)

        # Print summary
        print_separator()
        print("✓ GENERATION COMPLETE!")
        print(f"  Final word count: {result['metadata']['final_word_count']} words")
        print(f"  Revisions needed: {result['metadata']['iterations_needed'] - 1}")
        print(f"  Quality check: {'✓ Passed' if result['metadata']['validation_passed'] else '✗ See review'}")
        print_separator()

        return result

    except KeyboardInterrupt:
        print("\n\n✗ Generation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n✗ Error during generation: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure GOOGLE_API_KEY environment variable is set")
        print("  2. Check your internet connection (Wikipedia access needed)")
        print("  3. Verify all dependencies are installed")
        sys.exit(1)


if __name__ == "__main__":
    # Handle command line arguments
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a technical report using an agentic AI system"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        default=None,
        help="Topic for the report (e.g., 'MLOps Best Practices')"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save report to file"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate and display workflow diagram"
    )

    args = parser.parse_args()

    main(
        topic=args.topic,
        save_to_file=not args.no_save,
        visualize=args.visualize
    )
