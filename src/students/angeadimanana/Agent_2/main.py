"""
Main execution file for the agentic report system
"""

from config import GOOGLE_API_KEY, MODEL_NAME, TEMPERATURE
from langchain_google_genai import ChatGoogleGenerativeAI
from workflow.report_workflow import ReportWorkflow


def main():
    """
    Main function to run the report generation workflow
    """
    # Initialize the LLM
    print("Initializing LLM...")
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        google_api_key=GOOGLE_API_KEY
    )
    print("✓ LLM initialized")

    # Get topic from user
    print("\n" + "=" * 60)
    topic = input("Enter the topic for the report: ").strip()

    if not topic:
        topic = "RESTful APIs"  # Default topic
        print(f"No topic provided. Using default: {topic}")

    # Create workflow instance
    workflow = ReportWorkflow(llm)

    # Run the workflow
    results = workflow.run(topic, verbose=True)

    # Display final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Status: {results['status']}")
    print(f"QA Score: {results['qa_score']:.2f}/1.0")
    print(f"Word Count: {results['word_count']} words")
    print(f"Iterations: {results['iterations']}")
    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(results['final_report'])
    print("\n" + "=" * 60)
    print("QA EVALUATION")
    print("=" * 60)
    print(results['qa_results'])

    # Save to file
    save_option = input("\n\nSave report to file? (y/n): ").strip().lower()
    if save_option == 'y':
        filename = input("Enter filename (default: report.md): ").strip()
        if not filename:
            filename = "report.md"

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Report: {topic}\n\n")
            f.write(f"**Status:** {results['status']}\n")
            f.write(f"**QA Score:** {results['qa_score']:.2f}/1.0\n")
            f.write(f"**Word Count:** {results['word_count']} words\n")
            f.write(f"**Iterations:** {results['iterations']}\n\n")
            f.write("---\n\n")
            f.write(results['final_report'])
            f.write("\n\n---\n\n")
            f.write("## QA Evaluation\n\n")
            f.write(results['qa_results'])

        print(f"✓ Report saved to {filename}")


if __name__ == "__main__":
    main()
