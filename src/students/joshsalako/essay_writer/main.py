"""
This is the main entry point for the essay writer agent.
It takes a topic from the user and generates a report.
"""
from agent import run_agent
from tools import count_words


def main():
    """
    Main function to run the report generation agent.
    """
    topic = input("Please enter the topic for the report: ")
    print(f"\nGenerating a report on '{topic}'. This may take a few minutes...")

    result = run_agent(topic)

    report = result.get("final_report", "No report generated.")
    review = result.get("review", "No review available.")
    word_count = count_words(report)

    print("\n\n--- FINAL REPORT ---")
    print(report)
    print("\n\n--- REVIEW ---")
    print(review)
    print(f"\n\nWord Count: {word_count}")

    # You can also save the report to a file
    with open(f"report_on_{topic.replace(' ', '_')}.md", "w") as f:
        f.write(f"# Report on: {topic}\n\n")
        f.write(report)
        f.write("\n\n---\n\n")
        f.write("## Review\n\n")
        f.write(review)

    print(f"\nReport saved to report_on_{topic.replace(' ', '_')}.md")


if __name__ == "__main__":
    main()
