"""
This is the main entry point for the essay writer agent.
It takes a topic from the user and generates a report.
"""
from .agent import run_agent


def main(topic):
    """
    Main function to run the report generation agent.
    """
    print(f"\nGenerating a report on '{topic}'. This may take a few minutes...")

    result = run_agent(topic)
    return result
