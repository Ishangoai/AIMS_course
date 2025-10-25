from ..tools import get_wikipedia_tool


def test_tool():
    """
    Initializes the Wikipedia tool and prints its output for a test query.
    """
    print("--- Testing Wikipedia Tool Output ---")

    wikipedia_tool = get_wikipedia_tool()

    query = "Large Language Models"
    print(f"Query: '{query}'")

    result = wikipedia_tool._run(query)

    print("\n--- Tool Result ---")
    print(result)
    print("\n--- End of Test ---")


if __name__ == "__main__":
    test_tool()
