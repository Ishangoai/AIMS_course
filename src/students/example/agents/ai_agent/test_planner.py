from agents import planner_agent
from state import create_initial_state, get_state_summary

# Create initial state
topic = "The Impact of Artificial Intelligence on Software Engineering"
state = create_initial_state(topic=topic)

print("1. Initial State:")
print(get_state_summary(state))
print()

# Run the planner agent
print("2. Running PlannerAgent...")
try:
    updated_state = planner_agent(state)

    print("3. After Planning:")
    print(get_state_summary(updated_state))
    print()

    print("4. Generated Outline:")
    for i, section in enumerate(updated_state["outline"], 1):
        print(f"   {i}. {section}")
    print()

    print("5. Agent Messages:")
    for msg in updated_state["messages"]:
        print(f"   {msg['role']}: {msg['content']}")
    print()

    if updated_state["errors"]:
        print("6. Errors:")
        for error in updated_state["errors"]:
            print(f"   - {error}")

    if updated_state["warnings"]:
        print("7. Warnings:")
        for warning in updated_state["warnings"]:
            print(f"   - {warning}")

    print("🎉 PlannerAgent run completed successfully!")

except Exception as e:
    print(f"❌ PlannerAgent run failed: {e}")
    import traceback

    traceback.print_exc()
