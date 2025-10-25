import langgraph.graph as graph
import .actors as models

def build_graph():
    g = graph.StateGraph()

    classifier = models.make_classifier()
    retriever = models.make_retriever()
    generator = models.make_generator()
    checker = models.make_checker()

    g.add_node("classifier", classifier)
    g.add_node("retriever", retriever)
    g.add_node("generator", generator)
    g.add_node("checker", checker)

    g.add_edge("classifier", "retriever")
    g.add_edge("retriever", "generator")
    g.add_edge("generator", "checker")

    def loop_condition(state):
        return graph.END if state.get("ok", False) else "generator"

    g.add_conditional_edges("checker", loop_condition)
    g.set_entry_point("classifier")

    visualize_graph()
    return g.compile()