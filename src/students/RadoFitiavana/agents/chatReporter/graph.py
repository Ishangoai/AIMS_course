import langgraph.graph as graph
from . import actors as models
from .ChatState import ChatState

def build_graph():
    g = graph.StateGraph(ChatState)

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

    def loop_condition(state: ChatState):

        if state.ok:
            state.answer = state.last_generator_answer
            print(state.last_generator_answer)
            return graph.END

        if state.iteration_count >= 4:
            state.answer = state.last_generator_answer
            return graph.END

        return "generator"

    g.add_conditional_edges("checker", loop_condition)

    # Entry point
    g.set_entry_point("classifier")

    return g.compile()
