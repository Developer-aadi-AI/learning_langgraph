from langgraph.graph import StateGraph, START, END
from states import PGState
from node_func import get_context, MakePost, EvaluatePost, OptimizePost, initialize


graph = StateGraph(PGState)



def check_context(state: PGState):
    return "context" if state.file is not None else "post"


def check_post(state: PGState):
    if state.evaluation == 'Approved' or state.iteration >= state.max_iteration:
        return "Approved"
    elif state.evaluation == "Needs Improvement":
        return "Optimize"
    else:
        return "Approved"
    
graph.add_node("Initialize", initialize)
graph.add_node("Get_Context", get_context)
graph.add_node("Make_Post", MakePost)
graph.add_node("Evaluate_Post", EvaluatePost)
graph.add_node("Optimize_Post", OptimizePost)


graph.add_edge(START, "Initialize")
graph.add_conditional_edges("Initialize", check_context, {"context": "Get_Context", "post":"Make_Post"})
graph.add_edge("Get_Context", "Make_Post")
graph.add_edge("Make_Post","Evaluate_Post")
graph.add_conditional_edges("Evaluate_Post",check_post, {"Approved":END, "Optimize":"Optimize_Post"})
graph.add_edge("Optimize_Post","Evaluate_Post")



workflow = graph.compile()