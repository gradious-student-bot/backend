import logging
from langgraph.graph import StateGraph, END

from agent.state import LeadState
from agent.nodes.intent_router import intent_router_node
from agent.nodes.questionnaire_node import questionnaire_node
from agent.nodes.faq_node import faq_node
from agent.nodes.minor_intent_node import minor_intent_node
from agent.nodes.repeat_node import repeat_node
from agent.nodes.end_node import end_node
from agent.nodes.airtable_node import airtable_node

logger = logging.getLogger(__name__)


def route_after_intent(state: LeadState) -> str:
    """
    Determine the next node based on the intent extracted from the user's input.
    """
    intent = state.get("next_node", "answer")
    logger.info(f"Routing after intent: {intent}")
    
    routing = {
        "answer":           "questionnaire",    # default to questionnaire if no specific intent is found
        "query":            "faq",              # route to FAQ for general queries
        "answer_and_query": "questionnaire",    # route to questionnaire first, then FAQ
        "repeat":           "repeat",           # route to repeat node for repeating the last question
        "human_agent":      "questionnaire",    # route to questionnaire for human agent requests
        "rude":             "minor_intent",     # route to minor_intent for rude or inappropriate inputs
        "irrelevant":       "minor_intent",     # route to minor_intent for irrelevant inputs
        "confused":         "minor_intent",     # route to minor_intent for confused user
        "small_talk":       "minor_intent",     # route to minor_intent for small talk
        "not_interested":   "end",              # route to end node for disinterest
        "end_call":         "end",              # route to end node for explicit end call requests
    }

    route = routing.get(intent, "questionnaire")
    state["last_agent_node"] = route
    
    logger.info(f"Route selected: {route}")

    return route


def route_after_questionnaire(state: LeadState) -> str:
    """
    Determine the next node based on the state after processing a questionnaire turn.
    """
    # Check if the next node is set to faq_after_answer
    if state.get("next_node") == "faq_after_answer":
        logger.info("Routing questionnaire → faq_after_answer")
        return "faq_after_answer"
    
    # Check if the next node is set to not_interested
    if state.get("next_node") == "not_interested":
        logger.info("Routing questionnaire → end (not_interested from greeting)")
        return "end"
    
    # Check if the call has ended
    if state.get("call_ended"):
        logger.info("Call ended, routing to airtable")
        return "airtable"
    
    logger.info("Waiting for next user turn")
    return END  # wait for next user turn


def route_after_end(state: LeadState) -> str:
    """
    Determine the next node after the end node, typically routing to Airtable for logging.
    """
    logger.info("Routing from end_node to airtable")
    return "airtable"


def build_graph() -> StateGraph:
    """
    Build and return the state graph for the agent.
    """
    logger.info("Building agent graph")
    graph = StateGraph(LeadState)

    graph.add_node("intent_router",     intent_router_node)
    graph.add_node("questionnaire",     questionnaire_node)
    graph.add_node("faq",               faq_node)
    graph.add_node("faq_after_answer",  faq_node)
    graph.add_node("minor_intent",      minor_intent_node)
    graph.add_node("repeat",            repeat_node)
    graph.add_node("end",               end_node)
    graph.add_node("airtable",          airtable_node)

    # Initial entry point is the intent router
    graph.set_entry_point("intent_router")

    graph.add_conditional_edges("intent_router", route_after_intent, {
        "questionnaire":    "questionnaire",
        "faq":              "faq",
        "minor_intent":     "minor_intent",
        "repeat":           "repeat",
        "end":              "end",
    })

    graph.add_conditional_edges("questionnaire", route_after_questionnaire, {
        "faq_after_answer": "faq_after_answer",
        "end":              "end",
        "airtable":         "airtable",
        END:                END,
    })

    # faq_after_answer and faq are both terminal for the turn
    graph.add_edge("faq",               END)
    graph.add_edge("faq_after_answer",  END)
    graph.add_edge("minor_intent",      END)
    graph.add_edge("repeat",            END)
    graph.add_edge("end",               "airtable")
    graph.add_edge("airtable",          END)

    logger.info("Graph built successfully")
    return graph.compile()


agent_graph: StateGraph = build_graph()
