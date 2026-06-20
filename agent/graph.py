import logging
from langgraph.graph import StateGraph, END
from agent.state import LeadState
from agent.nodes.intent_router import intent_router_node
from agent.nodes.questionnaire_node import questionnaire_node
from agent.nodes.faq_node import faq_node
from agent.nodes.repeat_node import repeat_node
from agent.nodes.end_node import end_node
from agent.nodes.airtable_node import airtable_node

logger = logging.getLogger(__name__)


def route_after_intent(state: LeadState) -> str:
    intent = state.get("next_node", "answer")
    logger.info(f"Routing after intent: {intent}")
    routing = {
        "answer":           "questionnaire",
        "query":            "faq",
        "answer_and_query": "questionnaire",   # Task 2: questionnaire handles extraction, then routes to faq_after_answer
        "repeat":           "repeat",
        "human_agent":      "questionnaire",
        "rude":             "questionnaire",
        "irrelevant":       "questionnaire",
        "confused":         "questionnaire",
        "not_interested":   "end",
        "end_call":         "end",
    }
    route = routing.get(intent, "questionnaire")
    logger.info(f"Route selected: {route}")
    return route


def route_after_questionnaire(state: LeadState) -> str:
    # Task 2: questionnaire node signals faq_after_answer via next_node
    if state.get("next_node") == "faq_after_answer":
        logger.info("Routing questionnaire → faq_after_answer")
        return "faq_after_answer"
    # Task 8: if not_interested set during greeting flow, route to end
    if state.get("next_node") == "not_interested":
        logger.info("Routing questionnaire → end (not_interested from greeting)")
        return "end"
    if state.get("call_ended"):
        logger.info("Call ended, routing to airtable")
        return "airtable"
    logger.info("Waiting for next user turn")
    return END  # wait for next user turn


def route_after_end(state: LeadState) -> str:
    logger.info("Routing from end_node to airtable")
    return "airtable"


def build_graph() -> StateGraph:
    logger.info("Building agent graph")
    graph = StateGraph(LeadState)

    graph.add_node("intent_router",   intent_router_node)
    graph.add_node("questionnaire",   questionnaire_node)
    graph.add_node("faq",             faq_node)
    graph.add_node("faq_after_answer", faq_node)   # Task 2: same function, separate node reference
    graph.add_node("repeat",          repeat_node)
    graph.add_node("end",             end_node)
    graph.add_node("airtable",        airtable_node)

    graph.set_entry_point("intent_router")

    graph.add_conditional_edges("intent_router", route_after_intent, {
        "questionnaire": "questionnaire",
        "faq":           "faq",
        "repeat":        "repeat",
        "end":           "end",
    })

    graph.add_conditional_edges("questionnaire", route_after_questionnaire, {
        "faq_after_answer": "faq_after_answer",
        "end":              "end",
        "airtable":         "airtable",
        END:                END,
    })

    # faq_after_answer and faq are both terminal for the turn
    graph.add_edge("faq",             END)
    graph.add_edge("faq_after_answer", END)
    graph.add_edge("repeat",          END)
    graph.add_edge("end",             "airtable")
    graph.add_edge("airtable",        END)

    logger.info("Graph built successfully")
    return graph.compile()


agent_graph = build_graph()
