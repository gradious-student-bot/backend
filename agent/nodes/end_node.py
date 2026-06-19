import logging
from langchain_core.messages import AIMessage
from agent.state import LeadState

logger = logging.getLogger(__name__)


NOT_INTERESTED_RESPONSE = (
    "Ok, no problem. If you change your mind, feel free to reach out to us at "
    "admissions@gradious.com. Have a good day!"
)

END_CALL_RESPONSE = (
    "Thanks for your time. We'll be in touch. Have a good day!"
)


def end_node(state: LeadState) -> LeadState:
    intent = state.get("next_node", "end_call")
    logger.info(f"[End Node] Ending call for lead {state['lead_id']} with intent: {intent}")

    if intent == "not_interested":
        state["disposition"] = "not_interested"
        response = NOT_INTERESTED_RESPONSE
        logger.info(f"[End Node] Lead {state['lead_id']} marked as not_interested")
    else:
        state["disposition"] = state.get("disposition") or "end_call"
        response = END_CALL_RESPONSE
        logger.info(f"[End Node] Lead {state['lead_id']} call ended with disposition: {state['disposition']}")

    state["call_ended"] = True
    state["last_agent_response"] = response
    state["messages"].append(AIMessage(content=response))
    return state