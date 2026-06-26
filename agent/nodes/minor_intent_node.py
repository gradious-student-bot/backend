import logging

from agent.nodes.questionnaire_node import (
    ALL_FIELDS, 
    _get_next_field,
    _set_response
)
from agent.state import LeadState
from agent.prompts.questionnaire_prompt import (
    DE_ESCALATE_SYSTEM_PROMPT,
    IRRELEVANT_SYSTEM_PROMPT,
    CONFUSED_SYSTEM_PROMPT,
    SMALL_TALK_SYSTEM_PROMPT
)
from services.llm_service import llm

logger = logging.getLogger(__name__)

# minor_intent_node.py
MINOR_PROMPT_MAP = {
    "rude":       DE_ESCALATE_SYSTEM_PROMPT,
    "irrelevant": IRRELEVANT_SYSTEM_PROMPT,
    "confused":   CONFUSED_SYSTEM_PROMPT,
    "small_talk": SMALL_TALK_SYSTEM_PROMPT,
}

def minor_intent_node(state: LeadState):
    intent = state.get("next_node", "irrlevant")
    
    prompt_template = MINOR_PROMPT_MAP.get(intent, IRRELEVANT_SYSTEM_PROMPT)
    next_key = _get_next_field(state["answered_fields"])
    
    desc = ALL_FIELDS.get(next_key or state["current_question_key"], "our current question")

    logger.info(f"[MINOR INTENT] Handling intent={intent}")
    
    response = llm.invoke_json(
        [{"role": "system", "content": prompt_template.format(pending_question_description=desc)}]
        + llm.get_recent_messages(state)
        + [{"role": "user", "content": state["messages"][-1].content}]
    )

    _set_response(state, response.get("response", "Let me know when you're ready to continue."))

    return state