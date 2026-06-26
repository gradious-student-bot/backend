from agent.nodes.questionnaire_node import ALL_FIELDS, _get_next_field, get_recent_messages
from agent.state import LeadState
from agent.prompts.questionnaire_prompt import (
    DE_ESCALATE_SYSTEM_PROMPT,
    IRRELEVANT_SYSTEM_PROMPT,
    CONFUSED_SYSTEM_PROMPT,
    SMALL_TALK_SYSTEM_PROMPT
)

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

    recent = get_recent_messages(state, 6)