import logging
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from agent.state import LeadState
from agent.prompts.questionnaire_prompt import (
    QUESTIONNAIRE_SYSTEM_PROMPT,
    DE_ESCALATE_RESPONSE,
    IRRELEVANT_RESPONSE,
    CONFUSED_SYSTEM_PROMPT,
)

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# Ordered question flow
QUESTION_FLOW = [
    "course_interest",
    "student_status",
    "education_details",   # branches: current_year+passout_year+dept (student) or passout_year+dept (graduated)
    "training_mode",
    "class_type",          # only if training_mode == "online"
    "budget_range",
    "referral_source",
    "interested",
    "join_date",           # only if interested == True
    "callback_requested",
    "callback_time",       # only if callback_requested == True
]

QUESTION_PROMPTS = {
    "course_interest": "Which course are you interested in — Full Stack Development, AI & Machine Learning, or DSA?",
    "student_status": "Are you currently a student or have you already graduated?",
    "current_year": "Which year are you in currently?",
    "passout_year": "Which year did you graduate or are you passing out?",
    "department": "What is your branch or department?",
    "training_mode": "Are you looking for online training or classroom training?",
    "class_type": "Would you prefer self-paced learning or live classes?",
    "budget_range": "What budget range are you looking at for the course?",
    "referral_source": "How did you hear about Gradious — social media, a friend, or somewhere else?",
    "interested": "Are you interested in joining one of our programs?",
    "join_date": "When are you planning to start?",
    "callback_requested": "Would you like a callback from our admissions team for more details?",
    "callback_time": "What time works best for the callback?",
}

COURSE_KEY_MAP = {
    "full stack": "fullstack_batch",
    "fullstack": "fullstack_batch",
    "full-stack": "fullstack_batch",
    "ai": "ai_batch",
    "ml": "ai_batch",
    "machine learning": "ai_batch",
    "artificial intelligence": "ai_batch",
    "dsa": "dsa_batch",
    "data structures": "dsa_batch",
}


def _extract_course_key(text: str) -> str | None:
    text_lower = text.lower()
    for kw, key in COURSE_KEY_MAP.items():
        if kw in text_lower:
            return key
    return None


def _get_next_question_key(state: LeadState) -> str | None:
    af = state["answered_fields"]

    if "course_interest" not in af:
        return "course_interest"
    if "student_status" not in af:
        return "student_status"

    status = af.get("student_status", "")
    if "student" in status.lower():
        if "current_year" not in af:
            return "current_year"
        if "passout_year" not in af:
            return "passout_year"
    if "passout_year" not in af:
        return "passout_year"
    if "department" not in af:
        return "department"
    if "training_mode" not in af:
        return "training_mode"

    mode = af.get("training_mode", "")
    if "online" in mode.lower() and "class_type" not in af:
        return "class_type"

    if "budget_range" not in af:
        return "budget_range"
    if "referral_source" not in af:
        return "referral_source"
    if "interested" not in af:
        return "interested"

    interested = af.get("interested", "")
    if str(interested).lower() in ("yes", "true") and "join_date" not in af:
        return "join_date"

    if "callback_requested" not in af:
        return "callback_requested"

    cb = af.get("callback_requested", "")
    if str(cb).lower() in ("yes", "true") and "callback_time" not in af:
        return "callback_time"

    return None  # all done


def _store_answer(state: LeadState, key: str, value: str):
    state["answered_fields"][key] = value

    # Mirror into top-level state fields
    field_map = {
        "course_interest": "course_interest",
        "student_status": "student_status",
        "current_year": "current_year",
        "passout_year": "passout_year",
        "department": "department",
        "training_mode": "training_mode",
        "class_type": "class_type",
        "budget_range": "budget_range",
        "referral_source": "referral_source",
        "join_date": "join_date",
        "callback_time": "callback_time",
    }
    if key in field_map:
        state[field_map[key]] = value  # type: ignore

    if key == "course_interest":
        state["course_interest"] = _extract_course_key(value) or value
    if key == "interested":
        state["interested"] = value.lower() in ("yes", "true", "yeah", "yep")
    if key == "callback_requested":
        state["callback_requested"] = value.lower() in ("yes", "true", "yeah", "yep")


def questionnaire_node(state: LeadState) -> LeadState:
    intent = state.get("next_node", "answer")
    user_text = state["messages"][-1].content if state["messages"] else ""
    current_q_key = state.get("current_question_key", "")
    current_q_text = QUESTION_PROMPTS.get(current_q_key, "")

    logger.info(f"[Questionnaire] Processing intent '{intent}' for lead {state['lead_id']}, question key: {current_q_key}")
    logger.debug(f"User text: {user_text}")

    # ── Human agent short-circuit ──────────────────────────────────────
    if state.get("human_agent_requested") or intent == "human_agent":
        logger.info(f"[Questionnaire] Human agent requested for lead {state['lead_id']}")
        state["human_agent_requested"] = True
        state["disposition"] = "human_agent_callback"

        if not state.get("callback_time"):
            response = (
                "Sure! Our admissions team can walk you through everything in detail. "
                "What time works best for a callback?"
            )
            state["current_question_key"] = "callback_time"
        else:
            # callback_time just collected
            _store_answer(state, "callback_time", user_text)
            state["callback_requested"] = True
            state["call_ended"] = True
            response = (
                f"Done! Someone from our team will call you at {user_text}. "
                "Talk soon!"
            )

        state["last_agent_response"] = response
        state["messages"].append(AIMessage(content=response))
        return state

    # ── De-escalate rude ──────────────────────────────────────────────
    if intent == "rude":
        logger.warning(f"[Questionnaire] Rude intent detected for lead {state['lead_id']}")
        response = DE_ESCALATE_RESPONSE.format(current_question=current_q_text)
        state["last_agent_response"] = response
        state["messages"].append(AIMessage(content=response))
        return state

    # ── Redirect irrelevant ───────────────────────────────────────────
    if intent == "irrelevant":
        logger.info(f"[Questionnaire] Irrelevant message from lead {state['lead_id']}")
        response = IRRELEVANT_RESPONSE.format(current_question=current_q_text)
        state["last_agent_response"] = response
        state["messages"].append(AIMessage(content=response))
        return state

    # ── Clarify confused ──────────────────────────────────────────────
    if intent == "confused":
        logger.debug(f"[Questionnaire] User confused for lead {state['lead_id']}")
        prompt = CONFUSED_SYSTEM_PROMPT.format(current_question=current_q_text)
        ai_response = llm.invoke([{"role": "system", "content": prompt}])
        response = ai_response.content.strip()
        state["last_agent_response"] = response
        state["messages"].append(AIMessage(content=response))
        return state

    # ── Normal answer: store and move to next question ────────────────
    if current_q_key:
        logger.debug(f"[Questionnaire] Storing answer for {current_q_key}: {user_text[:50]}...")
        _store_answer(state, current_q_key, user_text)

    next_key = _get_next_question_key(state)

    if next_key is None:
        # All questions answered
        logger.info(f"[Questionnaire] All questions answered for lead {state['lead_id']}")
        state["call_ended"] = True
        if state.get("callback_requested"):
            state["disposition"] = "callback"
        elif state.get("interested"):
            state["disposition"] = "interested"
        else:
            state["disposition"] = "not_interested"

        logger.info(f"[Questionnaire] Call disposition set to {state['disposition']} for lead {state['lead_id']}")

        response = (
            "Thanks for your time! Our admissions team will be in touch with you soon. "
            "Have a good day!"
        )
        state["last_agent_response"] = response
        state["messages"].append(AIMessage(content=response))
        return state

    # Ask next question
    logger.debug(f"[Questionnaire] Moving to next question: {next_key}")
    state["current_question_key"] = next_key
    ack = _build_ack(user_text, current_q_key)
    next_q_text = QUESTION_PROMPTS[next_key]
    response = f"{ack}{next_q_text}" if ack else next_q_text

    state["last_agent_response"] = response
    state["messages"].append(AIMessage(content=response))
    return state


def _build_ack(user_text: str, previous_key: str) -> str:
    """Simple acknowledgement — no over-the-top adjectives."""
    if not user_text or not previous_key:
        return ""
    acks = ["Ok! ", "Got it. ", "Sure. ", "Noted. "]
    import hashlib
    idx = int(hashlib.md5(user_text.encode()).hexdigest(), 16) % len(acks)
    return acks[idx]