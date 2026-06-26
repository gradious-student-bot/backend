import json
import logging
from datetime import date, datetime
from openai import OpenAI
from agent.state import LeadState
from config import OPENAI_API_KEY
from services.email_service import send_onboarding_email
from agent.prompts.questionnaire_prompt import (
    EXTRACT_SYSTEM_PROMPT,
    NEXT_QUESTION_SYSTEM_PROMPT,
    GREETING_SYSTEM_PROMPT,
    POST_CONFIRM_SYSTEM_PROMPT,
    CONFIRM_TIMING_SYSTEM_PROMPT,
    INTRO_WITH_FIRST_QUESTION_PROMPT,
    CONFIRM_INTEREST_SYSTEM_PROMPT,
    WRAP_UP_SYSTEM_PROMPT,
    HUMAN_AGENT_SYSTEM_PROMPT,
    HUMAN_AGENT_CONFIRM_SYSTEM_PROMPT,
    DE_ESCALATE_SYSTEM_PROMPT,
    IRRELEVANT_SYSTEM_PROMPT,
    CONFUSED_SYSTEM_PROMPT,
    SMALL_TALK_SYSTEM_PROMPT,
    CONFIRM_SWITCH_SYSTEM_PROMPT,
    BAD_TIMING_SYSTEM_PROMPT,
    ONBOARDING_EMAIL_SYSTEM_PROMPT
)

logger = logging.getLogger(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# Field schema sent to LLM so it knows what to extract and what's pending
# ─────────────────────────────────────────────────────────────────────────────

ALL_FIELDS = {
    "course_interest":      "Course the student wants — one of: fullstack_batch, ai_batch, dsa_batch",
    "student_status":       "Whether student is currently studying, already graduated, or a working professional — values: student | graduated | working_professional",
    "current_year":         "Current academic year if student (e.g. 2nd year, 3rd year) — only for student status",
    "passout_year":         "Year of graduation or expected passout (e.g. 2026, 2027)",
    "department":           "Branch or department (e.g. CSE, ECE, IT, Mechanical)",
    "training_mode":        "Preferred training mode — values: online | offline",
    "class_type":           "If online: self_paced or live — only relevant when training_mode is online",
    "looking_for_job":      "Whether the person is currently looking for a job (only for graduated/working_professional)",
    "interested":           "Whether the student is interested in joining — values: yes | no",
    "join_date":            "When the student plans to start — only if interested is yes",
    "onboarding_requested": "Whether the student wants the onboarding form sent to their email address — values: yes | no",
    "callback_requested":   "Whether student wants a callback from admissions team — values: yes | no",
    "callback_time":        "Preferred time for the callback — only if callback_requested is yes",
}

# Fields that are conditionally required
CONDITIONAL_FIELDS = {
    "current_year":         lambda af: af.get("student_status") == "student",
    "class_type":           lambda af: af.get("training_mode") == "online",
    "looking_for_job":      lambda af: af.get("student_status") in ("graduated", "working_professional"),
    "join_date":            lambda af: str(af.get("interested", "")).lower() == "yes",
    "callback_time":        lambda af: str(af.get("callback_requested", "")).lower() == "yes",
    "onboarding_requested": lambda af: str(af.get("interested", "")).lower() == "yes",
}

SWITCHABLE_FIELDS = {"course_interest", "training_mode"}

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _llm_json(messages: list) -> dict:
    """
    Calls the LLM with a list of messages and returns the parsed JSON response.
    """
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.4,
        response_format={"type": "json_object"},
        messages=messages,
    )
    return json.loads(resp.choices[0].message.content)


def _filled_summary(af: dict) -> str:
    """Returns a string summary of the fields that have been filled so far, for LLM prompt context."""
    if not af:
        return "None yet."
    return "\n".join(f"  {k}: {v}" for k, v in af.items())


def _fields_description() -> str:
    """Returns a string describing all fields and their expected values, for LLM prompt context."""
    return "\n".join(f"  {k}: {v}" for k, v in ALL_FIELDS.items())


def _get_next_field(af: dict) -> str | None:
    """
    Returns the next field key that has not yet been filled, in the order defined by ALL_FIELDS.
    """
    order = [
        "course_interest", "student_status", "current_year",
        "passout_year", "department", "training_mode", "class_type",
        "looking_for_job", "interested",
        "join_date","onboarding_requested", "callback_requested", "callback_time",
    ]
    for field in order:
        if field in af:
            continue
        if field in CONDITIONAL_FIELDS:
            if not CONDITIONAL_FIELDS[field](af):
                continue  # condition not met — skip this field
        return field
    return None


def _apply_extracted(state: LeadState, extracted: dict):
    """
    Applies the extracted fields to the state, updating answered_fields and top-level state fields.
    """
    af = state["answered_fields"]
    for key, value in extracted.items():
        if value is None or value == "null":
            continue

        af[key] = value

        # Mirror to top-level state fields
        if key in (
            "course_interest", "student_status", "current_year", "passout_year",
            "department", "training_mode", "class_type",
            "join_date", "callback_time",
        ):
            state[key] = value  # type: ignore

        if key == "looking_for_job":
            state["looking_for_job"] = bool(value)

        if key == "interested":
            state["interested"] = str(value).lower() == "yes"

        if key == "onboarding_requested":
            state["onboarding_requested"] = str(value).lower() == "yes"

        if key == "onboarding_email_sent":
            state["onboarding_email_sent"] = str(value).lower() == "yes"

        if key == "callback_requested":
            state["callback_requested"] = str(value).lower() == "yes"


def _set_response(state: LeadState, text: str):
    """
    Sets the last_agent_response and appends an AIMessage to the messages list.
    """
    from langchain_core.messages import AIMessage
    state["last_agent_response"] = text
    state["messages"].append(AIMessage(content=text))


def get_recent_messages(state: LeadState, n: int = 10) -> list:
    """
    Returns the last n messages from state["messages"] formatted as OpenAI
    chat message dicts: {"role": "user"|"assistant", "content": "..."}.
    Skips the very last message (which is the current user input, already handled separately).
    """
    from langchain_core.messages import HumanMessage, AIMessage
    
    msgs = state.get("messages", [])
    history = msgs[:-1] if len(msgs) > 1 else []
    result = []

    for m in history[-n:]:

        if isinstance(m, HumanMessage):
            result.append({"role": "user", "content": m.content})

        elif isinstance(m, AIMessage):
            result.append({"role": "assistant", "content": m.content})

    logger.info(f"Conversation history: {result}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main node
# ─────────────────────────────────────────────────────────────────────────────

def questionnaire_node(state: LeadState) -> LeadState:
    from langchain_core.messages import AIMessage

    intent = state.get("next_node", "answer")
    user_text = state["messages"][-1].content if state["messages"] else ""
    current_q_key = state.get("current_question_key", "")
    
    today = date.today()

    GREETING_KEYS = {"confirm_identity", "confirm_timing", "confirm_interest"}

    logger.info(f"[Questionnaire] Lead={state['lead_id']} intent={intent} q_key={current_q_key}")

    # ── 1. Human agent short-circuit ─────────────────────────────────────────
    if state.get("human_agent_requested") or intent == "human_agent":
        logger.info("[Questionnaire] User requested for human agent")

        state["human_agent_requested"] = True
        state["disposition"] = "human_agent_callback"

        if current_q_key != "callback_time":
            logger.info("[Questionnaire] Asking callback time to user")
            
            result = _llm_json([{"role": "system", "content": HUMAN_AGENT_SYSTEM_PROMPT}])
            response = result.get("response", "Sure! What time works best for a callback from our team?")
            
            _set_response(state, response)
            state["current_question_key"] = "callback_time"
            
            return state
        else:
            logger.info("[Questionnaire] Acknowledging user after taking callback time")
            
            _apply_extracted(state, {"callback_time": user_text})
            state["callback_requested"] = True
            state["call_ended"] = True
            
            result = _llm_json([{
                "role": "system",
                "content": HUMAN_AGENT_CONFIRM_SYSTEM_PROMPT.format(callback_time=user_text),
            }])
            
            response = result.get("response", f"Done! Our team will call you at {user_text}. Talk soon!")
            _set_response(state, response)
            
            return state
        
    # Only proceed with greeting and questionnaire flow if not in human_agent, not_interested, rude, irrelevant, confused, or small_talk intents, and if greeting_step < 3
    if current_q_key in GREETING_KEYS and intent not in ["human_agent", "not_interested", "rude", "irrelevant", "confused", "small_talk"]:
        
        # Step 1 — confirm_identity
        if current_q_key == "confirm_identity":
        #     affirmatives = {"yes", "yeah", "yep", "sure", "correct", "right", "speaking", "that's me", "this is"}
        #     is_yes = any(w in user_text.lower() for w in affirmatives)

        #     if is_yes:
        #         result = _llm_json([{"role": "system", "content": POST_CONFIRM_SYSTEM_PROMPT}])
        #         response = result.get("response", "Hi, this is Bindhu from Gradious. Is this a good time to speak?")
        #         state["greeting_step"] = 1
        #         state["current_question_key"] = "confirm_timing"
        #     else:
        #         response = f"Sorry about that — if you are {state.get('lead_name', 'the person we registered')}, please let me know so I can continue."
        #     _set_response(state, response)
        #     return state
            logger.info(f"[Questionnaire] Confirming identity for lead {state['lead_id']}")

            lead_name = state.get('lead_name', 'the student')
            confirm_identity_prompt = f"""## ROLE
    You are Bindhu, a phone-based admissions counselor at Gradious.

    ## OBJECTIVE
    The user has just responded to the greeting "Am I speaking with {lead_name}?"
    First detect if the user confirmed their identity (yes) or denied it (no).
    Then generate the appropriate response.

    ## USER'S REPLY
    "{user_text}"

    ## RULES
    - If YES (user confirmed identity):
    - Introduce yourself as Bindhu from Gradious (1 short sentence).
    - Ask: "Is this a good time to speak?"
    - Keep it under 2 sentences. Sound natural, not scripted.
    - If NO or UNCLEAR (user denied or response is ambiguous):
    - Politely clarify and ask them to confirm if they are {lead_name}.
    - Keep it brief and natural.
    - Do not use "Absolutely!", "Certainly!", "Great question!", or "Definitely!".

    ## OUTPUT FORMAT
    Return STRICT JSON only.
    {{
    "is_yes": true or false,
    "response": ""
    }}"""
            result = _llm_json([{"role": "system", "content": confirm_identity_prompt}])
            is_yes = result.get("is_yes", False)
            response = result.get(
                "response",
                "Hi, this is Bindhu from Gradious. Is this a good time to speak?" if is_yes
                else f"Sorry about that — if you are {lead_name}, please let me know so I can continue."
            )
            if is_yes:
                state["greeting_step"] = 1
                state["current_question_key"] = "confirm_timing"
            _set_response(state, response)
            return state
        
        # Step 2 — confirm_timing
        if current_q_key == "confirm_timing":
        #     affirmatives = {"yes", "yeah", "yep", "sure", "go ahead", "ok", "okay", "of course",
        #                     "good time", "fine", "speak", "yes please"}
        #     negatives = {"no", "nope", "busy", "not now", "bad time", "later", "call back",
        #                  "not a good time", "can't", "cannot"}
        #     text_lower = user_text.lower()
        #     is_yes = any(w in text_lower for w in affirmatives)
        #     is_no = any(w in text_lower for w in negatives)

        #     if is_no or (not is_yes and any(w in text_lower for w in ["later", "another time", "call back"])):
        #         logger.info("[Questionnaire] Student unavailable — requesting callback time")
        #         state["human_agent_requested"] = True
        #         state["disposition"] = "human_agent_callback"
        #         result = _llm_json([{"role": "system", "content": BAD_TIMING_SYSTEM_PROMPT}])
        #         response = result.get("response", "No problem at all — when would be a good time for me to call you back?")
        #         state["current_question_key"] = "callback_time"
        #     else:
        #         result = _llm_json([{"role": "system", "content": CONFIRM_TIMING_SYSTEM_PROMPT}])
        #         response = result.get("response", "Ok, so I'm calling because you showed interest in our programs. Would you like to know more?")
        #         state["greeting_step"] = 2
        #         state["current_question_key"] = "confirm_interest"
        #     _set_response(state, response)
        #     return state
            logger.info(f"[Questionnaire] Confirming timing for lead {state['lead_id']}")

            confirm_timing_prompt = f"""## ROLE
    You are Bindhu, a phone-based admissions counselor at Gradious.

    ## OBJECTIVE
    The user has just responded to "Is this a good time to speak?"
    First detect if the user said it's a good time (yes) or not (no/busy/later).
    Then generate the appropriate response.

    ## USER'S REPLY
    "{user_text}"

    ## RULES
    - If YES (good time to talk):
    - Briefly mention the user showed interest in our tech training programs.
    - Ask: "Would you like to know more about what we offer?"
    - Keep it under 3 sentences. Sound natural.
    - If NO (not a good time / busy / wants callback):
    - Acknowledge with understanding (1 sentence).
    - Ask for a preferred callback time (1 sentence).
    - If UNCLEAR, treat as YES.
    - Do not use "Absolutely!", "Certainly!", "Great question!", or "Definitely!".

    ## OUTPUT FORMAT
    Return STRICT JSON only.
    {{
    "is_yes": true or false,
    "response": ""
    }}"""
            result = _llm_json([{"role": "system", "content": confirm_timing_prompt}])
            is_yes = result.get("is_yes", True)
            response = result.get(
                "response",
                "Ok, so I'm calling because you showed interest in our programs. Would you like to know more?" if is_yes
                else "No problem at all — when would be a good time for me to call you back?"
            )
            if is_yes:
                state["greeting_step"] = 2
                state["current_question_key"] = "confirm_interest"
            else:
                logger.info("[Questionnaire] Student unavailable — requesting callback time")
                state["human_agent_requested"] = True
                state["disposition"] = "human_agent_callback"
                state["current_question_key"] = "callback_time"
            _set_response(state, response)
            return state

        # Step 3 — confirm_interest
        if current_q_key == "confirm_interest":
        #     affirmatives = {"yes", "yeah", "yep", "sure", "ok", "okay", "of course", "interested",
        #                     "go ahead", "yes please", "tell me"}
        #     text_lower = user_text.lower()
        #     is_yes = any(w in text_lower for w in affirmatives)

        #     if is_yes:
        #         result = _llm_json([
        #             {"role": "system", "content": CONFIRM_INTEREST_SYSTEM_PROMPT}
        #         ])

        #         response = result.get(
        #             "response",
        #             "Got it. Let me get a few details to help point you in the right direction."
        #         )

        #         state["greeting_step"] = 3
        #         state["current_question_key"] = "course_interest"

        #         _set_response(state, response)
        #         return state
        #     else:
        #         logger.info("[Questionnaire] Student not interested after greeting → routing to end")
        #         state["next_node"] = "not_interested"
        #         state["disposition"] = "not_interested"
        #         state["call_ended"] = False  # end_node will set this
        #         return state
        
            logger.info(f"[Questionnaire] Confirming interest for lead {state['lead_id']}")
            confirm_interest_prompt = f"""## ROLE
    You are Bindhu, a phone-based admissions counselor at Gradious.

    ## OBJECTIVE
    The user has just responded to "Would you like to know more about what we offer?"
    First detect if the user expressed interest (yes) or declined (no/not interested).
    Then generate the appropriate response.

    ## USER'S REPLY
    "{user_text}"

    ## RULES
    - If YES (interested / wants to know more):
    - Acknowledge their interest briefly.
    - Transition naturally into asking about which course they're interested in.
    - Course options: Full Stack + Gen AI, AI Stack, or DSA course.
    - Keep it under 3 sentences. Sound natural and conversational.
    - Do NOT say "Let me ask you a few questions" and pause — just ask the course question directly.
    - If NO (not interested):
    - Politely acknowledge their response.
    - Thank them for their time (1 short sentence).
    - If UNCLEAR, treat as YES.
    - Do not use "Absolutely!", "Certainly!", "Great question!", or "Definitely!".
    - Do not create new courses — only mention Full Stack + Gen AI, AI Stack and DSA course.

    ## OUTPUT FORMAT
    Return STRICT JSON only.
    {{
    "is_yes": true or false,
    "response": ""
    }}"""
            result = _llm_json([{"role": "system", "content": confirm_interest_prompt}])
            is_yes = result.get("is_yes", True)
            response = result.get("response", "Got it. Let me get a few details to help point you in the right direction.")
            if is_yes:
                state["greeting_step"] = 3
                state["current_question_key"] = "course_interest"
                _set_response(state, response)
                return state
            else:
                logger.info("[Questionnaire] Student not interested after greeting → routing to end")
                _set_response(state, response)
                state["next_node"] = "not_interested"
                state["disposition"] = "not_interested"
                state["call_ended"] = False  # end_node will set this
                return state
    
    # Handle pending_switch confirmation [IMPROVEMENT] — Entity extraction from LLM
    if current_q_key == "confirm_switch":
        pending = state.get("pending_switch")
        affirmatives = {"yes", "yeah", "yep", "sure", "correct", "right", "switch", "change", "update"}
        negatives = {"no", "nope", "keep", "stay", "don't", "old", "original", "cancel"}
        text_lower = user_text.lower()
        is_yes = any(w in text_lower for w in affirmatives)
        is_no = any(w in text_lower for w in negatives)

        if pending:
            if is_yes and not is_no:
                logger.info(f"[Questionnaire] Switch confirmed: {pending['field']} → {pending['new_value']}")
                _apply_extracted(state, {pending["field"]: pending["new_value"]})
            else:
                logger.info(f"[Questionnaire] Switch denied: keeping old {pending['field']} value")

        state["pending_switch"] = None
        next_key = _get_next_field(state["answered_fields"])
        state["current_question_key"] = next_key or current_q_key
        current_q_key = state["current_question_key"]

    # ── 3. Rude ───────────────────────────────────────────────────────────────
    if intent == "rude":
        next_key = _get_next_field(state["answered_fields"])
        desc = ALL_FIELDS.get(next_key or current_q_key, "our current question")
        recent = get_recent_messages(state, 6)
        result = _llm_json(
            [{"role": "system", "content": DE_ESCALATE_SYSTEM_PROMPT.format(pending_question_description=desc)}]
            + recent
            + [{"role": "user", "content": user_text}]
        )
        _set_response(state, result.get("response", "I understand. Let me know when you're ready to continue."))
        return state

    # ── 4. Irrelevant ─────────────────────────────────────────────────────────
    if intent == "irrelevant":
        next_key = _get_next_field(state["answered_fields"])
        desc = ALL_FIELDS.get(next_key or current_q_key, "our current question")
        recent = get_recent_messages(state, 6)
        result = _llm_json(
            [{"role": "system", "content": IRRELEVANT_SYSTEM_PROMPT.format(pending_question_description=desc)}]
            + recent
            + [{"role": "user", "content": user_text}]
        )
        _set_response(state, result.get("response", "Let's get back on track — " + desc))
        return state

    # ── 5. Confused ───────────────────────────────────────────────────────────
    if intent == "confused":
        next_key = _get_next_field(state["answered_fields"])
        desc = ALL_FIELDS.get(next_key or current_q_key, "the current question")
        recent = get_recent_messages(state, 6)
        result = _llm_json(
            [{"role": "system", "content": CONFUSED_SYSTEM_PROMPT.format(pending_question_description=desc)}]
            + recent
            + [{"role": "user", "content": user_text}]
        )
        _set_response(state, result.get("response", desc))
        return state

    # ── 6. Small talk ─────────────────────────────────────────────────────────
    if intent == "small_talk":
        logger.info(f"[Questionnaire] Small talk detected — responding with small talk prompt")

        recent = get_recent_messages(state, 6)
        result = _llm_json(
            [{"role": "system", "content": SMALL_TALK_SYSTEM_PROMPT}]
            + recent
            + [{"role": "user", "content": user_text}]
        )
        _set_response(state, result.get("response", "I appreciate your input! Let's continue with our discussion."))
        return state

    # ── 7. Extract fields from user answer ───────────────────────────────────
    if user_text and current_q_key not in ("", "confirm_identity", "confirm_timing", "confirm_interest"):
        try:
            extract_prompt = EXTRACT_SYSTEM_PROMPT.format(
                today=today.isoformat(),
                today_year=today.year,
                fields_description=_fields_description(),
                filled_fields=_filled_summary(state["answered_fields"]),
            )

            logger.info(f"[Questionnaire] Current fields: {state['answered_fields']}")

            recent_extract = get_recent_messages(state, 10)
            extract_result = client.chat.completions.create(
                model="gpt-4.1-mini",
                temperature=0,
                response_format={"type": "json_object"},
                messages=(
                    [{"role": "system", "content": extract_prompt}]
                    + recent_extract
                    + [{"role": "user", "content": f"Agent asked: {state.get('last_agent_response', '')}\nStudent replied: {user_text}"}]
                ),
            )
            extracted = json.loads(extract_result.choices[0].message.content).get("extracted", {})
            logger.info(f"[Questionnaire] Extracted fields: {extracted}")

            # Detect if any switchable field is being changed mid-conversation
            af = state["answered_fields"]
            switch_detected = False

            for field in SWITCHABLE_FIELDS:
                new_val = extracted.get(field)

                # If the new value is different from the old value, prompt for confirmation
                # Except if extracted has true for std_course_change, where we don't ask for confirmation
                # We just inform the user that only online self-paced is available for 1st, 2nd, or 3rd year students.
                # Also ask the next question within that response.
                if field == "course_interest" and extracted.get("std_course_change") is True:
                    old_val = af.get(field)
                    logger.info(f"[Questionnaire] Course interest changed due to student status: {old_val} → {new_val}")

                    _apply_extracted(state, {field: new_val})
                    break

                if new_val and new_val != "null" and field in af and af[field] != new_val:
                    old_val = af[field]
                    logger.info(f"[Questionnaire] Switch detected: {field} {old_val} → {new_val}")
                    
                    state["pending_switch"] = {"field": field, "new_value": new_val}
                    state["current_question_key"] = "confirm_switch"
                    
                    switch_result = _llm_json([{
                        "role": "system",
                        "content": CONFIRM_SWITCH_SYSTEM_PROMPT.format(
                            field_label=field.replace("_", " ").title(),
                            old_value=old_val,
                            new_value=new_val,
                        ),
                    }])

                    switch_response = switch_result.get(
                        "response",
                        f"You had selected {old_val} earlier — did you want to switch to {new_val}?"
                    )
                    _set_response(state, switch_response)

                    switch_detected = True
                    break  # only handle one switch at a time

            if switch_detected:
                return state

            # No switch — apply extracted fields normally
            _apply_extracted(state, extracted)

            # ---------------------------------------------------
            # SEND ONBOARDING EMAIL
            # ---------------------------------------------------

            onboarding_yes = (
                str(extracted.get("onboarding_requested", "")).lower() == "yes"
                or state.get("onboarding_requested") is True
                or str(state["answered_fields"].get("onboarding_requested", "")).lower() == "yes"
            )

            if onboarding_yes and not state.get("onboarding_email_sent"):

                receiver_email = (
                    state.get("email")
                    or state.get("lead_email")
                    or state.get("student_email")
                )

                if not receiver_email:

                    logger.warning(
                        f"No email found for lead {state.get('lead_id')}"
                    )

                    result = _llm_json([{
                        "role": "system",
                        "content": ONBOARDING_EMAIL_SYSTEM_PROMPT.format(
                            lead_name=state.get("lead_name", ""),
                            email="",
                            email_status="missing_email",
                            answered_fields=_filled_summary(state.get("answered_fields", {})),
                            last_user_reply=user_text,
                        ),
                    }])

                    response = result.get(
                        "response",
                        "Could you please share your email address so I can send the onboarding form?"
                    )

                    _set_response(state, response)
                    return state

                logger.info(
                    f"Sending onboarding email to {receiver_email}"
                )

                success = send_onboarding_email(
                    student_name=state.get("lead_name", ""),
                    receiver_email=receiver_email,
                )

                if success:

                    logger.info(
                        f"Onboarding email sent to {receiver_email}"
                    )

                    state["onboarding_email_sent"] = True
                    state["answered_fields"]["onboarding_email_sent"] = True

                    result = _llm_json([{
                        "role": "system",
                        "content": ONBOARDING_EMAIL_SYSTEM_PROMPT.format(
                            lead_name=state.get("lead_name", ""),
                            email=receiver_email,
                            email_status="sent_success",
                            answered_fields=_filled_summary(state.get("answered_fields", {})),
                            last_user_reply=user_text,
                        ),
                    }])

                    response = result.get(
                        "response",
                        "I've sent the onboarding form to your email. Would you like someone from our admissions team to give you a callback?"
                    )

                    state["current_question_key"] = "callback_requested"

                    _set_response(state, response)
                    return state

                else:

                    logger.error(
                        f"Failed sending onboarding email to {receiver_email}"
                    )

                    result = _llm_json([{
                        "role": "system",
                        "content": ONBOARDING_EMAIL_SYSTEM_PROMPT.format(
                            lead_name=state.get("lead_name", ""),
                            email=receiver_email,
                            email_status="sent_failed",
                            answered_fields=_filled_summary(state.get("answered_fields", {})),
                            last_user_reply=user_text,
                        ),
                    }])

                    response = result.get(
                        "response",
                        "I'm sorry, I couldn't send the onboarding form right now. Our team will try again shortly."
                    )

                    _set_response(state, response)
                    return state
                
        except Exception:
            logger.exception(
                "[Questionnaire] Failed to extract fields from user response. Proceeding without extraction."
            )
            # Retry counter logic: on extraction failure, log the miss and proceed (retry counter handles retries)

    # ── EMAIL SENT CONFIRMATION ───────────────────────────────────────────────
    if state["answered_fields"].get("email_confirmation_pending"):
        state["answered_fields"]["email_confirmation_pending"] = False
        response = (
            "I've sent the onboarding form to your email address. "
            "Could you please check and let me know whether you've received it?"
        )
        state["current_question_key"] = "onboarding_email_sent"
        _set_response(state, response)
        return state

    # ── Answer_and_query — hand off to faq_after_answer ──────────────
    if intent == "answer_and_query" and state.get("pending_sub_query"):
        next_key = _get_next_field(state["answered_fields"])
        if next_key:
            next_q_result = _llm_json(
                [{"role": "system", "content": NEXT_QUESTION_SYSTEM_PROMPT.format(
                    filled_fields=_filled_summary(state["answered_fields"]),
                    next_field=next_key,
                    field_description=ALL_FIELDS[next_key],
                    last_user_reply=user_text,
                    std_course_change=state["answered_fields"].get("std_course_change", False)
                )}]
                + get_recent_messages(state, 6)
            )
            state["pending_next_question_text"] = next_q_result.get("response", ALL_FIELDS[next_key])
            state["current_question_key"] = next_key
            state["answered_fields"]["std_course_change"] = False  # reset after using it in prompt
        else:
            state["pending_next_question_text"] = None
        state["next_node"] = "faq_after_answer"
        logger.info("[Questionnaire] answer_and_query → routing to faq_after_answer")
        return state
    


    # ── 7. Determine next field ───────────────────────────────────────────────
    next_key = _get_next_field(state["answered_fields"])

    # Retry counter logic
    if next_key is not None:
        retry_counts = state.get("question_retry_counts") or {}
        retry_counts[next_key] = retry_counts.get(next_key, 0) + 1
        state["question_retry_counts"] = retry_counts

        if retry_counts[next_key] > 2:
            logger.warning(f"[Questionnaire] Skipping field {next_key} after 2 retries")
            state["answered_fields"][next_key] = "__skipped__"
            next_key = _get_next_field(state["answered_fields"])

    if next_key is None:
        # All done
        state["call_ended"] = True
        af = state["answered_fields"]
        if str(af.get("callback_requested", "")).lower() == "yes":
            state["disposition"] = "callback"
        elif state.get("interested"):
            state["disposition"] = "interested"
        else:
            state["disposition"] = "not_interested"

        result = _llm_json([{
            "role": "system",
            "content": WRAP_UP_SYSTEM_PROMPT.format(disposition=state["disposition"]),
        }])
        _set_response(state, result.get("response", "Thanks for your time. We'll be in touch soon!"))
        return state

    # ── 8. Ask next question ──────────────────────────────────────────────────
    if intent != "human_agent" and not state.get("human_agent_requested"):
        logger.info("[Questionnaire] Asking next question")
        state["current_question_key"] = next_key

        recent_q = get_recent_messages(state, 6)
        result = _llm_json(
            [{"role": "system", "content": NEXT_QUESTION_SYSTEM_PROMPT.format(
                filled_fields=_filled_summary(state["answered_fields"]),
                next_field=next_key,
                field_description=ALL_FIELDS[next_key],
                last_user_reply=user_text,
                std_course_change=state["answered_fields"].get("std_course_change", False)
            )}]
            + recent_q
        )
        state["answered_fields"]["std_course_change"] = False  # reset after using it in prompt
        response = result.get("response", ALL_FIELDS[next_key])
        _set_response(state, response)
        return state


def generate_greeting(lead_name: str) -> str:
    """Called at session init to generate a dynamic LLM greeting."""
    try:
        result = _llm_json([{
            "role": "system",
            "content": GREETING_SYSTEM_PROMPT.format(name=lead_name),
        }])
        return result.get("response", f"Hi, am I speaking with {lead_name}?")
    except Exception as e:
        logger.error(f"[Questionnaire] Greeting generation error: {e}")
        return f"Hi, am I speaking with {lead_name}?"
