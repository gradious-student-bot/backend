import json
import logging
from datetime import date, datetime
from openai import OpenAI

from services.llm_service import llm
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


def _course_display_name(course_value: str | None) -> str | None:
    """Converts internal course ids / Airtable values into user-friendly names."""
    if not course_value:
        return None

    value = str(course_value).strip()
    normalized = value.lower().replace("-", "_").replace(" ", "_")

    course_map = {
        "fullstack_batch": "Full Stack + Gen AI",
        "full_stack_gen_ai": "Full Stack + Gen AI",
        "full_stack_+_gen_ai": "Full Stack + Gen AI",
        "full_stack": "Full Stack + Gen AI",
        "campus_fullstack": "Campus Full Stack + Gen AI",
        "ai_batch": "AI Stack",
        "ai_stack": "AI Stack",
        "campus_ai": "Campus AI Stack",
        "dsa_batch": "DSA",
        "dsa": "DSA",
    }

    return course_map.get(normalized, value.replace("_", " ").title())


def _known_course_from_state(state: LeadState) -> str | None:
    """Gets course interest either from answered_fields or top-level state."""
    af = state.get("answered_fields") or {}
    return af.get("course_interest") or state.get("course_interest")


def _sync_known_course_to_answered_fields(state: LeadState) -> None:
    """If Airtable already gave us course_interest, mark it as answered so the bot will not ask again."""
    course = state.get("course_interest")
    if course:
        state.setdefault("answered_fields", {})["course_interest"] = course


def _set_response(state: LeadState, text: str):
    """
    Sets the last_agent_response and appends an AIMessage to the messages list.
    """
    from langchain_core.messages import AIMessage
    state["last_agent_response"] = text
    state["messages"].append(AIMessage(content=text))



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

    # If Airtable loaded course_interest into state, mark it answered so we do not ask it again.
    _sync_known_course_to_answered_fields(state)

    # Step 1 — confirm_identity
    if current_q_key == "confirm_identity":
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
        result = llm.invoke_json(
            [{"role": "system", "content": confirm_identity_prompt}]
            + llm.get_recent_messages(state)
            + [{"role": "user", "content": user_text}]
        )
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
        course_value = _known_course_from_state(state)
        course_name = _course_display_name(course_value)
        course_context = (
            f"the {course_name} training program"
            if course_name
            else "our training programs — Full Stack + Gen AI, AI Stack, and DSA"
        )

        confirm_timing_prompt = f"""## ROLE
You are Bindhu, a phone-based admissions counselor at Gradious.

## OBJECTIVE
The user has just responded to "Is this a good time to speak?"
First detect if the user said it's a good time (yes) or not (no/busy/later).
Then generate the appropriate response.

## KNOWN COURSE CONTEXT
{course_context}

    ## USER'S REPLY
    "{user_text}"

## RULES
- If YES (good time to talk):
  - Mention that the user had shown interest in {course_context}.
  - Ask if they would like to know more about it.
  - If a specific course is known, do NOT list all courses and do NOT ask which course they are interested in.
  - If no specific course is known, briefly mention the available programs: Full Stack + Gen AI, AI Stack, and DSA.
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
        result = llm.invoke_json(
            [{"role": "system", "content": confirm_timing_prompt}]
            + llm.get_recent_messages(state)
            + [{"role": "user", "content": user_text}]
        )
        is_yes = result.get("is_yes", True)
        fallback_yes = (
            f"Ok, I'm calling because you had shown interest in {course_context}. "
            "Would you like me to quickly explain it?"
        )
        response = result.get(
            "response",
            fallback_yes if is_yes
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
        course_value = _known_course_from_state(state)
        course_name = _course_display_name(course_value)
        known_course_text = course_name or "Not available"

        confirm_interest_prompt = f"""## ROLE
You are Bindhu, a phone-based admissions counselor at Gradious.

## OBJECTIVE
The user has just responded to whether they want to know more about the Gradious program.
First detect if the user expressed interest (yes) or declined (no/not interested).
Then generate the appropriate response.

## KNOWN COURSE
{known_course_text}

## USER'S REPLY
"{user_text}"

## RULES
- If YES and KNOWN COURSE is available:
  - Do NOT ask which course they are interested in.
  - Briefly mention the known course by name.
  - Give one simple line about the course benefit.
  - Then ask the next required detail: whether they are currently studying, graduated, or working professional.
- If YES and KNOWN COURSE is "Not available":
  - Ask which course they are interested in.
  - Course options: Full Stack + Gen AI, AI Stack, and DSA.
- If NO (not interested):
  - Politely acknowledge their response.
  - Thank them for their time in one short sentence.
- If UNCLEAR, treat as YES.
- Do not say "Great to hear you're interested!".
- Do not use "Absolutely!", "Certainly!", "Great question!", or "Definitely!".
- Keep it under 3 sentences. Sound natural and conversational.

## OUTPUT FORMAT
Return STRICT JSON only.
{{
"is_yes": true or false,
"response": ""
}}"""
        result = llm.invoke_json(
            [{"role": "system", "content": confirm_interest_prompt}]
            + llm.get_recent_messages(state)
            + [{"role": "user", "content": user_text}]
        )
        is_yes = result.get("is_yes", True)

        if is_yes:
            logger.info("User said yes in confirm interest")
            state["greeting_step"] = 3

            confirm_interest_prompt = CONFIRM_INTEREST_SYSTEM_PROMPT.format(
                course_interest=state.get("course_interest") or "Not available"
            )

            result = llm.invoke_json([
                {"role": "system", "content": confirm_interest_prompt}
            ])

            response = result.get("response")


            if state.get("course_interest"):
                state["answered_fields"]["course_interest"] = state["course_interest"]

            next_key = _get_next_field(state["answered_fields"])
            state["current_question_key"] = next_key or "student_status"

            _set_response(state, response)
            return state
        else:
            logger.info("[Questionnaire] Student not interested after greeting → routing to end")
            
            response = result.get("response", "No problem. Thanks for your time, and I wish you the best.")
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
    
    # ── 2. Human agent short-circuit ─────────────────────────────────────────
    if state.get("human_agent_requested") or intent == "human_agent":
        state["human_agent_requested"] = True
        state["disposition"] = "human_agent_callback"

        if not state.get("callback_time"):
            result = llm.invoke_json([{"role": "system", "content": HUMAN_AGENT_SYSTEM_PROMPT}])
            response = result.get("response", "Sure! What time works best for a callback from our team?")
            state["current_question_key"] = "callback_time"
        else:
            _apply_extracted(state, {"callback_time": user_text})
            state["callback_requested"] = True
            state["call_ended"] = True
            result = llm.invoke_json([{
                "role": "system",
                "content": HUMAN_AGENT_CONFIRM_SYSTEM_PROMPT.format(callback_time=user_text),
            }])
            response = result.get("response", f"Done! Our team will call you at {user_text}. Talk soon!")
            _set_response(state, response)
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

            extracted = llm.invoke_json(
                messages=(
                    [{"role": "system", "content": extract_prompt}]
                    + llm.get_recent_messages(state)
                    + [{"role": "user", "content": f"Agent asked: {state.get('last_agent_response', '')}\nStudent replied: {user_text}"}]
                )
            )
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
                    
                    switch_result = llm.invoke_json([{
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

                    result = llm.invoke_json([{
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

                    result = llm.invoke_json([{
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

                    result = llm.invoke_json([{
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
            next_q_result = llm.invoke_json(
                [{"role": "system", "content": NEXT_QUESTION_SYSTEM_PROMPT.format(
                    filled_fields=_filled_summary(state["answered_fields"]),
                    next_field=next_key,
                    field_description=ALL_FIELDS[next_key],
                    last_user_reply=user_text,
                    std_course_change=state["answered_fields"].get("std_course_change", False)
                )}]
                + llm.get_recent_messages(state)
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

        result = llm.invoke_json([{
            "role": "system",
            "content": WRAP_UP_SYSTEM_PROMPT.format(disposition=state["disposition"]),
        }])
        _set_response(state, result.get("response", "Thanks for your time. We'll be in touch soon!"))
        return state

    # ── 8. Ask next question ──────────────────────────────────────────────────
    if intent != "human_agent" and not state.get("human_agent_requested"):
        logger.info("[Questionnaire] Asking next question")
        state["current_question_key"] = next_key

        next_que_prompt = NEXT_QUESTION_SYSTEM_PROMPT.format(
            filled_fields=_filled_summary(state["answered_fields"]),
            next_field=next_key,
            field_description=ALL_FIELDS[next_key],
            last_user_reply=user_text,
            std_course_change=state["answered_fields"].get("std_course_change", False)
        )

        result = llm.invoke_json(
            [{"role": "system", "content": next_que_prompt}]
            + llm.get_recent_messages(state)
            + [{"role": "user", "content": user_text}]
        )

        state["answered_fields"]["std_course_change"] = False  # reset after using it in prompt
        response = result.get("response", ALL_FIELDS[next_key])
        _set_response(state, response)
        return state


def generate_greeting(lead_name: str) -> str:
    """Called at session init to generate a dynamic LLM greeting."""
    try:
        result = llm.invoke_json([{
            "role": "system",
            "content": GREETING_SYSTEM_PROMPT.format(name=lead_name),
        }])
        return result.get("response", f"Hi, am I speaking with {lead_name}?")
    except Exception as e:
        logger.error(f"[Questionnaire] Greeting generation error: {e}")
        return f"Hi, am I speaking with {lead_name}?"
