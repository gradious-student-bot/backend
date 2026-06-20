import json
import logging
from datetime import date
from openai import OpenAI
from agent.state import LeadState
from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# Field schema sent to LLM so it knows what to extract and what's pending
# ─────────────────────────────────────────────────────────────────────────────

ALL_FIELDS = {
    "course_interest":   "Course the student wants — one of: fullstack_batch, ai_batch, dsa_batch",
    "student_status":    "Whether student is currently studying or already graduated — values: student | graduated",
    "current_year":      "Current academic year if student (e.g. 2nd year, 3rd year) — only for student status",
    "passout_year":      "Year of graduation or expected passout (e.g. 2026, 2027)",
    "department":        "Branch or department (e.g. CSE, ECE, IT, Mechanical)",
    "training_mode":     "Preferred training mode — values: online | offline",
    "class_type":        "If online: self_paced or live — only relevant when training_mode is online",
    "budget_range":      "Budget the student can spend on the course (e.g. 15000-20000, around 20k)",
    "referral_source":   "How the student heard about Gradious (e.g. Instagram, friend, YouTube)",
    "interested":        "Whether the student is interested in joining — values: yes | no",
    "join_date":         "When the student plans to start — only if interested is yes",
    "callback_requested":"Whether student wants a callback from admissions team — values: yes | no",
    "callback_time":     "Preferred time for the callback — only if callback_requested is yes",
}

# Fields that are conditionally required
CONDITIONAL_FIELDS = {
    "current_year":      lambda af: af.get("student_status") == "student",
    "class_type":        lambda af: af.get("training_mode") == "online",
    "join_date":         lambda af: str(af.get("interested", "")).lower() == "yes",
    "callback_time":     lambda af: str(af.get("callback_requested", "")).lower() == "yes",
}

EXTRACT_SYSTEM_PROMPT = """\
## ROLE
You are a data extraction assistant for an admissions counseling agent at Gradious,
a tech training institute. Today's date is {today}.

## OBJECTIVE
Given the agent's last question and the student's latest reply, extract any lead information
the student has provided. A student may answer multiple fields in a single reply
(e.g. "I'm a 3rd year CSE student" answers student_status, current_year, department, and passout_year).

## FIELDS TO EXTRACT
{fields_description}

## CURRENT STATE OF FILLED FIELDS
{filled_fields}

## RULES
1. Extract ONLY fields that the student has clearly provided in their reply.
2. For course_interest: map to fullstack_batch / ai_batch / dsa_batch based on what student says.
3. For student_status: map to "student" or "graduated".
4. For training_mode: map to "online" or "offline".
5. For class_type: map to "self_paced" or "live".
6. For interested / callback_requested: map to "yes" or "no".
7. For passout_year: if student is in Nth year of a 4-year degree and mentions current year,
   compute passout_year = {today_year} + (4 - current_year_number).
   Example: 3rd year in 2026 → passout_year = 2027.
8. Only extract fields with high confidence. Do not guess.
9. Return null for any field you cannot confidently extract.

## OUTPUT FORMAT
Return STRICT JSON only. No explanation, no markdown.
{{
  "extracted": {{
    "course_interest": "<value or null>",
    "student_status": "<value or null>",
    "current_year": "<value or null>",
    "passout_year": "<value or null>",
    "department": "<value or null>",
    "training_mode": "<value or null>",
    "class_type": "<value or null>",
    "budget_range": "<value or null>",
    "referral_source": "<value or null>",
    "interested": "<value or null>",
    "join_date": "<value or null>",
    "callback_requested": "<value or null>",
    "callback_time": "<value or null>"
  }}
}}"""

NEXT_QUESTION_SYSTEM_PROMPT = """\
## ROLE
You are a phone-based admissions counselor at Gradious, a tech training institute in Hyderabad.

## OBJECTIVE
Ask the next pending question to the student in a natural, conversational phone call style.

## TONE GUIDELINES
- Simple and professional. No excessive praise or adjectives.
- Brief acknowledgements only: "Ok.", "Got it.", "Sure." — nothing more.
- Do not repeat the student's name repeatedly.
- Keep the question short — this is a phone call, not a form.
- Sound natural and vary phrasing across the conversation.

## FILLED FIELDS (already collected — do NOT ask again)
{filled_fields}

## NEXT QUESTION TO ASK
Field: {next_field}
Description: {field_description}

## STUDENT'S LAST REPLY (for context to phrase acknowledgement)
"{last_user_reply}"

## OUTPUT FORMAT
Return STRICT JSON only.
{{
  "response": "<your natural conversational response — acknowledgement + question>"
}}"""

GREETING_SYSTEM_PROMPT = """\
## ROLE
You are a phone-based admissions counselor at Gradious, a tech training institute.

## OBJECTIVE
Generate a natural, warm but brief call-opening message. This is the very first thing
the agent says on the call.

## RULES
- Confirm you are speaking to the correct person by asking "Am I speaking with {name}?"
- Keep it short — one sentence only.
- Sound natural, not scripted.
- Do not add anything else — no introduction, no pitch yet.

## OUTPUT FORMAT
Return STRICT JSON only.
{{
  "response": "<your greeting>"
}}"""

POST_CONFIRM_SYSTEM_PROMPT = """\
## ROLE
You are a phone-based admissions counselor at Gradious, a tech training institute in Hyderabad.

## OBJECTIVE
The student has confirmed their identity. Now introduce yourself briefly and ask the first question.

## RULES
- Introduce yourself as Bindhu from Gradious (1 short sentence).
- Mention you are calling about their course registration.
- Ask whether they are currently a student or have already graduated.
- Keep the whole message under 3 sentences.
- Sound natural, not scripted.

## OUTPUT FORMAT
Return STRICT JSON only.
{{
  "response": "<your response>"
}}"""

WRAP_UP_SYSTEM_PROMPT = """\
## ROLE
You are a phone-based admissions counselor at Gradious.

## OBJECTIVE
All lead information has been collected. Close the call warmly and professionally.

## DISPOSITION
{disposition}

## RULES
- Keep it brief (2 sentences max).
- If disposition is "interested": confirm someone will follow up.
- If disposition is "callback": confirm the callback is scheduled.
- If disposition is "not_interested": politely wish them well, leave door open.
- Sound natural and genuine.

## OUTPUT FORMAT
Return STRICT JSON only.
{{
  "response": "<your closing message>"
}}"""

HUMAN_AGENT_SYSTEM_PROMPT = """\
## ROLE
You are a phone-based admissions counselor at Gradious.

## OBJECTIVE
The student wants to speak to a human admissions expert.
Acknowledge their request warmly and ask for a preferred callback time.

## RULES
- Acknowledge the request with understanding (1 sentence).
- Ask for a good time for the callback (1 sentence).
- Keep it brief and natural.

## OUTPUT FORMAT
Return STRICT JSON only.
{{
  "response": "<your response>"
}}"""

HUMAN_AGENT_CONFIRM_SYSTEM_PROMPT = """\
## ROLE
You are a phone-based admissions counselor at Gradious.

## OBJECTIVE
The student has provided their preferred callback time. Confirm and close the call.

## CALLBACK TIME PROVIDED
{callback_time}

## RULES
- Confirm the callback time naturally.
- Close the call warmly (2 sentences max).

## OUTPUT FORMAT
Return STRICT JSON only.
{{
  "response": "<your response>"
}}"""

DE_ESCALATE_SYSTEM_PROMPT = """\
## ROLE
You are a calm, professional admissions counselor at Gradious on a phone call.

## OBJECTIVE
The student has been rude or hostile. De-escalate calmly without being defensive,
then gently re-ask the pending question.

## PENDING QUESTION
{pending_question_description}

## RULES
- One calm acknowledgement sentence. No confrontation.
- Redirect to the pending question naturally.
- Keep it very brief.

## OUTPUT FORMAT
Return STRICT JSON only.
{{
  "response": "<your response>"
}}"""

IRRELEVANT_SYSTEM_PROMPT = """\
## ROLE
You are a phone-based admissions counselor at Gradious.

## OBJECTIVE
The student said something unrelated to courses or Gradious. Politely redirect them
back to the conversation and re-ask the pending question.

## PENDING QUESTION
{pending_question_description}

## OUTPUT FORMAT
Return STRICT JSON only.
{{
  "response": "<your response>"
}}"""

CONFUSED_SYSTEM_PROMPT = """\
## ROLE
You are a phone-based admissions counselor at Gradious.

## OBJECTIVE
The student seems confused or unsure about the question. Rephrase it more simply
with a brief hint to help them answer.

## ORIGINAL QUESTION
{pending_question_description}

## RULES
- Rephrase simply — do not add new information.
- Keep it short (1–2 sentences).
- Sound natural.

## OUTPUT FORMAT
Return STRICT JSON only.
{{
  "response": "<your rephrased question>"
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _llm_json(messages: list[dict]) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.4,
        response_format={"type": "json_object"},
        messages=messages,
    )
    return json.loads(resp.choices[0].message.content)


def _filled_summary(af: dict) -> str:
    logger.info(f"[Questionnaire] Filled fields summary: {af}")
    if not af:
        return "None yet."
    return "\n".join(f"  {k}: {v}" for k, v in af.items())


def _fields_description() -> str:
    return "\n".join(f"  {k}: {v}" for k, v in ALL_FIELDS.items())


def _get_next_field(af: dict) -> str | None:
    order = [
        "course_interest", "student_status", "current_year",
        "passout_year", "department", "training_mode", "class_type",
        "budget_range", "referral_source", "interested",
        "join_date", "callback_requested", "callback_time",
    ]
    for field in order:
        if field in af:
            continue
        # Check conditional skip
        if field in CONDITIONAL_FIELDS:
            if not CONDITIONAL_FIELDS[field](af):
                continue  # condition not met — skip this field
        return field
    return None


# Need to check this
def _apply_extracted(state: LeadState, extracted: dict):
    af = state["answered_fields"]
    for key, value in extracted.items():
        if value is None or value == "null":
            continue
        af[key] = value
        # Mirror to top-level
        if key in ("course_interest", "student_status", "current_year", "passout_year",
                   "department", "training_mode", "class_type", "budget_range",
                   "referral_source", "join_date", "callback_time"):
            state[key] = value  # type: ignore
        if key == "interested":
            state["interested"] = str(value).lower() == "yes"
        if key == "callback_requested":
            state["callback_requested"] = str(value).lower() == "yes"


def _set_response(state: LeadState, text: str):
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

    logger.info(f"[Questionnaire] Lead={state['lead_id']} intent={intent} q_key={current_q_key}")

    # ── 1. Identity confirmation ──────────────────────────────────────────────
    if current_q_key == "confirm_identity":
        affirmatives = {"yes", "yeah", "yep", "sure", "correct", "right", "speaking", "that's me", "this is"}
        is_yes = any(w in user_text.lower() for w in affirmatives)

        if is_yes:
            result = _llm_json([{"role": "system", "content": POST_CONFIRM_SYSTEM_PROMPT}])
            response = result.get("response", "This is Bindhu from Gradious. Are you currently a student or have you graduated?")
            state["current_question_key"] = "student_status"
        else:
            response = f"Sorry about that — if you are {state.get('lead_name', 'the person we registered')}, please let me know so I can continue."
        _set_response(state, response)
        return state

    # ── 2. Human agent short-circuit ─────────────────────────────────────────
    if state.get("human_agent_requested") or intent == "human_agent":
        state["human_agent_requested"] = True
        state["disposition"] = "human_agent_callback"

        if not state.get("callback_time"):
            result = _llm_json([{"role": "system", "content": HUMAN_AGENT_SYSTEM_PROMPT}])
            response = result.get("response", "Sure! What time works best for a callback from our team?")
            state["current_question_key"] = "callback_time"
        else:
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

    # ── 3. Rude ───────────────────────────────────────────────────────────────
    if intent == "rude":
        next_key = _get_next_field(state["answered_fields"])
        desc = ALL_FIELDS.get(next_key or current_q_key, "our current question")
        result = _llm_json([{
            "role": "system",
            "content": DE_ESCALATE_SYSTEM_PROMPT.format(pending_question_description=desc),
        }])
        _set_response(state, result.get("response", "I understand. Let me know when you're ready to continue."))
        return state

    # ── 4. Irrelevant ─────────────────────────────────────────────────────────
    if intent == "irrelevant":
        next_key = _get_next_field(state["answered_fields"])
        desc = ALL_FIELDS.get(next_key or current_q_key, "our current question")
        result = _llm_json([{
            "role": "system",
            "content": IRRELEVANT_SYSTEM_PROMPT.format(pending_question_description=desc),
        }])
        _set_response(state, result.get("response", "Let's get back on track — " + desc))
        return state

    # ── 5. Confused ───────────────────────────────────────────────────────────
    if intent == "confused":
        next_key = _get_next_field(state["answered_fields"])
        desc = ALL_FIELDS.get(next_key or current_q_key, "the current question")
        result = _llm_json([{
            "role": "system",
            "content": CONFUSED_SYSTEM_PROMPT.format(pending_question_description=desc),
        }])
        _set_response(state, result.get("response", desc))
        return state

    # ── 6. Extract fields from user answer ───────────────────────────────────
    if user_text and current_q_key not in ("", "confirm_identity"):
        try:
            extract_prompt = EXTRACT_SYSTEM_PROMPT.format(
                today=today.isoformat(),
                today_year=today.year,
                fields_description=_fields_description(),
                filled_fields=_filled_summary(state["answered_fields"]),
            )
            extract_result = client.chat.completions.create(
                model="gpt-4.1-mini",
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": extract_prompt},
                    {"role": "user", "content": f"Agent asked: {state.get('last_agent_response', '')}\nStudent replied: {user_text}"},
                ],
            )
            extracted = json.loads(extract_result.choices[0].message.content).get("extracted", {})
            logger.info(f"[Questionnaire] Extracted fields: {extracted}")
            _apply_extracted(state, extracted)
        except Exception as e:
            logger.error(f"[Questionnaire] Extraction error: {e}")

    # ── 7. Determine next field ───────────────────────────────────────────────
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
    state["current_question_key"] = next_key
    result = _llm_json([{
        "role": "system",
        "content": NEXT_QUESTION_SYSTEM_PROMPT.format(
            filled_fields=_filled_summary(state["answered_fields"]),
            next_field=next_key,
            field_description=ALL_FIELDS[next_key],
            last_user_reply=user_text,
        ),
    }])
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