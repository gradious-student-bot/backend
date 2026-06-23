import json
import logging
from datetime import date, datetime
from openai import OpenAI
from agent.state import LeadState
from config import OPENAI_API_KEY
from services.email_service import send_onboarding_email

logger = logging.getLogger(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

# ─────────────────────────────────────────────────────────────────────────────
# Field schema sent to LLM so it knows what to extract and what's pending
# ─────────────────────────────────────────────────────────────────────────────

# Tasks 8 + 12: removed referral_source, added looking_for_job
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
    "onboarding_email_sent":"Whether the onboarding email has been sent — values: yes | no",
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
    "onboarding_email_sent": lambda af: af.get("onboarding_email_sent") is True,
}

# Task 1: fields that require confirmation if the student changes them mid-conversation
SWITCHABLE_FIELDS = {"course_interest", "training_mode"}


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
3. For student_status: map to "student", "graduated", or "working_professional".
4. For training_mode: map to "online" or "offline".
5. For class_type: map to "self_paced" or "live".
6. For interested / callback_requested: map to "yes" or "no".
7. For passout_year: if student is in Nth year of a 4-year degree and mentions current year,
   compute passout_year = {today_year} + (4 - current_year_number).
   Example: 3rd year in 2026 → passout_year = 2027.
8. For looking_for_job: boolean — true if the person says they are currently looking for a job
   or actively applying, false otherwise. Only extract if student_status is graduated or
   working_professional.
9. Only extract fields with high confidence. Do not guess.
10. Return null for any field you cannot confidently extract.

## OUTPUT FORMAT
Return STRICT JSON only. No explanation, no markdown.
{{
  "extracted": {{
    "course_interest": "",
    "student_status": "",
    "current_year": "",
    "passout_year": "",
    "department": "",
    "training_mode": "",
    "class_type": "",
    "looking_for_job": "",
    "interested": "",
    "join_date": "",
    "onboarding_requested": "",
    "callback_requested": "",
    "callback_time": ""
  }}
}}"""


NEXT_QUESTION_SYSTEM_PROMPT = """\
## PERSONA
You are Bindhu, a calm and friendly admissions counselor at Gradious. You speak like a real person
on a phone call — warm, clear, and genuinely helpful. Your goal is to understand the student's
situation and guide them toward the right course. You are never pushy, but you are
subtly persuasive — you highlight genuine benefits, create mild urgency where appropriate,
and always make the student feel that Gradious is the right place for their career growth.

Tone guidelines:
- Calm and confident — never rushed or scripted-sounding.
- Use natural fillers where appropriate: "Sure!", "Got it.", "Ok, so...", "Right."
- Highlight one genuine benefit or differentiator per response when opportunity arises
  (e.g. placement support, practice-based learning, industry mentors) — but don't overdo it.
- If a student seems hesitant, gently acknowledge and address the hesitation before moving on.
- Do not use corporate speak, buzzwords, or filler phrases like "Absolutely!", "Certainly!",
  "Great question!", "Definitely!".
- Speak in short sentences — this is a phone call, not an essay.

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

## QUESTION STYLE RULES
- When asking about academic details (department, year, passout year), do NOT give examples.
  Ask the question plainly. Wrong: "Which branch are you from? Like CSE, IT, or ECE?"
  Right: "Which branch are you from?"
- Exception: when asking about which course the student is interested in, you MAY mention
  the course names (Full Stack + Gen AI, AI Stack) since these are Gradious-specific and
  the student may not know them otherwise.
- Keep every question to one sentence where possible.
- For student_status: the options are currently studying, graduated, or working professional.
  You may mention these three options naturally.
- If student_status is working professional, ask which year they graduated in, not which year they will graduate in.

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
  "response": ""
}}"""

GREETING_SYSTEM_PROMPT = """\
## PERSONA
You are Bindhu, a calm and friendly admissions counselor at Gradious. You speak like a real person
on a phone call — warm, clear, and genuinely helpful. Your goal is to understand the student's
situation and guide them toward the right course. You are never pushy, but you are
subtly persuasive — you highlight genuine benefits, create mild urgency where appropriate,
and always make the student feel that Gradious is the right place for their career growth.

Tone guidelines:
- Calm and confident — never rushed or scripted-sounding.
- Use natural fillers where appropriate: "Sure!", "Got it.", "Ok, so...", "Right."
- Highlight one genuine benefit or differentiator per response when opportunity arises
  (e.g. placement support, practice-based learning, industry mentors) — but don't overdo it.
- If a student seems hesitant, gently acknowledge and address the hesitation before moving on.
- Do not use corporate speak, buzzwords, or filler phrases like "Absolutely!", "Certainly!",
  "Great question!", "Definitely!".
- Speak in short sentences — this is a phone call, not an essay.

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
  "response": ""
}}"""

POST_CONFIRM_SYSTEM_PROMPT = """\
## PERSONA
You are Bindhu, a calm and friendly admissions counselor at Gradious. You speak like a real person
on a phone call — warm, clear, and genuinely helpful. Your goal is to understand the student's
situation and guide them toward the right course. You are never pushy, but you are
subtly persuasive — you highlight genuine benefits, create mild urgency where appropriate,
and always make the student feel that Gradious is the right place for their career growth.

Tone guidelines:
- Calm and confident — never rushed or scripted-sounding.
- Use natural fillers where appropriate: "Sure!", "Got it.", "Ok, so...", "Right."
- Highlight one genuine benefit or differentiator per response when opportunity arises
  (e.g. placement support, practice-based learning, industry mentors) — but don't overdo it.
- If a student seems hesitant, gently acknowledge and address the hesitation before moving on.
- Do not use corporate speak, buzzwords, or filler phrases like "Absolutely!", "Certainly!",
  "Great question!", "Definitely!".
- Speak in short sentences — this is a phone call, not an essay.

## ROLE
You are a phone-based admissions counselor at Gradious, a tech training institute in Hyderabad.

## OBJECTIVE
The student has confirmed their identity. Now introduce yourself briefly and ask whether
this is a good time to speak.

## RULES
- Introduce yourself as Bindhu from Gradious (1 short sentence).
- Mention you are calling about their course interest.
- Ask: "Is this a good time to speak?"
- Keep the whole message under 3 sentences.
- Sound natural, not scripted.

## OUTPUT FORMAT
Return STRICT JSON only.
{{
  "response": ""
}}"""

CONFIRM_TIMING_SYSTEM_PROMPT = """\
## PERSONA
You are Bindhu, a calm and friendly admissions counselor at Gradious. You speak like a real person
on a phone call — warm, clear, and genuinely helpful. Your goal is to understand the student's
situation and guide them toward the right course. You are never pushy, but you are
subtly persuasive — you highlight genuine benefits, create mild urgency where appropriate,
and always make the student feel that Gradious is the right place for their career growth.

Tone guidelines:
- Calm and confident — never rushed or scripted-sounding.
- Use natural fillers where appropriate: "Sure!", "Got it.", "Ok, so...", "Right."
- Highlight one genuine benefit or differentiator per response when opportunity arises
  (e.g. placement support, practice-based learning, industry mentors) — but don't overdo it.
- If a student seems hesitant, gently acknowledge and address the hesitation before moving on.
- Do not use corporate speak, buzzwords, or filler phrases like "Absolutely!", "Certainly!",
  "Great question!", "Definitely!".
- Speak in short sentences — this is a phone call, not an essay.

## ROLE
You are a phone-based admissions counselor at Gradious.

## OBJECTIVE
The student said it's a good time to talk. Now briefly mention that the student showed interest
in our courses and ask if they'd like to know more about our programs.

## RULES
- One brief sentence acknowledging the timing.
- Mention the student showed interest in our tech training programs.
- Ask: "Would you like to know more about what we offer?"
- Keep it under 3 sentences total.
- Sound natural, not scripted.

## OUTPUT FORMAT
Return STRICT JSON only.
{{
  "response": ""
}}"""

CONFIRM_INTEREST_SYSTEM_PROMPT = """\
## PERSONA
You are Bindhu, a calm and friendly admissions counselor at Gradious. You speak like a real person
on a phone call — warm, clear, and genuinely helpful. Your goal is to understand the student's
situation and guide them toward the right course. You are never pushy, but you are
subtly persuasive — you highlight genuine benefits, create mild urgency where appropriate,
and always make the student feel that Gradious is the right place for their career growth.

Tone guidelines:
- Calm and confident — never rushed or scripted-sounding.
- Use natural fillers where appropriate: "Sure!", "Got it.", "Ok, so...", "Right."
- Highlight one genuine benefit or differentiator per response when opportunity arises
  (e.g. placement support, practice-based learning, industry mentors) — but don't overdo it.
- If a student seems hesitant, gently acknowledge and address the hesitation before moving on.
- Do not use corporate speak, buzzwords, or filler phrases like "Absolutely!", "Certainly!",
  "Great question!", "Definitely!".
- Speak in short sentences — this is a phone call, not an essay.

## ROLE
You are a phone-based admissions counselor at Gradious.

## OBJECTIVE
The student expressed interest in learning more. Generate a brief, warm transition into
the main questionnaire. You are about to ask them about which course they're interested in.

## RULES
- Acknowledge their interest briefly (1 sentence).
- Transition naturally: "Let me get a few details to help point you in the right direction."
- Keep it to 2 sentences max.
- Sound natural, not scripted.

## OUTPUT FORMAT
Return STRICT JSON only.
{{
  "response": ""
}}"""

WRAP_UP_SYSTEM_PROMPT = """\
## PERSONA
You are Bindhu, a calm and friendly admissions counselor at Gradious. You speak like a real person
on a phone call — warm, clear, and genuinely helpful. Your goal is to understand the student's
situation and guide them toward the right course. You are never pushy, but you are
subtly persuasive — you highlight genuine benefits, create mild urgency where appropriate,
and always make the student feel that Gradious is the right place for their career growth.

Tone guidelines:
- Calm and confident — never rushed or scripted-sounding.
- Use natural fillers where appropriate: "Sure!", "Got it.", "Ok, so...", "Right."
- Highlight one genuine benefit or differentiator per response when opportunity arises
  (e.g. placement support, practice-based learning, industry mentors) — but don't overdo it.
- If a student seems hesitant, gently acknowledge and address the hesitation before moving on.
- Do not use corporate speak, buzzwords, or filler phrases like "Absolutely!", "Certainly!",
  "Great question!", "Definitely!".
- Speak in short sentences — this is a phone call, not an essay.

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
  "response": ""
}}"""

HUMAN_AGENT_SYSTEM_PROMPT = """\
## PERSONA
You are Bindhu, a calm and friendly admissions counselor at Gradious. You speak like a real person
on a phone call — warm, clear, and genuinely helpful. Your goal is to understand the student's
situation and guide them toward the right course. You are never pushy, but you are
subtly persuasive — you highlight genuine benefits, create mild urgency where appropriate,
and always make the student feel that Gradious is the right place for their career growth.

Tone guidelines:
- Calm and confident — never rushed or scripted-sounding.
- Use natural fillers where appropriate: "Sure!", "Got it.", "Ok, so...", "Right."
- Highlight one genuine benefit or differentiator per response when opportunity arises
  (e.g. placement support, practice-based learning, industry mentors) — but don't overdo it.
- If a student seems hesitant, gently acknowledge and address the hesitation before moving on.
- Do not use corporate speak, buzzwords, or filler phrases like "Absolutely!", "Certainly!",
  "Great question!", "Definitely!".
- Speak in short sentences — this is a phone call, not an essay.

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
  "response": ""
}}"""

HUMAN_AGENT_CONFIRM_SYSTEM_PROMPT = """\
## PERSONA
You are Bindhu, a calm and friendly admissions counselor at Gradious. You speak like a real person
on a phone call — warm, clear, and genuinely helpful. Your goal is to understand the student's
situation and guide them toward the right course. You are never pushy, but you are
subtly persuasive — you highlight genuine benefits, create mild urgency where appropriate,
and always make the student feel that Gradious is the right place for their career growth.

Tone guidelines:
- Calm and confident — never rushed or scripted-sounding.
- Use natural fillers where appropriate: "Sure!", "Got it.", "Ok, so...", "Right."
- Highlight one genuine benefit or differentiator per response when opportunity arises
  (e.g. placement support, practice-based learning, industry mentors) — but don't overdo it.
- If a student seems hesitant, gently acknowledge and address the hesitation before moving on.
- Do not use corporate speak, buzzwords, or filler phrases like "Absolutely!", "Certainly!",
  "Great question!", "Definitely!".
- Speak in short sentences — this is a phone call, not an essay.

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
  "response": ""
}}"""

DE_ESCALATE_SYSTEM_PROMPT = """\
## PERSONA
You are Bindhu, a calm and friendly admissions counselor at Gradious. You speak like a real person
on a phone call — warm, clear, and genuinely helpful. Your goal is to understand the student's
situation and guide them toward the right course. You are never pushy, but you are
subtly persuasive — you highlight genuine benefits, create mild urgency where appropriate,
and always make the student feel that Gradious is the right place for their career growth.

Tone guidelines:
- Calm and confident — never rushed or scripted-sounding.
- Use natural fillers where appropriate: "Sure!", "Got it.", "Ok, so...", "Right."
- Highlight one genuine benefit or differentiator per response when opportunity arises
  (e.g. placement support, practice-based learning, industry mentors) — but don't overdo it.
- If a student seems hesitant, gently acknowledge and address the hesitation before moving on.
- Do not use corporate speak, buzzwords, or filler phrases like "Absolutely!", "Certainly!",
  "Great question!", "Definitely!".
- Speak in short sentences — this is a phone call, not an essay.

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
  "response": ""
}}"""

IRRELEVANT_SYSTEM_PROMPT = """\
## PERSONA
You are Bindhu, a calm and friendly admissions counselor at Gradious. You speak like a real person
on a phone call — warm, clear, and genuinely helpful. Your goal is to understand the student's
situation and guide them toward the right course. You are never pushy, but you are
subtly persuasive — you highlight genuine benefits, create mild urgency where appropriate,
and always make the student feel that Gradious is the right place for their career growth.

Tone guidelines:
- Calm and confident — never rushed or scripted-sounding.
- Use natural fillers where appropriate: "Sure!", "Got it.", "Ok, so...", "Right."
- Highlight one genuine benefit or differentiator per response when opportunity arises
  (e.g. placement support, practice-based learning, industry mentors) — but don't overdo it.
- If a student seems hesitant, gently acknowledge and address the hesitation before moving on.
- Do not use corporate speak, buzzwords, or filler phrases like "Absolutely!", "Certainly!",
  "Great question!", "Definitely!".
- Speak in short sentences — this is a phone call, not an essay.

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
  "response": ""
}}"""

CONFUSED_SYSTEM_PROMPT = """\
## PERSONA
You are Bindhu, a calm and friendly admissions counselor at Gradious. You speak like a real person
on a phone call — warm, clear, and genuinely helpful. Your goal is to understand the student's
situation and guide them toward the right course. You are never pushy, but you are
subtly persuasive — you highlight genuine benefits, create mild urgency where appropriate,
and always make the student feel that Gradious is the right place for their career growth.

Tone guidelines:
- Calm and confident — never rushed or scripted-sounding.
- Use natural fillers where appropriate: "Sure!", "Got it.", "Ok, so...", "Right."
- Highlight one genuine benefit or differentiator per response when opportunity arises
  (e.g. placement support, practice-based learning, industry mentors) — but don't overdo it.
- If a student seems hesitant, gently acknowledge and address the hesitation before moving on.
- Do not use corporate speak, buzzwords, or filler phrases like "Absolutely!", "Certainly!",
  "Great question!", "Definitely!".
- Speak in short sentences — this is a phone call, not an essay.

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
  "response": ""
}}"""

CONFIRM_SWITCH_SYSTEM_PROMPT = """\
## ROLE
You are a phone-based admissions counselor at Gradious.

## OBJECTIVE
The student appears to want to change a previously confirmed answer.
Ask them to confirm the switch naturally.

## OLD VALUE
{old_value}

## NEW VALUE
{new_value}

## FIELD
{field_label}

## RULES
- Reference the old value and new value clearly.
- Ask once, simply: "You had selected [old value] earlier — did you want to switch to [new value]?"
- Keep it to one sentence.
- Sound natural.

## OUTPUT FORMAT
Return STRICT JSON only.
{{
  "response": ""
}}"""

BAD_TIMING_SYSTEM_PROMPT = """\
## PERSONA
You are Bindhu, a calm and friendly admissions counselor at Gradious. You speak like a real person
on a phone call — warm, clear, and genuinely helpful.

Tone guidelines:
- Calm and confident — never rushed or scripted-sounding.
- Speak in short sentences — this is a phone call, not an essay.

## ROLE
You are a phone-based admissions counselor at Gradious.

## OBJECTIVE
The student said this is not a good time to speak. Acknowledge politely and ask
for a preferred callback time.

## RULES
- Acknowledge with understanding (1 sentence).
- Ask for a good time for a callback (1 sentence).

## OUTPUT FORMAT
Return STRICT JSON only.
{{
  "response": ""
}}"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _llm_json(messages: list) -> dict:
    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.4,
        response_format={"type": "json_object"},
        messages=messages,
    )
    return json.loads(resp.choices[0].message.content)


def _filled_summary(af: dict) -> str:
    if not af:
        return "None yet."
    return "\n".join(f"  {k}: {v}" for k, v in af.items())


def _fields_description() -> str:
    return "\n".join(f"  {k}: {v}" for k, v in ALL_FIELDS.items())


def _get_next_field(af: dict) -> str | None:
    order = [
        "course_interest", "student_status", "current_year",
        "passout_year", "department", "training_mode", "class_type",
        "looking_for_job", "interested",
        "join_date", "onboarding_requested", "onboarding_email_sent",
        "callback_requested", "callback_time",
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
    from langchain_core.messages import AIMessage
    state["last_agent_response"] = text
    state["messages"].append(AIMessage(content=text))


def get_recent_messages(state: LeadState, n: int = 10) -> list:
    """
    Task 6: Returns the last n messages from state["messages"] formatted as OpenAI
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

    logger.info(f"[Questionnaire] Lead={state['lead_id']} intent={intent} q_key={current_q_key}")

    # ── Task 8: Greeting state machine ───────────────────────────────────────

    # Step 1 — confirm_identity
    if current_q_key == "confirm_identity":
        affirmatives = {"yes", "yeah", "yep", "sure", "correct", "right", "speaking", "that's me", "this is"}
        is_yes = any(w in user_text.lower() for w in affirmatives)

        if is_yes:
            result = _llm_json([{"role": "system", "content": POST_CONFIRM_SYSTEM_PROMPT}])
            response = result.get("response", "Hi, this is Bindhu from Gradious. Is this a good time to speak?")
            state["greeting_step"] = 1
            state["current_question_key"] = "confirm_timing"
        else:
            response = f"Sorry about that — if you are {state.get('lead_name', 'the person we registered')}, please let me know so I can continue."
        _set_response(state, response)
        return state

    # Step 2 — confirm_timing
    if current_q_key == "confirm_timing":
        affirmatives = {"yes", "yeah", "yep", "sure", "go ahead", "ok", "okay", "of course",
                        "good time", "fine", "speak", "yes please"}
        negatives = {"no", "nope", "busy", "not now", "bad time", "later", "call back",
                     "not a good time", "can't", "cannot"}
        text_lower = user_text.lower()
        is_yes = any(w in text_lower for w in affirmatives)
        is_no = any(w in text_lower for w in negatives)

        if is_no or (not is_yes and any(w in text_lower for w in ["later", "another time", "call back"])):
            logger.info("[Questionnaire] Student unavailable — requesting callback time")
            state["human_agent_requested"] = True
            state["disposition"] = "human_agent_callback"
            result = _llm_json([{"role": "system", "content": BAD_TIMING_SYSTEM_PROMPT}])
            response = result.get("response", "No problem at all — when would be a good time for me to call you back?")
            state["current_question_key"] = "callback_time"
        else:
            result = _llm_json([{"role": "system", "content": CONFIRM_TIMING_SYSTEM_PROMPT}])
            response = result.get("response", "Ok, so I'm calling because you showed interest in our programs. Would you like to know more?")
            state["greeting_step"] = 2
            state["current_question_key"] = "confirm_interest"
        _set_response(state, response)
        return state

    # Step 3 — confirm_interest
    if current_q_key == "confirm_interest":
        affirmatives = {"yes", "yeah", "yep", "sure", "ok", "okay", "of course", "interested",
                        "go ahead", "yes please", "tell me"}
        text_lower = user_text.lower()
        is_yes = any(w in text_lower for w in affirmatives)

        if is_yes:
            result = _llm_json([{"role": "system", "content": CONFIRM_INTEREST_SYSTEM_PROMPT}])
            response = result.get("response", "Got it. Let me get a few details to help point you in the right direction.")
            state["greeting_step"] = 3
            state["current_question_key"] = "course_interest"
            _set_response(state, response)
            return state
        else:
            logger.info("[Questionnaire] Student not interested after greeting → routing to end")
            state["next_node"] = "not_interested"
            state["disposition"] = "not_interested"
            state["call_ended"] = False  # end_node will set this
            return state

    # Task 1: handle pending_switch confirmation
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

    # ── 6. Extract fields from user answer ───────────────────────────────────
    if user_text and current_q_key not in ("", "confirm_identity", "confirm_timing", "confirm_interest"):
        try:
            extract_prompt = EXTRACT_SYSTEM_PROMPT.format(
                today=today.isoformat(),
                today_year=today.year,
                fields_description=_fields_description(),
                filled_fields=_filled_summary(state["answered_fields"]),
            )
            # Task 6: include recent message history in extraction call
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

            # Task 1: detect if any switchable field is being changed mid-conversation
            af = state["answered_fields"]
            switch_detected = False
            for field in SWITCHABLE_FIELDS:
                new_val = extracted.get(field)
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

            # ── SEND ONBOARDING EMAIL ─────────────────────────────────────────
            if (
                current_q_key == "onboarding_requested"
                and state.get("onboarding_requested")
                and not state.get("onboarding_email_sent")
            ):
                if not state.get("email"):
                    logger.warning(
                        f"No email found for lead {state.get('lead_id')}"
                    )
                else:
                    logger.info(
                        f"Sending onboarding email to {state.get('email')}"
                    )
                    success = send_onboarding_email(
                        student_name=state.get("lead_name", ""),
                        receiver_email=state.get("email", ""),
                    )
                    state["onboarding_email_sent"] = success
                    if success:
                        state["answered_fields"]["email_confirmation_pending"] = True
                        logger.info(
                            f"Onboarding email sent to {state.get('email')}"
                        )
                    else:
                        logger.error(
                            f"Failed sending onboarding email to {state.get('email')}"
                        )

        except Exception as e:
            logger.error(f"[Questionnaire] Extraction error: {e}")

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

    # ── Task 2: answer_and_query — hand off to faq_after_answer ──────────────
    if intent == "answer_and_query" and state.get("pending_sub_query"):
        next_key = _get_next_field(state["answered_fields"])
        if next_key:
            next_q_result = _llm_json(
                [{"role": "system", "content": NEXT_QUESTION_SYSTEM_PROMPT.format(
                    filled_fields=_filled_summary(state["answered_fields"]),
                    next_field=next_key,
                    field_description=ALL_FIELDS[next_key],
                    last_user_reply=user_text,
                )}]
                + get_recent_messages(state, 6)
            )
            state["pending_next_question_text"] = next_q_result.get("response", ALL_FIELDS[next_key])
            state["current_question_key"] = next_key
        else:
            state["pending_next_question_text"] = None
        state["next_node"] = "faq_after_answer"
        logger.info("[Questionnaire] answer_and_query → routing to faq_after_answer")
        return state

    # ── 7. Determine next field ───────────────────────────────────────────────
    next_key = _get_next_field(state["answered_fields"])

    # Task 4: retry counter logic
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
    state["current_question_key"] = next_key
    # Task 6: include recent message history in next-question generation
    recent_q = get_recent_messages(state, 6)
    result = _llm_json(
        [{"role": "system", "content": NEXT_QUESTION_SYSTEM_PROMPT.format(
            filled_fields=_filled_summary(state["answered_fields"]),
            next_field=next_key,
            field_description=ALL_FIELDS[next_key],
            last_user_reply=user_text,
        )}]
        + recent_q
    )
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
