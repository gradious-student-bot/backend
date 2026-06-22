import json
import logging
from openai import OpenAI
from agent.state import LeadState
from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

END_SYSTEM_PROMPT = """\
## PERSONA
You are Ava, a calm and friendly admissions counselor at Gradious. You speak like a real person
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
You are a phone-based admissions counselor at Gradious closing a call.

## OBJECTIVE
Close the call naturally based on the reason below.

## REASON
{reason}

## RULES
- 2 sentences max.
- not_interested: politely wish them well, say they can reach out anytime.
- end_call: thank them and say the team will be in touch.
- Warm but brief.

## OUTPUT FORMAT
Return STRICT JSON only.
{{
  "response": "<closing message>"
}}"""


def end_node(state: LeadState) -> LeadState:
    from langchain_core.messages import AIMessage

    intent = state.get("next_node", "end_call")
    logger.info(f"[End] Lead={state['lead_id']} | Reason={intent}")

    if intent == "not_interested":
        state["disposition"] = "not_interested"
        reason = "Student said they are not interested in the courses."
    else:
        state["disposition"] = state.get("disposition") or "end_call"
        reason = "Student is ending the call politely."

    try:
        result = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.4,
            response_format={"type": "json_object"},
            messages=[{
                "role": "system",
                "content": END_SYSTEM_PROMPT.format(reason=reason),
            }],
        )
        parsed = json.loads(result.choices[0].message.content)
        response = parsed.get("response", "Thanks for your time. Have a good day!")
    except Exception as e:
        logger.error(f"[End] LLM error: {e}")
        response = "Thanks for your time. Have a good day!"

    state["call_ended"] = True
    state["last_agent_response"] = response
    state["messages"].append(AIMessage(content=response))
    return state
