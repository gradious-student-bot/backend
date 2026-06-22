import json
import logging
from openai import OpenAI
from agent.state import LeadState
from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

REPEAT_SYSTEM_PROMPT = """\
## ROLE
You are a phone-based admissions counselor at Gradious on a call with a student.

## OBJECTIVE
The student asked you to repeat or said they didn't understand your previous response.
Rephrase it naturally and conversationally — do not read it back word for word.

## RULES
1. Start with a brief natural acknowledgement: "Sure!", "Of course!", "No problem!"
2. Rephrase the content more clearly — expand slightly if it helps clarity.
3. Do NOT add any new information not present in the previous response.
4. Keep it short and suitable for spoken conversation.
5. Return STRICT JSON only.

## PREVIOUS RESPONSE TO REPHRASE
{last_agent_response}

## OUTPUT FORMAT
{{
  "response": "<your rephrased response>"
}}"""


def repeat_node(state: LeadState) -> LeadState:
    from langchain_core.messages import AIMessage

    last_response = state.get("last_agent_response", "")
    logger.info(f"[Repeat] Lead={state['lead_id']} | Rephrasing last response")

    if not last_response:
        response = "Sorry, I don't have anything to repeat. Could you let me know what you'd like to know?"
        state["messages"].append(AIMessage(content=response))
        return state

    try:
        result = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.3,
            response_format={"type": "json_object"},
            messages=[{
                "role": "system",
                "content": REPEAT_SYSTEM_PROMPT.format(last_agent_response=last_response),
            }],
        )
        parsed = json.loads(result.choices[0].message.content)
        response = parsed.get("response", last_response)
    except Exception as e:
        logger.error(f"[Repeat] LLM error: {e}")
        response = f"Sure! {last_response}"

    # Do NOT update last_agent_response — repeat doesn't replace the original
    state["messages"].append(AIMessage(content=response))
    return state