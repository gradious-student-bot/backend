import json
import logging
from openai import OpenAI
from agent.state import LeadState

logger = logging.getLogger(__name__)
client = OpenAI()

INTENT_SYSTEM_PROMPT = """\
## ROLE
You are an intelligent intent classifier for a phone-based admissions counseling bot at Gradious,
a tech training institute in Hyderabad offering courses in Full Stack, AI/ML, and DSA.

## OBJECTIVE
Identify the intent of the student's latest message based on the full conversation context
and the intent descriptions below. Return a strict JSON response.

## INTENT NAMES AND DESCRIPTIONS

1. **answer**
   Return "answer" if the student is responding to a question the agent just asked.
   This includes direct answers like "CSE", "3rd year", "Online", "Yes", "No", "Next month",
   "Through Instagram", "Full Stack", and also compound answers like
   "I'm a 3rd year CSE student" or "I prefer online, self-paced".
   IMPORTANT: If the agent just asked a question and the student's reply contains information
   that answers it — even partially — classify as "answer", not "query".

2. **query**
   Return "query" ONLY if the student is asking a genuine question about Gradious offerings,
   courses, fees, duration, placements, platform, office location, timings, or any
   factual information about the institute — WITHOUT also providing an answer to the agent's
   pending question.
   Examples: "What is the fee?", "Tell me about the Full Stack course",
   "How long is the AI course?", "Where is your office?", "Do you have placement support?"
   Do NOT classify as "query" if the student is answering a question the agent just asked.

3. **answer_and_query**
   Return "answer_and_query" when the student's message contains BOTH:
     a. A clear answer to the question the agent just asked, AND
     b. A separate question or request for information about Gradious.
   Examples:
   - "Yes, I prefer online — also, what is the fee for Full Stack?"
   - "I'm a 3rd year CSE student. Can you tell me more about the AI course?"
   - "Full Stack, also how long is the course?"
   When returning this intent, you MUST also populate the "sub_query" field with the
   isolated question portion of the student's message.

4. **repeat**
   Return "repeat" if the student is asking to hear the previous message again.
   Examples: "Say that again", "Can you repeat?", "Didn't catch that", "What did you say?", "Huh?"

5. **human_agent**
   Return "human_agent" if the student explicitly wants to speak to a human representative.
   Examples: "Talk to a real person", "Connect me to HR", "I want to speak to someone",
   "Can I talk to your team?", "Get me your advisor"

6. **not_interested**
   Return "not_interested" if the student clearly does not want to continue or enroll.
   Examples: "I'm not interested", "Don't call me again", "Remove my number",
   "I don't want this", "Please don't contact me"

7. **rude**
   Return "rude" if the student uses hostile, abusive, or inappropriate language toward the agent.

8. **irrelevant**
   Return "irrelevant" if the student's message is completely unrelated to Gradious, courses,
   education, or career — such as asking about weather, politics, sports, or other random topics.

9. **confused**
   Return "confused" if the student is genuinely unsure or unclear about how to respond.
   Examples: "I don't know", "Not sure", "Maybe?", "I'm confused", "Can you explain?"

10. **end_call**
    Return "end_call" if the student is politely wrapping up the conversation.
    Examples: "Bye", "Thanks, goodbye", "I'll call back", "Talk later", "That's all"

## CRITICAL DISAMBIGUATION RULES

1. If the agent's last message was a question AND the student's reply contains an answer
   to that question AND a separate query — classify as **answer_and_query**.
2. If the agent's last message was a question AND the student's reply contains only an answer
   to that question — classify as **answer**, even if the reply also contains other information.
3. Only classify as **query** if the student is genuinely asking for information unprompted
   by the agent's last question (no answer present).
4. Short replies after agent questions ("Yes", "No", "CSE", "Online") are always **answer**.
5. If unsure between "answer" and "query", prefer **answer**.

## CONVERSATION CONTEXT
The agent's last message (the question that was just asked) is provided below.
Use it to correctly determine if the student is answering that question or asking something new.

Agent's last message: {last_agent_message}

## OUTPUT FORMAT
Return STRICT JSON only. No explanation, no markdown, no extra text.
{{
  "intent": "answer | query | answer_and_query | repeat | human_agent | not_interested | rude | irrelevant | confused | end_call",
  "reasoning": "<one short sentence explaining why>",
  "sub_query": "<the isolated question portion of the message if intent is answer_and_query, otherwise null>"
}}"""


def intent_router_node(state: LeadState) -> LeadState:
    user_message = state["messages"][-1].content
    logger.info(f"[IntentRouter] Lead={state['lead_id']} | Input: {user_message[:80]}")

    # Identity confirmation phase — always treat as answer
    if state.get("current_question_key") == "confirm_identity":
        logger.info("[IntentRouter] confirm_identity phase → forcing 'answer'")
        state["next_node"] = "answer"
        return state

    last_agent_msg = state.get("last_agent_response", "")

    # Task 6: build recent message history for context
    from langchain_core.messages import HumanMessage, AIMessage as LCAIMessage
    recent = []
    msgs = state.get("messages", [])
    # Exclude last message (current user input)
    history = msgs[:-1] if len(msgs) > 1 else []
    for m in history[-6:]:
        if isinstance(m, HumanMessage):
            recent.append({"role": "user", "content": m.content})
        elif isinstance(m, LCAIMessage):
            recent.append({"role": "assistant", "content": m.content})

    try:
        messages_payload = (
            [{"role": "system", "content": INTENT_SYSTEM_PROMPT.format(last_agent_message=last_agent_msg)}]
            + recent
            + [{"role": "user", "content": user_message}]
        )
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=messages_payload,
        )
        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        intent = parsed.get("intent", "answer").strip().lower()
        reasoning = parsed.get("reasoning", "")
        sub_query = parsed.get("sub_query", None)
        logger.info(f"[IntentRouter] Intent={intent} | Reason: {reasoning}")
        if sub_query:
            logger.info(f"[IntentRouter] sub_query detected: {sub_query[:80]}")
    except Exception as e:
        logger.error(f"[IntentRouter] LLM error: {e}. Defaulting to 'answer'")
        intent = "answer"
        sub_query = None

    valid_intents = {
        "answer", "query", "answer_and_query", "repeat", "human_agent",
        "not_interested", "rude", "irrelevant", "confused", "end_call",
    }
    if intent not in valid_intents:
        logger.warning(f"[IntentRouter] Unknown intent '{intent}' → defaulting to 'answer'")
        intent = "answer"
        sub_query = None

    state["next_node"] = intent

    # Task 2: store sub_query in state for answer_and_query handling
    if intent == "answer_and_query" and sub_query:
        state["pending_sub_query"] = sub_query
    else:
        state["pending_sub_query"] = None

    return state
