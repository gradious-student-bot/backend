import json
import logging
from openai import OpenAI
from agent.state import LeadState
from config import OPENAI_API_KEY
from services.llm_service import llm

logger = logging.getLogger(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

INTENT_SYSTEM_PROMPT = """## PERSONA
You are Bindhu, a calm and friendly admissions counselor at Gradious. You speak like a real person on a phone call — warm, clear, and genuinely helpful.

## ROLE
You are an intelligent intent classifier for a phone-based admissions counseling bot at Gradious, a tech training institute in Hyderabad offering courses in Full Stack, AI/ML, and DSA.

## OBJECTIVE
Identify the intent of the student's latest message based on the full conversation context and the intent descriptions below. Return a strict JSON response.

## INTENT NAMES AND DESCRIPTIONS
1. **answer**:
    - Return "answer" when user is responding/answering to a questionnaire question asked by assistant.
    - Assistant will ask questions from a questionnaire for user details and also follow-up questions in response of a user query.
    - A questionnaire question is when assistant asks for user details like course interest/academic details etc.
    - A follow-up question is when assistant asks user if they'd like to know more about a course/information.
    - Correctly differentiate which type of question assistant asked and return "answer" if user is answering to a questionnaire question.
    - Only return "answer" when user answers a question from questionnaire.

2. **query**:
    - Return "query" when user only asks a question/query about Gradious offerings, course details, coaching details.
    - Such queries includes asking details about factual information of institute, course, etc.
    - When user responds to a question assistant asked and also asks a query at a time, the intent is "answer_and_query".
    - When assistant responds to a user query in previous turn and adds a follow-up question like if user wants to know more, then when user answers this question, set intent as "query".
    Examples for "query" intent user utterances:
        - "What's the fee and duration of this?"
        - "Tell me about this course?"
        - "Where is the office and it's timings?"

3. **answer_and_query**:
    - Return "answer_and_query" when the user's utterance contains BOTH:
        a. A clear answer to the question the agent just asked, AND
        b. A separate question or request for information about Gradious.
    - When returning this intent, you MUST also populate the "sub_query" field with the isolated question/query portion of the user's message.
   Examples:
       - "Yes, I prefer online and also, what's fee for this?"
       - "I'm 3rd year CSE student. Tell me more about AI course."
       - "Full Stack, also how long is the course?"

4. **repeat**:
   - Return "repeat" if the user is asking to hear the previous message again.
   Examples: "Say that again", "Can you repeat?", "Didn't catch that!", "What did you say?", "Huh?"

5. **human_agent**
   - Return "human_agent" if the user prefers to speak with a human representative.
   - Such intent is when user isn't interested in talking to AI agent[You] and would like to speak with Human representative.
   Examples: "Talk to a real person", "Connect me to HR", "I want to speak to someone", "Can I talk to your team?", "Get me your advisor".

6. **not_interested**
   - Return "not_interested" if user doesn't want to continue the call or enroll into the course.
   - Such intent is when user is clearly disinterested in continuing the call/conversation, and would like to cut the call.
   Examples: "I'm not interested", "Don't call me again", "Remove my number", "I don't want this", "Please don't contact me".

7. **rude**
   - Return "rude" if the user uses hostile, abusive, or inappropriate language toward the assistant.
   - Such intent is when user is clearly angry and shows hostile behaviour towards assistant.

8. **irrelevant**
   - Return "irrelevant" if the student's message is completely unrelated to Gradious, courses, education, or career — such as asking about weather, politics, sports, or other random topics.

9. **confused**
   - Return "confused" if the student is genuinely unsure or unclear about how to respond.
   Examples: "I don't know", "Not sure", "Maybe?", "I'm confused", "Can you explain?".

10. **end_call**
    - Return "end_call" if the student is politely wrapping up the conversation.
    Examples: "Bye", "Thanks, goodbye", "I'll call back", "Talk later", "That's all"

11. **small_talk**
    - Return "small_talk" when user's utterance is generic, conversational, greeting, identification-related, purpose knowing, confused about the call.
    - Such intent is when user is:
        - Asking generic questions.
        - Greeting the assistant.
        - Having a conversation to assistant related to courses/institute.
        - Asking assistant about it's identity or institute.
        - Asking assistant about the purpose of the call.
        - Confusion about call or assistant response like Huh?, Sorry?, What?, I can't hear you.
        
## CRITICAL RULES
- Correctly classify the intent based on conversation history, previous turns.

## OUTPUT FORMAT
Return STRICT JSON only. No explanation, no markdown, no extra text.
{
  "intent": "answer | query | answer_and_query | repeat | human_agent | not_interested | rude | irrelevant | confused | end_call | small_talk",
  "reasoning": "<one short sentence explaining why>",
  "sub_query": "<the isolated question portion of the message if intent is answer_and_query, otherwise null>"
}"""

def intent_router_node(state: LeadState) -> LeadState:
    user_message = state["messages"][-1].content
    logger.info(f"[IntentRouter] Lead={state['lead_id']} | Input: {user_message[:80]}")

    last_agent_msg = state.get("last_agent_response", "")

    intent_prompt = INTENT_SYSTEM_PROMPT
    
    try:

        parsed = llm.invoke_json(
            messages= (
                [{"role": "system", "content": intent_prompt}]
                + llm.get_recent_messages(state)
                + [{"role": "user", "content": user_message}]
            )
        )
        
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
        "not_interested", "rude", "irrelevant", "confused", "end_call", "small_talk",
    }

    if intent not in valid_intents:
        logger.warning(f"[IntentRouter] Unknown intent '{intent}' → defaulting to 'answer'")
        intent = "answer"
        sub_query = None

    state["next_node"] = intent

    # Store sub_query in state for answer_and_query handling
    if intent == "answer_and_query" and sub_query:
        state["pending_sub_query"] = sub_query
    else:
        state["pending_sub_query"] = None

    return state
