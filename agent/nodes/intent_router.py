import logging
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from agent.state import LeadState

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

INTENT_SYSTEM_PROMPT = """
You are an intent classifier for a phone-based counseling bot at Gradious, a tech training institute.

Classify the student's message into exactly one of these intents:

- answer        : Student is answering a question asked by the agent (e.g. "I'm in 3rd year", "CSE", "Yes", "Online", "Full Stack Development", "Next month", "No, thanks")
- query         : Student is asking about courses, fees, duration, placements, platform, office, or anything about Gradious
- repeat        : Student wants the previous message repeated (e.g. "Say that again", "Didn't catch that", "Can you repeat?", "What?")
- human_agent   : Student wants to speak to a human (e.g. "Talk to a real person", "Connect me to HR", "I want to speak to someone")
- not_interested: Student clearly does not want to continue (e.g. "I'm not interested", "Don't call me", "Remove my number")
- rude          : Student is hostile, abusive, or using inappropriate language
- irrelevant    : Student's message is completely unrelated to courses or Gradious (e.g. "What's the weather?", "Who is the PM?")
- confused      : Student is unsure or unclear (e.g. "I don't know", "Not sure", "Maybe?", "I'm confused")
- end_call      : Student is ending the conversation politely (e.g. "Bye", "Thanks, goodbye", "Talk later")

Reply with ONLY the intent word. No explanation.
"""

def intent_router_node(state: LeadState) -> LeadState:
    user_message = state["messages"][-1].content
    logger.info(f"[Intent Router] Processing message for lead {state['lead_id']}")
    logger.debug(f"User message: {user_message}")

    response = llm.invoke([
        {"role": "system", "content": INTENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ])
    intent = response.content.strip().lower()
    logger.debug(f"Initial intent classification: {intent}")

    # Fallback safety
    valid_intents = {
        "answer", "query", "repeat", "human_agent",
        "not_interested", "rude", "irrelevant", "confused", "end_call"
    }
    if intent not in valid_intents:
        logger.warning(f"Invalid intent '{intent}' received, defaulting to 'answer'")
        intent = "answer"

    logger.info(f"[Intent Router] Intent determined: {intent} for lead {state['lead_id']}")
    state["next_node"] = intent
    return state