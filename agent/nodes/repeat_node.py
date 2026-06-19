import logging
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from agent.state import LeadState
from agent.prompts.repeat_prompt import REPEAT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


def repeat_node(state: LeadState) -> LeadState:
    last_response = state.get("last_agent_response", "")
    logger.info(f"[Repeat Node] Repeat requested for lead {state['lead_id']}")
    logger.debug(f"Last response length: {len(last_response)} chars")

    if not last_response:
        logger.warning(f"[Repeat Node] No previous response to repeat for lead {state['lead_id']}")
        response = "Sorry, I don't have anything to repeat. Could you let me know what you'd like to know?"
        state["messages"].append(AIMessage(content=response))
        state["last_agent_response"] = response
        return state

    prompt = REPEAT_SYSTEM_PROMPT.format(last_agent_response=last_response)

    logger.debug(f"[Repeat Node] Invoking LLM to rephrase message")
    response = llm.invoke([{"role": "system", "content": prompt}])
    rephrased = response.content.strip()
    logger.debug(f"[Repeat Node] Message rephrased for lead {state['lead_id']}")

    # Note: last_agent_response is NOT updated here — repeat shouldn't
    # replace the original; next user answer should still map to current_question_key
    state["messages"].append(AIMessage(content=rephrased))
    return state