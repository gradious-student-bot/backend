import logging
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from agent.state import LeadState
from agent.prompts.faq_prompt import build_faq_system_prompt
from knowledge.context_builders import build_faq_context, resolve_courses_for_faq

logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o", temperature=0.4)


def faq_node(state: LeadState) -> LeadState:
    user_query = state["messages"][-1].content
    logger.info(f"[FAQ] Processing query for lead {state['lead_id']}")
    logger.debug(f"Query: {user_query}")

    # Resolve which courses to include in context
    course_keys = resolve_courses_for_faq(
        user_query=user_query,
        course_interest=state.get("course_interest"),
    )
    logger.debug(f"[FAQ] Resolved courses: {course_keys}")

    # Build context
    context = build_faq_context(course_keys)
    logger.debug(f"[FAQ] Context built with {len(course_keys)} course(s)")

    # The pending question is whatever the agent last asked
    pending_question = state.get("last_agent_response") if state.get("current_question_key") else None

    # Build system prompt
    system_prompt = build_faq_system_prompt(context, pending_question)

    # Single LLM call — answer + graceful transition back to pending Q
    logger.debug(f"[FAQ] Invoking LLM for FAQ response")
    response = llm.invoke([
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_query},
    ])

    answer = response.content.strip()
    logger.info(f"[FAQ] FAQ response generated for lead {state['lead_id']}")
    state["last_agent_response"] = answer
    state["messages"].append(AIMessage(content=answer))
    return state