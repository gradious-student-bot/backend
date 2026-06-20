import json
import logging
from openai import OpenAI
from agent.state import LeadState
from knowledge.context_builders import build_faq_context, resolve_courses_for_faq

logger = logging.getLogger(__name__)
client = OpenAI()

FAQ_SYSTEM_PROMPT = """\
## ROLE
You are a phone-based admissions counselor at Gradious, a tech training institute in Hyderabad.

## OBJECTIVE
Answer the student's question clearly and concisely using ONLY the knowledge base provided.
You are on a phone call — keep answers short and natural, suitable for spoken conversation.

## TONE GUIDELINES
- Warm and positive but not over-the-top. No excessive adjectives.
- Highlight Gradious strengths naturally — LMS Leap platform, mentors, placement support.
- When mentioning the platform, say something brief like:
  "You'll be doing this course on our own LMS platform called Leap"
- Do NOT read out long bullet lists — summarise in 2-3 sentences max.
- At the end, always ask if they'd like to know more OR offer to return to the previous question.

## KNOWLEDGE BASE
{context}

## PENDING QUESTION (if any — the question the agent had just asked before the student's query)
{pending_question}

## RESPONSE RULES
1. Answer the student's question in 2-3 natural spoken sentences.
2. Include a brief mention of Leap LMS and/or placement support where relevant.
3. End with: "Would you like to know more, or shall we get back to [pending question summary]?"
   If there is no pending question, ask: "Is there anything else you'd like to know?"
4. Do NOT include bullet points, headers, or long lists.
5. Return STRICT JSON only.

## OUTPUT FORMAT
{{
  "response": "<your natural spoken response>"
}}"""


def faq_node(state: LeadState) -> LeadState:
    from langchain_core.messages import AIMessage

    user_query = state["messages"][-1].content
    logger.info(f"[FAQ] Lead={state['lead_id']} | Query: {user_query[:80]}")

    course_keys = resolve_courses_for_faq(
        user_query=user_query,
        course_interest=state.get("course_interest"),
    )
    context = build_faq_context(course_keys)

    # Pending question = whatever the agent last said (which was a question)
    pending_question = ""
    if state.get("current_question_key") and state.get("current_question_key") != "confirm_identity":
        pending_question = state.get("last_agent_response", "")

    system_prompt = FAQ_SYSTEM_PROMPT.format(
        context=context,
        pending_question=pending_question if pending_question else "None",
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.4,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
        )
        parsed = json.loads(response.choices[0].message.content)
        answer = parsed.get("response", "").strip()
    except Exception as e:
        logger.error(f"[FAQ] LLM error: {e}")
        answer = "I'm sorry, I had trouble fetching that information. Could you ask again?"

    logger.info(f"[FAQ] Response length: {len(answer)} chars")
    state["last_agent_response"] = answer
    state["messages"].append(AIMessage(content=answer))
    return state