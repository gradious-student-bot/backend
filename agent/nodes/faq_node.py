import json
import logging
from openai import OpenAI
from agent.state import LeadState
from knowledge.context_builders import build_faq_context, resolve_courses_for_faq
from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)
client = OpenAI(api_key=OPENAI_API_KEY)

FAQ_SYSTEM_PROMPT = """\
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
Answer the student's question clearly and concisely using ONLY the knowledge base provided.
You are on a phone call — keep answers short and natural, suitable for spoken conversation.

## TONE GUIDELINES
- Warm and positive but not over-the-top. No excessive adjectives.
- Do NOT read out long bullet lists — summarise in 2-3 sentences max.
- At the end, always ask if they'd like to know more OR offer to return to the previous question.

## LMS MENTION RULES
- Do NOT mention the Leap platform in every response.
- Only mention Leap when:
  a. The student specifically asks about the platform, portal, or learning experience.
  b. The student asks for more details about a course (after the initial brief answer).
- When mentioning Leap, NEVER use the term "LMS". Say "our learning portal called Leap" instead.
- When Leap is relevant, mention 1-2 specific differentiators, e.g.:
  "You'll learn by doing — Leap has hands-on coding exercises built into every lesson, not just video lectures."
- Do not list all platform features — pick the most compelling 1-2 for the context.

## SCOPE RULES
You ONLY answer questions about:
- Gradious courses (Full Stack, AI/ML, DSA): content, structure, syllabus, duration, fees, modes
- Gradious platform (Leap): how it works, features, access
- Gradious placements: companies, process, rates, packages
- Gradious company info: location, timings, contact
- Enrollment process and next steps

If the student asks a general knowledge question unrelated to Gradious
(e.g. "What is Machine Learning?", "Explain React", "What is DSA?"),
DO NOT answer the general definition. Instead, redirect the student toward the relevant
Gradious course naturally.

Examples of redirection:
- "What is ML?" → "We cover Machine Learning in depth in our AI Stack course — from Python basics all the way to Generative AI. Would you like to know more about that course?"
- "What is React?" → "React is one of the core topics in our Full Stack course. Want me to walk you through what the Full Stack program covers?"
- "What is DSA?" → "DSA is a focused program we offer — great for interview preparation. Want me to tell you more about it?"

Always tie the answer back to a Gradious offering.

## FEE AND DISCOUNT RULES
- Answer fee questions factually from the knowledge base.
- If the student tries to negotiate fees or asks for a discount, do NOT negotiate.
  Instead, respond with: "For fee-related discussions and any special options, our admissions
  expert can help you out. Would you like me to schedule a call with them?"
  Set your intent in the response to indicate human_agent handoff is needed.

## KNOWLEDGE BASE
{context}

## PENDING QUESTION (if any — the question the agent had just asked before the student's query)
{pending_question}

## RESPONSE RULES
1. Answer the student's question in 2-3 natural spoken sentences.
2. Only mention Leap platform when the student asks about it specifically or requests details.
3. End with: "Would you like to know more, or shall we get back to [pending question summary]?"
   If there is no pending question, ask: "Is there anything else you'd like to know?"
4. Do NOT include bullet points, headers, or long lists.
5. Return STRICT JSON only.

## OUTPUT FORMAT
{{
  "response": "<your natural spoken response>",
  "needs_human_agent": false
}}"""

# Keywords that indicate the user is asking specifically about the platform/learning experience
_LMS_KEYWORDS = {
    "platform", "portal", "leap", "lms", "learning experience",
    "how do i learn", "how does it work", "how will i learn",
}


def faq_node(state: LeadState) -> LeadState:
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.messages import AIMessage as LCAIMessage

    # Task 2: use pending_sub_query if set (answer_and_query flow), else last message
    if state.get("pending_sub_query"):
        user_query = state["pending_sub_query"]
        logger.info(f"[FAQ] Using pending_sub_query: {user_query[:80]}")
    else:
        user_query = state["messages"][-1].content
    logger.info(f"[FAQ] Lead={state['lead_id']} | Query: {user_query[:80]}")

    course_keys = resolve_courses_for_faq(
        user_query=user_query,
        course_interest=state.get("course_interest"),
    )

    # Task 9: detect if user is asking specifically about the platform
    query_lower = user_query.lower()
    include_lms_detail = any(kw in query_lower for kw in _LMS_KEYWORDS)
    if include_lms_detail:
        logger.info("[FAQ] LMS-specific query detected — including full LMS detail context")
    context = build_faq_context(course_keys, include_lms_detail=True)

    # Task 2: use pending_next_question_text if set, otherwise fall back to last_agent_response
    if state.get("pending_next_question_text"):
        pending_question = state["pending_next_question_text"]
        logger.info(f"[FAQ] Using pending_next_question_text as pending question")
    elif state.get("current_question_key") and state.get("current_question_key") != "confirm_identity":
        pending_question = state.get("last_agent_response", "")
    else:
        pending_question = ""

    system_prompt = FAQ_SYSTEM_PROMPT.format(
        context=context,
        pending_question=pending_question if pending_question else "None",
    )

    # Task 6: build recent message history for context
    msgs = state.get("messages", [])
    history = msgs[:-1] if len(msgs) > 1 else []
    recent = []
    for m in history[-10:]:
        if isinstance(m, HumanMessage):
            recent.append({"role": "user", "content": m.content})
        elif isinstance(m, LCAIMessage):
            recent.append({"role": "assistant", "content": m.content})

    messages_payload = (
        [{"role": "system", "content": system_prompt}]
        + recent
        + [{"role": "user", "content": user_query}]
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0.4,
            response_format={"type": "json_object"},
            messages=messages_payload,
        )
        parsed = json.loads(response.choices[0].message.content)
        answer = parsed.get("response", "").strip()

        # Task 7: if LLM signals human agent needed (fee negotiation), set flag
        if parsed.get("needs_human_agent"):
            logger.info("[FAQ] Human agent handoff triggered (fee negotiation or special request)")
            state["human_agent_requested"] = True
            state["next_node"] = "human_agent"

    except Exception as e:
        logger.error(f"[FAQ] LLM error: {e}")
        answer = "I'm sorry, I had trouble fetching that information. Could you ask again?"

    logger.info(f"[FAQ] Response length: {len(answer)} chars")
    state["last_agent_response"] = answer
    state["messages"].append(AIMessage(content=answer))

    # Task 2: clear pending_sub_query and pending_next_question_text after use
    state["pending_sub_query"] = None
    state["pending_next_question_text"] = None

    return state
