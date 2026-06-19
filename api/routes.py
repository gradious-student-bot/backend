import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from langchain_core.messages import HumanMessage, AIMessage
from agent.graph import agent_graph
from agent.state import LeadState
from models.schemas import TurnRequest, TurnResponse, InitRequest, InitResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory session store — replace with Redis for production
_sessions: dict[str, LeadState] = {}

GREETING_TEMPLATE = (
    "Hi, am I speaking with {name}? This is Ava from Gradious. "
    "We received your registration for our training programs. "
    "Are you currently a student or have you already graduated?"
)


def _init_state(lead_id: str, lead_name: str, phone: str, email: Optional[str]) -> LeadState:
    greeting = GREETING_TEMPLATE.format(name=lead_name)
    logger.info(f"[Routes] Initializing session for lead {lead_id}")
    return LeadState(
        lead_id=lead_id,
        lead_name=lead_name,
        phone=phone,
        email=email,
        messages=[AIMessage(content=greeting)],
        last_agent_response=greeting,
        next_node="",
        current_question_key="student_status",
        answered_fields={},
        human_agent_requested=False,
        course_interest=None,
        student_status=None,
        current_year=None,
        passout_year=None,
        department=None,
        training_mode=None,
        class_type=None,
        budget_range=None,
        referral_source=None,
        interested=None,
        join_date=None,
        callback_requested=None,
        callback_time=None,
        disposition="in_progress",
        call_ended=False,
    )


# ─── REST Endpoint ───────────────────────────────────────────────────────────

@router.post("/agent/init", response_model=InitResponse)
async def init_session(req: InitRequest):
    logger.info(f"[Routes] POST /agent/init - Creating session for lead {req.lead_id}")
    state = _init_state(req.lead_id, req.lead_name, req.phone, req.email)
    _sessions[req.session_id] = state
    logger.debug(f"[Routes] Session {req.session_id} created successfully")
    return InitResponse(
        agent_text=state["last_agent_response"],
        session_id=req.session_id,
    )


@router.post("/agent/turn", response_model=TurnResponse)
async def agent_turn(req: TurnRequest):
    logger.debug(f"[Routes] POST /agent/turn - Processing turn for session {req.session_id}")
    state = _sessions.get(req.session_id)
    if not state:
        logger.warning(f"[Routes] Session {req.session_id} not found")
        raise HTTPException(status_code=404, detail="Session not found. Call /agent/init first.")

    if state.get("call_ended"):
        logger.warning(f"[Routes] Attempted turn on ended call for session {req.session_id}")
        raise HTTPException(status_code=400, detail="Call has already ended.")

    logger.debug(f"[Routes] User input for {state['lead_id']}: {req.user_text[:50]}...")
    state["messages"].append(HumanMessage(content=req.user_text))

    result = agent_graph.invoke(state)
    _sessions[req.session_id] = result
    
    logger.debug(f"[Routes] Turn processed for session {req.session_id}, call_ended: {result.get('call_ended')}")

    agent_text = result["messages"][-1].content
    return TurnResponse(
        agent_text=agent_text,
        call_ended=result.get("call_ended", False),
        disposition=result.get("disposition"),
    )


# ─── WebSocket Endpoint ───────────────────────────────────────────────────────

@router.websocket("/ws/agent/{session_id}")
async def websocket_agent(websocket: WebSocket, session_id: str):
    logger.info(f"[Routes] WebSocket connection initiated for session {session_id}")
    await websocket.accept()
    logger.info(f"[Routes] WebSocket accepted for session {session_id}")

    # Expect first message to be init payload: {"lead_id": ..., "lead_name": ..., "phone": ...}
    try:
        init_data: dict[str, str] = await websocket.receive_json()
        logger.info(f"[Routes] WebSocket init data received for lead {init_data.get('lead_id')}")
        
        state = _init_state(
            lead_id=init_data["lead_id"],
            lead_name=init_data["lead_name"],
            phone=init_data["phone"],
            email=init_data.get("email"),
        )
        _sessions[session_id] = state
        
        logger.debug(f"[Routes] Sending initial greeting via WebSocket")
        await websocket.send_json({
            "agent_text": state["last_agent_response"],
            "call_ended": False,
        })

        while True:
            data = await websocket.receive_json()
            user_text = data.get("text", "")

            if not user_text:
                logger.debug(f"[Routes] Empty message received")
                continue

            logger.debug(f"[Routes] WebSocket message received for {state['lead_id']}: {user_text[:50]}...")
            state["messages"].append(HumanMessage(content=user_text))
            result = agent_graph.invoke(state)
            _sessions[session_id] = result

            agent_text = result["messages"][-1].content
            call_ended = result.get("call_ended", False)
            
            logger.debug(f"[Routes] Sending WebSocket response, call_ended: {call_ended}")

            await websocket.send_json({
                "agent_text": agent_text,
                "call_ended": call_ended,
                "disposition": result.get("disposition"),
            })

            if call_ended:
                logger.info(f"[Routes] Call ended for session {session_id}")
                break

    except WebSocketDisconnect:
        logger.info(f"[Routes] WebSocket disconnected for session {session_id}")
    except Exception as e:
        logger.error(f"[Routes] WebSocket error for session {session_id}: {str(e)}", exc_info=True)
    finally:
        _sessions.pop(session_id, None)
        logger.info(f"[Routes] Session {session_id} cleaned up")