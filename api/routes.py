import logging
from typing import Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from langchain_core.messages import HumanMessage, AIMessage

from agent.graph import agent_graph
from agent.state import LeadState
from agent.nodes.questionnaire_node import generate_greeting
from datetime import datetime, timezone

from models.schemas import TurnRequest, TurnResponse, InitRequest, InitResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# In-memory session store — replace with Redis for production
_sessions: dict[str, LeadState] = {}


def _init_state(lead_id: str, lead_name: str, phone: str, email: Optional[str], session_id: str, course_interest: Optional[str] = None) -> LeadState:
    """
    Initialize a new LeadState for a given lead.
    """
    greeting = generate_greeting(lead_name)
    logger.info(f"[Routes] Initializing session for lead {lead_id}")
    return LeadState(
        lead_id=lead_id,
        lead_name=lead_name,
        phone=phone,
        email=email,
        conversation_id=session_id,
        call_start_time=datetime.now(timezone.utc).isoformat(),
        call_end_time=None,
        call_duration=None,
        call_transcript=None,
        call_summary=None,
        messages=[AIMessage(content=greeting)],
        last_agent_response=greeting,
        next_node="",
        current_question_key="confirm_identity",
        answered_fields={"course_interest": course_interest} if course_interest else {},
        human_agent_requested=False,
        greeting_step=0,
        question_retry_counts={},
        pending_switch=None,
        pending_sub_query=None,
        pending_next_question_text=None,
        course_interest=course_interest,
        student_status=None,
        current_year=None,
        passout_year=None,
        department=None,
        training_mode=None,
        class_type=None,
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
    """
    API endpoint to initialize a new session for a lead.
    """
    logger.info(f"[Routes] POST /agent/init - Creating session for lead {req.lead_id}")

    # Initialize the state for the new session
    state = _init_state(req.lead_id, req.lead_name, req.phone, req.email, req.session_id, req.course_interest)
    
    print("REQ COURSE =", req.course_interest)
    print("STATE COURSE =", state.get("course_interest"))
    print("ANSWERED FIELDS =", state.get("answered_fields"))

    _sessions[req.session_id] = state
    logger.info(f"[Routes] Session {req.session_id} created successfully")

    # Return the initial greeting and session ID
    return InitResponse(
        agent_text=state["last_agent_response"],
        session_id=req.session_id,
    )


@router.post("/agent/turn", response_model=TurnResponse)
async def agent_turn(req: TurnRequest):
    """
    API endpoint to process a turn for an existing session.
    """
    logger.info(f"[Routes] POST /agent/turn - Processing turn for session {req.session_id}")

    # Retrieve the state for the given session ID
    state = _sessions.get(req.session_id)
    if not state:
        # Session not found, return 404
        logger.warning(f"[Routes] Session {req.session_id} not found")
        raise HTTPException(status_code=404, detail="Session not found. Call /agent/init first.")

    # Check if the call has already ended
    if state.get("call_ended"):
        logger.warning(f"[Routes] Attempted turn on ended call for session {req.session_id}")
        raise HTTPException(status_code=400, detail="Call has already ended.")

    # Process the user input and update the state
    logger.info(f"[Routes] User input for {state['lead_id']}: [{req.user_text[:50]}]")
    state["messages"].append(HumanMessage(content=req.user_text))

    # Invoke the agent graph to process the turn
    result = agent_graph.invoke(state)
    _sessions[req.session_id] = result

    logger.info(f"[Routes] Turn processed for session {req.session_id}, call_ended: {result.get('call_ended')}")

    # Return the agent's response and call status
    agent_text = result["messages"][-1].content
    return TurnResponse(
        agent_text=agent_text,
        call_ended=result.get("call_ended", False),
        disposition=result.get("disposition"),
    )


# ─── WebSocket Endpoint ───────────────────────────────────────────────────────

@router.websocket("/ws/agent/{session_id}")
async def websocket_agent(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for handling real-time communication with the agent.
    """
    logger.info(f"[Routes] WebSocket connection initiated for session {session_id}")

    # Accept the WebSocket connection
    await websocket.accept()
    logger.info(f"[Routes] WebSocket accepted for session {session_id}")

    # Expect first message to be init payload: {"lead_id": ..., "lead_name": ..., "phone": ...}
    try:
        init_data: dict[str, str] = await websocket.receive_json()
        logger.info(f"[Routes] WebSocket init data received for lead {init_data.get('lead_id')}")

        # Initialize the state for the new session
        state = _init_state(
            lead_id=init_data["lead_id"],
            lead_name=init_data["lead_name"],
            phone=init_data["phone"],
            email=init_data.get("email"),
            session_id=session_id,
            course_interest=init_data.get("course_interest")
        )
        _sessions[session_id] = state

        # Send the initial greeting to the client
        logger.info(f"[Routes] Sending initial greeting via WebSocket")
        await websocket.send_json({
            "agent_text": state["last_agent_response"],
            "call_ended": False,
        })

        # Main loop to handle incoming messages from the client
        while True:
            data = await websocket.receive_json()
            user_text = data.get("text", "")

            if not user_text:
                logger.info(f"[Routes] Empty message received")
                continue
            
            # Process the user input and update the state
            state = _sessions.get(session_id)
            logger.info(f"[Routes] WebSocket message received for {state['lead_id']}: {user_text[:50]}...")
            state["messages"].append(HumanMessage(content=user_text))

            # Invoke the agent graph to process the turn
            result = agent_graph.invoke(state)
            _sessions[session_id] = result

            # Log the agent's response and call status
            agent_text = result["messages"][-1].content
            call_ended = result.get("call_ended", False)

            # Send the agent's response back to the client via WebSocket
            logger.info(f"[Routes] Sending WebSocket response, call_ended: {call_ended}")
            await websocket.send_json({
                "agent_text": agent_text,
                "call_ended": call_ended,
                "disposition": result.get("disposition"),
            })

            # If the call has ended, break the loop and close the WebSocket
            if call_ended:
                logger.info(f"[Routes] Call ended for session {session_id}")
                break

    except WebSocketDisconnect:
        logger.info(f"[Routes] WebSocket disconnected for session {session_id}")

    except Exception as e:
        logger.error(f"[Routes] WebSocket error for session {session_id}: {str(e)}", exc_info=True)

    finally:
        # Clean up the session state when the WebSocket disconnects or an error occurs
        _sessions.pop(session_id, None)
        logger.info(f"[Routes] Session {session_id} cleaned up")
