from pydantic import BaseModel
from typing import Optional


class TurnRequest(BaseModel):
    """
    Request model for a single turn in the conversation.
    """
    session_id: str
    user_text: str


class TurnResponse(BaseModel):
    """
    Response model for a single turn in the conversation.
    """
    agent_text: str
    call_ended: bool
    disposition: Optional[str] = None


class InitRequest(BaseModel):
    """
    Request model for initializing a new conversation session.
    """
    session_id: str
    lead_name: str
    phone: str
    lead_id: str
    email: Optional[str] = None


class InitResponse(BaseModel):
    """
    Response model for initializing a new conversation session.
    """
    agent_text: str
    session_id: str


class WebSocketMessage(BaseModel):
    """
    Model for messages sent over WebSocket.
    """
    text: str