from pydantic import BaseModel
from typing import Optional


class TurnRequest(BaseModel):
    session_id: str
    user_text: str


class TurnResponse(BaseModel):
    agent_text: str
    call_ended: bool
    disposition: Optional[str] = None


class InitRequest(BaseModel):
    session_id: str
    lead_name: str
    phone: str
    lead_id: str
    email: Optional[str] = None


class InitResponse(BaseModel):
    agent_text: str
    session_id: str


class WebSocketMessage(BaseModel):
    text: str