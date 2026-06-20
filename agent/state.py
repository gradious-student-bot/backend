from typing import Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage


class LeadState(TypedDict):
    # Identity
    lead_id: str
    lead_name: str
    phone: str
    email: Optional[str]

    # Conversation
    messages: list[BaseMessage]
    last_agent_response: str
    next_node: str  # set by intent_router to direct graph routing

    # Questionnaire tracking
    current_question_key: str
    answered_fields: dict
    human_agent_requested: bool

    # Collected Lead Data
    course_interest: Optional[str]       # "ai_batch" | "fullstack_batch" | "dsa_batch"
    student_status: Optional[str]        # "student" | "graduated"
    current_year: Optional[str]          # if student
    passout_year: Optional[str]
    department: Optional[str]
    training_mode: Optional[str]         # "online" | "offline"
    class_type: Optional[str]            # "self_paced" | "live"
    budget_range: Optional[str]
    referral_source: Optional[str]
    interested: Optional[bool]
    join_date: Optional[str]
    callback_requested: Optional[bool]
    callback_time: Optional[str]

    # Meta
    disposition: str   # "interested" | "not_interested" | "callback" | "human_agent_callback" | "end_call"
    call_ended: bool