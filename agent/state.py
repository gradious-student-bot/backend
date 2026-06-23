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
    greeting_step: int  # 0 = not started, 1 = identity confirmed, 2 = timing confirmed, 3 = interest confirmed → proceed to Q flow
    question_retry_counts: dict  # field name → number of times asked (Task 4)

    # Entity switch detection (Task 1)
    pending_switch: Optional[dict]  # {"field": "<field_name>", "new_value": "<new_value>"}

    # Multi-intent handling (Task 2)
    pending_sub_query: Optional[str]
    pending_next_question_text: Optional[str]

    # Collected Lead Data
    course_interest: Optional[str]       # "ai_batch" | "fullstack_batch" | "dsa_batch"
    student_status: Optional[str]        # "student" | "graduated" | "working_professional"
    current_year: Optional[str]          # if student
    passout_year: Optional[str]
    department: Optional[str]
    training_mode: Optional[str]         # "online" | "offline"
    class_type: Optional[str]            # "self_paced" | "live"
    interested: Optional[bool]
    join_date: Optional[str]
    onboarding_requested: Optional[bool]
    onboarding_email_sent: Optional[bool]
    callback_requested: Optional[bool]
    callback_time: Optional[str]         # ISO 8601 datetime string (e.g. "2026-06-25T15:00:00")

    # Meta
    disposition: str   # "interested" | "not_interested" | "callback" | "human_agent_callback" | "end_call"
    call_ended: bool

    # New fields (Batch 3)
    corrected_name: Optional[str]        # Task 6 — name provided when wrong person answers
    looking_for_job: Optional[bool]      # Task 12 — job hunting status for graduated/working professional
    lead_score: int                      # Task 11 — computed lead score 0-100
    lead_classification: str             # Task 11 — "Hot Lead" | "Warm Lead" | "Cold Lead"
    callback_time_raw: Optional[str]     # Task 13 — raw natural language callback time phrase
    join_date_raw: Optional[str]         # Task 13 — raw natural language join date phrase
