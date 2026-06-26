import logging
from agent.state import LeadState
from agent.utils.scoring import compute_lead_score
from services.airtable_client import write_lead_table, write_lead_metrics
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def airtable_node(state: LeadState) -> LeadState:
    score, classification = compute_lead_score(state)

    state["lead_score"] = score
    state["lead_classification"] = classification

    end_time = datetime.now(timezone.utc)

    if not state.get("call_end_time"):
        state["call_end_time"] = end_time.isoformat()

    if state.get("call_start_time"):
        start_time = datetime.fromisoformat(state["call_start_time"])
        state["call_duration"] = int((end_time - start_time).total_seconds())

    if not state.get("conversation_id"):
        state["conversation_id"] = state.get("session_id") or state.get("lead_id")

    transcript_lines = []
    for msg in state.get("messages", []):
        role = getattr(msg, "type", "message")
        content = getattr(msg, "content", "")
        transcript_lines.append(f"{role}: {content}")

    state["call_transcript"] = "\n".join(transcript_lines)

    state["call_summary"] = (
        f"Lead {state.get('lead_name')} discussed "
        f"{state.get('course_interest')}. "
        f"Disposition: {state.get('disposition')}. "
        f"Interested: {state.get('interested')}. "
        f"Callback requested: {state.get('callback_requested')}."
    )

    lead_data = {
        "lead_id": state.get("lead_id"),
        "name": state.get("lead_name"),
        "phone": state.get("phone"),
        "email": state.get("email"),
        "city": state.get("city"),
        "interested_course": state.get("course_interest"),
        "student_status": state.get("student_status"),
        "callback_requested": state.get("callback_requested"),
        "interested": state.get("interested"),
        "join_date": state.get("join_date"),
        "callback_time": state.get("callback_time"),
        "passout_year": state.get("passout_year")if state.get("passout_year") is not None else None,
        "department": state.get("department")if state.get("department") is not None else None,
        "academic_details": state.get("academic_details"),
        "training_mode": state.get("training_mode"),
        "disposition": state.get("disposition", "unknown"),
        "current_year": state.get("current_year"),
        "class_type": state.get("class_type"),
        "onboarding_requested": state.get("onboarding_requested"),
        "onboarding_email_sent": state.get("onboarding_email_sent"),
        "Join Date (Raw)": state.get("join_date_raw"),
        "Callback Time (Raw)": state.get("callback_time_raw"),
        "Looking for Job": state.get("looking_for_job"),
        "Corrected Name": state.get("corrected_name"),
        "Lead Score": score,
        "Lead Classification": classification,
    }

    metrics_data = {
        "conversation_id": state.get("conversation_id"),
        "lead_id": state.get("lead_id"),
        "call_transcript": state.get("call_transcript"),
        "call_summary": state.get("call_summary"),
        "call_start_time": state.get("call_start_time"),
        "call_end_time": state.get("call_end_time"),
        "call_duration": state.get("call_duration"),
        "Lead Score": score,
        "Lead Classification": classification,
    }

    try:
        logger.info(f"[Airtable] Writing Lead_Table for {state.get('lead_id')}")
        write_lead_table(lead_data)
        logger.info(f"[Airtable] Lead_Table write success for {state.get('lead_id')}")
    except Exception as e:
        logger.error(
            f"[Airtable] Lead_Table write failed for {state.get('lead_id')}: {str(e)}",
            exc_info=True,
        )

    try:
        logger.info(f"[Airtable] Writing Lead_metrics for {state.get('conversation_id')}")
        write_lead_metrics(metrics_data)
        logger.info(
            f"[Airtable] Lead_metrics write success for {state.get('conversation_id')}"
        )
    except Exception as e:
        logger.error(
            f"[Airtable] Lead_metrics write failed for {state.get('conversation_id')}: {str(e)}",
            exc_info=True,
        )

    return state