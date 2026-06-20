import logging
from agent.state import LeadState
from services.airtable_client import write_lead

logger = logging.getLogger(__name__)


def airtable_node(state: LeadState) -> LeadState:
    lead_data = {
        "lead_id":             state.get("lead_id"),
        "name":                state.get("lead_name"),
        "phone":               state.get("phone"),
        "email":               state.get("email"),
        "interested_course":     state.get("course_interest"),
        "student_status":      state.get("student_status"),
        "current_year":        state.get("current_year"),
        "passout_year":        state.get("passout_year"),
        "department":          state.get("department"),
        "training_mode":       state.get("training_mode"),
        "class_type":          state.get("class_type"),
        "budget_range":        state.get("budget_range"),
        "referral_source":     state.get("referral_source"),
        "interested":          state.get("interested"),
        "join_date":           state.get("join_date"),
        "callback_requested":  state.get("callback_requested"),
        "callback_time":       state.get("callback_time"),
        "disposition":         state.get("disposition", "unknown"),
    }

    logger.info(f"[Airtable] Writing lead record for {state.get('lead_id')} with disposition {state.get('disposition')}")
    logger.info(f"Lead data: {lead_data}")

    try:
        write_lead(lead_data)
        logger.info(f"[Airtable] Successfully wrote lead record for {state.get('lead_id')}")
    except Exception as e:
        # Log but don't crash — call data is more important than Airtable write
        logger.error(f"[Airtable] Write failed for lead {state.get('lead_id')}: {str(e)}", exc_info=True)

    return state