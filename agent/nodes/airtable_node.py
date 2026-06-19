import logging
from agent.state import LeadState
from services.airtable_client import write_lead

logger = logging.getLogger(__name__)


def airtable_node(state: LeadState) -> LeadState:
    lead_data = {
        "Lead ID":             state.get("lead_id"),
        "Name":                state.get("lead_name"),
        "Phone":               state.get("phone"),
        "Course Interest":     state.get("course_interest"),
        "Student Status":      state.get("student_status"),
        "Current Year":        state.get("current_year"),
        "Passout Year":        state.get("passout_year"),
        "Department":          state.get("department"),
        "Training Mode":       state.get("training_mode"),
        "Class Type":          state.get("class_type"),
        "Budget Range":        state.get("budget_range"),
        "Referral Source":     state.get("referral_source"),
        "Interested":          state.get("interested"),
        "Join Date":           state.get("join_date"),
        "Callback Requested":  state.get("callback_requested"),
        "Callback Time":       state.get("callback_time"),
        "Disposition":         state.get("disposition", "unknown"),
    }

    logger.info(f"[Airtable] Writing lead record for {state.get('lead_id')} with disposition {state.get('disposition')}")
    logger.debug(f"Lead data: {lead_data}")

    try:
        write_lead(lead_data)
        logger.info(f"[Airtable] Successfully wrote lead record for {state.get('lead_id')}")
    except Exception as e:
        # Log but don't crash — call data is more important than Airtable write
        logger.error(f"[Airtable] Write failed for lead {state.get('lead_id')}: {str(e)}", exc_info=True)

    return state