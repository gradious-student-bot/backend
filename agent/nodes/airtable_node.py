import logging
from agent.state import LeadState
from agent.nodes.scoring import compute_lead_score
from services.airtable_client import write_lead

logger = logging.getLogger(__name__)


def airtable_node(state: LeadState) -> LeadState:
    # Task 11: compute lead score and classification before writing
    score, classification = compute_lead_score(state)
    state["lead_score"] = score
    state["lead_classification"] = classification

    lead_data = {
        "lead_id":                state.get("lead_id"),
        "name":                   state.get("lead_name"),
        "phone":                  state.get("phone"),
        "email":                  state.get("email"),
        "interested_course":      state.get("course_interest"),
        "student_status":         state.get("student_status"),
        "current_year":           state.get("current_year"),
        "passout_year":           state.get("passout_year"),
        "department":             state.get("department"),
        "training_mode":          state.get("training_mode"),
        "class_type":             state.get("class_type"),
        "interested":             state.get("interested"),
        "join_date":              state.get("join_date"),          # ISO datetime
        "onboarding_requested": state.get("onboarding_requested"),
        "onboarding_email_sent": state.get("onboarding_email_sent"),
        "Join Date (Raw)":        state.get("join_date_raw"),      # user's phrase
        "callback_requested":     state.get("callback_requested"),
        "callback_time":          state.get("callback_time"),      # ISO datetime
        "Callback Time (Raw)":    state.get("callback_time_raw"),  # user's phrase
        "disposition":            state.get("disposition", "unknown"),
        # Task 11: scoring
        "Lead Score":             score,
        "Lead Classification":    classification,
        "Looking for Job":        state.get("looking_for_job"),
        # Task 6: corrected name
        "Corrected Name":         state.get("corrected_name"),
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
