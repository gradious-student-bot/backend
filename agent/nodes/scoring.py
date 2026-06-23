"""
Lead scoring and classification logic.
Called by airtable_node before writing the Airtable record.
"""
from agent.state import LeadState
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


def compute_lead_score(state: LeadState) -> tuple[int, str]:
    """
    Computes a lead score 0-100 and returns (score, classification).

    Scoring rules:
    +20  Actively job hunting
         (student_status == "working_professional" or "graduated") AND looking_for_job == True
    +15  Graduated or working professional
         (student_status in ["graduated", "working_professional"])
    +15  Responded to call
         (call was answered — always True if we reach this node)
    +20  Attended counseling
         (answered_fields has at least 4 fields filled — indicates engagement through the call)
    +20  Interested in joining
         (state["interested"] == True)
    +10  Interested in joining within 30 days
         (state["join_date"] is set AND parseable date is within 30 days from today)

    Classification:
    80-100 → "Hot Lead"
    50-79  → "Warm Lead"
    0-49   → "Cold Lead"
    """
    score = 0

    # +15: Graduated or working professional
    if state.get("student_status") in ("graduated", "working_professional"):
        score += 15
        logger.info("[Score] +15: Graduated/Working Professional")

    # +20: Actively job hunting
    if state.get("looking_for_job") is True and state.get("student_status") in ("graduated", "working_professional"):
        score += 20
        logger.info("[Score] +20: Actively job hunting")

    # +15: Responded to call (always true when we reach scoring)
    score += 15
    logger.info("[Score] +15: Responded to call")

    # +20: Attended counseling — at least 4 answered fields
    answered = {k: v for k, v in (state.get("answered_fields") or {}).items() if v and v != "__skipped__"}
    if len(answered) >= 4:
        score += 20
        logger.info(f"[Score] +20: Attended counseling ({len(answered)} fields answered)")

    # +20: Interested in joining
    if state.get("interested") is True:
        score += 20
        logger.info("[Score] +20: Interested in joining")

    # +10: Interested within 30 days
    join_date_str = state.get("join_date")
    if join_date_str:
        try:
            join_dt = datetime.fromisoformat(join_date_str)
            today = datetime.now(timezone.utc).replace(tzinfo=None)
            if hasattr(join_dt, "tzinfo") and join_dt.tzinfo:
                today = datetime.now(timezone.utc)
            delta = (join_dt - today).days
            if 0 <= delta <= 30:
                score += 10
                logger.info(f"[Score] +10: Join date within 30 days ({delta} days away)")
        except Exception as e:
            logger.warning(f"[Score] Could not parse join_date '{join_date_str}': {e}")

    score = min(score, 100)

    if score >= 80:
        classification = "Hot Lead"
    elif score >= 50:
        classification = "Warm Lead"
    else:
        classification = "Cold Lead"

    logger.info(f"[Score] Final: {score} → {classification}")
    return score, classification
