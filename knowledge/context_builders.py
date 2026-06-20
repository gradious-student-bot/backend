import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_KB_PATH = Path(__file__).parent / "fasttrack.json"

logger.info(f"Loading knowledge base from {_KB_PATH}")
with open(_KB_PATH, "r", encoding="utf-8") as f:
    kb = json.load(f)
logger.info(f"Knowledge base loaded successfully with {len(kb.get('courses', {}))} courses")


# ─────────────────────────────────────────────
# Base Builders
# ─────────────────────────────────────────────

def build_base_context(kb: dict) -> str:
    logger.info("Building base context")
    company = kb["company"]
    platform = kb["platform"]
    return f"""
ORGANIZATION DETAILS
Name           : {company["name"]}
Location       : {company["office_location"]}
Office Timings : {company["office_timings"]}
Email          : {company["contact_email"]}
Website        : {company["website"]}

LEARNING PLATFORM (Leap LMS)
LMS            : {platform["lms"]}
Mentorship     : {platform["mentors"]}
Doubt Sessions : {platform["doubt_sessions"]}
Daily Standups : {platform["standups"]}
Certificate    : {platform["certificate"]}
""".strip()


def build_placement_context(kb: dict) -> str:
    logger.info("Building placement context")
    p = kb["placements"]
    companies = ", ".join(p["top_hiring_companies"])
    return f"""

PLACEMENT SUPPORT
Placement Assistance : {"Yes" if p["assistance"] else "No"}
Average Package      : {p["avg_package"]}
Placement Rate       : {p["placement_rate"]}
Top Hiring Companies : {companies}
""".strip()


# ─────────────────────────────────────────────
# Course-Specific Builders
# ─────────────────────────────────────────────

def build_course_overview(kb: dict, course: str) -> str:
    logger.info(f"Building course overview for {course}")
    c = kb["courses"][course]
    topics = "\n".join(f"  • {t}" for t in c["course_overview"]["what_you_will_learn"])
    return (
        build_base_context(kb)
        + f"""

COURSE OVERVIEW
Course Name        : {c["name"]}
What You'll Learn  :
{topics}
"""
    )


def build_course_fee(kb: dict, course: str) -> str:
    logger.info(f"Building course fee context for {course}")
    c = kb["courses"][course]
    return (
        build_base_context(kb)
        + f"""

COURSE FEES
Course         : {c["name"]}
Self-Paced Fee : {c["self"]["fee_structure"]}
Live Class Fee : {c["live"]["fee_structure"]}
"""
    )


def build_course_duration(kb: dict, course: str) -> str:
    logger.info(f"Building course duration context for {course}")
    c = kb["courses"][course]
    self_dur = c["self"].get("course_duration", c.get("course_duration", "N/A"))
    live_dur = c["live"].get("course_duration", c.get("course_duration", "N/A"))
    return (
        build_base_context(kb)
        + f"""

COURSE DURATION
Course     : {c["name"]}
Self-Paced : {self_dur}
Live       : {live_dur}
"""
    )


def build_course_platform_access(kb: dict, course: str) -> str:
    logger.info(f"Building platform access context for {course}")
    c = kb["courses"][course]
    if "platform_access" in c:
        self_acc = live_acc = c["platform_access"]
    else:
        self_acc = c["self"]["platform_access"]
        live_acc = c["live"]["platform_access"]
    return (
        build_base_context(kb)
        + f"""

PLATFORM ACCESS
Course     : {c["name"]}
Self-Paced : {self_acc}
Live       : {live_acc}
"""
    )


def build_course_eligibility(kb: dict, course: str) -> str:
    logger.info(f"Building course eligibility context for {course}")
    c = kb["courses"][course]
    return (
        build_base_context(kb)
        + f"""

ELIGIBILITY
Course      : {c["name"]}
Eligibility : {c["eligibility"]}
"""
    )


def build_complete_course_context(kb: dict, course: str) -> str:
    logger.info(f"Building complete course context for {course}")
    c = kb["courses"][course]
    topics = "\n".join(f"  • {t}" for t in c["course_overview"]["what_you_will_learn"])
    self_dur = c["self"].get("course_duration", c.get("course_duration", "N/A"))
    live_dur = c["live"].get("course_duration", c.get("course_duration", "N/A"))
    if "platform_access" in c:
        self_acc = live_acc = c["platform_access"]
    else:
        self_acc = c["self"]["platform_access"]
        live_acc = c["live"]["platform_access"]
    return f"""

COURSE DETAILS — {c["name"]}
Eligibility     : {c["eligibility"]}
Modes Available : {", ".join(c["modes"])}
Topics Covered  :
{topics}

  SELF-PACED
  Duration        : {self_dur}
  Fee             : {c["self"]["fee_structure"]}
  Platform Access : {self_acc}

  LIVE CLASSES
  Duration        : {live_dur}
  Fee             : {c["live"]["fee_structure"]}
  Platform Access : {live_acc}
"""


# ─────────────────────────────────────────────
# FAQ Context Builder (used by faq_node)
# ─────────────────────────────────────────────

def build_faq_context(course_keys: list[str]) -> str:
    """
    Always includes: base context + placement context.
    Variable: complete course block for each key in course_keys.
    If course_keys is empty, includes all courses.
    """
    logger.info(f"Building FAQ context for courses: {course_keys if course_keys else 'ALL'}")
    context = build_base_context(kb)
    context += "\n\n" + build_placement_context(kb)

    keys = course_keys if course_keys else list(kb["courses"].keys())
    logger.info(f"Adding {len(keys)} course(s) to FAQ context")
    for key in keys:
        context += build_complete_course_context(kb, key)

    logger.info(f"FAQ context built, length: {len(context)} chars")
    return context


def detect_course_from_text(text: str) -> list[str]:
    """
    Simple keyword-based course detection from user query text.
    Returns list of matched course keys.
    """
    logger.info(f"Detecting courses from text")
    text_lower = text.lower()
    matched = []
    keywords = {
        "fullstack_batch": ["full stack", "fullstack", "full-stack", "frontend", "backend", "react", "node"],
        "ai_batch": ["ai", "ml", "machine learning", "artificial intelligence", "generative", "deep learning", "nlp"],
        "dsa_batch": ["dsa", "data structures", "algorithms", "competitive", "problem solving"],
    }
    for key, kws in keywords.items():
        if any(kw in text_lower for kw in kws):
            matched.append(key)
    
    if matched:
        logger.info(f"Detected courses: {matched}")
    return matched


def resolve_courses_for_faq(user_query: str, course_interest: str | None) -> list[str]:
    """
    Priority 1: course already selected during questioning phase.
    Priority 2: course detected from the query text.
    Priority 3: no context — return all courses.
    """
    logger.info(f"Resolving courses for FAQ - course_interest: {course_interest}")
    if course_interest:
        logger.info(f"Using course_interest: {course_interest}")
        return [course_interest]
    detected = detect_course_from_text(user_query)
    if detected:
        logger.info(f"Using detected courses: {detected}")
        return detected
    logger.info("No course context - will use all courses")
    return []  # empty = all courses (handled in build_faq_context)