import json
import logging
from pathlib import Path
from openai import OpenAI
from config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

_KB_PATH = Path(__file__).parent / "fasttrack.json"

logger.info(f"Loading knowledge base from {_KB_PATH}")
with open(_KB_PATH, "r", encoding="utf-8") as f:
    kb = json.load(f)
logger.info(f"Knowledge base loaded with {len(kb.get('courses', {}))} courses")

_llm_client = OpenAI(api_key=OPENAI_API_KEY)

# ─────────────────────────────────────────────
# LLM-based Course Detection
# ─────────────────────────────────────────────

_COURSE_DETECT_PROMPT = """\
## ROLE
You are a course classifier for Gradious, a tech training institute.

## OBJECTIVE
Given a user query and optionally their year of study, identify which Gradious course(s) they are asking about or are eligible for.

## AVAILABLE COURSES
- fullstack_batch : Full Stack + Gen AI — for final year students and passed outs
- ai_batch        : AI Stack (ML + Generative AI) — for final year students and passed outs
- dsa_batch       : DSA (Data Structures and Algorithms) — for final year students and passed outs
- campus_fullstack: Campus Full Stack + Gen AI — ONLY for 1st, 2nd, or 3rd year students
- campus_ai       : Campus ML + Gen AI — ONLY for 1st, 2nd, or 3rd year students

## ELIGIBILITY RULES
- If student_year is 1, 2, or 3 → only campus_fullstack and campus_ai are eligible
- If student_year is 4, final year, or passed out → fullstack_batch, ai_batch, dsa_batch are eligible
- If student_year is unknown → do not filter by eligibility, detect from query only

## INSTRUCTIONS
1. Identify which course(s) the query is about based on keywords and context.
2. Apply eligibility filtering if student_year is provided.
3. If no specific course is mentioned, return an empty list (all courses will be shown).
4. Return STRICT JSON only.

## OUTPUT FORMAT
{
  "course_keys": ["fullstack_batch"] // list of matched course keys, or [] if none detected
}"""

_LMS_DETECT_PROMPT = """\
## ROLE
You are an intent classifier for Gradious, a tech training institute.

## OBJECTIVE
Determine if the user's query is specifically asking about the learning platform/portal/experience.

## INSTRUCTIONS
Return true if the user is asking about ANY of:
- The learning platform or portal (Leap)
- How the course is delivered or how learning works
- Videos, recordings, exercises, projects in the course
- Mentors, doubt sessions, standups, daily classes
- Platform access duration
- Learning experience details

Return false if the query is about course content/syllabus, fees, placements, or general course info.

## OUTPUT FORMAT
{
  "is_lms_query": true | false
}"""


def llm_detect_courses(user_query: str, student_year: int | None = None) -> list[str]:
    """
    Use LLM to detect which course keys are relevant to the user query.
    Falls back to keyword detection on LLM error.
    Returns list of course keys, or [] meaning all courses.
    """
    logger.info(f"[LLM] Detecting courses for query: {user_query[:80]!r} | year={student_year}")
    user_content = f"User query: {user_query}"
    if student_year is not None:
        user_content += f"\nStudent year: {student_year}"

    try:
        resp = _llm_client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _COURSE_DETECT_PROMPT},
                {"role": "user",   "content": user_content},
            ],
        )
        result = json.loads(resp.choices[0].message.content)
        keys = result.get("course_keys", [])
        # Validate keys exist in KB
        valid = [k for k in keys if k in kb["courses"]]
        logger.info(f"[LLM] Course detection result: {valid}")
        return valid
    except Exception as e:
        logger.warning(f"[LLM] Course detection failed ({e}), falling back to keyword detection")
        return _keyword_detect_courses(user_query)


def llm_detect_lms_query(user_query: str) -> bool:
    """
    Use LLM to detect if user is asking specifically about the learning platform/experience.
    Falls back to keyword matching on error.
    """
    logger.info(f"[LLM] Detecting LMS query intent for: {user_query[:80]!r}")
    try:
        resp = _llm_client.chat.completions.create(
            model="gpt-4.1-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _LMS_DETECT_PROMPT},
                {"role": "user",   "content": user_query},
            ],
        )
        result = json.loads(resp.choices[0].message.content)
        is_lms = bool(result.get("is_lms_query", False))
        logger.info(f"[LLM] LMS query detection: {is_lms}")
        return is_lms
    except Exception as e:
        logger.warning(f"[LLM] LMS detection failed ({e}), falling back to keyword detection")
        return _keyword_detect_lms(user_query)


# ─────────────────────────────────────────────
# Keyword Fallbacks
# ─────────────────────────────────────────────

_COURSE_KEYWORDS: dict[str, list[str]] = {
    "fullstack_batch":  ["full stack", "fullstack", "full-stack", "frontend", "backend", "react", "node", "gen ai full stack"],
    "ai_batch":         ["ai stack", "ml", "machine learning", "artificial intelligence", "generative ai", "deep learning", "nlp", "ai batch"],
    "dsa_batch":        ["dsa", "data structures", "algorithms", "competitive", "problem solving"],
    "campus_fullstack": ["campus full stack", "campus fullstack"],
    "campus_ai":        ["campus ai", "campus ml"],
}

_LMS_KEYWORDS: set[str] = {
    "platform", "portal", "leap", "lms", "learning experience",
    "how do i learn", "how does it work", "how will i learn",
    "videos", "recorded", "exercises", "projects", "doubt session",
    "standup", "mentor session", "platform access",
}


def _keyword_detect_courses(text: str) -> list[str]:
    text_lower = text.lower()
    matched = [key for key, kws in _COURSE_KEYWORDS.items() if any(kw in text_lower for kw in kws)]
    logger.info(f"[Keyword] Course detection: {matched}")
    return matched


def _keyword_detect_lms(text: str) -> bool:
    return any(kw in text.lower() for kw in _LMS_KEYWORDS)


# ─────────────────────────────────────────────
# Eligibility helper
# ─────────────────────────────────────────────

def get_eligible_course_keys(student_year: int | None) -> list[str]:
    """
    Returns the course keys a student is eligible for based on their study year.
    year 1/2/3 → campus courses only
    year 4 / None (passed out) → regular batches only
    year None (unknown) → all courses
    """
    if student_year is None:
        return list(kb["courses"].keys())
    if student_year in (1, 2, 3):
        return ["campus_fullstack", "campus_ai"]
    return ["fullstack_batch", "ai_batch", "dsa_batch"]


# ─────────────────────────────────────────────
# Base Builders
# ─────────────────────────────────────────────

def build_base_context(kb: dict) -> str:
    logger.info("Building base context")
    company  = kb["company"]
    platform = kb["platform"]
    mentors  = platform["mentors"]
    mentor_companies = ", ".join(mentors["background_companies"])

    return f"""
ORGANIZATION DETAILS
Name             : {company["name"]}
Location         : {company["office_location"]}
Office Timings   : {company["office_timings"]}
Email            : {company["contact_email"]}
Website          : {company["website"]}

LEARNING PLATFORM (Leap)
Platform         : {platform["lms"]}
Mentors          : {mentors["description"]} — from companies like {mentor_companies}
Doubt Sessions   : {platform["doubt_sessions"]}
Daily Standups   : {platform["standups"]}
Certificate      : {platform["certificate"]}

TRAINING MODES
Online           : {kb["training_modes"]["online"]["description"]}
Offline          : {kb["training_modes"]["offline"]["description"]}
Note             : {kb["training_modes"]["note"]}
""".strip()


def build_lms_detail_context(kb: dict) -> str:
    """
    Returns a detailed block about the Leap platform's learning approach, features, and mentors.
    Used only when the student specifically asks about the learning experience or platform.
    """
    logger.info("Building LMS detail context")
    platform = kb["platform"]
    mentors  = platform["mentors"]
    features = "\n".join(f"  • {f}" for f in platform.get("features", []))
    mentor_things = "\n".join(f"  • {t}" for t in mentors["what_they_do"])
    mentor_companies = ", ".join(mentors["background_companies"])

    return f"""
LEAP PLATFORM — LEARNING EXPERIENCE
Learning Approach : {platform.get("learning_approach")}
Key Features      :
{features}

MENTORS
Background        : Professionals from {mentor_companies}
What They Do      :
{mentor_things}
Offline Advantage : {mentors["offline_note"]}
""".strip()


def build_placement_context(kb: dict) -> str:
    """
    Returns a block about Gradious' placement support, packages, and partner companies.
    """
    logger.info("Building placement context")
    p = kb["placements"]
    return f"""
PLACEMENT SUPPORT
Assistance            : {"Yes" if p["assistance"] else "No"}
Highest Package       : {p.get("highest_package", "N/A")}
Average Package       : {p.get("avg_package", "N/A")}
Placement Rate        : {p.get("placement_rate", "N/A")}
Partner Companies     : {p.get("partnered_companies_type", "N/A")}
About Placements      : {p.get("placement_description", "")}
For More Details      : {p.get("for_more_details", "")}
""".strip()


# ─────────────────────────────────────────────
# Course-Specific Builders
# ─────────────────────────────────────────────

def _get_duration(c: dict, mode: str) -> str:
    return c[mode].get("course_duration") or c.get("course_duration") or "N/A"


def _get_platform_access(c: dict, mode: str) -> str:
    return c[mode].get("platform_access") or c.get("platform_access") or "N/A"


def build_complete_course_context(kb: dict, course: str) -> str:
    logger.info(f"Building complete course context for {course}")
    c = kb["courses"][course]
    topics     = "\n".join(f"  • {t}" for t in c["course_overview"]["what_you_will_learn"])
    gen_ai_note = c["course_overview"].get("gen_ai_note", "")
    campus_note = c.get("campus_note", "")
    class_types = c.get("class_types", ["self-paced", "live-classes"])
    modes       = c.get("modes", ["online", "offline"])

    # batch_schedule block
    batch_sched = c.get("batch_schedule")
    batch_block = ""
    if batch_sched:
        batch_block = f"""
BATCH SCHEDULE
Live Classes     : {batch_sched.get("live", "N/A")}
Self-Paced       : {batch_sched.get("self_paced", "N/A")}"""

    # Campus courses only have self-paced
    if "live" in c:
        live_block = f"""
LIVE CLASSES
Duration         : {_get_duration(c, "live")}
Fee              : {c["live"]["fee_structure"]}
Platform Access  : {_get_platform_access(c, "live")}"""
    else:
        live_block = "\nLIVE CLASSES : Not available for this program"

    note_lines = ""
    if campus_note:
        note_lines += f"\nNote             : {campus_note}"
    if gen_ai_note:
        note_lines += f"\nGen AI Module    : {gen_ai_note}"

    return f"""

COURSE DETAILS — {c["name"]}
Eligibility      : {c["eligibility"]}
Modes Available  : {", ".join(modes)}
Class Types      : {", ".join(class_types)}
Topics Covered   :
{topics}{note_lines}

SELF-PACED
Duration         : {_get_duration(c, "self")}
Fee              : {c["self"]["fee_structure"]}
Platform Access  : {_get_platform_access(c, "self")}
{live_block}{batch_block}
"""


# ─────────────────────────────────────────────
# FAQ Context Builder (used by faq_node)
# ─────────────────────────────────────────────

def build_faq_context(course_keys: list[str], include_lms_detail: bool = False) -> str:
    """
    Assembles the knowledge base context to inject into the FAQ prompt.

    Always includes: company/platform base + placement context.
    Variable: complete course block for each key in course_keys.
    If course_keys is empty, includes all courses.
    include_lms_detail: if True, appends the full Leap platform detail block.
    """
    logger.info(
        f"Building FAQ context | courses={course_keys if course_keys else 'ALL'} "
        f"| lms_detail={include_lms_detail}"
    )
    context  = build_base_context(kb)
    context += "\n\n" + build_placement_context(kb)

    if include_lms_detail:
        logger.info("Including full LMS detail block")
        context += "\n\n" + build_lms_detail_context(kb)

    keys = course_keys if course_keys else list(kb["courses"].keys())
    logger.info(f"Adding {len(keys)} course(s) to context")
    for key in keys:
        if key in kb["courses"]:
            context += build_complete_course_context(kb, key)
        else:
            logger.warning(f"Course key '{key}' not found in KB — skipping")

    logger.info(f"FAQ context built | length={len(context)} chars")
    return context


# ─────────────────────────────────────────────
# Resolve Courses for FAQ node
# ─────────────────────────────────────────────

def resolve_courses_for_faq(
    user_query: str,
    course_interest: str | None,
    student_year: int | None = None,
) -> list[str]:
    """
    Priority 1: course_interest already stored in state (student confirmed a course).
    Priority 2: LLM course detection from query text, filtered by student eligibility.
    Priority 3: return eligible courses for student year (or all if unknown).
    """
    logger.info(
        f"Resolving courses for FAQ | course_interest={course_interest} | year={student_year}"
    )

    if course_interest and course_interest in kb["courses"]:
        logger.info(f"Using stored course_interest: {course_interest}")
        return [course_interest]

    detected = llm_detect_courses(user_query, student_year=student_year)
    if detected:
        logger.info(f"Using LLM-detected courses: {detected}")
        return detected

    # Fallback: return eligible courses based on year so context is not overwhelming
    eligible = get_eligible_course_keys(student_year)
    logger.info(f"No specific course detected — using eligible courses: {eligible}")
    return eligible
