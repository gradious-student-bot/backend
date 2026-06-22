import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, ToolUseBlock
from dotenv import load_dotenv
from openpyxl import load_workbook
load_dotenv()

CWD_PATH = r"C:\Users\vishw\Gradious Files\Gradious bot\backend_claude_run"

SYSTEM_PROMPT = """
# Gradious Lead Agent — Claude Code Task Prompt (Batch 3)

## Project Overview
This is a Python-based voice agent for lead conversion at Gradious, a tech training institute.
The agent makes outbound calls to student leads, collects qualification data via a questionnaire,
answers course-related queries using a knowledge base, and writes results to Airtable.
Built with FastAPI, LangGraph, and OpenAI (gpt-4.1-mini).

## File Structure
.
├── .env                              # Environment variables — OpenAI key, Airtable token, base ID, table name
├── main.py                           # FastAPI app entry point — mounts router, sets up CORS, configures logging
├── config.py                         # Pydantic settings loader — reads .env and exposes typed config to all modules
├── requirements.txt                  # Python dependencies — openai, fastapi, langgraph, pyairtable, pydantic-settings
│
├── agent/
│   ├── state.py                      # LeadState TypedDict — single source of truth for all session data across nodes
│   ├── graph.py                      # LangGraph StateGraph — wires all nodes together with conditional routing edges
│   │
│   ├── nodes/
│   │   ├── intent_router.py          # Classifies every user utterance into one of 9 intents via GPT-4.1-mini JSON call
│   │   ├── questionnaire_node.py     # Core Q&A flow — LLM field extraction, dynamic question generation, branch logic
│   │   ├── faq_node.py               # Answers course/fee/placement queries using KB context, appends pending question
│   │   ├── repeat_node.py            # Rephrases last agent response naturally when student asks to repeat
│   │   ├── end_node.py               # Generates LLM closing message for not_interested or end_call intents
│   │   └── airtable_node.py          # Writes final lead record to Airtable at end of every call
│   │
│   └── prompts/
│       ├── questionnaire_prompt.py   # Legacy static prompt strings — superseded by inline prompts in questionnaire_node
│       ├── faq_prompt.py             # FAQ system prompt builder — injects KB context and pending question dynamically
│       └── repeat_prompt.py          # Repeat node system prompt — instructs LLM to rephrase, not replay verbatim
│
├── api/
│   └── routes.py                     # FastAPI routes — POST /agent/init, POST /agent/turn, WS /ws/agent/{session_id}
│
├── knowledge/
│   ├── fasttrack.json                # Course knowledge base — courses, fees, platform (Leap LMS), placements, company info
│   └── context_builders.py           # KB accessor functions — builds structured text context blocks for FAQ LLM prompts
│
├── models/
│   └── schemas.py                    # Pydantic request/response models — InitRequest, TurnRequest, TurnResponse, etc.
│
├── services/
│   ├── airtable_client.py            # Airtable API wrapper — initialises pyairtable client, exposes write_lead()
│   └── llm.py                        # Shared OpenAI client instance — import from here to avoid re-initialising per node
│
└── voice/
    ├── agent.py                       # Voice integration layer — bridges STT/TTS calling service with agent WebSocket
    └── trigger_call.py                # Outbound call trigger — initiates a call to a lead's phone via calling service API

---

## General Principles (Apply to ALL tasks)

- All LLM calls that produce a final spoken response to the user must use `temperature=0.85`.
  LLM calls that classify, extract, or route (intent_router, field extraction) must keep `temperature=0`.
- All new system prompts must include the `## PERSONA` block (copy from faq_node.py's existing PERSONA section).
- All LLM calls use `response_format={"type": "json_object"}` and the `openai` package directly.
- Never hardcode spoken response strings — every message the user hears must come from an LLM call.
- Do not modify files not explicitly mentioned in each task's scope.

---

## Task 1 — Increase Temperature for Final Response LLM Calls (agent/nodes/questionnaire_node.py, agent/nodes/faq_node.py, agent/nodes/end_node.py, agent/nodes/repeat_node.py)

**Problem:** Responses feel robotic because LLM calls generating spoken output use low temperature (0.2-0.4), making them deterministic and flat.

**What to change:**

Go through every `client.chat.completions.create(...)` call in the following files and apply these temperature rules:

| File | Call Purpose | New Temperature |
|---|---|---|
| `questionnaire_node.py` | Greeting / identity / timing / interest prompts | 0.85 |
| `questionnaire_node.py` | Next question generation | 0.85 |
| `questionnaire_node.py` | Wrap-up / closing message | 0.85 |
| `questionnaire_node.py` | Human agent scheduling confirmation | 0.85 |
| `questionnaire_node.py` | De-escalation / confused / irrelevant / rude responses | 0.85 |
| `questionnaire_node.py` | Field extraction (EXTRACT_SYSTEM_PROMPT) | 0.0 (classification — do not change) |
| `questionnaire_node.py` | Entity switch confirmation | 0.85 |
| `faq_node.py` | FAQ answer generation | 0.85 |
| `end_node.py` | Closing message generation | 0.85 |
| `repeat_node.py` | Rephrase generation | 0.85 |

Do not change temperature for intent_router.py or any other classification/extraction call.

---

## Task 2 — Small Talk Handling at Greeting Step (agent/nodes/questionnaire_node.py, agent/nodes/intent_router.py)

**Problem:** When the agent asks "Am I speaking with [Name]?" and the user says "Hi" or asks "What are you?", the agent doesn't handle this gracefully — it either misclassifies or ignores the greeting.

**What to implement:**

**In `intent_router.py`:**
- Add a new intent `"small_talk"` to the intent classification system prompt with this description:
  ```
  "small_talk": Return "small_talk" if the user's message is a greeting (Hi, Hello, Hey),
  a pleasantry (How are you?), or a meta-question about the agent itself
  (What are you?, Who am I speaking to?, Are you a bot?, Are you a real person?).
  This also applies when the user says "Hi" or similar instead of answering the current question.
  ```
- `small_talk` should be valid at any point in the conversation, not just at the greeting step.
- Add `"small_talk"` to the intent enum in the OUTPUT FORMAT JSON.
- Add a routing rule: `"small_talk"` → route to `"questionnaire"` (the questionnaire node will handle the response inline).

**In `questionnaire_node.py`:**
- At the top of `questionnaire_node`, add a check: if `intent == "small_talk"`:
  1. Call a dedicated LLM prompt `SMALL_TALK_SYSTEM_PROMPT` to generate a spoken response.
  2. The prompt should instruct the LLM:
     - Greet the user back warmly if they said Hi/Hello.
     - If user asks "What are you?" or "Are you a bot?", be honest: say it's an AI assistant from Gradious, here to help with course information and collect a few details.
     - After the small talk response, naturally re-ask the current pending question (inject `state["last_agent_response"]` as the pending question).
  3. The JSON output format: `{"response": "..."}`.
  4. Set `state["last_agent_response"]` and append to `state["messages"]`, then return — do not proceed to field extraction or next question generation.

---

## Task 3 — Ask First Question Immediately After Identity Confirmation (agent/nodes/questionnaire_node.py)

**Problem:** After the agent confirms identity and the user says yes, the agent says a transition phrase ("Thanks for your interest...") and then waits for the next user turn before asking the first question. This wastes a turn and feels unnatural.

**What to implement:**

In `questionnaire_node.py`, in the `confirm_timing` step (Step 2 of greeting, reached after identity confirmed):

When generating the transition response after identity is confirmed (the `POST_CONFIRM_SYSTEM_PROMPT` call), modify the system prompt instruction to combine the transition phrase AND the first question in a single response.

Update `POST_CONFIRM_SYSTEM_PROMPT` to say:
```
After the warm introduction sentence, immediately ask the first question of the questionnaire
in the same response. The first question is about which course the student is interested in.
Do not wait — ask it right away in a natural, flowing way.
Example style: "Great to connect with you, [Name]. I'm calling from Gradious — we noticed you showed
interest in our courses and wanted to help. So, which course are you looking at — Full Stack or AI?"
(This is style inspiration only — LLM should generate its own version.)
```

The `current_question_key` should be set to `"course_interest"` immediately after this response so the next user turn is correctly processed as answering the course question.

---

## Task 4 — Live Batch Start Date Handling (agent/nodes/faq_node.py, knowledge/fasttrack.json)

**What to implement:**

**In `knowledge/fasttrack.json`:**
Add a `"batch_schedule"` field under each course that has live classes (`fullstack_batch`, `ai_batch`, `dsa_batch`):
```json
"batch_schedule": {
  "live": "New batches start every 2nd week of the month",
  "self_paced": "Start anytime — self-paced courses have no fixed batch date"
}
```
Also add a top-level `"admissions"` block in the JSON:
```json
"admissions": {
  "batch_frequency": "New live batches start every 2nd week of the month",
  "enrollment_process": "Contact our admissions team or schedule a call with an expert for exact upcoming batch dates and seat availability."
}
```

**In `knowledge/context_builders.py`:**
Update `build_complete_course_context()` to include `batch_schedule` in the rendered context block when present:
```
BATCH SCHEDULE
Live Classes     : New batches start every 2nd week of the month
Self-Paced       : Start anytime — no fixed batch date
```

**In `agent/nodes/faq_node.py` — `FAQ_SYSTEM_PROMPT`:**
Add a section `## BATCH DATE RULES`:
```
## BATCH DATE RULES
- If the student asks about when a batch starts or live class dates:
  Say: "Live batches start every 2nd week of the month."
- Do NOT give a specific calendar date — you don't have that information.
- Follow up by offering to connect them with an admissions expert for exact dates and seat availability.
- This should naturally flow into offering to schedule a callback: "Would you like me to have
  our admissions expert give you a call with the exact upcoming dates?"
```

---

## Task 5 — Remove Examples from Academic Detail Questions (agent/nodes/questionnaire_node.py)

**Problem:** When asking academic questions, the agent adds examples like "Which department are you from? Like CSE, ECE?" — this sounds scripted and adds unnecessary words on a phone call.

**What to implement:**

In `questionnaire_node.py`, in `NEXT_QUESTION_SYSTEM_PROMPT`, add an explicit instruction:
```
## QUESTION STYLE RULES
- When asking about academic details (department, year, passout year), do NOT give examples.
  Ask the question plainly. Wrong: "Which branch are you from? Like CSE, IT, or ECE?"
  Right: "Which branch are you from?"
- Exception: when asking about which course the student is interested in, you MAY mention
  the course names (Full Stack + Gen AI, AI Stack) since these are Gradious-specific and
  the student may not know them otherwise.
- Keep every question to one sentence where possible.
```

---

## Task 6 — Wrong Person Flow: Ask for Correct Name (agent/nodes/questionnaire_node.py, agent/state.py)

**Problem:** When the agent asks "Am I speaking with [Name]?" and the user says "No" or "Wrong number", the current logic either loops (re-asks the same identity question) or ends the call. Instead, it should apologize, ask for the correct name, store it, and continue normally.

**What to implement:**

**In `agent/state.py`:**
Add a new field:
```python
corrected_name: Optional[str]   # Set when the lead's name is corrected during identity confirmation
```

**In `questionnaire_node.py` — `confirm_identity` handling:**
Currently, when user confirms identity → proceed to `confirm_timing`. When user denies → currently undefined/looping.

Add a new `current_question_key` value: `"ask_corrected_name"`.

When processing `confirm_identity`:
- If user says yes/affirmative → proceed normally to `confirm_timing` (existing logic).
- If user says no/negative (detect via a simple LLM extraction call asking "did the user confirm or deny identity? Return: `{"confirmed": true/false}`"):
  1. Set `current_question_key = "ask_corrected_name"`.
  2. Generate an LLM response using a new `ASK_CORRECTED_NAME_PROMPT`:
     ```
     The person on the call is not the person we expected. Apologize briefly and naturally,
     then ask for their name. Keep it short and warm.
     Example style: "Oh, I'm sorry about that! Could I get your name please?"
     (Style inspiration only — LLM generates its own version.)
     ```
  3. Set response, append to messages, return.

When processing `ask_corrected_name`:
- Run field extraction to pull the name from the user's reply. Store in `state["corrected_name"]`.
- Also update `state["lead_name"]` to the corrected name so all subsequent LLM prompts use the right name.
- Then proceed normally: set `current_question_key = "confirm_timing"` and generate the Step 2 greeting response (same `POST_CONFIRM_SYSTEM_PROMPT` flow as normal identity confirmation).

---

## Task 7 — Out-of-Scope: Restrict to Courses and Coaching Only (agent/nodes/faq_node.py)

**Problem:** The current `## SCOPE RULES` in `FAQ_SYSTEM_PROMPT` redirects general questions toward Gradious courses, but the language is too flexible and the agent sometimes still partially answers off-scope questions.

**What to change in `FAQ_SYSTEM_PROMPT` — `## SCOPE RULES`:**

Replace the existing section with:
```
## SCOPE RULES
You can ONLY help with:
- Gradious courses: content, syllabus, structure, duration, fees, modes, batches
- Gradious learning platform (Leap): how it works, features
- Gradious placements: companies, process, packages
- Gradious company info: location, timings, contact
- Enrollment and next steps

If the student asks ANYTHING outside this scope — general knowledge, career advice,
coding help, definitions of technologies, other institutes, job market advice, etc. —
do NOT answer it at all. Say clearly but warmly that you can only help with
Gradious course and coaching information right now.

Example phrasings for out-of-scope:
- "I can only help with details about our courses and coaching programs right now.
   Is there something specific about our courses you'd like to know?"
- "That's a bit outside what I can help with on this call — I'm here specifically to
   help with Gradious course and enrollment information. Want me to tell you about
   our programs instead?"

NEVER attempt to answer general knowledge questions even partially.
Always redirect back to Gradious offerings.
```

---

## Task 8 — Remove Referral Source Question (agent/nodes/questionnaire_node.py, agent/state.py, agent/nodes/airtable_node.py)

**What to remove:**

1. **`questionnaire_node.py`:** Remove `"referral_source"` from `ALL_FIELDS`, from the ordered field list in `_get_next_field()`, and from `_apply_extracted()`.
2. **`agent/state.py`:** Remove `referral_source: Optional[str]` from `LeadState`.
3. **`airtable_node.py`:** Remove `"Referral Source": state.get("referral_source")` from `lead_data`.

---

## Task 9 — Update Placement Details and Remove Company Names from Responses (knowledge/fasttrack.json, knowledge/context_builders.py, agent/nodes/faq_node.py)

**What to change in `knowledge/fasttrack.json`:**

Update the `"placements"` object to:
```json
"placements": {
  "assistance": true,
  "highest_package": "40 LPA",
  "avg_package": "6.4 LPA",
  "placement_rate": "70% of eligible students placed within 3 months",
  "partnered_companies_type": "Product-based and tech-first companies",
  "placement_description": "We have partnered with product-based companies that actively hire trained candidates from Gradious. Our placement support includes resume preparation, mock interviews, and direct referrals to our hiring partners.",
  "for_more_details": "For specific company names, active openings, and detailed placement process — our admissions expert can walk you through everything on a call."
}
```
Remove the `"partnered_companies"` array entirely — company names should not be mentioned in responses.

**What to change in `knowledge/context_builders.py` — `build_placement_context()`:**
Update to render the new fields. Do not render a company names list. Output:
```
PLACEMENT SUPPORT
Assistance            : Yes
Highest Package       : 40 LPA
Average Package       : 6.4 LPA
Placement Rate        : 70% of eligible students placed within 3 months
Partner Companies     : Product-based and tech-first companies
About Placements      : We have partnered with product-based companies...
For More Details      : For company names, openings, and placement process — schedule a call with our admissions expert.
```

**What to change in `agent/nodes/faq_node.py` — `FAQ_SYSTEM_PROMPT`:**
Replace the existing placement instruction in `## SCOPE RULES` with:
```
## PLACEMENT RESPONSE RULES
- When answering placement questions, say we partner with product-based companies.
  Mention the packages: highest 40 LPA, average 6.4 LPA, placement rate 70%.
- Do NOT name specific companies.
- If the student asks for company names, specific openings, or placement process details,
  say: "For that level of detail, our admissions expert would be the right person to speak
  to — they can walk you through the exact companies and process. Want me to schedule a
  quick call with them?"
- This should set needs_human_agent: true in the response JSON.
```

---

## Task 10 — Add Mentor Details to Knowledge Base (knowledge/fasttrack.json)

**What to change in `knowledge/fasttrack.json`:**

The `"mentors"` field under `"platform"` should already be a structured object (added in a previous batch). Ensure it exactly matches:
```json
"mentors": {
  "description": "Industry professionals with hands-on experience at top tech and financial companies",
  "background_companies": ["Microsoft", "JP Morgan", "Gradious AI", "Kore.ai"],
  "what_they_do": [
    "1:1 doubt clearing sessions",
    "Code reviews and project feedback",
    "Mock interview preparation",
    "Career guidance and resume building",
    "Real-world problem-solving walkthroughs"
  ],
  "offline_note": "Students in offline mode get direct in-person access to mentors and industry professionals at the Gradious office"
}
```
If it already exists with this shape, verify the `background_companies` list matches exactly: `["Microsoft", "JP Morgan", "Gradious AI", "Kore.ai"]`. Update if different.

No other file changes needed for this task — `context_builders.py` already renders this structure.

---

## Task 11 — Lead Scoring and Classification (agent/state.py, agent/nodes/airtable_node.py)

**What to implement:**

**In `agent/state.py`:**
Add two new fields:
```python
lead_score: int           # Computed score 0-100
lead_classification: str  # "Hot Lead" | "Warm Lead" | "Cold Lead"
```

**Create a new file `agent/nodes/scoring.py`:**
```python
'''
Lead scoring and classification logic.
Called by airtable_node before writing the Airtable record.
'''
from agent.state import LeadState
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)


def compute_lead_score(state: LeadState) -> tuple[int, str]:
    '''
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
    '''
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
```

**In `agent/nodes/airtable_node.py`:**
- Import `compute_lead_score` from `agent.nodes.scoring`.
- Before building `lead_data`, call:
  ```python
  score, classification = compute_lead_score(state)
  state["lead_score"] = score
  state["lead_classification"] = classification
  ```
- Add to `lead_data`:
  ```python
  "Lead Score":          score,
  "Lead Classification": classification,
  "Looking for Job":     state.get("looking_for_job"),
  ```

---

## Task 12 — Working Professional Support (agent/state.py, agent/nodes/questionnaire_node.py)

**Problem:** The questionnaire only handles "student" and "graduated". Working professionals should be treated like graduated people but also asked if they're currently looking for a job.

**What to implement:**

**In `agent/state.py`:**
Add:
```python
looking_for_job: Optional[bool]   # True if graduated/working professional is actively job hunting
```
Update the `student_status` field comment to:
```python
student_status: Optional[str]     # "student" | "graduated" | "working_professional"
```

**In `agent/nodes/questionnaire_node.py`:**

1. **Field list / `ALL_FIELDS`:** Add `"looking_for_job"` with description:
   `"Whether the person is currently looking for a job (only for graduated/working_professional)"`

2. **`_get_next_field()` branching logic:** After `student_status` is collected:
   - If `student_status == "student"` → branch: `current_year`, `passout_year`, `department`
   - If `student_status in ("graduated", "working_professional")` → branch: `passout_year`, `department`, `looking_for_job`
   So `looking_for_job` is only asked when `student_status` is `"graduated"` or `"working_professional"`. Never ask it for students.

3. **`_apply_extracted()` mirror map:** Add `"looking_for_job"` → `state["looking_for_job"]`. The value should be a boolean — the field extraction prompt should extract `true` if the person says they're looking for a job, `false` if not.

4. **`EXTRACT_SYSTEM_PROMPT`:** Add `looking_for_job` to the extractable fields list with description:
   `"looking_for_job: boolean — true if the person says they are currently looking for a job or actively applying, false otherwise. Only extract if student_status is graduated or working_professional."`

5. **Intent router / small_talk identity handling:** When asking `student_status`, the options should include "working professional" as a valid answer. Update `NEXT_QUESTION_SYSTEM_PROMPT` to note that student status options are: currently studying, graduated, or working professional.

---

## Task 13 — Extract Callback Date and Time as ISO Datetime (agent/nodes/questionnaire_node.py, agent/state.py)

**Problem:** The `callback_time` field stores whatever the user says as free text (e.g. "tomorrow at 3pm"). We need this parsed into a structured ISO datetime string.

**What to implement:**

**In `agent/state.py`:**
- Keep `callback_time: Optional[str]` but clarify in comment: stores ISO 8601 datetime string (e.g. `"2026-06-25T15:00:00"`)
- Add `callback_time_raw: Optional[str]` — stores the user's original natural language phrase before parsing

**In `agent/nodes/questionnaire_node.py`:**

After the field extraction step extracts `callback_time` as a raw string, add a dedicated datetime parsing step. Add a helper function `_parse_callback_datetime(raw: str) -> str | None`:

```python
def _parse_callback_datetime(raw: str) -> str | None:
    '''
    Uses LLM to parse a natural language time expression into ISO 8601 datetime.
    Sends today's date as context so relative expressions like "tomorrow" resolve correctly.
    Returns ISO string or None if parsing fails.
    '''
```

Implementation:
- Import and use `datetime.now()` to get today's date.
- Call `client.chat.completions.create` with `temperature=0` and `response_format={"type": "json_object"}`.
- System prompt:
  ```
  ## ROLE
  You are a datetime parser.

  ## OBJECTIVE
  Convert a natural language time expression into an ISO 8601 datetime string.

  ## CONTEXT
  Today's date and time: {today_iso}
  The user is located in Hyderabad, India (IST, UTC+5:30).

  ## INSTRUCTIONS
  1. Parse the given time expression relative to today's date.
  2. If only a time is given (e.g. "3pm"), assume today if it's in the future, otherwise tomorrow.
  3. If only a day is given (e.g. "tomorrow", "Monday"), assume 10:00 AM IST.
  4. If the expression is ambiguous or unparseable, return null.
  5. Return STRICT JSON only.

  ## OUTPUT FORMAT
  {"iso_datetime": "2026-06-25T15:00:00" | null}
  ```
- User message: the raw callback_time string.
- On success: return `parsed["iso_datetime"]`
- On failure or null: return `None`, log a warning.

In `questionnaire_node`, when `callback_time` is extracted:
1. Store the raw string in `state["callback_time_raw"]`.
2. Call `_parse_callback_datetime(raw)` and store result in `state["callback_time"]`.
3. If parsing returns `None`, keep `callback_time_raw` and log: `logger.warning("[Questionnaire] Could not parse callback datetime from: {raw}")`

**In `agent/nodes/airtable_node.py`:**
- Add `"Callback Time (Raw)": state.get("callback_time_raw")` to `lead_data`.
- The existing `"Callback Time": state.get("callback_time")` now stores the ISO datetime.

Same pattern should apply to `join_date` if it is also a date expression. Apply `_parse_callback_datetime` to `join_date` extraction as well. Store raw in `join_date_raw: Optional[str]` (add to `LeadState`).

---

## Task 14 — Schedule Expert Call for Unknown/Complex Queries (agent/nodes/faq_node.py, agent/nodes/questionnaire_node.py)

**Problem:** When the agent can't answer something — either because it's out of scope, too specific, or the KB doesn't have enough detail — it should offer to schedule a call with an admissions expert rather than giving a vague or unhelpful response.

**What to implement:**

**In `agent/nodes/faq_node.py` — `FAQ_SYSTEM_PROMPT`, add a `## FALLBACK RULES` section:**
## FALLBACK RULES
If you cannot answer the student's question confidently from the knowledge base:
- Do NOT guess or make up information.
- Do NOT give a vague answer.
- Instead, say something like:
  "That's a great question — I don't have all the details on that right now. Let me connect
   you with our admissions expert who can give you the full picture. Should I schedule a
   quick call with them?"
- Set "needs_human_agent": true in your response JSON.
- This applies to: specific batch dates, specific company names, scholarship details,
  installment plans, customised training, anything not explicitly in your knowledge base.

**In `agent/nodes/questionnaire_node.py`:**

In the `CONFUSED_SYSTEM_PROMPT` and `IRRELEVANT_SYSTEM_PROMPT`, add an instruction:
If the user's message suggests they need help beyond what this call can provide
(e.g. they mention a very specific technical question, pricing negotiation, or complex
eligibility scenario), offer to schedule a call with an admissions expert.
In that case, set "schedule_expert": true in the JSON.

Update the JSON output format for those prompts to include `"schedule_expert": false` as a default field.

In `questionnaire_node`, after generating the confused/irrelevant response, check if `parsed.get("schedule_expert")` is `True`. If so, set `state["human_agent_requested"] = True` and `state["next_node"] = "human_agent"`.

---

## State Changes Summary (agent/state.py)

Ensure `LeadState` has ALL of the following fields after all tasks are applied:

```python
# Existing fields (keep as-is)
lead_id: str
lead_name: str
phone: str
email: Optional[str]
messages: list
last_agent_response: str
next_node: str
current_question_key: str
answered_fields: dict
human_agent_requested: bool
pending_switch: Optional[dict]
pending_sub_query: Optional[str]
pending_next_question_text: Optional[str]
question_retry_counts: dict
greeting_step: int
course_interest: Optional[str]
student_status: Optional[str]
current_year: Optional[str]
passout_year: Optional[str]
department: Optional[str]
training_mode: Optional[str]
class_type: Optional[str]
interested: Optional[bool]
join_date: Optional[str]
callback_requested: Optional[bool]
callback_time: Optional[str]
disposition: str
call_ended: bool

# New fields added in this batch
corrected_name: Optional[str]        # Task 6 — name provided when wrong person answers
looking_for_job: Optional[bool]      # Task 12 — job hunting status for graduated/working
lead_score: int                      # Task 11 — computed lead score 0-100
lead_classification: str             # Task 11 — Hot/Warm/Cold Lead
callback_time_raw: Optional[str]     # Task 13 — raw natural language callback time
join_date_raw: Optional[str]         # Task 13 — raw natural language join date
```

---

## Airtable Field Summary (agent/nodes/airtable_node.py)

After all tasks, `lead_data` must include these fields (add new ones, do not remove existing):
```python
"Lead Score":             state.get("lead_score"),
"Lead Classification":    state.get("lead_classification"),
"Looking for Job":        state.get("looking_for_job"),
"Corrected Name":         state.get("corrected_name"),
"Callback Time":          state.get("callback_time"),        # ISO datetime
"Callback Time (Raw)":    state.get("callback_time_raw"),    # user's phrase
"Join Date":              state.get("join_date"),            # ISO datetime
"Join Date (Raw)":        state.get("join_date_raw"),        # user's phrase
```
Remove `"Referral Source"` (Task 8).
Remove `"Budget Range"` if still present (removed in previous batch).

"""


import pandas as pd
from datetime import datetime
import os

class CostTracker:
    def __init__(self, model_name="claude-sonnet-4.6"):
        self.model = model_name
        self.tool_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.steps = []
        self.start_time = datetime.now()

        self.input_cost_per_token = 3 / 1_000_000
        self.output_cost_per_token = 15 / 1_000_000

    def log_tool_call(self):
        self.tool_calls += 1

    def log_tokens(self, usage):
        if not usage:
            return

        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)

        self.input_tokens += in_tok
        self.output_tokens += out_tok

        self.steps.append({
            "timestamp": datetime.now(),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "step_cost_usd": self.calculate_cost(in_tok, out_tok)
        })

    def calculate_cost(self, in_tok, out_tok):
        return round(
            (in_tok * self.input_cost_per_token) +
            (out_tok * self.output_cost_per_token),
            8
        )

    def total_cost(self):
        return self.calculate_cost(self.input_tokens, self.output_tokens)

    def save_to_excel(self):
        print("🚨 save_to_excel CALLED")
        print("steps:", len(self.steps))
        print("input_tokens:", self.input_tokens)
        print("output_tokens:", self.output_tokens)
        os.makedirs("costing", exist_ok=True)
        file_path = "costing/agent_costs.xlsx"

        df_summary = pd.DataFrame([{
            "timestamp": datetime.now(),
            "model": self.model,
            "tool_calls": self.tool_calls,
            "total_input_tokens": self.input_tokens,
            "total_output_tokens": self.output_tokens,
            "total_cost_usd": self.total_cost(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds()
        }])

        df_steps = pd.DataFrame(self.steps)

        # ---- FIRST TIME CREATE ----
        if not os.path.exists(file_path):
            with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
                df_summary.to_excel(writer, sheet_name="runs", index=False)
                df_steps.to_excel(writer, sheet_name="steps", index=False)
            print("✅ Excel created")
            return

        # ---- APPEND ----
        with pd.ExcelWriter(
            file_path,
            engine="openpyxl",
            mode="a",
            if_sheet_exists="overlay"
        ) as writer:

            # ---- RUNS ----
            try:
                existing_runs = pd.read_excel(file_path, sheet_name="runs")
                startrow_runs = len(existing_runs) + 1
            except:
                startrow_runs = 0

            df_summary.to_excel(
                writer,
                sheet_name="runs",
                index=False,
                header=(startrow_runs == 0),
                startrow=startrow_runs
            )

            # ---- STEPS ----
            try:
                existing_steps = pd.read_excel(file_path, sheet_name="steps")
                startrow_steps = len(existing_steps) + 1
            except:
                startrow_steps = 0

            df_steps.to_excel(
                writer,
                sheet_name="steps",
                index=False,
                header=(startrow_steps == 0),
                startrow=startrow_steps
            )

        print("✅ Excel appended successfully")

async def main():
    tracker = CostTracker("claude-sonnet-4.6")

    async for message in query(
        prompt=SYSTEM_PROMPT,
        options=ClaudeAgentOptions(
            # model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            model="us.anthropic.claude-sonnet-4-6",
            allowed_tools=["Read", "Edit", "Bash"],
            permission_mode="acceptEdits",
            cwd = r"C:\Users\vishw\Gradious Files\Gradious bot\backend_claude_run"
        )
    ):

        if hasattr(message, "usage"):
            tracker.log_tokens(message.usage)

 
        if hasattr(message, "content") and message.content:
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    tracker.log_tool_call()
                    print(f"\n🔧 TOOL: {block.name}")
                    print(f"INPUT: {block.input}")

        elif hasattr(message, "result"):
            print(f"\n✅ FINAL RESULT:\n{message.result}")

    tracker.save_to_excel()

asyncio.run(main())