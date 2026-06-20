import asyncio
from claude_agent_sdk import query, ClaudeAgentOptions, ToolUseBlock
from dotenv import load_dotenv
from openpyxl import load_workbook
load_dotenv()

CWD_PATH = r"C:\Users\chand\Desktop\Claude agent bot\backend"

SYSTEM_PROMPT = f"""
# Gradious Lead Agent — Claude Code Task Prompt

## Project Overview
This is a Python-based voice agent for lead conversion at Gradious, a tech training institute.
The agent makes outbound calls to student leads, collects qualification data via a questionnaire,
answers course-related queries using a knowledge base, and writes results to Airtable.
Built with FastAPI, LangGraph, and OpenAI.

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
│   └── routes.py                     # FastAPI routes — POST /agent/init, POST /agent/turn, WS /ws/agent/{{session_id}}
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

## Tasks

### Task 1 — Entity Switching Confirmation (agent/nodes/questionnaire_node.py)

**Problem:** When a user changes a previously confirmed answer — such as switching their selected course from DSA to Full Stack, or changing training mode from online to offline — the agent silently overwrites the stored value without confirming. This can cause incorrect data.

**What to implement:**

The switchable/confirmable entities are:
- `course_interest` (e.g. "Actually, I want Full Stack, not DSA")
- `training_mode` (e.g. "Wait, I prefer offline actually")

In the LLM field extraction step (the `EXTRACT_SYSTEM_PROMPT` call inside `questionnaire_node`), add detection for when the extracted value for `course_interest` or `training_mode` **differs from the already-stored value** in `answered_fields`.

When a conflict is detected:
1. Do NOT overwrite the stored value immediately.
2. Add a new field to `LeadState` called `pending_switch` (type: `dict | None`) that stores `{{"field": "<field_name>", "new_value": "<new_value>"}}`.
3. Set `state["current_question_key"] = "confirm_switch"` and generate an LLM response asking the user to confirm the switch.
   Example style: "You had selected [old value] earlier. Did you want to switch to [new value]?"
4. Add handling in `questionnaire_node` for `current_question_key == "confirm_switch"`:
   - If user confirms (affirmative): apply the new value from `pending_switch`, clear `pending_switch`, resume normal flow by calling `_get_next_field()`.
   - If user denies: discard `pending_switch`, resume normal flow keeping the old value.
5. Add `pending_switch: Optional[dict]` to `LeadState` in `agent/state.py`.

---

### Task 2 — Multi-Intent: Answer + Query in Same Utterance (agent/nodes/intent_router.py, agent/nodes/questionnaire_node.py, agent/nodes/faq_node.py)

**Problem:** When a student says something like "Yes, also tell me about the Full Stack course", the current system routes to either `answer` or `query` — it cannot handle both in the same turn. The answer gets lost or the query goes unanswered.

**What to implement:**

**In `intent_router.py`:**
- Add a new intent: `"answer_and_query"` — returned when the student's message contains BOTH a clear answer to the pending question AND a separate question or request for information.
- Update the intent classification system prompt to describe this intent clearly with examples.
- Update the JSON output format to include a `"sub_query"` field: the isolated question part of the message (e.g. `"Tell me about the Full Stack course"`). This is `null` for all other intents.
- Update the JSON output schema to:
  ```json
  {{
    "intent": "...",
    "reasoning": "...",
    "sub_query": "<isolated query string or null>"
  }}
  ```
- Store `sub_query` in state: add `pending_sub_query: Optional[str]` to `LeadState` in `agent/state.py`.

**In `agent/graph.py`:**
- Add routing for `"answer_and_query"` → route to `"questionnaire"` node (it will handle extraction first, then hand off to FAQ).

**In `questionnaire_node.py`:**
- At the top of `questionnaire_node`, detect if `intent == "answer_and_query"`.
- If so:
  1. Run the field extraction step on the answer portion of the user's message (same extraction LLM call as normal).
  2. Store extracted fields via `_apply_extracted()`.
  3. Determine the next question using `_get_next_field()`.
  4. Do NOT generate the next question yet. Instead, store it in `state["pending_next_question_text"]` (add this field to `LeadState` as `Optional[str]`).
  5. Pass control to `faq_node` by setting `state["next_node"] = "faq_after_answer"` and returning.

**In `agent/graph.py`:**
- Add a new node reference `"faq_after_answer"` that points to the same `faq_node` function.
- Add edge: `"questionnaire"` → conditional → `"faq_after_answer"` when `state["next_node"] == "faq_after_answer"`.

**In `faq_node.py`:**
- Check if `state.get("pending_sub_query")` is set. If so, use that as the `user_query` instead of the last message content.
- Check if `state.get("pending_next_question_text")` is set. If so, use that as the `pending_question` injected at the end of the FAQ response (instead of `last_agent_response`).
- After generating the response, clear both `pending_sub_query` and `pending_next_question_text` from state.

---

### Task 3 — Knowledge Base Fixes (knowledge/fasttrack.json, knowledge/context_builders.py)

**What to fix in `fasttrack.json`:**

1. Fix the typo in `platform.lms`: remove the stray `r` at the end — `"Proprietary LMS with video lectures, quizzes, and assignmentsr"` should be `"Proprietary LMS with video lectures, quizzes, and assignments"`.

2. Add richer LMS platform details under `"platform"` to better distinguish Gradious from competitors. Add these new fields:
   ```json
   "learning_approach": "Practice-first learning — learn by doing, not just watching videos",
   "features": [
     "Hands-on coding exercises embedded in lessons",
     "Project-based learning with real-world assignments",
     "Progress tracking and performance dashboards",
     "Recorded sessions available for revision",
     "Mobile-friendly access"
   ]
   ```

3. Under `"placements"`, rename `"top_hiring_companies"` to `"partnered_companies"` for clarity — update all references in `context_builders.py` accordingly.

**What to fix in `context_builders.py`:**

1. Update `build_placement_context()` to use `partnered_companies` instead of `top_hiring_companies`.
2. Add a new context builder function `build_lms_detail_context(kb: dict) -> str` that returns a paragraph about the Leap platform's learning approach and features — used when the user specifically asks about the platform or requests more details.
3. Update `build_faq_context()` to accept an optional `include_lms_detail: bool = False` parameter. Only include the full LMS detail block when this is `True`. The base context should only include a one-line platform mention.

---

### Task 4 — Prevent Question Looping (agent/nodes/questionnaire_node.py)

**Problem:** In some scenarios, the agent asks the same question multiple times in a row — the field extraction fails silently and `_get_next_field()` returns the same field again, causing an infinite loop.

**What to fix:**

1. Add a new field `question_retry_counts: dict` to `LeadState` in `agent/state.py`. This is a dict mapping field name → number of times it has been asked. Default: {{}}.

2. In `questionnaire_node`, after calling `_get_next_field()` to determine the next field to ask:
   - Increment `state["question_retry_counts"][next_key]` by 1.
   - If `question_retry_counts[next_key]` exceeds `2` (asked more than 2 times):
     - Skip that field: mark it in `answered_fields` as `"__skipped__"` so `_get_next_field()` won't return it again.
     - Log a warning: `logger.warning(f"[Questionnaire] Skipping field {{next_key}} after 2 retries")`.
     - Call `_get_next_field()` again to get the actual next field.

3. In the field extraction step, if the extracted value for the current question key comes back as `null` or the extraction call throws an exception, do NOT re-ask the same question immediately on the next turn. Instead log the miss and proceed to try `_get_next_field()` — the retry counter above will handle retrying it up to 2 times.

---

### Task 5 — Out-of-Scope Query Handling (agent/nodes/faq_node.py)

**Problem:** The FAQ node answers general knowledge questions that are unrelated to Gradious or its offerings (e.g. "What is Machine Learning?", "Explain React"). The agent should not function as a general knowledge bot.

**What to implement:**

In `faq_node.py`, update the `FAQ_SYSTEM_PROMPT` to include a section called `## SCOPE RULES` with the following instructions:

## SCOPE RULES
You ONLY answer questions about:
- Gradious courses (Full Stack, AI/ML, DSA): content, structure, syllabus, duration, fees, modes
- Gradious platform (Leap LMS): how it works, features, access
- Gradious placements: companies, process, rates, packages
- Gradious company info: location, timings, contact
- Enrollment process and next steps

If the student asks a general knowledge question unrelated to Gradious
(e.g. "What is Machine Learning?", "Explain React", "What is DSA?"),
DO NOT answer the general definition. Instead, redirect the student toward the relevant
Gradious course naturally.

Examples of redirection:
- "What is ML?" → "We cover Machine Learning in depth in our AI Stack course — from Python basics all the way to Generative AI. Would you like to know more about that course?"
- "What is React?" → "React is one of the core topics in our Full Stack course. Want me to walk you through what the Full Stack program covers?"
- "What is DSA?" → "DSA is a focused program we offer — great for interview preparation. Want me to tell you more about it?"

Always tie the answer back to a Gradious offering.

---

### Task 6 — Conversation History Context (agent/nodes/intent_router.py, agent/nodes/questionnaire_node.py, agent/nodes/faq_node.py)

**Problem:** LLM calls in the nodes are made with only the current system prompt and latest user message. The LLM has no memory of prior conversation turns, causing it to ask questions already answered, lose track of context, and make poor decisions.

**What to implement:**

Create a helper function in `agent/nodes/questionnaire_node.py` (and import it in other nodes):

def get_recent_messages(state: LeadState, n: int = 10) -> list[dict]:
    Returns the last n messages from state["messages"] formatted as OpenAI
    chat message dicts: {{"role": "user" | "assistant", "content": "..."}}
    Skips the very last message (which is the current user input, already handled separately).

- Map `HumanMessage` → `{{"role": "user", "content": ...}}`
- Map `AIMessage` → `{{"role": "assistant", "content": ...}}`
- Return the last `n` messages excluding the most recent one (current turn).

**Apply this in the following LLM calls:**

1. **`questionnaire_node.py` — field extraction call:**
   Insert `get_recent_messages(state, 10)` between the system prompt message and the final user message in the messages list.

2. **`questionnaire_node.py` — next question generation call:**
   Insert `get_recent_messages(state, 6)` — a shorter window is enough here.

3. **`questionnaire_node.py` — confused/rude/irrelevant branch calls:**
   Insert `get_recent_messages(state, 6)`.

4. **`faq_node.py` — FAQ answer generation call:**
   Insert `get_recent_messages(state, 10)` before the final user message.

5. **`intent_router.py` — intent classification call:**
   Insert `get_recent_messages(state, 6)` — enough context to disambiguate follow-up answers from new queries.

The final message list structure for each call should always be:
```
[system_prompt, ...recent_messages, current_user_message]
```

---

### Task 7 — Remove Budget Question (agent/nodes/questionnaire_node.py, agent/state.py)

**What to change:**

1. Remove `"budget_range"` from the `ALL_FIELDS` dict in `questionnaire_node.py`.
2. Remove `"budget_range"` from the ordered field list inside `_get_next_field()`.
3. Remove `budget_range: Optional[str]` from `LeadState` in `agent/state.py`.
4. Remove `"budget_range"` from the `_apply_extracted()` mirror map.
5. Remove `"Budget Range"` from the `lead_data` dict in `airtable_node.py`.
6. In `faq_node.py`: if a user asks about fees or pricing, answer with the course fee from the knowledge base. If the user then tries to negotiate or asks for a discount, the agent should NOT negotiate. Instead, respond with something like: "For fee-related discussions and any special options, our admissions expert can help you out. Would you like me to schedule a call with them?" — this should route to the `human_agent` branch.

---

### Task 8 — Multi-Step Greeting Flow (api/routes.py, agent/nodes/questionnaire_node.py, agent/state.py)

**Problem:** The current greeting is a single static message. We need a 3-step greeting flow before the main questionnaire begins.

**What to implement:**

Add a `greeting_step` field to `LeadState` in `agent/state.py`:
```python
greeting_step: int  # 0 = not started, 1 = identity confirmed, 2 = timing confirmed, 3 = interest confirmed → proceed to Q flow
```

In `api/routes.py`, `_init_state()`:
- Set `greeting_step = 0` and `current_question_key = "confirm_identity"`.
- The initial greeting (sent immediately on call start) should be Step 1: an LLM-generated message asking "Am I speaking with {{name}}?". This already exists — keep it.

In `questionnaire_node.py`, handle the greeting steps as a state machine before the main Q flow:

**Step 1 — `confirm_identity` (already exists):**
- If user confirms → generate Step 2 response (LLM): introduce as calling from Gradious, ask "Is this a good time to speak?"
- Set `greeting_step = 1`, `current_question_key = "confirm_timing"`.

**Step 2 — `confirm_timing` (new):**
- If user says it's a good time (affirmative) → generate Step 3 response (LLM): briefly mention the student showed interest in courses, ask "Would you like to know more about our programs?"
- Set `greeting_step = 2`, `current_question_key = "confirm_interest"`.
- If user says it's NOT a good time → ask for a callback time, set `human_agent_requested = True`, proceed to callback scheduling, end call gracefully.

**Step 3 — `confirm_interest` (new):**
- If user says yes → generate brief LLM transition ("Great, let me get some details from you") → set `greeting_step = 3`, `current_question_key = "course_interest"`, proceed to normal questionnaire flow.
- If user says no → route to `end_node` with `not_interested` disposition.

All 3 step responses must be LLM-generated using `_llm_json()`. Write dedicated system prompts for Step 2 and Step 3 inline in `questionnaire_node.py` (similar to how `POST_CONFIRM_SYSTEM_PROMPT` is defined).

---

### Task 9 — LMS Mention Fix in FAQ (agent/nodes/faq_node.py, knowledge/context_builders.py, knowledge/fasttrack.json)

**Problem:** The FAQ node mentions the LMS platform (Leap) in almost every response, even when it's irrelevant. Also, users unfamiliar with "LMS" need friendlier language.

**What to change:**

1. **`faq_node.py` — `FAQ_SYSTEM_PROMPT`:**
   Replace the current LMS mention instruction with:
   ```
   ## LMS MENTION RULES
   - Do NOT mention the Leap platform in every response.
   - Only mention Leap when:
     a. The student specifically asks about the platform, portal, or learning experience.
     b. The student asks for more details about a course (after the initial brief answer).
   - When mentioning Leap, NEVER use the term "LMS". Say "our learning portal called Leap" instead.
   - When Leap is relevant, mention 1–2 specific differentiators, e.g.:
     "You'll learn by doing — Leap has hands-on coding exercises built into every lesson, not just video lectures."
   - Do not list all platform features — pick the most compelling 1–2 for the context.
   ```

2. **`faq_node.py` — add a trigger for detailed LMS context:**
   Before calling `build_faq_context()`, detect if the user query is specifically about the platform/portal/learning experience using keyword matching (words like: "platform", "portal", "leap", "lms", "learning experience", "how do I learn", "how does it work").
   If detected: call `build_faq_context(course_keys, include_lms_detail=True)`.
   Otherwise: call `build_faq_context(course_keys, include_lms_detail=False)` (default).

3. **`knowledge/fasttrack.json`:**
   Already covered in Task 3 — the `learning_approach` and `features` fields added there will power this richer context.

---

### Task 10 — Sales Persona for Agent Responses (agent/nodes/questionnaire_node.py, agent/nodes/faq_node.py, agent/nodes/end_node.py)

**Problem:** Agent responses feel robotic and transactional. They need to feel like a calm, persuasive human sales counselor who genuinely wants to help the student make the right decision.

**What to change:**

In every system prompt that generates a spoken response (questionnaire question generation, FAQ answer, end node closing, greeting steps), add a `## PERSONA` section with the following:

## PERSONA
You are Ava, a calm and friendly admissions counselor at Gradious. You speak like a real person
on a phone call — warm, clear, and genuinely helpful. Your goal is to understand the student's
situation and guide them toward the right course. You are never pushy, but you are
subtly persuasive — you highlight genuine benefits, create mild urgency where appropriate,
and always make the student feel that Gradious is the right place for their career growth.

Tone guidelines:
- Calm and confident — never rushed or scripted-sounding.
- Use natural fillers where appropriate: "Sure!", "Got it.", "Ok, so...", "Right."
- Highlight one genuine benefit or differentiator per response when opportunity arises
  (e.g. placement support, practice-based learning, industry mentors) — but don't overdo it.
- If a student seems hesitant, gently acknowledge and address the hesitation before moving on.
- Do not use corporate speak, buzzwords, or filler phrases like "Absolutely!", "Certainly!",
  "Great question!", "Definitely!".
- Speak in short sentences — this is a phone call, not an essay.

Apply this `## PERSONA` section to the following system prompts in `questionnaire_node.py`:
- `NEXT_QUESTION_SYSTEM_PROMPT`
- `GREETING_SYSTEM_PROMPT`
- `POST_CONFIRM_SYSTEM_PROMPT`
- `WRAP_UP_SYSTEM_PROMPT`
- `HUMAN_AGENT_SYSTEM_PROMPT`
- `HUMAN_AGENT_CONFIRM_SYSTEM_PROMPT`
- `DE_ESCALATE_SYSTEM_PROMPT`
- `CONFUSED_SYSTEM_PROMPT`
- `IRRELEVANT_SYSTEM_PROMPT`
- The new Step 2 (`confirm_timing`) and Step 3 (`confirm_interest`) prompts from Task 8.

Apply it to `FAQ_SYSTEM_PROMPT` in `faq_node.py`.
Apply it to `END_SYSTEM_PROMPT` in `end_node.py`.

---

## General Instructions for Claude Code

- Make all changes surgical — only modify what each task specifies. Do not refactor unrelated code.
- After all changes, verify that `LeadState` in `agent/state.py` has all newly added fields.
- After all changes, verify that `agent/graph.py` has edges/routing for all new paths introduced (e.g. `answer_and_query`, `faq_after_answer`, new greeting steps).
- Do not remove existing logging statements — add new ones where new logic is introduced.
- All new LLM calls must use `response_format={{"type": "json_object"}}` and the `openai` package directly (not LangChain wrappers).
- All new system prompts must follow the `## SECTION_NAME` heading format used in existing prompts.
- Do not change any file not mentioned in a task's scope.

"""


import pandas as pd
from datetime import datetime
import os

class CostTracker:
    def __init__(self, model_name="claude-haiku-4.5"):
        self.model = model_name
        self.tool_calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.steps = []
        self.start_time = datetime.now()

        self.input_cost_per_token = 1 / 1_000_000
        self.output_cost_per_token = 5 / 1_000_000

    def log_tool_call(self):
        self.tool_calls += 1

    def log_tokens(self, usage):
        if not usage:
            return

        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)

        self.input_tokens += in_tok
        self.output_tokens += out_tok

        self.steps.append({{
            "timestamp": datetime.now(),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "step_cost_usd": self.calculate_cost(in_tok, out_tok)
        }})

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

        df_summary = pd.DataFrame([{{
            "timestamp": datetime.now(),
            "model": self.model,
            "tool_calls": self.tool_calls,
            "total_input_tokens": self.input_tokens,
            "total_output_tokens": self.output_tokens,
            "total_cost_usd": self.total_cost(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds()
        }}])

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
    tracker = CostTracker("claude-haiku-4.5")

    async for message in query(
        prompt=SYSTEM_PROMPT,
        options=ClaudeAgentOptions(
            # model="us.anthropic.claude-haiku-4-5-20251001-v1:0",
            model="us.anthropic.claude-sonnet-4-6",
            allowed_tools=["Read", "Edit", "Bash"],
            permission_mode="acceptEdits",
            cwd = r"C:\Users\chand\Desktop\Claude doc gen"
        )
    ):

        if hasattr(message, "usage"):
            tracker.log_tokens(message.usage)

 
        if hasattr(message, "content") and message.content:
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    tracker.log_tool_call()
                    print(f"\n🔧 TOOL: {{block.name}}")
                    print(f"INPUT: {{block.input}}")

        elif hasattr(message, "result"):
            print(f"\n✅ FINAL RESULT:\n{{message.result}}")

    tracker.save_to_excel()

asyncio.run(main())