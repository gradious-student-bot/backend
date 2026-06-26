# Gradious Lead Bot Backend

A conversational AI backend for Gradious admissions and lead qualification. The system uses a stateful LangGraph workflow to talk with leads, answer course-related questions, collect qualification details, and route interesting prospects to the right next step.

## Overview

This project powers an intelligent lead bot that can:
- greet and qualify incoming leads,
- answer FAQ-style questions about Gradious courses and training programs,
- detect intent such as repetition requests, disinterest, rude/irrelevant input, or human-agent requests,
- store lead and call information in Airtable,
- send onboarding emails,
- support both web chat and voice-driven calling workflows.

## Main Features

### 1. Stateful lead conversation agent
The agent uses a LangGraph-based workflow with specialized nodes for:
- intent routing,
- questionnaire-based lead qualification,
- FAQ answering,
- repeating the last question,
- ending the call gracefully,
- Airtable logging.

This gives the bot the ability to maintain context across a full conversation rather than treating each message independently.

### 2. Intelligent lead qualification
The questionnaire flow collects structured information such as:
- course interest,
- student status,
- current or passout year,
- department,
- preferred training mode,
- interest in joining,
- callback requests,
- onboarding preferences.

### 3. FAQ and course guidance
The FAQ node answers Gradious-specific questions using an LLM and a knowledge base built from course and program information. It is designed to:
- reply in a natural spoken style,
- stay within the supported scope,
- redirect out-of-scope requests back to Gradious offerings,
- trigger a human handoff for sensitive or unsupported topics.

### 4. Human agent escalation
The bot can detect cases where a live admissions expert should take over, including:
- fee negotiation requests,
- detailed placement questions,
- unsupported or unclear topics,
- callback requests.

### 5. Airtable integration
Lead and call metrics are written to Airtable for reporting and follow-up. The backend includes dedicated clients for:
- lead record creation,
- call/metrics logging.

### 6. Onboarding email support
When the conversation indicates that a lead wants onboarding information, the system can send an onboarding email using SMTP.

### 7. Web and voice interfaces
The backend includes:
- a FastAPI service for REST and WebSocket interactions,
- a chat UI template,
- voice integration modules for LiveKit and calling workflows.

## Architecture

The project is organized into these main areas:

- app entry point: main.py
- API layer: api/routes.py
- agent workflow: agent/
- knowledge base and context builders: knowledge/
- LLM integration: llm/
- Airtable and email services: services/
- voice calling integration: voice/
- chat UI template: templates/

## Tech Stack

- Python
- FastAPI
- LangGraph
- OpenAI / LLM-based conversation logic
- Pydantic
- Airtable API
- LiveKit / voice calling integration
- SMTP email sending

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables
Create a .env file and set values for:

```env
OPENAI_API_KEY=your_openai_key
AIRTABLE_TOKEN=your_airtable_token
AIRTABLE_BASE_ID=your_airtable_base_id
AIRTABLE_LEAD_TABLE=Lead_Table
AIRTABLE_METRICS_TABLE=Lead_metrics
SMTP_EMAIL=your_email
SMTP_PASSWORD=your_email_password
```

Optional voice-related environment values may also be required for the LiveKit and calling workflow.

### 3. Run the backend

```bash
python main.py
```

or with uvicorn:

```bash
uvicorn main:app --reload
```

### 4. Access the service
- Health check: /health
- Chat UI: /chat
- Agent init endpoint: /agent/init
- Agent turn endpoint: /agent/turn
- WebSocket endpoint: /ws/agent/{session_id}

## Typical Conversation Flow

1. The user starts a session and receives an initial greeting.
2. The agent routes the message to either:
   - a questionnaire node for lead qualification,
   - an FAQ node for course-related questions,
   - a repeat node for clarification,
   - an end node when the conversation concludes.
3. The bot stores the updated conversation state and logs the outcome.
4. If the lead is interested, the system can proceed with onboarding or callback handling.

## Notes

This backend is designed as a flexible conversational assistant for admissions and lead follow-up. It is suitable for both chat-based and voice-based deployment scenarios.
