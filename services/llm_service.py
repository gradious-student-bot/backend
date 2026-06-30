import json
import logging
from openai import OpenAI

from agent.state import LeadState
from config import (
    OPENAI_API_KEY, 
    OPENAI_THINK_MODEL,
    OPENAI_MODEL, 
    OPENAI_FAST_MODEL
)

logger = logging.getLogger(__name__)

class LLM_Client:
    def __init__(self, model_name: str = OPENAI_MODEL):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name

    def invoke(self, messages: list[dict[str, str]]) -> str:
        """
        Calls the LLM with a list of messages and returns the string response.
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.6,
                messages=messages
            )

            return resp.choices[0].message.content
        except Exception as e:
            logger.exception("Exception occured while making LLM call", exc_info=True)
            return ""

    def invoke_json(self, messages: list[dict[str, str]]) -> dict[str, str]:
        """
        Calls the LLM with a list of messages and returns the parsed JSON response.
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model_name,
                temperature=0.4,
                response_format={"type": "json_object"},
                messages=messages,
            )
            return json.loads(resp.choices[0].message.content)
        except json.JSONDecodeError:
            logger.exception("Exception while decoding the JSON response from LLM", exc_info=True)
            return {}
    
    @staticmethod
    def get_recent_messages(state: LeadState, n: int = 10) -> list[dict[str, str]]:
        """
        Returns the last n messages from state["messages"] formatted as OpenAI
        chat message dicts: 
        
        `{"role": "user"|"assistant", "content": "..."}`.
        
        Skips the very last message (which is the current user input, already handled separately).
        """
        from langchain_core.messages import HumanMessage, AIMessage
        
        msgs = state.get("messages", [])
        history = msgs[:-1] if len(msgs) > 1 else []
        result: list[dict[str, str]] = []

        for m in history[-n:]:

            if isinstance(m, HumanMessage):
                result.append({"role": "user", "content": m.content})

            elif isinstance(m, AIMessage):
                result.append({"role": "assistant", "content": m.content})
        return result

llm = LLM_Client()