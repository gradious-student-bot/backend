import json
from openai import OpenAI


from config import (
    OPENAI_API_KEY, 
    OPENAI_THINK_MODEL,
    OPENAI_MODEL, 
    OPENAI_FAST_MODEL
)

class LLM_Client:
    def __init__(self, model_name: str = OPENAI_MODEL):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model_name = model_name

    def invoke(self, messages: list[dict[str, str]]) -> str:
        """
        Calls the LLM with a list of messages and returns the string response.
        """
        resp = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.6,
            messages=messages
        )

        return resp.choices[0].message.content

    def invoke_json(self, messages: list[dict[str, str]]) -> dict:
        """
        Calls the LLM with a list of messages and returns the parsed JSON response.
        """
        resp = self.client.chat.completions.create(
            model=self.model_name,
            temperature=0.4,
            response_format={"type": "json_object"},
            messages=messages,
        )
        return json.loads(resp.choices[0].message.content)