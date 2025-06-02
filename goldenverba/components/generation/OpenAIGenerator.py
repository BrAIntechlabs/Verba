import os
from dotenv import load_dotenv
from goldenverba.components.interfaces import Generator
from goldenverba.components.types import InputConfig
from goldenverba.components.util import get_environment, get_token
from typing import List
import httpx
import json
from wasabi import msg

load_dotenv()


class OpenAIGenerator(Generator):
    """
    OpenAI Generator.
    """

    def __init__(self):
        super().__init__()
        self.name = "OpenAI"
        self.description = "Using OpenAI LLM models to generate answers to queries"
        self.context_window = 10000

        api_key = get_token("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        models = self.get_models(api_key, base_url)
        default_model = os.getenv("OPENAI_MODEL", models[0])

        self.config["Model"] = InputConfig(
            type="dropdown",
            value=default_model,
            description="Select an OpenAI Model",
            values=models,
        )

        self.config["System Message"] = InputConfig(
            type="textarea",
            value=(
                "You are a professional and concise customer support assistant for an online service.\n\n"
                "Your primary goals are:\n"
                "1. Provide clear, direct, and helpful answers.\n"
                "2. Avoid unnecessary details or filler phrases.\n"
                "3. Use simple language, even for technical topics.\n"
                "4. Give examples when it helps clarify the solution.\n"
                "5. Never make up information. Say 'I don't know' if unsure.\n\n"
                "Examples:\n"
                "User: How can I reset my password?\n"
                "Assistant: Click on 'Forgot password' at login and follow the instructions.\n\n"
                "User: I want to cancel my subscription.\n"
                "Assistant: Go to 'Account Settings' → 'Billing' → click 'Cancel Subscription'.\n\n"
                "If the user is frustrated, stay calm and offer solutions politely.\n"
                "Always stay on topic and respond in a maximum of 2 to 3 short sentences."
            ),
            description="System prompt that defines the assistants behavior and tone.",
            values=[],
        )


        if get_token("OPENAI_API_KEY") is None:
            self.config["API Key"] = InputConfig(
                type="password",
                value="",
                description="You can set your OpenAI API Key here or set it as environment variable `OPENAI_API_KEY`",
                values=[],
            )
        if os.getenv("OPENAI_BASE_URL") is None:
            self.config["URL"] = InputConfig(
                type="text",
                value="https://api.openai.com/v1",
                description="You can change the Base URL here if needed",
                values=[],
            )

    async def generate_stream(
        self,
        config: dict,
        query: str,
        context: str,
        conversation: list[dict] = [],
    ):
        system_message = config.get("System Message").value
        model = config.get("Model", {"value": "gpt-3.5-turbo"}).value
        openai_key = get_environment(
            config, "API Key", "OPENAI_API_KEY", "No OpenAI API Key found"
        )
        openai_url = get_environment(
            config, "URL", "OPENAI_BASE_URL", "https://api.openai.com/v1"
        )

        messages = self.prepare_messages(query, context, conversation, system_message)

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {openai_key}",
        }
        data = {
            "messages": messages,
            "model": model,
            "stream": True,
            "temperature": 0.1,
            "max_tokens": 300
        }

        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                f"{openai_url}/chat/completions",
                json=data,
                headers=headers,
                timeout=None,
            ) as response:
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        if line.strip() == "data: [DONE]":
                            break
                        json_line = json.loads(line[6:])
                        choice = json_line["choices"][0]
                        if "delta" in choice and "content" in choice["delta"]:
                            yield {
                                "message": choice["delta"]["content"],
                                "finish_reason": choice.get("finish_reason"),
                            }
                        elif "finish_reason" in choice:
                            yield {
                                "message": "",
                                "finish_reason": choice["finish_reason"],
                            }

    def prepare_messages(
        self, query: str, context: str, conversation: list[dict], system_message: str
    ) -> list[dict]:
        messages = [
            {
                "role": "system",
                "content": system_message,
            }
        ]

        for message in conversation:
            messages.append({"role": message.type, "content": message.content})

        messages.append(
            {
                "role": "user",
                "content": f"Answer this query: '{query}' with this provided context: {context}",
            }
        )

        return messages

    def get_models(self, token: str, url: str) -> List[str]:
        """Fetch available embedding models from OpenAI API."""
        default_models = ["gpt-4o", "gpt-3.5-turbo"]
        try:
            if token is None:
                return default_models

            import requests

            headers = {"Authorization": f"Bearer {token}"}
            response = requests.get(f"{url}/models", headers=headers)
            response.raise_for_status()
            return [
                model["id"]
                for model in response.json()["data"]
                if not "embedding" in model["id"]
            ]
        except Exception as e:
            msg.info(f"Failed to fetch OpenAI models: {str(e)}")
            return default_models
