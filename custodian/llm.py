"""A thin wrapper around the OpenAI-compatible client so the rest of the code
does not rely on the SDK's response objects (allows for easy testing)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import Settings


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # Raw JSON string, exactly as the model produced it


@dataclass
class ChatMessage:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)

    def to_history(self) -> dict[str, Any]:
        """Serialize the assistant turn so it can be sent back in the next request."""
        message: dict[str, Any] = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        return message


class ChatModel(Protocol):
    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> ChatMessage: ...


class MuseClient:
    """Calls Muse Spark (Meta Model API) or Muse Spark Open (Baseten) with retries."""

    def __init__(self, settings: Settings, retries: int = 3):
        from openai import OpenAI  # Lazy import so tests do not need the SDK

        settings.require_api_key()
        self._client = OpenAI(base_url=settings.base_url, api_key=settings.api_key)
        self.model = settings.model
        self.retries = retries

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> ChatMessage:
        kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = self._client.chat.completions.create(**kwargs)
                break
            except Exception as error:  # Rate limits and transient network errors
                last_error = error
                time.sleep(2**attempt)
        else:
            raise RuntimeError(f"Model call failed after {self.retries} attempts") from last_error

        message = response.choices[0].message
        calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (message.tool_calls or [])
        ]
        return ChatMessage(content=message.content, tool_calls=calls)
