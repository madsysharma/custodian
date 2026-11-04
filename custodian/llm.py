"""A thin wrapper around the OpenAI-compatible client so the rest of the code
does not depend on the SDK's response objects (which also makes testing easy)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from .config import Settings

# Fireworks rejects non-streaming requests whose max_tokens is above this value,
# so larger requests are streamed and reassembled into a single message.
FIREWORKS_NON_STREAMING_LIMIT = 4096


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model produced it


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
    """Calls Muse Spark (Meta Model API) or Muse Spark Open (Fireworks) with retries."""

    def __init__(self, settings: Settings, retries: int = 3):
        from openai import OpenAI  # imported lazily so tests do not need the SDK

        settings.require_api_key()
        self._client = OpenAI(base_url=settings.base_url, api_key=settings.api_key)
        self.model = settings.model
        self.max_tokens = settings.max_tokens
        self.retries = retries
        self.stream = (
            settings.provider == "fireworks" and settings.max_tokens > FIREWORKS_NON_STREAMING_LIMIT
        )
        # Sent in the request body so it works regardless of the installed SDK version.
        self.extra_body: dict[str, Any] = (
            {"reasoning_effort": settings.reasoning_effort} if settings.reasoning_effort else {}
        )

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> ChatMessage:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            # Without a cap, some providers reserve the model's full output limit (131,072
            # tokens) per request and reject it if the account cannot cover that worst case.
            "max_tokens": self.max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        if self.extra_body:
            kwargs["extra_body"] = self.extra_body

        last_error: Exception | None = None
        for attempt in range(self.retries):
            try:
                content, calls, finish_reason = (
                    self._streamed(kwargs) if self.stream else self._single(kwargs)
                )
                break
            except Exception as error:
                last_error = error
                if not _is_retryable(error):
                    # Billing, authentication and bad-request errors will not fix themselves.
                    raise RuntimeError(f"Model call failed: {error}") from error
                time.sleep(2**attempt)
        else:
            raise RuntimeError(
                f"Model call failed after {self.retries} attempts: {last_error}"
            ) from last_error

        if finish_reason == "length" and not calls and not (content or "").strip():
            # Reasoning and the answer share max_tokens; the model ran out while still thinking.
            raise RuntimeError(
                f"The model used all {self.max_tokens} output tokens on reasoning and returned no "
                "answer. Raise CUST_MAX_TOKENS or lower CUST_REASONING_EFFORT."
            )
        return ChatMessage(content=content, tool_calls=calls)

    def _single(self, kwargs: dict[str, Any]) -> tuple[str | None, list[ToolCall], str | None]:
        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments or "{}")
            for tc in (choice.message.tool_calls or [])
        ]
        return choice.message.content, calls, choice.finish_reason

    def _streamed(self, kwargs: dict[str, Any]) -> tuple[str | None, list[ToolCall], str | None]:
        """Stream a response and rebuild the text and tool calls from the chunks.

        Tool calls arrive in pieces: the first chunk for each call carries its index, id and
        name, and later chunks with the same index append to its JSON arguments.
        """
        text: list[str] = []
        partial: dict[int, dict[str, Any]] = {}
        finish_reason: str | None = None
        for chunk in self._client.chat.completions.create(**kwargs, stream=True):
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta
            if delta is None:
                continue
            if delta.content:
                text.append(delta.content)
            for tc in delta.tool_calls or []:
                slot = partial.setdefault(tc.index, {"id": "", "name": "", "arguments": []})
                if tc.id:
                    slot["id"] = tc.id
                if tc.function is not None:
                    if tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function.arguments:
                        slot["arguments"].append(tc.function.arguments)
        calls = [
            ToolCall(
                id=slot["id"] or f"call_{index}",
                name=slot["name"],
                arguments="".join(slot["arguments"]) or "{}",
            )
            for index, slot in sorted(partial.items())
        ]
        return ("".join(text) or None), calls, finish_reason


def _is_retryable(error: Exception) -> bool:
    """Retry rate limits, timeouts and server errors; fail fast on everything else."""
    status = getattr(error, "status_code", None)
    if status is None:
        return True  # network errors and timeouts have no status code
    return status in (408, 409, 429) or status >= 500