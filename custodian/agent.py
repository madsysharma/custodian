"""A small, transparent tool-calling loop."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .llm import ChatModel
from .tools import Toolbox

FINALIZE = (
    "You have used your step budget. Do not call any more tools. "
    "Reply now with your final answer in the required JSON format."
)


@dataclass
class AgentResult:
    final_text: str
    steps: int
    tool_log: list[dict[str, Any]] = field(default_factory=list)


class Agent:
    def __init__(self, model: ChatModel, toolbox: Toolbox, system_prompt: str, max_steps: int = 15):
        self.model = model
        self.toolbox = toolbox
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    def run(self, user_content: str | list[dict[str, Any]]) -> AgentResult:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        tools = self.toolbox.schemas()
        for step in range(1, self.max_steps + 1):
            reply = self.model.chat(messages, tools=tools)
            messages.append(reply.to_history())
            if not reply.tool_calls:
                return AgentResult(reply.content or "", step, self.toolbox.log)
            for call in reply.tool_calls:
                output = self.toolbox.call(call.name, call.arguments)
                messages.append({"role": "tool", "tool_call_id": call.id, "content": output})
        messages.append({"role": "user", "content": FINALIZE})
        reply = self.model.chat(messages, tools=None)
        return AgentResult(reply.content or "", self.max_steps + 1, self.toolbox.log)
