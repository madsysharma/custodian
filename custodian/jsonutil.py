"""Helpers for extracting a JSON object out of a model's final answer."""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def extract_json(text: str | None) -> dict[str, Any]:
    """Return the first JSON object found in ``text``.

    The function first looks for a fenced ```json block and then falls back to the
    first balanced ``{...}`` span. It raises ``ValueError`` if nothing parses.
    """
    if not text:
        raise ValueError("Empty model output")
    for match in _FENCE.finditer(text):
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : index + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("No JSON object found in model output")
