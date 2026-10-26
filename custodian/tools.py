"""Tools the agent can call. All of them are read-only or local to the sandbox checkout."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .github_client import GitHub
from .sandbox import RunResult, run_command
from .search import IssueIndex
from .workspace import RepoWorkspace

MAX_TOOL_OUTPUT = 8000


@dataclass
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Toolbox:
    def __init__(self, tools: list[Tool]):
        self.tools = {tool.name: tool for tool in tools}
        self.log: list[dict[str, Any]] = []

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self.tools.values()]

    def call(self, name: str, raw_arguments: str) -> str:
        tool = self.tools.get(name)
        if tool is None:
            output = f"ERROR: unknown tool '{name}'. Available: {sorted(self.tools)}"
        else:
            try:
                args = json.loads(raw_arguments or "{}")
                if not isinstance(args, dict):
                    raise ValueError("arguments must be a JSON object")
                output = str(tool.fn(**args))
            except Exception as error:  # Report errors to the model so it can recover
                output = f"ERROR: {type(error).__name__}: {error}"
        if len(output) > MAX_TOOL_OUTPUT:
            output = output[:MAX_TOOL_OUTPUT] + "\n... (output truncated)"
        self.log.append({"tool": name, "arguments": raw_arguments, "output": output[:500]})
        return output


def _obj(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


def repo_tools(ws: RepoWorkspace) -> list[Tool]:
    return [
        Tool(
            "list_files",
            "List files in the repository, optionally under a subdirectory and filtered by a glob on the file name.",
            _obj(
                {
                    "subdir": {"type": "string", "description": "Directory relative to repo root. Default '.'"},
                    "pattern": {"type": "string", "description": "Glob such as '*.py'. Default '*'"},
                },
                [],
            ),
            ws.list_files,
        ),
        Tool(
            "read_file",
            "Read a file with line numbers. Reads at most 400 lines per call.",
            _obj(
                {
                    "path": {"type": "string"},
                    "start": {"type": "integer", "description": "First line, 1-based"},
                    "end": {"type": "integer", "description": "Last line, inclusive"},
                },
                ["path"],
            ),
            ws.read_file,
        ),
        Tool(
            "grep",
            "Search file contents with a Python regular expression. Returns path:line: text.",
            _obj(
                {
                    "pattern": {"type": "string"},
                    "glob": {"type": "string", "description": "Restrict to file names matching this glob"},
                },
                ["pattern"],
            ),
            ws.grep,
        ),
    ]


def issue_tools(github: GitHub, index: IssueIndex, current: int) -> list[Tool]:
    def search_issues(query: str, k: int = 5) -> str:
        hits = index.search(query, k=min(k, 10), exclude=current)
        if not hits:
            return "(no similar issues)"
        return "\n".join(
            f"#{issue.number} [{issue.state}] {issue.title} (score {score:.1f})" for issue, score in hits
        )

    def get_issue(number: int) -> str:
        issue = github.get_issue(number)
        return f"#{issue.number} [{issue.state}] {issue.title}\nLabels: {issue.labels}\n\n{issue.body[:4000]}"

    return [
        Tool(
            "search_issues",
            "Keyword search over this repository's recent issues to find possible duplicates.",
            _obj({"query": {"type": "string"}, "k": {"type": "integer"}}, ["query"]),
            search_issues,
        ),
        Tool(
            "get_issue",
            "Fetch the full title and body of another issue by number.",
            _obj({"number": {"type": "integer"}}, ["number"]),
            get_issue,
        ),
    ]


def sandbox_tools(ws: RepoWorkspace, allowed: list[list[str]], timeout: int) -> tuple[list[Tool], list[RunResult]]:
    runs: list[RunResult] = []

    def run_tests(command: str) -> str:
        result = run_command(ws.root, command, allowed, timeout)
        runs.append(result)
        return result.summary()

    tools = [
        Tool(
            "write_test_file",
            "Create or overwrite a TEST file in the local sandbox checkout. Nothing is pushed to GitHub.",
            _obj({"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]),
            ws.write_test_file,
        ),
        Tool(
            "run_tests",
            "Run an allow-listed test command in the sandbox, e.g. 'pytest tests/test_bug.py -x -q'.",
            _obj({"command": {"type": "string"}}, ["command"]),
            run_tests,
        ),
    ]
    return tools, runs
