import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from custodian.config import Settings  # noqa: E402
from custodian.github_client import Issue  # noqa: E402
from custodian.llm import ChatMessage, ToolCall  # noqa: E402
from custodian.search import IssueIndex  # noqa: E402
from custodian.workflows import Context  # noqa: E402
from custodian.workspace import RepoWorkspace  # noqa: E402


class ScriptedModel:
    """Replays a fixed list of assistant turns and records what it was sent."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def chat(self, messages, tools=None):
        self.requests.append({"messages": list(messages), "tools": tools})
        return self.turns.pop(0)


def call(name, **args):
    return ChatMessage(content=None, tool_calls=[ToolCall(id=f"c_{name}", name=name, arguments=json.dumps(args))])


def final(obj):
    return ChatMessage(content="Here you go:\n```json\n" + json.dumps(obj) + "\n```")


class FakeGitHub:
    repo = "octo/demo"

    def __init__(self, issues):
        self.issues = {i.number: i for i in issues}

    def get_issue(self, number):
        return self.issues[number]


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "demo"
    (root / "demo").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "demo" / "__init__.py").write_text("")
    (root / "demo" / "mathx.py").write_text(
        "def clamp(x, lo, hi):\n    # BUG: upper bound is never applied\n    return max(x, lo)\n"
    )
    (root / "tests" / "__init__.py").write_text("")
    (root / "tests" / "test_existing.py").write_text(
        "from demo.mathx import clamp\n\ndef test_low():\n    assert clamp(-1, 0, 5) == 0\n"
    )
    return RepoWorkspace(root)


ISSUES = [
    Issue(10, "clamp ignores upper bound", "clamp(9, 0, 5) returns 9 ![shot](https://img.example/a.png)", "open"),
    Issue(4, "clamp does not respect the hi argument", "clamp returns values above hi", "closed"),
    Issue(7, "Add docs for install", "The README lacks install steps", "open"),
]


@pytest.fixture
def make_ctx(repo):
    def _make(model):
        settings = Settings(
            allowed_commands=[[sys.executable, "-m", "pytest"]], command_timeout=60, max_steps=6
        )
        github = FakeGitHub(ISSUES)
        return Context(
            settings=settings,
            model=model,
            github=github,
            issue=ISSUES[0],
            comments=[],
            repo_labels={"bug", "math", "good first issue", "docs"},
            index=IssueIndex(ISSUES),
            workspace=repo,
        )

    return _make
