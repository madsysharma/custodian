"""ActionGate is the only path from the agent to GitHub.

The agent never holds write access. It returns a structured proposal, deterministic code
verifies it and the gate either records the action for a maintainer (dry run, the default)
or executes it (``--apply``). Closing, merging, assigning and deleting are not possible at all.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .github_client import GitHub

ALLOWED_ACTIONS = {"post_comment", "add_labels"}
MAX_COMMENT_CHARS = 6000
MAX_LABELS = 4


@dataclass
class Action:
    kind: str
    args: dict[str, Any]
    status: str = "proposed"  # proposed | executed | blocked | failed
    detail: str = ""


@dataclass
class ActionGate:
    github: GitHub | None
    issue_number: int
    repo_labels: set[str]
    apply: bool = False
    actions: list[Action] = field(default_factory=list)

    def check(self, kind: str, args: dict[str, Any]) -> str | None:
        """Return a reason to block the action, or None if it is allowed."""
        if kind not in ALLOWED_ACTIONS:
            return f"'{kind}' is not an allowed action"
        if kind == "add_labels":
            labels = args.get("labels") or []
            if not labels:
                return "No labels given"
            if len(labels) > MAX_LABELS:
                return f"At most {MAX_LABELS} labels per issue"
            unknown = [label for label in labels if label not in self.repo_labels]
            if unknown:
                return f"Labels do not exist in the repository: {unknown}"
        if kind == "post_comment":
            body = (args.get("body") or "").strip()
            if not body:
                return "Empty comment"
            if len(body) > MAX_COMMENT_CHARS:
                return f"Comment longer than {MAX_COMMENT_CHARS} characters"
        return None

    def propose(self, kind: str, **args: Any) -> Action:
        action = Action(kind=kind, args=args)
        reason = self.check(kind, args)
        if reason:
            action.status, action.detail = "blocked", reason
        elif self.apply and self.github is not None:
            try:
                if kind == "post_comment":
                    action.detail = self.github.post_comment(self.issue_number, args["body"])
                else:
                    self.github.add_labels(self.issue_number, args["labels"])
                    action.detail = "labels added"
                action.status = "executed"
            except Exception as error:  # Surface API errors without crashing the run
                action.status, action.detail = "failed", str(error)
        self.actions.append(action)
        return action

    def report(self) -> list[dict[str, Any]]:
        return [asdict(a) for a in self.actions]
