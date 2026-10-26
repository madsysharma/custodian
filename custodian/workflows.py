"""The three workflows: triage, reproduce, and onboard.

Each workflow follows the same pattern, embodying the core idea of the project:
the agent explores and proposes, deterministic code verifies and the ActionGate decides
what (if anything) reaches GitHub.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import prompts
from .agent import Agent
from .config import Settings
from .github_client import GitHub, Issue
from .jsonutil import extract_json
from .llm import ChatModel
from .policy import ActionGate
from .sandbox import CommandNotAllowed, parse_command, run_command
from .search import IssueIndex
from .tools import Toolbox, issue_tools, repo_tools, sandbox_tools
from .workspace import RepoWorkspace

IMAGE_PATTERNS = [
    re.compile(r"!\[[^\]]*\]\((https?://[^)\s]+)\)"),
    re.compile(r"<img[^>]+src=\"(https?://[^\"]+)\"", re.IGNORECASE),
]
DIFFICULTIES = {"good-first-issue", "intermediate", "advanced"}
MAX_IMAGES = 4


@dataclass
class Context:
    settings: Settings
    model: ChatModel
    github: GitHub
    issue: Issue
    comments: list[dict[str, str]]
    repo_labels: set[str]
    index: IssueIndex
    workspace: RepoWorkspace


@dataclass
class WorkflowResult:
    workflow: str
    verified: dict[str, Any]
    warnings: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    tool_log: list[dict[str, Any]] = field(default_factory=list)
    steps: int = 0


# ---------------------------------------------------------------------------------------
# Context building
# ---------------------------------------------------------------------------------------


def load_issue_index(github: GitHub, settings: Settings, refresh: bool = False) -> list[Issue]:
    cache = settings.workdir / "cache" / f"{github.repo.replace('/', '__')}_issues.json"
    if cache.exists() and not refresh:
        return [Issue(**item) for item in json.loads(cache.read_text())]
    issues = github.list_issues(max_issues=settings.max_index_issues)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps([issue.__dict__ for issue in issues]))
    return issues


def build_context(
    settings: Settings,
    model: ChatModel,
    repo: str,
    number: int,
    include_comments: bool = True,
    workspace_path: Path | None = None,
) -> Context:
    github = GitHub(repo, settings.github_token)
    issue = github.get_issue(number)
    comments = github.get_comments(number) if include_comments else []
    workspace = (
        RepoWorkspace(workspace_path) if workspace_path else RepoWorkspace.clone(repo, settings.workdir)
    )
    return Context(
        settings=settings,
        model=model,
        github=github,
        issue=issue,
        comments=comments,
        repo_labels=set(github.list_labels()),
        index=IssueIndex(load_issue_index(github, settings)),
        workspace=workspace,
    )


def image_urls(text: str) -> list[str]:
    urls: list[str] = []
    for pattern in IMAGE_PATTERNS:
        for url in pattern.findall(text):
            if url not in urls:
                urls.append(url)
    return urls[:MAX_IMAGES]


def issue_message(ctx: Context, extra: str = "", with_images: bool = True) -> list[dict[str, Any]]:
    """Build a multimodal user message: the issue text plus any screenshots it links to."""
    issue = ctx.issue
    comment_text = "\n\n".join(f"@{c['author']}: {c['body'][:1500]}" for c in ctx.comments[:10])
    text = (
        f"Repository: {ctx.github.repo}\n"
        f"Issue #{issue.number}: {issue.title}\n\n"
        f"<issue_body>\n{issue.body[:8000]}\n</issue_body>\n\n"
        + (f"<comments>\n{comment_text}\n</comments>\n\n" if comment_text else "")
        + extra
    )
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    if with_images:
        for url in image_urls(issue.body):
            content.append({"type": "image_url", "image_url": {"url": url}})
    return content


def _gate(ctx: Context, apply: bool) -> ActionGate:
    return ActionGate(ctx.github, ctx.issue.number, ctx.repo_labels, apply=apply)


# ---------------------------------------------------------------------------------------
# Triage
# ---------------------------------------------------------------------------------------


def verify_triage(raw: dict[str, Any], ctx: Context) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    labels = [label for label in raw.get("labels") or [] if isinstance(label, str)]
    kept = [label for label in labels if label in ctx.repo_labels]
    if len(kept) != len(labels):
        warnings.append(f"Dropped labels that do not exist: {sorted(set(labels) - set(kept))}")

    difficulty = raw.get("difficulty")
    if difficulty not in DIFFICULTIES:
        warnings.append(f"Unknown difficulty {difficulty!r}; set to null")
        difficulty = None

    duplicate = raw.get("duplicate_of")
    known_numbers = {issue.number for issue in ctx.index.issues}
    if duplicate is not None:
        try:
            duplicate = int(duplicate)
        except (TypeError, ValueError):
            duplicate = None
        if duplicate == ctx.issue.number or duplicate not in known_numbers:
            warnings.append(f"Discarded duplicate_of={raw.get('duplicate_of')!r}: not a known other issue")
            duplicate = None

    verified = {
        "summary": str(raw.get("summary", "")),
        "area": str(raw.get("area", "")),
        "difficulty": difficulty,
        "labels": kept,
        "duplicate_of": duplicate,
        "duplicate_reason": str(raw.get("duplicate_reason", "")) if duplicate else "",
        "missing_info": [str(q) for q in raw.get("missing_info") or []][:5],
        "screenshot_findings": str(raw.get("screenshot_findings", "")),
    }
    return verified, warnings


def triage_comment(v: dict[str, Any]) -> str:
    parts = [f"Thanks for the report! Here is a quick triage.\n\n**Summary:** {v['summary']}"]
    if v["area"]:
        parts.append(f"**Likely area:** {v['area']}")
    if v["screenshot_findings"]:
        parts.append(f"**From the screenshot:** {v['screenshot_findings']}")
    if v["duplicate_of"]:
        parts.append(
            f"**Possible duplicate of #{v['duplicate_of']}.** {v['duplicate_reason']} "
            "If that issue covers your case, please follow it there; if not, let us know what is different."
        )
    if v["missing_info"]:
        questions = "\n".join(f"- {q}" for q in v["missing_info"])
        parts.append(f"**To help maintainers act on this, could you share:**\n{questions}")
    return "\n\n".join(parts)


def triage(ctx: Context, apply: bool = False) -> WorkflowResult:
    candidates = ctx.index.search(f"{ctx.issue.title} {ctx.issue.body}", k=8, exclude=ctx.issue.number)
    candidate_text = "\n".join(f"#{i.number} [{i.state}] {i.title}" for i, _ in candidates) or "(none)"
    extra = (
        f"Existing repository labels: {sorted(ctx.repo_labels)}\n\n"
        f"BM25 duplicate candidates:\n{candidate_text}\n"
    )
    toolbox = Toolbox(repo_tools(ctx.workspace) + issue_tools(ctx.github, ctx.index, ctx.issue.number))
    agent = Agent(ctx.model, toolbox, prompts.TRIAGE, ctx.settings.max_steps)
    result = agent.run(issue_message(ctx, extra))

    try:
        raw = extract_json(result.final_text)
    except ValueError as error:
        return WorkflowResult("triage", {}, [f"Could not parse model output: {error}"], [], result.tool_log, result.steps)

    verified, warnings = verify_triage(raw, ctx)
    gate = _gate(ctx, apply)
    labels = list(verified["labels"])
    if verified["difficulty"] == "good-first-issue" and "good first issue" in ctx.repo_labels:
        labels.append("good first issue")
    if labels:
        gate.propose("add_labels", labels=sorted(set(labels)))
    gate.propose("post_comment", body=triage_comment(verified))
    return WorkflowResult("triage", verified, warnings, gate.report(), result.tool_log, result.steps)


# ---------------------------------------------------------------------------------------
# Reproduce
# ---------------------------------------------------------------------------------------

PYTEST_NOT_A_FAILURE = {2: "interrupted or collection error", 4: "usage error", 5: "no tests collected"}


def verify_reproduction(raw: dict[str, Any], ctx: Context) -> tuple[dict[str, Any], list[str]]:
    """Independently re-run the agent's test. A reproduction only counts if WE observe it fail."""
    warnings: list[str] = []
    verified = {
        "claimed": bool(raw.get("reproduced")),
        "reproduced": False,
        "test_path": str(raw.get("test_path", "")),
        "command": str(raw.get("command", "")),
        "explanation": str(raw.get("explanation", "")),
        "failure_excerpt": "",
        "test_code": "",
    }
    if not verified["claimed"]:
        return verified, warnings

    path, command = verified["test_path"], verified["command"]
    if not path or not ctx.workspace.exists(path):
        warnings.append("Agent claimed a reproduction, but the test file does not exist")
        return verified, warnings
    if path not in command:
        warnings.append("The command does not run the agent's test file")
        return verified, warnings
    try:
        argv = parse_command(command, ctx.settings.allowed_commands)
    except CommandNotAllowed as error:
        warnings.append(f"Command rejected: {error}")
        return verified, warnings

    run = run_command(ctx.workspace.root, command, ctx.settings.allowed_commands, ctx.settings.command_timeout)
    is_pytest = "pytest" in argv
    if run.timed_out or run.returncode in (0, 127):
        warnings.append(f"Re-run did not show a failure ({'timeout' if run.timed_out else f'exit {run.returncode}'})")
    elif is_pytest and run.returncode in PYTEST_NOT_A_FAILURE:
        warnings.append(f"pytest exit {run.returncode}: {PYTEST_NOT_A_FAILURE[run.returncode]}, not a real failure")
    elif any(marker in run.output for marker in ("ModuleNotFoundError", "ImportError while importing")):
        warnings.append("The test fails on an import error, which does not reproduce the bug")
    else:
        verified["reproduced"] = True
        verified["failure_excerpt"] = run.output[-1500:]
        verified["test_code"] = ctx.workspace.resolve(path).read_text()[:4000]
    return verified, warnings


def reproduce(ctx: Context, apply: bool = False) -> WorkflowResult:
    ctx.workspace.reset()
    tools, _runs = sandbox_tools(ctx.workspace, ctx.settings.allowed_commands, ctx.settings.command_timeout)
    toolbox = Toolbox(repo_tools(ctx.workspace) + tools)
    allowed = ", ".join(" ".join(c) for c in ctx.settings.allowed_commands)
    agent = Agent(ctx.model, toolbox, prompts.REPRODUCE, ctx.settings.max_steps)
    result = agent.run(issue_message(ctx, f"Allowed test commands start with: {allowed}\n"))

    try:
        raw = extract_json(result.final_text)
    except ValueError as error:
        ctx.workspace.reset()
        return WorkflowResult("reproduce", {}, [f"Could not parse model output: {error}"], [], result.tool_log, result.steps)

    verified, warnings = verify_reproduction(raw, ctx)
    gate = _gate(ctx, apply)
    if verified["reproduced"]:
        body = (
            f"I was able to reproduce this with a minimal failing test.\n\n{verified['explanation']}\n\n"
            f"<details><summary><code>{verified['test_path']}</code></summary>\n\n"
            f"```\n{verified['test_code']}\n```\n</details>\n\n"
            f"Run it with `{verified['command']}`. Key output:\n\n```\n{verified['failure_excerpt'][-800:]}\n```"
        )
        gate.propose("post_comment", body=body)
    ctx.workspace.reset()
    return WorkflowResult("reproduce", verified, warnings, gate.report(), result.tool_log, result.steps)


# ---------------------------------------------------------------------------------------
# Onboard
# ---------------------------------------------------------------------------------------

_PATHISH = re.compile(r"`([\w./-]+\.[A-Za-z0-9]{1,6}|[\w.-]+/[\w./-]+)`")


def verify_onboarding(raw: dict[str, Any], ctx: Context) -> tuple[dict[str, Any], list[str]]:
    """Drop any file the agent mentions that does not actually exist (hallucination check)."""
    warnings: list[str] = []
    files = []
    for item in raw.get("files") or []:
        path = str(item.get("path", "")) if isinstance(item, dict) else ""
        if path and ctx.workspace.exists(path):
            files.append({"path": path, "why": str(item.get("why", ""))})
        else:
            warnings.append(f"Removed non-existent file from guide: {path!r}")

    def scrub(text: str) -> str:
        def check(match: re.Match[str]) -> str:
            if ctx.workspace.exists(match.group(1)):
                return match.group(0)
            warnings.append(f"Unverified path in text: {match.group(1)!r}")
            return match.group(1) + " (unverified)"

        return _PATHISH.sub(check, text)

    verified = {
        "overview": scrub(str(raw.get("overview", ""))),
        "files": files,
        "steps": [scrub(str(s)) for s in raw.get("steps") or []][:8],
        "test_command": str(raw.get("test_command", "")),
    }
    return verified, warnings


def onboarding_comment(v: dict[str, Any]) -> str:
    files = "\n".join(f"- `{f['path']}`: {f['why']}" for f in v["files"])
    steps = "\n".join(f"{i}. {s}" for i, s in enumerate(v["steps"], 1))
    parts = ["### New to this codebase? Here is a starting map for this issue", v["overview"]]
    if files:
        parts.append(f"**Files to read first**\n{files}")
    if steps:
        parts.append(f"**Getting started**\n{steps}")
    if v["test_command"]:
        parts.append(f"**Run the relevant tests:** `{v['test_command']}`")
    return "\n\n".join(parts)


def onboard(ctx: Context, apply: bool = False) -> WorkflowResult:
    toolbox = Toolbox(repo_tools(ctx.workspace))
    agent = Agent(ctx.model, toolbox, prompts.ONBOARD, ctx.settings.max_steps)
    result = agent.run(issue_message(ctx, with_images=False))
    try:
        raw = extract_json(result.final_text)
    except ValueError as error:
        return WorkflowResult("onboard", {}, [f"Could not parse model output: {error}"], [], result.tool_log, result.steps)
    verified, warnings = verify_onboarding(raw, ctx)
    gate = _gate(ctx, apply)
    gate.propose("post_comment", body=onboarding_comment(verified))
    return WorkflowResult("onboard", verified, warnings, gate.report(), result.tool_log, result.steps)


WORKFLOWS = {"triage": triage, "reproduce": reproduce, "onboard": onboard}
