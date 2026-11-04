"""The three workflows: triage, reproduce, and onboard.

Each workflow follows the same pattern, which is the core idea of the project:
the agent explores and proposes, deterministic code verifies, and the ActionGate decides
what (if anything) reaches GitHub.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import prompts
from .agent import Agent, AgentResult
from .config import Settings
from .failures import PYTEST_VERIFY_FLAGS, classify
from .github_client import GitHub, Issue
from .images import fetch_image_data_url
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
MAX_TRIAGE_LABELS = 3


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
    label_descriptions: dict[str, str] = field(default_factory=dict)


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
    extra_issue_numbers: list[int] | None = None,
) -> Context:
    """Build everything a workflow needs for one issue.

    ``extra_issue_numbers`` adds specific issues to the duplicate-search index. The evaluation
    uses it so an older original issue is searchable, as it would be in a deployment that
    indexes a repository's full history rather than only its most recent issues.
    """
    github = GitHub(repo, settings.github_token)
    labels = github.list_labels_with_descriptions()
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
        repo_labels=set(labels),
        index=IssueIndex(_with_extra_issues(github, load_issue_index(github, settings), extra_issue_numbers)),
        workspace=workspace,
        label_descriptions=labels,
    )


def describe_labels(ctx: Context) -> str:
    """One line per label, with the maintainers' description when there is one."""
    lines = []
    for name in sorted(ctx.repo_labels):
        description = ctx.label_descriptions.get(name, "")
        lines.append(f"- {name}: {description}" if description else f"- {name}")
    return "\n".join(lines)


def _with_extra_issues(github: GitHub, issues: list[Issue], numbers: list[int] | None) -> list[Issue]:
    known = {issue.number for issue in issues}
    extra = []
    for number in numbers or []:
        if number not in known:
            try:
                extra.append(github.get_issue(number))
            except Exception:  # a deleted or transferred issue simply stays out of the index
                continue
    return issues + extra


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
            data_url = fetch_image_data_url(url, ctx.settings.github_token)
            if data_url:
                content.append({"type": "image_url", "image_url": {"url": data_url}})
    return content


def run_with_image_fallback(agent: Agent, ctx: Context, extra: str) -> tuple[AgentResult, list[str]]:
    """Run with screenshots, and retry from the text alone if the provider rejects an image."""
    message = issue_message(ctx, extra)
    if len(message) == 1:  # no usable images
        return agent.run(message), []
    try:
        return agent.run(message), []
    except RuntimeError as error:
        if "image" not in str(error).lower():
            raise
        agent.toolbox.log.clear()
        return agent.run(issue_message(ctx, extra, with_images=False)), [
            "The provider could not read the issue's images, so triage used the text only"
        ]


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
    if len(kept) > MAX_TRIAGE_LABELS:
        warnings.append(f"Kept only the first {MAX_TRIAGE_LABELS} labels: dropped {kept[MAX_TRIAGE_LABELS:]}")
        kept = kept[:MAX_TRIAGE_LABELS]

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
        "Existing repository labels, with the maintainers' descriptions of when to use each one.\n"
        "Follow these descriptions closely:\n"
        f"{describe_labels(ctx)}\n\n"
        f"BM25 duplicate candidates:\n{candidate_text}\n"
    )
    toolbox = Toolbox(repo_tools(ctx.workspace) + issue_tools(ctx.github, ctx.index, ctx.issue.number))
    agent = Agent(ctx.model, toolbox, prompts.TRIAGE, ctx.settings.max_steps)
    result, image_warnings = run_with_image_fallback(agent, ctx, extra)

    try:
        raw = extract_json(result.final_text)
    except ValueError as error:
        return WorkflowResult(
            "triage", {}, image_warnings + [f"Could not parse model output: {error}"], [], result.tool_log, result.steps
        )

    verified, warnings = verify_triage(raw, ctx)
    warnings = image_warnings + warnings
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

# Output that means the test never ran properly, for pytest and for unittest-style runners
# such as Django's tests/runtests.py.
NOT_A_REAL_FAILURE = (
    "ModuleNotFoundError",
    "ImportError while importing",
    "Failed to import test module",
    "SyntaxError",
)


def test_labels(path: str) -> set[str]:
    """Ways a command can name a test file: its path, or a dotted module label.

    Django's runner takes labels relative to the tests/ folder, so
    tests/utils_tests/test_text.py can be run as utils_tests.test_text.
    """
    normalized = path.replace("\\", "/").removeprefix("./")
    module = normalized.removesuffix(".py")
    labels = {normalized, module.replace("/", ".")}
    for prefix in ("tests/", "test/"):
        if module.startswith(prefix):
            labels.add(module.removeprefix(prefix).replace("/", "."))
    return labels


def runs_test_file(command: str, path: str) -> bool:
    tokens = command.replace("\\", "/").split()
    for label in test_labels(path):
        for token in tokens:
            # A label may be followed by a class or method, e.g. utils_tests.test_x.SlugifyTests
            if token == label or token.startswith(label + ".") or token.startswith(label + "::"):
                return True
    return False


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
    if not runs_test_file(command, path):
        warnings.append("The command does not run the agent's test file")
        return verified, warnings
    try:
        argv = parse_command(command, ctx.settings.allowed_commands)
    except CommandNotAllowed as error:
        warnings.append(f"Command rejected: {error}")
        return verified, warnings

    is_pytest = "pytest" in argv
    # For pytest, force the short summary on so the failure type can always be read.
    rerun = f"{command} {' '.join(PYTEST_VERIFY_FLAGS)}" if is_pytest else command
    run = run_command(ctx.workspace.root, rerun, ctx.settings.allowed_commands, ctx.settings.command_timeout)
    if run.timed_out or run.returncode in (0, 127):
        warnings.append(f"Re-run did not show a failure ({'timeout' if run.timed_out else f'exit {run.returncode}'})")
    elif is_pytest and run.returncode in PYTEST_NOT_A_FAILURE:
        warnings.append(f"pytest exit {run.returncode}: {PYTEST_NOT_A_FAILURE[run.returncode]}, not a real failure")
    elif any(marker in run.output for marker in NOT_A_REAL_FAILURE):
        warnings.append("The test fails on an import or syntax error, which does not reproduce the bug")
    else:
        verdict = classify(run.output, is_pytest)
        if verdict.is_assertion_failure:
            verified["reproduced"] = True
            verified["failure_excerpt"] = run.output[-1500:]
            verified["test_code"] = ctx.workspace.resolve(path).read_text()[:4000]
        else:
            warnings.append(verdict.reason)
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