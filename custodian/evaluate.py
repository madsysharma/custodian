"""Offline evaluation on a repository's past, already-resolved issues.

The ground truth comes from what maintainers actually did: the labels they applied and the
issues they closed as duplicates. Comments are hidden from the agent during evaluation so the
maintainers' own triage replies cannot leak the answer.

Every run saves a full report (settings, per-issue predictions and warnings) so mistakes can be
inspected and the scores recomputed with ``custodian rescore`` without calling the model again.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .github_client import GitHub
from .llm import ChatModel
from .workflows import build_context, reproduce, triage

# Process and status labels describe where an issue is in the maintainers' workflow, not what it
# is about, so they are removed from BOTH the maintainers' labels and the agent's predictions.
PROCESS_LABELS = re.compile(
    r"duplicate|wontfix|won't fix|invalid|stale|needs|triage|question|answered|closed|"
    r"waiting|blocked|in progress|help wanted|good first issue|hacktoberfest|gssoc|closing candidate",
    re.I,
)

# Ways maintainers point at the original issue when closing a duplicate.
DUPLICATE_OF = re.compile(
    r"(?:duplicate of|dupe of|dup of|duplicates|closing in favou?r of)\s+"
    r"(?:issue\s+)?(?:https://github\.com/[\w.-]+/[\w.-]+/issues/|#)(\d+)",
    re.I,
)


# Many projects prefix workflow labels, e.g. black's "R: not a bug" (resolution) and
# "S: accepted" (status), or "status: confirmed" and "resolution: fixed".
PROCESS_PREFIX = re.compile(r"^\s*(r|s|status|resolution|state)\s*[:/]", re.I)


def is_process_label(label: str) -> bool:
    return bool(PROCESS_LABELS.search(label) or PROCESS_PREFIX.search(label))


def topic_labels(labels: list[str]) -> list[str]:
    return sorted({label for label in labels if not is_process_label(label)})


# ---------------------------------------------------------------------------------------
# Building the dataset
# ---------------------------------------------------------------------------------------


def find_original(github: GitHub, number: int) -> int | None:
    """Return the issue a closed duplicate points to, read from the maintainers' comments."""
    for comment in github.get_comments(number):
        for match in DUPLICATE_OF.finditer(comment["body"]):
            original = int(match.group(1))
            if original != number:
                return original
    return None


def build_dataset(github: GitHub, out: Path, limit: int = 50, duplicates: int = 0) -> dict[str, int]:
    """Write recent labeled closed issues, plus up to ``duplicates`` known duplicate issues.

    Duplicates are rare in any recent sample, so they are searched for separately. Mixing both
    kinds in one dataset measures duplicate recall and the false-alarm rate in a single run.
    """
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()

    for issue in github.list_issues(max_issues=limit * 4, state="closed"):
        if len(rows) >= limit:
            break
        gold = topic_labels(issue.labels)
        original = None
        if any("duplicate" in label.lower() for label in issue.labels):
            original = find_original(github, issue.number)
        if gold or original:
            rows.append({"number": issue.number, "labels": gold, "duplicate_of": original})
            seen.add(issue.number)

    added_duplicates = 0
    if duplicates:
        candidates: list[int] = []
        for query in ("is:closed label:duplicate", 'is:closed "duplicate of" in:comments'):
            try:
                candidates.extend(github.search_issue_numbers(query, max_results=duplicates * 4))
            except Exception:  # search is rate-limited; use whatever was found
                continue
        for number in dict.fromkeys(candidates):  # keeps order, removes repeats
            if added_duplicates >= duplicates:
                break
            if number in seen:
                continue
            original = find_original(github, number)
            if not original:
                continue
            issue = github.get_issue(number)
            rows.append({"number": number, "labels": topic_labels(issue.labels), "duplicate_of": original})
            seen.add(number)
            added_duplicates += 1

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return {"issues": len(rows), "duplicates": sum(1 for r in rows if r["duplicate_of"])}


# ---------------------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------------------


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def prf(self) -> dict[str, float]:
        p = self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0
        r = self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3)}


def compute_metrics(per_issue: list[dict[str, Any]], repro: list[dict[str, Any]]) -> dict[str, Any]:
    """Score saved predictions. Process labels are removed from both sides first."""
    labels = Counts()
    dup_hits = dup_total = false_alarms = non_dup_total = 0
    labels_per_issue = []
    for row in per_issue:
        gold = set(topic_labels(row["labels"]))
        predicted = set(topic_labels(row.get("predicted_labels_raw", row.get("predicted_labels", []))))
        labels.tp += len(gold & predicted)
        labels.fp += len(predicted - gold)
        labels.fn += len(gold - predicted)
        labels_per_issue.append(len(predicted))
        predicted_dup = row.get("predicted_duplicate")
        if row.get("duplicate_of"):
            dup_total += 1
            dup_hits += int(predicted_dup == row["duplicate_of"])
        else:
            non_dup_total += 1
            false_alarms += int(predicted_dup is not None)

    metrics: dict[str, Any] = {
        "scored_issues": len(per_issue),
        "labels": {
            **labels.prf(),
            "avg_labels_predicted": round(sum(labels_per_issue) / len(labels_per_issue), 2) if labels_per_issue else 0,
        },
        "duplicates": {
            "recall_on_true_duplicates": round(dup_hits / dup_total, 3) if dup_total else None,
            "false_alarm_rate": round(false_alarms / non_dup_total, 3) if non_dup_total else None,
            "true_duplicates": dup_total,
            "flagged_non_duplicates": false_alarms,
        },
    }
    if repro:
        claimed = sum(int(r["claimed"]) for r in repro)
        verified = sum(int(r["reproduced"]) for r in repro)
        metrics["reproduction"] = {
            "attempted": len(repro),
            "verified_reproductions": verified,
            "claimed_by_agent": claimed,
            # How often verification caught the agent over-claiming success.
            "overclaims_caught": claimed - verified,
        }
    return metrics


# ---------------------------------------------------------------------------------------
# Running
# ---------------------------------------------------------------------------------------


def run_eval(
    settings: Settings,
    model: ChatModel,
    repo: str,
    dataset: Path,
    with_reproduction: list[int] | None = None,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    originals = sorted({row["duplicate_of"] for row in rows if row.get("duplicate_of")})
    per_issue: list[dict[str, Any]] = []
    repro: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []

    for position, row in enumerate(rows, 1):
        print(f"[{position}/{len(rows)}] triaging #{row['number']}", flush=True)
        try:
            ctx = build_context(
                settings, model, repo, row["number"], include_comments=False, extra_issue_numbers=originals
            )
            result = triage(ctx, apply=False)
        except Exception as error:  # one bad issue should not end a long, paid evaluation run
            failed.append({"number": row["number"], "stage": "triage", "error": str(error)[:300]})
            _stop_if_setup_is_broken(failed)
            continue
        per_issue.append(
            {
                **row,
                "predicted_labels": result.verified.get("labels", []),
                "predicted_duplicate": result.verified.get("duplicate_of"),
                "duplicate_reason": result.verified.get("duplicate_reason", ""),
                "warnings": result.warnings,
                "steps": result.steps,
            }
        )

    for number in with_reproduction or []:
        print(f"reproducing #{number}", flush=True)
        try:
            ctx = build_context(settings, model, repo, number, include_comments=False)
            result = reproduce(ctx, apply=False)
        except Exception as error:
            failed.append({"number": number, "stage": "reproduce", "error": str(error)[:300]})
            continue
        repro.append(
            {
                "number": number,
                "claimed": bool(result.verified.get("claimed", False)),
                "reproduced": bool(result.verified.get("reproduced", False)),
                "warnings": result.warnings,
            }
        )

    return {
        "run": {
            "repo": repo,
            "dataset": str(dataset),
            "provider": settings.provider,
            "model": settings.model,
            "reasoning_effort": settings.reasoning_effort or "provider default",
            "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "issues": len(rows),
        "failed": failed,
        **compute_metrics(per_issue, repro),
        "per_issue": per_issue,
        "per_reproduction": repro,
    }


SETUP_FAILURE_LIMIT = 3


class EvaluationAborted(RuntimeError):
    pass


def _stop_if_setup_is_broken(failed: list[dict[str, Any]]) -> None:
    """Stop early when the first issues all fail with the same error.

    That pattern means the setup is broken (a missing file, a bad key, a wrong model name),
    not that individual issues are hard, so continuing would only repeat the same failure.
    """
    first = failed[:SETUP_FAILURE_LIMIT]
    if len(failed) == SETUP_FAILURE_LIMIT and len({f["error"] for f in first}) == 1:
        raise EvaluationAborted(
            f"The first {SETUP_FAILURE_LIMIT} issues all failed with the same error, so the run was "
            f"stopped. Fix this and run again: {first[0]['error']}"
        )


def rescore(report_path: Path) -> dict[str, Any]:
    """Recompute the scores of a saved run with the current scoring rules (no model calls)."""
    report = json.loads(report_path.read_text())
    return {
        "run": report.get("run", {}),
        **compute_metrics(report.get("per_issue", []), report.get("per_reproduction", [])),
    }


def default_report_path(settings: Settings, repo: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"{repo.replace('/', '__')}_{settings.provider}_{stamp}.json"
    return settings.workdir / "reports" / name


def summary(report: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in report.items() if k not in {"per_issue", "per_reproduction"}}