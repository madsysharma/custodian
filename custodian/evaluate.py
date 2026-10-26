"""Offline evaluation on a repository's past, already-resolved issues.

The ground truth comes from what maintainers actually did: the labels they applied and the
issues they closed as duplicates. Comments are hidden from the agent during evaluation so the
maintainers' own triage replies cannot leak the answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import Settings
from .github_client import GitHub
from .llm import ChatModel
from .workflows import build_context, reproduce, triage

STATUS_LABELS = re.compile(r"duplicate|wontfix|invalid|stale|needs|triage|question|answered|closed", re.I)
DUPLICATE_OF = re.compile(r"duplicate of #(\d+)", re.I)


def build_dataset(github: GitHub, out: Path, limit: int = 50) -> int:
    rows: list[dict[str, Any]] = []
    for issue in github.list_issues(max_issues=limit * 4, state="closed"):
        if len(rows) >= limit:
            break
        gold_labels = sorted(l for l in issue.labels if not STATUS_LABELS.search(l))
        duplicate_of = None
        if any("duplicate" in l.lower() for l in issue.labels):
            for comment in github.get_comments(issue.number):
                match = DUPLICATE_OF.search(comment["body"])
                if match:
                    duplicate_of = int(match.group(1))
                    break
        if gold_labels or duplicate_of:
            rows.append({"number": issue.number, "labels": gold_labels, "duplicate_of": duplicate_of})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return len(rows)


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


def score_labels(gold: list[str], predicted: list[str], counts: Counts) -> None:
    g, p = set(gold), set(predicted)
    counts.tp += len(g & p)
    counts.fp += len(p - g)
    counts.fn += len(g - p)


def run_eval(
    settings: Settings,
    model: ChatModel,
    repo: str,
    dataset: Path,
    with_reproduction: list[int] | None = None,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]
    labels = Counts()
    dup_hits = dup_total = dup_false_alarms = non_dup_total = 0
    per_issue: list[dict[str, Any]] = []

    for row in rows:
        ctx = build_context(settings, model, repo, row["number"], include_comments=False)
        result = triage(ctx, apply=False)
        predicted = result.verified.get("labels", [])
        score_labels(row["labels"], predicted, labels)
        predicted_dup = result.verified.get("duplicate_of")
        if row["duplicate_of"]:
            dup_total += 1
            dup_hits += int(predicted_dup == row["duplicate_of"])
        else:
            non_dup_total += 1
            dup_false_alarms += int(predicted_dup is not None)
        per_issue.append({**row, "predicted_labels": predicted, "predicted_duplicate": predicted_dup})

    repro: dict[str, Any] = {}
    if with_reproduction:
        claimed = verified = 0
        for number in with_reproduction:
            ctx = build_context(settings, model, repo, number, include_comments=False)
            result = reproduce(ctx, apply=False)
            claimed += int(result.verified.get("claimed", False))
            verified += int(result.verified.get("reproduced", False))
        repro = {
            "attempted": len(with_reproduction),
            "verified_reproductions": verified,
            "claimed_by_agent": claimed,
            # How often verification caught the agent over-claiming success.
            "overclaims_caught": claimed - verified,
        }

    return {
        "issues": len(rows),
        "labels": labels.prf(),
        "duplicates": {
            "recall_on_true_duplicates": round(dup_hits / dup_total, 3) if dup_total else None,
            "false_alarm_rate": round(dup_false_alarms / non_dup_total, 3) if non_dup_total else None,
            "true_duplicates": dup_total,
        },
        "reproduction": repro,
        "per_issue": per_issue,
    }
