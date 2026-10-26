"""Command-line entry point for Custodian.

Examples:
  custodian triage    --repo owner/name --issue 123
  custodian all       --repo owner/name --issue 123 --apply
  custodian build-eval --repo owner/name --limit 50
  custodian eval      --repo owner/name --dataset .custodian/eval/owner__name.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .github_client import GitHub


def _print_result(result) -> None:
    print(f"\n=== {result.workflow} ({result.steps} steps, {len(result.tool_log)} tool calls) ===")
    print(json.dumps(result.verified, indent=2))
    for warning in result.warnings:
        print(f"  ! verification: {warning}")
    for action in result.actions:
        print(f"  -> {action['kind']} [{action['status']}] {action['detail']}")
        if action["kind"] == "post_comment" and action["status"] == "proposed":
            print("     " + action["args"]["body"].replace("\n", "\n     "))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="custodian", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("triage", "reproduce", "onboard", "all"):
        p = sub.add_parser(name, help=f"run the {name} workflow on one issue")
        p.add_argument("--repo", required=True, help="owner/name")
        p.add_argument("--issue", required=True, type=int)
        p.add_argument("--apply", action="store_true", help="actually post to GitHub (default: dry run)")
        p.add_argument("--workspace", type=Path, help="use an existing local checkout instead of cloning")
        p.add_argument("--json-out", type=Path, help="save the full results, including tool logs")

    p = sub.add_parser("build-eval", help="build an evaluation set from closed issues")
    p.add_argument("--repo", required=True)
    p.add_argument("--limit", type=int, default=50)

    p = sub.add_parser("eval", help="score triage (and optionally reproduction) on an evaluation set")
    p.add_argument("--repo", required=True)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--repro-issues", type=int, nargs="*", help="issue numbers to attempt reproduction on")
    p.add_argument("--json-out", type=Path)

    args = parser.parse_args(argv)
    settings = Settings.from_env()

    if args.command == "build-eval":
        out = settings.workdir / "eval" / f"{args.repo.replace('/', '__')}.jsonl"
        count = build_dataset_cmd(settings, args.repo, out, args.limit)
        print(f"Wrote {count} issues to {out}")
        return 0

    from .llm import MuseClient

    model = MuseClient(settings)

    if args.command == "eval":
        from .evaluate import run_eval

        report = run_eval(settings, model, args.repo, args.dataset, args.repro_issues)
        summary = {k: v for k, v in report.items() if k != "per_issue"}
        print(json.dumps(summary, indent=2))
        if args.json_out:
            args.json_out.write_text(json.dumps(report, indent=2))
        return 0

    from .workflows import WORKFLOWS, build_context

    ctx = build_context(settings, model, args.repo, args.issue, workspace_path=args.workspace)
    names = list(WORKFLOWS) if args.command == "all" else [args.command]
    if not args.apply:
        print("Dry run: proposed actions are printed, nothing is posted. Use --apply to post.")
    results = []
    for name in names:
        result = WORKFLOWS[name](ctx, apply=args.apply)
        _print_result(result)
        results.append(result.__dict__)
    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2, default=str))
    return 0


def build_dataset_cmd(settings: Settings, repo: str, out: Path, limit: int) -> int:
    from .evaluate import build_dataset

    return build_dataset(GitHub(repo, settings.github_token), out, limit)


if __name__ == "__main__":
    sys.exit(main())
