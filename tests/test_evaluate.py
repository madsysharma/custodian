"""Tests for evaluation: fair label scoring, finding duplicates, saved reports, and rescoring."""

import json

from custodian.evaluate import DUPLICATE_OF, build_dataset, compute_metrics, rescore, topic_labels
from custodian.github_client import Issue
from custodian.workflows import _with_extra_issues, triage

from conftest import ScriptedModel, final


def test_process_labels_are_ignored_on_both_sides():
    per_issue = [
        {"number": 1, "labels": ["Bug", "Needs Triage", "module:tree"], "duplicate_of": None,
         "predicted_labels": ["Bug", "Needs Triage", "Needs Investigation"], "predicted_duplicate": None},
    ]
    metrics = compute_metrics(per_issue, [])
    # Only topic labels count: gold {Bug, module:tree}, predicted {Bug}.
    assert metrics["labels"]["precision"] == 1.0
    assert metrics["labels"]["recall"] == 0.5
    assert metrics["labels"]["avg_labels_predicted"] == 1


def test_topic_labels():
    assert topic_labels(["Bug", "Needs Info", "good first issue", "Documentation", "Stale"]) == ["Bug", "Documentation"]


def test_duplicate_phrasings():
    for text, expected in [
        ("Duplicate of #123", 123),
        ("Closing as a duplicate of https://github.com/a/b/issues/45", 45),
        ("dupe of #9, thanks!", 9),
        ("Closing in favor of #77", 77),
    ]:
        assert int(DUPLICATE_OF.search(text).group(1)) == expected
    assert DUPLICATE_OF.search("See #12 for context") is None


class FakeEvalGitHub:
    repo = "o/r"

    def __init__(self):
        self.issues = {
            10: Issue(10, "a", "", "closed", ["Bug", "Needs Triage"]),
            11: Issue(11, "b", "", "closed", ["Needs Triage"]),  # no topic label, skipped
            12: Issue(12, "c", "", "closed", ["Documentation"]),
            3: Issue(3, "old dup", "", "closed", ["Duplicate", "Bug"]),
        }
        self.comments = {3: [{"author": "maint", "body": "Closing as duplicate of #1"}]}

    def list_issues(self, max_issues, state):
        return [self.issues[n] for n in (10, 11, 12)]

    def get_comments(self, number):
        return self.comments.get(number, [])

    def search_issue_numbers(self, query, max_results):
        return [3, 10]

    def get_issue(self, number):
        return self.issues[number]


def test_build_dataset_adds_known_duplicates(tmp_path):
    out = tmp_path / "set.jsonl"
    counts = build_dataset(FakeEvalGitHub(), out, limit=50, duplicates=5)
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert counts == {"issues": 3, "duplicates": 1}
    assert {"number": 3, "labels": ["Bug"], "duplicate_of": 1} in rows
    assert all(r["number"] != 11 for r in rows)


def test_rescore_matches_saved_predictions(tmp_path):
    report = {
        "run": {"provider": "fireworks"},
        "per_issue": [
            {"number": 1, "labels": ["Bug"], "duplicate_of": 5, "predicted_labels": ["Bug"], "predicted_duplicate": 5},
            {"number": 2, "labels": ["Docs"], "duplicate_of": None, "predicted_labels": ["Bug"], "predicted_duplicate": 1},
        ],
        "per_reproduction": [{"number": 7, "claimed": True, "reproduced": False}],
    }
    path = tmp_path / "r.json"
    path.write_text(json.dumps(report))
    scores = rescore(path)
    assert scores["duplicates"]["recall_on_true_duplicates"] == 1.0
    assert scores["duplicates"]["false_alarm_rate"] == 1.0
    assert scores["reproduction"]["overclaims_caught"] == 1
    assert scores["labels"]["f1"] == 0.5


def test_index_can_include_older_originals():
    class GH:
        def get_issue(self, number):
            if number == 404:
                raise RuntimeError("gone")
            return Issue(number, "old", "", "closed")

    recent = [Issue(50, "new", "", "open")]
    numbers = [i.number for i in _with_extra_issues(GH(), recent, [1, 50, 404])]
    assert numbers == [50, 1]


def test_triage_keeps_at_most_three_labels(make_ctx):
    ctx = make_ctx(ScriptedModel([final({"summary": "x", "labels": ["bug", "math", "docs", "good first issue"],
                                         "difficulty": "advanced"})]))
    result = triage(ctx)
    assert result.verified["labels"] == ["bug", "math", "docs"]
    assert any("first 3" in w for w in result.warnings)


def test_prefixed_workflow_labels_are_process_labels():
    assert topic_labels(["R: not a bug", "S: accepted", "status: confirmed", "T: bug", "C: crash", "spam / ai"]) == [
        "C: crash", "T: bug", "spam / ai"
    ]


def test_label_descriptions_reach_the_prompt(make_ctx):
    model = ScriptedModel([final({"summary": "x", "labels": ["bug"], "difficulty": "advanced"})])
    ctx = make_ctx(model)
    ctx.label_descriptions = {"bug": "Something is broken", "math": ""}
    triage(ctx)
    prompt = model.requests[0]["messages"][1]["content"][0]["text"]
    assert "- bug: Something is broken" in prompt
    assert "- math\n" in prompt


def test_eval_stops_when_setup_is_broken(tmp_path, monkeypatch):
    import pytest

    from custodian import evaluate
    from custodian.config import Settings

    dataset = tmp_path / "d.jsonl"
    dataset.write_text("\n".join(json.dumps({"number": n, "labels": ["Bug"], "duplicate_of": None}) for n in range(10)))
    calls = []

    def broken(*args, **kwargs):
        calls.append(1)
        raise AttributeError("'GitHub' object has no attribute 'list_labels_with_descriptions'")

    monkeypatch.setattr(evaluate, "build_context", broken)
    with pytest.raises(evaluate.EvaluationAborted, match="list_labels_with_descriptions"):
        evaluate.run_eval(Settings(), model=None, repo="o/r", dataset=dataset)
    assert len(calls) == 3


def _rows(tmp_path, numbers):
    dataset = tmp_path / "d.jsonl"
    dataset.write_text("\n".join(json.dumps({"number": n, "labels": ["Bug"], "duplicate_of": None}) for n in numbers))
    return dataset


def _fake_triage_result(labels):
    from custodian.workflows import WorkflowResult

    return WorkflowResult("triage", {"labels": labels, "duplicate_of": None})


def test_failures_after_a_success_do_not_stop_the_run(tmp_path, monkeypatch):
    from custodian import evaluate
    from custodian.config import Settings

    def context(settings, model, repo, number, **kwargs):
        if number >= 2:  # the provider becomes overloaded after the first issue
            raise RuntimeError("Model call failed after 6 attempts: Error code: 503 - service overloaded")
        return number

    monkeypatch.setattr(evaluate, "build_context", context)
    monkeypatch.setattr(evaluate, "triage", lambda ctx, apply: _fake_triage_result(["Bug"]))
    report = evaluate.run_eval(Settings(), None, "o/r", _rows(tmp_path, [1, 2, 3, 4, 5]))
    assert report["scored_issues"] == 1
    assert len(report["failed"]) == 4


def test_checkpoint_and_resume(tmp_path, monkeypatch):
    from custodian import evaluate
    from custodian.config import Settings

    checkpoint = tmp_path / "report.json"
    seen = []

    def context(settings, model, repo, number, **kwargs):
        seen.append(number)
        if number == 3:
            raise KeyboardInterrupt  # the run is interrupted part-way through
        return number

    monkeypatch.setattr(evaluate, "build_context", context)
    monkeypatch.setattr(evaluate, "triage", lambda ctx, apply: _fake_triage_result(["Bug"]))
    dataset = _rows(tmp_path, [1, 2, 3, 4])
    try:
        evaluate.run_eval(Settings(), None, "o/r", dataset, checkpoint=checkpoint)
    except KeyboardInterrupt:
        pass
    saved = json.loads(checkpoint.read_text())
    assert [i["number"] for i in saved["per_issue"]] == [1, 2]  # progress survived

    seen.clear()
    monkeypatch.setattr(evaluate, "build_context", lambda settings, model, repo, number, **kw: seen.append(number) or number)
    final = evaluate.run_eval(Settings(), None, "o/r", dataset, checkpoint=checkpoint, resume=saved)
    assert seen == [3, 4]  # finished issues were not paid for twice
    assert final["scored_issues"] == 4


def test_retry_delay_respects_retry_after():
    from types import SimpleNamespace

    from custodian.llm import retry_delay

    busy = SimpleNamespace(status_code=503, response=SimpleNamespace(headers={"retry-after": "7"}))
    assert retry_delay(busy, attempt=0) == 7
    plain = SimpleNamespace(status_code=503, response=None)
    assert 1.6 <= retry_delay(plain, attempt=0) <= 2.4
    assert retry_delay(plain, attempt=10) <= 72


def test_client_keeps_retrying_when_overloaded(monkeypatch):
    from types import SimpleNamespace as NS

    from custodian import llm
    from custodian.config import Settings

    class Overloaded(Exception):
        status_code = 503
        response = None

    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    client = llm.MuseClient(Settings(provider="meta", api_key="k", max_retries=6))
    attempts = []

    def create(**kwargs):
        attempts.append(1)
        if len(attempts) < 5:
            raise Overloaded("service overloaded")
        return NS(choices=[NS(finish_reason="stop", message=NS(content="ok", tool_calls=None))])

    client._client.chat.completions.create = create
    assert client.chat([{"role": "user", "content": "hi"}]).content == "ok"
    assert len(attempts) == 5