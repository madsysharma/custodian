from custodian.workflows import onboard, reproduce, triage

from conftest import ScriptedModel, call, final

FAILING_TEST = "from demo.mathx import clamp\n\ndef test_upper():\n    assert clamp(9, 0, 5) == 5\n"
PASSING_TEST = "def test_ok():\n    assert True\n"


def test_triage_verifies_labels_and_duplicates(make_ctx):
    model = ScriptedModel(
        [
            call("grep", pattern="def clamp"),
            call("search_issues", query="clamp upper bound"),
            final(
                {
                    "summary": "clamp() ignores its upper bound.",
                    "area": "demo/mathx.py",
                    "difficulty": "good-first-issue",
                    "labels": ["bug", "invented-label"],
                    "duplicate_of": 4,
                    "duplicate_reason": "#4 reports the same hi-bound problem.",
                    "missing_info": ["Which version are you using?"],
                    "screenshot_findings": "",
                }
            ),
        ]
    )
    ctx = make_ctx(model)
    result = triage(ctx)

    assert result.verified["labels"] == ["bug"]
    assert result.verified["duplicate_of"] == 4
    assert any("invented-label" in w for w in result.warnings)
    labels_action = result.actions[0]
    assert labels_action["args"]["labels"] == ["bug", "good first issue"]
    assert all(a["status"] == "proposed" for a in result.actions)  # dry run
    # The screenshot in the issue body is sent to the model as an image part.
    first_user = model.requests[0]["messages"][1]["content"]
    assert any(part["type"] == "image_url" for part in first_user)


def test_triage_rejects_self_duplicate(make_ctx):
    model = ScriptedModel([final({"summary": "x", "labels": [], "duplicate_of": 10, "difficulty": "advanced"})])
    result = triage(make_ctx(model))
    assert result.verified["duplicate_of"] is None


def test_reproduce_counts_only_verified_failures(make_ctx, repo):
    model = ScriptedModel(
        [
            call("read_file", path="demo/mathx.py"),
            call("write_test_file", path="tests/test_gk_repro.py", content=FAILING_TEST),
            call("run_tests", command="-m pytest tests/test_gk_repro.py -q"),  # rejected by allowlist
            final(
                {
                    "reproduced": True,
                    "test_path": "tests/test_gk_repro.py",
                    "command": "PYTHON -m pytest tests/test_gk_repro.py -q",
                    "explanation": "clamp(9, 0, 5) returns 9 instead of 5.",
                    "failure_excerpt": "",
                }
            ),
        ]
    )
    ctx = make_ctx(model)
    py = " ".join(ctx.settings.allowed_commands[0][:1])
    model.turns[-1].content = model.turns[-1].content.replace("PYTHON", py)
    result = reproduce(ctx)

    assert result.verified["reproduced"] is True, result.warnings
    assert "assert 9 == 5" in result.verified["failure_excerpt"]
    assert result.actions and result.actions[0]["kind"] == "post_comment"


def test_reproduce_catches_overclaim(make_ctx):
    model = ScriptedModel(
        [
            call("write_test_file", path="tests/test_gk_fake.py", content=PASSING_TEST),
            final(
                {
                    "reproduced": True,
                    "test_path": "tests/test_gk_fake.py",
                    "command": "PYTHON -m pytest tests/test_gk_fake.py -q",
                    "explanation": "trust me",
                    "failure_excerpt": "",
                }
            ),
        ]
    )
    ctx = make_ctx(model)
    model.turns[-1].content = model.turns[-1].content.replace("PYTHON", ctx.settings.allowed_commands[0][0])
    result = reproduce(ctx)

    assert result.verified["claimed"] is True
    assert result.verified["reproduced"] is False
    assert result.actions == []  # nothing is posted for an unverified claim


def test_onboard_removes_hallucinated_paths(make_ctx):
    model = ScriptedModel(
        [
            call("list_files"),
            final(
                {
                    "overview": "The logic lives in `demo/mathx.py` and is tested in `tests/core_test.py`.",
                    "files": [
                        {"path": "demo/mathx.py", "why": "clamp is defined here"},
                        {"path": "demo/ghost.py", "why": "does not exist"},
                    ],
                    "steps": ["Read `demo/mathx.py`.", "Add a test in tests/."],
                    "test_command": "pytest -q",
                }
            ),
        ]
    )
    result = onboard(make_ctx(model))
    assert [f["path"] for f in result.verified["files"]] == ["demo/mathx.py"]
    assert "tests/core_test.py (unverified)" in result.verified["overview"]
    assert len(result.warnings) == 2
