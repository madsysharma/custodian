"""Tests for telling real assertion failures apart from crashing tests, for both pytest and
unittest-style runners (plain unittest and Django's tests/runtests.py)."""

import sys

from custodian.failures import classify
from custodian.workflows import reproduce

from conftest import ScriptedModel, call, final

ASSERTS = "from demo.mathx import clamp\n\ndef test_upper():\n    assert clamp(9, 0, 5) == 5\n"
CRASHES = "from demo.mathx import clamp\n\ndef test_upper():\n    assert clamp(9, 0, 5) == hi\n"  # NameError
UNITTEST_ASSERTS = (
    "import unittest\nfrom demo.mathx import clamp\n\n"
    "class ClampTests(unittest.TestCase):\n"
    "    def test_upper(self):\n        self.assertEqual(clamp(9, 0, 5), 5)\n"
)
UNITTEST_CRASHES = (
    "import unittest\nfrom demo.mathx import clamp\n\n"
    "class ClampTests(unittest.TestCase):\n"
    "    def test_upper(self):\n        self.assertEqual(clamp(9, 0, 5), hi)\n"
)


def run_claim(make_ctx, content, command_tail, runner):
    """Have the scripted agent write a test, claim success, and let the verifier judge it."""
    path = "tests/test_gk_claim.py"
    model = ScriptedModel(
        [
            call("write_test_file", path=path, content=content),
            final({"reproduced": True, "test_path": path, "command": "CMD", "explanation": "x"}),
        ]
    )
    ctx = make_ctx(model)
    ctx.settings.allowed_commands = [[sys.executable, "-m", "pytest"], [sys.executable, "-m", "unittest"]]
    command = f"{sys.executable} -m {runner} {command_tail}"
    model.turns[-1].content = model.turns[-1].content.replace("CMD", command.replace("\\", "\\\\"))
    return reproduce(ctx)


# ---- end-to-end: the verifier really runs the tests ------------------------------------------


def test_pytest_assertion_failure_counts(make_ctx):
    result = run_claim(make_ctx, ASSERTS, "tests/test_gk_claim.py -q --tb=no", "pytest")
    assert result.verified["reproduced"] is True, result.warnings


def test_pytest_crash_is_rejected(make_ctx):
    result = run_claim(make_ctx, CRASHES, "tests/test_gk_claim.py -q", "pytest")
    assert result.verified["reproduced"] is False
    assert any("NameError" in w for w in result.warnings), result.warnings
    assert result.actions == []


def test_unittest_assertion_failure_counts(make_ctx):
    result = run_claim(make_ctx, UNITTEST_ASSERTS, "tests.test_gk_claim", "unittest")
    assert result.verified["reproduced"] is True, result.warnings


def test_unittest_crash_is_rejected(make_ctx):
    result = run_claim(make_ctx, UNITTEST_CRASHES, "tests.test_gk_claim", "unittest")
    assert result.verified["reproduced"] is False
    assert any("error" in w for w in result.warnings), result.warnings


# ---- the classifier on output samples --------------------------------------------------------

DJANGO_FAILURE = """Testing against Django installed in '/src/django' with up to 8 processes
Found 1 test(s).
System check identified no issues (0 silenced).
F
======================================================================
FAIL: test_lowercase (utils_tests.test_custodian_repro.SlugifyTests.test_lowercase)
----------------------------------------------------------------------
AssertionError: 'Hello-World' != 'hello-world'

----------------------------------------------------------------------
Ran 1 test in 0.002s

FAILED (failures=1)
"""

DJANGO_ERROR = DJANGO_FAILURE.replace("FAIL:", "ERROR:").replace(
    "AssertionError: 'Hello-World' != 'hello-world'", "AttributeError: module has no attribute 'slugfy'"
).replace("FAILED (failures=1)", "FAILED (errors=1)")


def test_django_runner_output():
    assert classify(DJANGO_FAILURE, is_pytest=False).is_assertion_failure
    assert not classify(DJANGO_ERROR, is_pytest=False).is_assertion_failure
    mixed = DJANGO_FAILURE.replace("FAILED (failures=1)", "FAILED (failures=1, errors=1)")
    assert not classify(mixed, is_pytest=False).is_assertion_failure


def test_pytest_summary_variants():
    assert classify("FAILED t.py::test_a - Failed: DID NOT RAISE <class 'ValueError'>", True).is_assertion_failure
    assert classify("FAILED t.py::test_a - AssertionError: wrong", True).is_assertion_failure
    assert not classify("ERROR t.py::test_a - fixture 'db' not found", True).is_assertion_failure
    assert not classify("1 failed in 0.1s", True).is_assertion_failure  # no summary line to confirm


def test_other_runners_need_an_assertion_message():
    assert classify("Expected: 5\nReceived: 9\nTests: 1 failed", False).is_assertion_failure
    assert not classify("TypeError: clamp is not a function\nTests: 1 failed", False).is_assertion_failure


def test_truncated_pytest_summary_is_read_correctly():
    # pytest shortens long lines; this is the exact shape seen on psf/black.
    line = "FAILED tests/test_repro_pylint.py::test_pylint_inline_stay_on_first_line - As..."
    assert classify(line, is_pytest=True).is_assertion_failure
    assert not classify("FAILED tests/t.py::test_x - At...", is_pytest=True).is_assertion_failure  # AttributeError


def test_long_test_names_keep_full_summary(make_ctx):
    long_name = "test_" + "very_long_descriptive_name_" * 6
    content = ASSERTS.replace("def test_upper", f"def {long_name}")
    result = run_claim(make_ctx, content, "tests/test_gk_claim.py -q", "pytest")
    assert result.verified["reproduced"] is True, result.warnings