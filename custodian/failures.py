"""Decide whether a failing test run shows a real assertion failure or just a crash.

A reproduction should fail because the code did the wrong thing (an assertion compared the
expected behavior with the actual behavior and they differed), not because the test itself is
broken (a typo, a missing fixture, an import error). Test runners report these two cases
differently, so each runner gets its own check:

* pytest: the short summary has one line per problem, such as
  "FAILED tests/test_x.py::test_y - assert 'A' == 'a'" (an assertion) or
  "FAILED tests/test_x.py::test_y - NameError: name 'z' is not defined" (a crash).
* unittest and Django's runtests.py: the final line reads "FAILED (failures=1)" for assertions
  and "FAILED (errors=1)" for crashes.
* Anything else (for example Jest or Vitest): best effort, by looking for assertion messages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Extra pytest flags added when the verifier re-runs a test, so the short summary is always
# printed (-rfE) and tracebacks stay readable, even if the agent's command turned them off.
PYTEST_VERIFY_FLAGS = ["-rfE", "--tb=short"]

_PYTEST_SUMMARY = re.compile(r"^(FAILED|ERROR) (\S+)(?: - (.*))?$", re.MULTILINE)
_UNITTEST_SUMMARY = re.compile(r"^FAILED \(([^)]*)\)\s*$", re.MULTILINE)
_PYTEST_ASSERTION = ("assert", "AssertionError", "Failed: DID NOT RAISE")
_GENERIC_ASSERTION = ("AssertionError", "Expected:", "DID NOT RAISE")


@dataclass
class Verdict:
    is_assertion_failure: bool
    reason: str


def classify(output: str, is_pytest: bool) -> Verdict:
    if is_pytest:
        return _classify_pytest(output)
    if _UNITTEST_SUMMARY.search(output):
        return _classify_unittest(output)
    return _classify_generic(output)


def _classify_pytest(output: str) -> Verdict:
    entries = _PYTEST_SUMMARY.findall(output)
    if not entries:
        return Verdict(False, "pytest printed no FAILED lines, so the failure could not be confirmed")
    errors = [node for kind, node, _ in entries if kind == "ERROR"]
    if errors:
        return Verdict(False, f"pytest reported setup or collection errors in {errors[0]}")
    crashes = []
    assertions = 0
    for _kind, node, message in entries:
        message = (message or "").strip()
        if _is_truncated_assertion(message) or message.startswith(_PYTEST_ASSERTION):
            assertions += 1
        else:
            crashes.append(f"{node}: {message or 'unknown error'}")
    if crashes:
        return Verdict(False, f"The test crashed instead of failing an assertion ({crashes[0]})")
    return Verdict(True, f"{assertions} assertion failure(s)")


def _is_truncated_assertion(message: str) -> bool:
    """pytest shortens long summary lines with "...". "As..." can only be "AssertionError"."""
    if not message.endswith("...") or len(message) < 5:
        return False
    head = message[:-3]
    return any(marker.startswith(head) for marker in ("AssertionError", "assert ", "Failed: DID NOT RAISE"))


def _classify_unittest(output: str) -> Verdict:
    summary = _UNITTEST_SUMMARY.findall(output)[-1]
    counts = {key.strip(): int(value) for key, value in re.findall(r"(\w+)=(\d+)", summary)}
    failures, errors = counts.get("failures", 0), counts.get("errors", 0)
    if errors:
        return Verdict(False, f"The test run had {errors} error(s), meaning tests crashed instead of failing an assertion")
    if not failures:
        return Verdict(False, f"The run failed without any assertion failures ({summary})")
    return Verdict(True, f"{failures} assertion failure(s)")


def _classify_generic(output: str) -> Verdict:
    if any(marker in output for marker in _GENERIC_ASSERTION):
        return Verdict(True, "assertion message found in the output")
    return Verdict(False, "No assertion failure found in the output, so the failure could not be confirmed")