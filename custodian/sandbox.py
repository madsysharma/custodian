"""Run allow-listed test commands inside the repository checkout.

This runs code from the target repository, so treat it as untrusted. For anything beyond
a local demo, run it inside a container or a Modal Sandbox (see docs/ARCHITECTURE.md).
"""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

MAX_OUTPUT_CHARS = 6000


@dataclass
class RunResult:
    command: str
    returncode: int
    output: str
    timed_out: bool = False

    def summary(self) -> str:
        status = "TIMED OUT" if self.timed_out else f"exit code {self.returncode}"
        return f"$ {self.command}\n[{status}]\n{self.output}"


class CommandNotAllowed(Exception):
    pass


def split_command(command: str) -> list[str]:
    """Split a command line. On Windows, backslashes in paths must survive, so POSIX
    escaping is turned off and surrounding quotes are stripped instead."""
    if os.name != "nt":
        return shlex.split(command)
    return [token.strip('"').strip("'") for token in shlex.split(command, posix=False)]


def parse_command(command: str, allowed: list[list[str]]) -> list[str]:
    argv = split_command(command)
    if not argv:
        raise CommandNotAllowed("Empty command")
    if any(token in {"&&", "||", ";", "|", ">", "<"} for token in argv):
        raise CommandNotAllowed("Shell operators are not allowed")
    if not any(argv[: len(prefix)] == prefix for prefix in allowed):
        options = ", ".join(" ".join(p) for p in allowed)
        raise CommandNotAllowed(f"Command must start with one of: {options}")
    return argv


def run_command(root: Path, command: str, allowed: list[list[str]], timeout: int) -> RunResult:
    argv = parse_command(command, allowed)
    try:
        # A wide virtual terminal stops pytest from cutting its summary lines short
        # (for example "AssertionError" shortened to "As..."), which the verifier relies on.
        env = {**os.environ, "COLUMNS": "400"}
        proc = subprocess.run(
            argv, cwd=root, capture_output=True, text=True, timeout=timeout, check=False, env=env
        )
    except subprocess.TimeoutExpired as error:
        partial = (error.stdout or "") if isinstance(error.stdout, str) else ""
        return RunResult(command, -1, _tail(partial), timed_out=True)
    except FileNotFoundError:
        return RunResult(command, 127, f"Executable not found: {argv[0]}")
    return RunResult(command, proc.returncode, _tail(proc.stdout + "\n" + proc.stderr))


def _tail(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    return "... (truncated)\n" + text[-MAX_OUTPUT_CHARS:]