"""Where reproduction tests run: on this machine, or in an isolated Modal Sandbox.

Reproduction runs code from the target repository, including tests the agent wrote, so it is
untrusted. ``LocalExecutor`` runs it on this machine and is meant for repositories you trust.
``ModalExecutor`` runs it in a Modal Sandbox: a separate container in the cloud with no network
access after setup, so a malicious repository cannot touch this machine or reach the internet.

How the Modal executor works:

1. The repository checkout is uploaded into a container image, and the setup commands
   (CUST_SANDBOX_SETUP, for example "pip install -e .") run once while the image is built.
   Files the agent wrote or edited are left out of the image, so the image stays the same
   between runs and Modal can reuse it from its cache instead of rebuilding it.
2. One sandbox is started per workflow, with networking blocked.
3. Before every command, the files the agent wrote or edited are copied into the sandbox, so
   the sandbox always matches the local checkout.
4. The sandbox is shut down when the workflow ends.

Using it needs ``pip install "custodian[modal]"`` and ``modal setup`` (which stores a token).
"""

from __future__ import annotations

import time
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .config import Settings
from .sandbox import TEST_ENV, RunResult, parse_command, run_command, tail
from .workspace import SKIP_DIRS, RepoWorkspace

REMOTE_ROOT = "/repo"


class Executor(Protocol):
    name: str

    def run(self, command: str, allowed: list[list[str]], timeout: int) -> RunResult: ...

    def close(self) -> None: ...


class LocalExecutor:
    name = "local"

    def __init__(self, workspace: RepoWorkspace):
        self.workspace = workspace

    def run(self, command: str, allowed: list[list[str]], timeout: int) -> RunResult:
        return run_command(self.workspace.root, command, allowed, timeout)

    def close(self) -> None:
        pass


class SandboxSetupError(RuntimeError):
    pass


class ModalExecutor:
    name = "modal"

    def __init__(
        self,
        workspace: RepoWorkspace,
        setup_commands: list[str],
        python_version: str = "3.12",
        app_name: str = "custodian",
        session_timeout: int = 1800,
        block_network: bool = True,
        modal_module: Any = None,
    ):
        self.workspace = workspace
        self.setup_commands = setup_commands
        self.python_version = python_version
        self.app_name = app_name
        self.session_timeout = session_timeout
        self.block_network = block_network
        self._modal = modal_module
        self._sandbox: Any = None

    # ---- lifecycle -------------------------------------------------------------------

    def _load_modal(self) -> Any:
        if self._modal is None:
            try:
                import modal  # optional dependency
            except ImportError as error:
                raise SandboxSetupError(
                    'CUST_SANDBOX=modal needs the Modal SDK: run pip install "custodian[modal]" '
                    "and then modal setup."
                ) from error
            self._modal = modal
        return self._modal

    def _ignored(self, relative: Path) -> bool:
        """Leave out version control, dependency folders, and every file the agent changed."""
        parts = relative.parts
        if any(part in SKIP_DIRS for part in parts):
            return True
        return PurePosixPath(*parts).as_posix() in self.workspace.modified

    def _ensure_sandbox(self) -> Any:
        if self._sandbox is not None:
            return self._sandbox
        modal = self._load_modal()
        image = modal.Image.debian_slim(python_version=self.python_version).add_local_dir(
            str(self.workspace.root), REMOTE_ROOT, copy=True, ignore=self._ignored
        )
        if self.setup_commands:
            image = image.run_commands(*[f"cd {REMOTE_ROOT} && {cmd}" for cmd in self.setup_commands])
        try:
            app = modal.App.lookup(self.app_name, create_if_missing=True)
            self._sandbox = modal.Sandbox.create(
                app=app,
                image=image,
                workdir=REMOTE_ROOT,
                timeout=self.session_timeout,
                block_network=self.block_network,
            )
        except Exception as error:
            raise SandboxSetupError(f"Could not start the Modal Sandbox: {error}") from error
        return self._sandbox

    def _sync_changed_files(self, sandbox: Any) -> None:
        for relative in self.workspace.modified:
            local = self.workspace.root / relative
            if not local.exists():
                continue
            remote = PurePosixPath(REMOTE_ROOT) / relative
            if str(remote.parent) != REMOTE_ROOT:
                sandbox.exec("mkdir", "-p", str(remote.parent)).wait()  # new test folders
            sandbox.filesystem.write_text(local.read_text(), str(remote))

    def close(self) -> None:
        if self._sandbox is not None:
            try:
                self._sandbox.terminate()
            finally:
                self._sandbox = None

    # ---- running ---------------------------------------------------------------------

    def run(self, command: str, allowed: list[list[str]], timeout: int) -> RunResult:
        argv = parse_command(command, allowed)  # the same allow-list as local runs
        sandbox = self._ensure_sandbox()
        self._sync_changed_files(sandbox)
        started = time.monotonic()
        process = sandbox.exec(*argv, workdir=REMOTE_ROOT, timeout=timeout, env=dict(TEST_ENV))
        returncode = process.wait()
        elapsed = time.monotonic() - started
        output = (process.stdout.read() or "") + "\n" + (process.stderr.read() or "")
        timed_out = elapsed >= timeout - 1
        if returncode is None:
            returncode = -1
        return RunResult(command, -1 if timed_out else returncode, tail(output), timed_out=timed_out)


def make_executor(settings: Settings, workspace: RepoWorkspace) -> Executor:
    if settings.sandbox == "modal":
        return ModalExecutor(
            workspace,
            setup_commands=settings.sandbox_setup,
            python_version=settings.sandbox_python,
            block_network=not settings.sandbox_network,
        )
    return LocalExecutor(workspace)
