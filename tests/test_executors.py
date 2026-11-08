"""Executors: the local runner's environment, and the Modal Sandbox runner with a fake Modal SDK."""

from pathlib import Path

from custodian import sandbox
from custodian.config import Settings
from custodian.executors import LocalExecutor, ModalExecutor, make_executor


class FakeStream:
    def __init__(self, text):
        self.text = text

    def read(self):
        return self.text


class FakeProcess:
    def __init__(self, returncode, out):
        self.returncode = returncode
        self.stdout, self.stderr = FakeStream(out), FakeStream("")

    def wait(self):
        return self.returncode


class FakeFilesystem:
    def __init__(self):
        self.files = {}

    def write_text(self, data, remote_path):
        self.files[remote_path] = data


class FakeSandbox:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.filesystem = FakeFilesystem()
        self.commands = []
        self.terminated = False

    def exec(self, *argv, workdir=None, timeout=None, env=None):
        if argv[:2] == ("mkdir", "-p"):
            return FakeProcess(0, "")
        self.commands.append((argv, workdir, env))
        return FakeProcess(1, "FAILED tests/test_x.py::test_a - assert 9 == 5")

    def terminate(self):
        self.terminated = True


class FakeImage:
    def __init__(self, steps=()):
        self.steps = list(steps)

    @classmethod
    def debian_slim(cls, python_version):
        return cls([("python", python_version)])

    def add_local_dir(self, local, remote, copy, ignore):
        image = FakeImage(self.steps + [("dir", local, remote, copy)])
        image.ignore = ignore
        return image

    def run_commands(self, *commands):
        image = FakeImage(self.steps + [("run", commands)])
        image.ignore = self.ignore
        return image


class FakeModal:
    def __init__(self):
        self.sandboxes = []
        outer = self

        class App:
            @staticmethod
            def lookup(name, create_if_missing):
                return f"app:{name}"

        class Sandbox:
            @staticmethod
            def create(**kwargs):
                created = FakeSandbox(**kwargs)
                outer.sandboxes.append(created)
                return created

        self.App, self.Sandbox, self.Image = App, Sandbox, FakeImage


def test_modal_executor_builds_once_syncs_changes_and_blocks_network(repo):
    modal = FakeModal()
    repo.write_test_file("tests/test_x.py", "def test_a():\n    assert 9 == 5\n")
    executor = ModalExecutor(repo, setup_commands=["pip install -e ."], modal_module=modal)

    first = executor.run("pytest tests/test_x.py -q", [["pytest"]], timeout=60)
    executor.run("pytest tests/test_x.py -q", [["pytest"]], timeout=60)
    [box] = modal.sandboxes  # one sandbox for the whole workflow
    image = box.kwargs["image"]

    assert box.kwargs["block_network"] is True
    assert ("run", ("cd /repo && pip install -e .",)) in image.steps
    # The agent's file is left out of the image (so the image stays cached) and synced instead.
    assert image.ignore(Path("tests/test_x.py")) is True
    assert image.ignore(Path("demo/mathx.py")) is False
    assert image.ignore(Path(".git/config")) is True
    assert box.filesystem.files["/repo/tests/test_x.py"].startswith("def test_a")
    assert box.commands[0][0] == ("pytest", "tests/test_x.py", "-q")
    assert box.commands[0][2]["COLUMNS"] == "400"
    assert first.returncode == 1 and "assert 9 == 5" in first.output

    executor.close()
    assert box.terminated


def test_modal_executor_enforces_the_allow_list(repo):
    executor = ModalExecutor(repo, setup_commands=[], modal_module=FakeModal())
    try:
        executor.run("rm -rf /", [["pytest"]], timeout=60)
    except sandbox.CommandNotAllowed:
        return
    raise AssertionError("disallowed commands must be refused before reaching Modal")


def test_make_executor_follows_settings(repo):
    assert isinstance(make_executor(Settings(), repo), LocalExecutor)
    modal_executor = make_executor(Settings(sandbox="modal", sandbox_setup=["npm ci"]), repo)
    assert isinstance(modal_executor, ModalExecutor) and modal_executor.setup_commands == ["npm ci"]
