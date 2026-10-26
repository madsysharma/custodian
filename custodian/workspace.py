"""A local checkout of the target repository that the agent can explore safely."""

from __future__ import annotations

import fnmatch
import re
import subprocess
from pathlib import Path

SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "dist", "build", ".tox"}
MAX_FILE_BYTES = 400_000
TEST_PATH = re.compile(r"(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]+\.py$|_test\.py$|\.(test|spec)\.[jt]sx?$")


class WorkspaceError(Exception):
    pass


class RepoWorkspace:
    def __init__(self, root: Path):
        self.root = root.resolve()
        if not self.root.is_dir():
            raise WorkspaceError(f"{self.root} is not a directory")

    @classmethod
    def clone(cls, repo: str, workdir: Path, refresh: bool = False) -> "RepoWorkspace":
        target = workdir / "repos" / repo.replace("/", "__")
        if target.exists() and not refresh:
            return cls(target)
        if target.exists():
            subprocess.run(["rm", "-rf", str(target)], check=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", f"https://github.com/{repo}.git", str(target)],
            check=True,
            capture_output=True,
        )
        return cls(target)

    def reset(self) -> None:
        """Discard any files the agent wrote, so every run starts from a clean tree."""
        if (self.root / ".git").exists():
            subprocess.run(["git", "checkout", "--", "."], cwd=self.root, capture_output=True)
            subprocess.run(["git", "clean", "-fd"], cwd=self.root, capture_output=True)

    # ---- path safety -----------------------------------------------------------------

    def resolve(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise WorkspaceError(f"Path escapes the repository: {path}")
        return candidate

    def exists(self, path: str) -> bool:
        try:
            return self.resolve(path).exists()
        except WorkspaceError:
            return False

    def _iter_files(self, base: Path):
        for path in sorted(base.rglob("*")):
            if any(part in SKIP_DIRS for part in path.relative_to(self.root).parts):
                continue
            if path.is_file():
                yield path

    # ---- Read tools ------------------------------------------------------------------

    def list_files(self, subdir: str = ".", pattern: str = "*", limit: int = 200) -> str:
        base = self.resolve(subdir)
        files = [
            str(p.relative_to(self.root))
            for p in self._iter_files(base)
            if fnmatch.fnmatch(p.name, pattern)
        ]
        shown = files[:limit]
        suffix = f"\n... ({len(files) - limit} more)" if len(files) > limit else ""
        return "\n".join(shown) + suffix if shown else "(no matching files)"

    def read_file(self, path: str, start: int = 1, end: int | None = None) -> str:
        target = self.resolve(path)
        if not target.is_file():
            raise WorkspaceError(f"No such file: {path}")
        lines = target.read_text(errors="replace").splitlines()
        start = max(start, 1)
        end = min(end or start + 299, len(lines), start + 399)
        numbered = [f"{i:>5}  {lines[i - 1]}" for i in range(start, end + 1)]
        header = f"{path} (lines {start}-{end} of {len(lines)})"
        return header + "\n" + "\n".join(numbered)

    def grep(self, pattern: str, glob: str = "*", limit: int = 50) -> str:
        try:
            regex = re.compile(pattern)
        except re.error as error:
            raise WorkspaceError(f"Invalid regex: {error}") from error
        hits: list[str] = []
        for path in self._iter_files(self.root):
            if not fnmatch.fnmatch(path.name, glob) or path.stat().st_size > MAX_FILE_BYTES:
                continue
            try:
                text = path.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    hits.append(f"{path.relative_to(self.root)}:{number}: {line.strip()[:200]}")
                    if len(hits) >= limit:
                        return "\n".join(hits) + "\n... (limit reached)"
        return "\n".join(hits) or "(no matches)"

    # ---- Local write tool (sandbox only, never pushed) --------------------------------

    def write_test_file(self, path: str, content: str) -> str:
        normalized = path.replace("\\", "/").lstrip("./")
        if not TEST_PATH.search(normalized):
            raise WorkspaceError(
                "The agent may only write test files (e.g. tests/test_x.py or foo.test.ts)."
            )
        target = self.resolve(normalized)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return f"Wrote {normalized} ({len(content.splitlines())} lines)"
