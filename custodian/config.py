"""Runtime configuration, loaded from environment variables.

Two providers are supported, and both align with the OpenAI-compatible chat completions format:

* ``meta``: Meta Model API (closed-weight Muse Spark), which is the default.
* ``baseten``: Muse Spark Open (open weights) served by Baseten, making the project
  eligible for the "Best Build on Meta Open Weight Models" bonus prize.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROVIDERS = {
    "meta": {
        "base_url": "https://api.meta.ai/v1",
        "model": "muse-spark-1.3",
        "key_env": "MODEL_API_KEY",
    },
    "baseten": {
        "base_url": "https://inference.baseten.co/v1",
        "model": "meta-models/muse-spark-open-1026",
        "key_env": "BASETEN_API_KEY",
    },
}


def _bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class Settings:
    provider: str = "meta"
    api_key: str = ""
    base_url: str = PROVIDERS["meta"]["base_url"]
    model: str = PROVIDERS["meta"]["model"]
    github_token: str = ""
    workdir: Path = field(default_factory=lambda: Path(".custodian"))
    # Commands the reproduction agent may run, as space-separated prefixes separated by ';'.
    allowed_commands: list[list[str]] = field(
        default_factory=lambda: [["pytest"], ["python", "-m", "pytest"]]
    )
    command_timeout: int = 180
    max_steps: int = 15
    max_index_issues: int = 300

    @classmethod
    def from_env(cls) -> "Settings":
        provider = os.getenv("GK_PROVIDER", "meta").lower()
        if provider not in PROVIDERS:
            raise ValueError(f"GK_PROVIDER must be one of {sorted(PROVIDERS)}, got {provider!r}")
        defaults = PROVIDERS[provider]
        raw_cmds = os.getenv("GK_ALLOWED_COMMANDS", "pytest;python -m pytest")
        return cls(
            provider=provider,
            api_key=os.getenv(defaults["key_env"], ""),
            base_url=os.getenv("GK_BASE_URL", defaults["base_url"]),
            model=os.getenv("GK_MODEL", defaults["model"]),
            github_token=os.getenv("GITHUB_TOKEN", ""),
            workdir=Path(os.getenv("GK_WORKDIR", ".custodian")),
            allowed_commands=[c.split() for c in raw_cmds.split(";") if c.strip()],
            command_timeout=int(os.getenv("GK_COMMAND_TIMEOUT", "180")),
            max_steps=int(os.getenv("GK_MAX_STEPS", "15")),
            max_index_issues=int(os.getenv("GK_MAX_INDEX_ISSUES", "300")),
        )

    def require_api_key(self) -> None:
        if not self.api_key:
            key_env = PROVIDERS[self.provider]["key_env"]
            raise RuntimeError(f"Set {key_env} to call the {self.provider} provider.")
