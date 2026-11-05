"""Runtime configuration, loaded from environment variables.

Two providers are supported, and both speak the OpenAI-compatible chat completions format:

* ``meta``: Meta Model API (closed-weight Muse Spark), the default.
* ``fireworks``: Muse Spark Open (open weights) served by Fireworks AI, one of the hackathon's
  inference partners.
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
    "fireworks": {
        "base_url": "https://api.fireworks.ai/inference/v1",
        "model": "accounts/fireworks/models/muse-spark-open",
        "key_env": "FIREWORKS_API_KEY",
    },
}


REASONING_EFFORTS = ("low", "medium", "high", "xhigh")


def load_dotenv(path: Path = Path(".env")) -> None:
    """Load KEY=value lines from a .env file into the environment.

    This works the same on Windows, macOS, and Linux, so no shell-specific setup is needed.
    Variables that are already set in the environment take priority over the file.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key and value and key not in os.environ:
            os.environ[key] = value


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
    # Muse Spark Open always reasons and defaults to "xhigh". Lower values respond faster.
    # Leave empty to use the provider's default.
    reasoning_effort: str = ""
    # Output cap per model call, shared by reasoning and the answer.
    max_tokens: int = 8192
    # Attempts per model call when the provider is busy or rate-limiting (HTTP 429/503).
    max_retries: int = 6

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        provider = os.getenv("CUST_PROVIDER", "meta").lower()
        if provider not in PROVIDERS:
            raise ValueError(f"CUST_PROVIDER must be one of {sorted(PROVIDERS)}, got {provider!r}")
        defaults = PROVIDERS[provider]
        raw_cmds = os.getenv("CUST_ALLOWED_COMMANDS", "pytest;python -m pytest")
        effort = os.getenv("CUST_REASONING_EFFORT", "").strip().lower()
        if effort and effort not in REASONING_EFFORTS:
            raise ValueError(f"CUST_REASONING_EFFORT must be one of {REASONING_EFFORTS}, got {effort!r}")
        return cls(
            provider=provider,
            api_key=os.getenv(defaults["key_env"], ""),
            base_url=os.getenv("CUST_BASE_URL", defaults["base_url"]),
            model=os.getenv("CUST_MODEL", defaults["model"]),
            github_token=os.getenv("GITHUB_TOKEN", ""),
            workdir=Path(os.getenv("CUST_WORKDIR", ".custodian")),
            allowed_commands=[c.split() for c in raw_cmds.split(";") if c.strip()],
            command_timeout=int(os.getenv("CUST_COMMAND_TIMEOUT", "180")),
            max_steps=int(os.getenv("CUST_MAX_STEPS", "15")),
            max_index_issues=int(os.getenv("CUST_MAX_INDEX_ISSUES", "300")),
            reasoning_effort=effort,
            max_tokens=int(os.getenv("CUST_MAX_TOKENS", "8192")),
            max_retries=max(1, int(os.getenv("CUST_MAX_RETRIES", "6"))),
        )

    def require_api_key(self) -> None:
        if not self.api_key:
            key_env = PROVIDERS[self.provider]["key_env"]
            raise RuntimeError(f"Set {key_env} to call the {self.provider} provider.")