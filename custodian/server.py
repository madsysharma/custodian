"""Optional GitHub webhook server: triage and onboard every newly opened issue.

Run with:  uvicorn custodian.server:app --port 8000
Requires:  pip install "custodian[server]"
Actions stay in dry-run mode (saved to .custodian/runs) unless GK_APPLY=1.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time

from .config import Settings


def valid_signature(secret: str, body: bytes, header: str | None) -> bool:
    if not secret or not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.removeprefix("sha256="))


def handle_opened_issue(repo: str, number: int) -> None:
    from .llm import MuseClient
    from .workflows import build_context, onboard, triage

    settings = Settings.from_env()
    apply = os.getenv("GK_APPLY") == "1"
    ctx = build_context(settings, MuseClient(settings), repo, number)
    results = [triage(ctx, apply=apply), onboard(ctx, apply=apply)]
    out = settings.workdir / "runs" / f"{repo.replace('/', '__')}_{number}_{int(time.time())}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([r.__dict__ for r in results], indent=2, default=str))


try:
    from fastapi import BackgroundTasks, FastAPI, HTTPException, Request

    app = FastAPI(title="Custodian webhook")

    @app.post("/webhook")
    async def webhook(request: Request, background: BackgroundTasks) -> dict[str, str]:
        body = await request.body()
        secret = os.getenv("GITHUB_WEBHOOK_SECRET", "")
        if not valid_signature(secret, body, request.headers.get("X-Hub-Signature-256")):
            raise HTTPException(status_code=401, detail="bad signature")
        payload = json.loads(body)
        if request.headers.get("X-GitHub-Event") == "issues" and payload.get("action") == "opened":
            background.add_task(
                handle_opened_issue, payload["repository"]["full_name"], payload["issue"]["number"]
            )
            return {"status": "queued"}
        return {"status": "ignored"}

except ImportError:  # The server extra is optional
    app = None
