"""Minimal GitHub REST client. Write methods are only ever called through the ActionGate."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx

API = "https://api.github.com"
FOOTER = (
    "\n\n---\n<sub>🤖 Drafted by Custodian, an AI maintainer assistant built on Meta's Muse "
    "models. A maintainer reviews every action it takes.</sub>"
)


@dataclass
class Issue:
    number: int
    title: str
    body: str
    state: str
    labels: list[str] = field(default_factory=list)
    url: str = ""
    author: str = ""

    @classmethod
    def from_api(cls, data: dict[str, Any]) -> "Issue":
        return cls(
            number=data["number"],
            title=data.get("title") or "",
            body=data.get("body") or "",
            state=data.get("state") or "",
            labels=[label["name"] for label in data.get("labels", [])],
            url=data.get("html_url") or "",
            author=(data.get("user") or {}).get("login", ""),
        )


class GitHub:
    def __init__(self, repo: str, token: str = "", client: httpx.Client | None = None):
        if repo.count("/") != 1:
            raise ValueError("repo must look like 'owner/name'")
        self.repo = repo
        headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        self._http = client or httpx.Client(base_url=API, headers=headers, timeout=30)

    def _get(self, path: str, **params: Any) -> Any:
        response = self._http.get(f"/repos/{self.repo}{path}", params=params)
        response.raise_for_status()
        return response.json()

    def _post(self, path: str, payload: dict[str, Any]) -> Any:
        response = self._http.post(f"/repos/{self.repo}{path}", json=payload)
        response.raise_for_status()
        return response.json()

    # ---- read operations -------------------------------------------------------------

    def get_issue(self, number: int) -> Issue:
        return Issue.from_api(self._get(f"/issues/{number}"))

    def get_comments(self, number: int, limit: int = 30) -> list[dict[str, str]]:
        data = self._get(f"/issues/{number}/comments", per_page=min(limit, 100))
        return [{"author": c["user"]["login"], "body": c.get("body") or ""} for c in data[:limit]]

    def list_issues(self, max_issues: int = 300, state: str = "all") -> list[Issue]:
        """Return recent issues (pull requests are filtered out)."""
        issues: list[Issue] = []
        page = 1
        while len(issues) < max_issues:
            data = self._get("/issues", state=state, per_page=100, page=page, sort="created")
            if not data:
                break
            issues.extend(Issue.from_api(d) for d in data if "pull_request" not in d)
            page += 1
        return issues[:max_issues]

    def search_issue_numbers(self, query: str, max_results: int = 100) -> list[int]:
        """Numbers of this repository's issues matching a GitHub search query."""
        numbers: list[int] = []
        page = 1
        while len(numbers) < max_results:
            response = self._http.get(
                "/search/issues",
                params={"q": f"repo:{self.repo} is:issue {query}", "per_page": 100, "page": page},
            )
            response.raise_for_status()
            items = response.json().get("items", [])
            if not items:
                break
            numbers.extend(item["number"] for item in items if "pull_request" not in item)
            page += 1
        return numbers[:max_results]

    def list_labels(self) -> list[str]:
        return list(self.list_labels_with_descriptions())

    def list_labels_with_descriptions(self) -> dict[str, str]:
        """Label names mapped to the descriptions maintainers wrote for them (may be empty)."""
        labels: dict[str, str] = {}
        page = 1
        while True:
            data = self._get("/labels", per_page=100, page=page)
            if not data:
                return labels
            for label in data:
                labels[label["name"]] = (label.get("description") or "").strip()
            page += 1

    # ---- write operations (call only through ActionGate) -----------------------------

    def post_comment(self, number: int, body: str) -> str:
        data = self._post(f"/issues/{number}/comments", {"body": body + FOOTER})
        return data.get("html_url", "")

    def add_labels(self, number: int, labels: list[str]) -> list[str]:
        data = self._post(f"/issues/{number}/labels", {"labels": labels})
        return [label["name"] for label in data]