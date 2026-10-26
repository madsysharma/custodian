"""A small, dependency-free BM25 index used to find candidate duplicate issues.

BM25 reduces hundreds of issues down to a short list; the model then judges whether any
candidate is a true duplicate. This keeps prompts small and makes the decision auditable.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from .github_client import Issue

_TOKEN = re.compile(r"[a-z0-9_]+")
_STOP = set(
    "a an and are as at be but by for from has have i if in is it its not of on or so that the "
    "this to was when which with you your we can cannot doesn't dont does do".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP and len(t) > 1]


class IssueIndex:
    def __init__(self, issues: list[Issue], k1: float = 1.5, b: float = 0.75):
        self.issues = issues
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(f"{i.title} {i.title} {i.body[:3000]}")) for i in issues]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg_len = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        df: Counter[str] = Counter()
        for doc in self.docs:
            df.update(doc.keys())
        n = len(self.docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def search(self, query: str, k: int = 5, exclude: int | None = None) -> list[tuple[Issue, float]]:
        terms = tokenize(query)
        scored: list[tuple[Issue, float]] = []
        for issue, doc, length in zip(self.issues, self.docs, self.lengths):
            if issue.number == exclude:
                continue
            score = 0.0
            for term in terms:
                tf = doc.get(term, 0)
                if not tf:
                    continue
                norm = tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * length / (self.avg_len or 1)))
                score += self.idf.get(term, 0.0) * norm
            if score > 0:
                scored.append((issue, score))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:k]
