"""Okapi BM25, written out rather than imported, so every constant is visible.

score(q, d) = sum over query terms t of
    idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

with the non-negative idf variant idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5)), so a
term that appears in most documents still counts a little instead of going negative.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence


class BM25:
    """An immutable BM25 index over pre-tokenised documents."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.2, b: float = 0.75):
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError(f"need k1 >= 0 and 0 <= b <= 1, got k1={k1}, b={b}")
        self.k1, self.b = k1, b
        self.lengths = [len(d) for d in docs]
        self.n = len(docs)
        self.avgdl = (sum(self.lengths) / self.n) if self.n else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = {}
        for i, d in enumerate(docs):
            for t, f in Counter(d).items():
                self.postings.setdefault(t, []).append((i, f))
        self.idf = {
            t: math.log(1 + (self.n - len(p) + 0.5) / (len(p) + 0.5))
            for t, p in self.postings.items()
        }

    def scores(self, query: Sequence[str]) -> list[float]:
        """One score per document, in document order. Repeated query terms count once."""
        out = [0.0] * self.n
        if not self.n or self.avgdl == 0:
            return out
        k1, b, avgdl = self.k1, self.b, self.avgdl
        for t in set(query):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i, f in self.postings[t]:
                norm = k1 * (1 - b + b * self.lengths[i] / avgdl)
                out[i] += idf * f * (k1 + 1) / (f + norm)
        return out

    def rank(self, query: Sequence[str]) -> list[int]:
        """Document indices by descending score; ties keep document order."""
        s = self.scores(query)
        return sorted(range(self.n), key=lambda i: -s[i])
