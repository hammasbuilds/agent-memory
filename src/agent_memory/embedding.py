"""Dense and hybrid retrieval (model arm): nomic-embed-text over turns, fused with the
store's BM25 retriever by reciprocal rank fusion."""

from __future__ import annotations

import math
from datetime import datetime
from weakref import WeakKeyDictionary

from agent_memory.context import line
from agent_memory.llm import EMBED_MODEL, LLM
from agent_memory.retrieval import History, Plan, Retriever, RetrieverConfig, Strategy

# nomic-embed-text is trained with task prefixes and degrades without them.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
RRF_K = 60


def _unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


class DenseIndex:
    def __init__(self, client: LLM, h: History, model: str = EMBED_MODEL):
        self.client, self.model = client, model
        texts = [DOC_PREFIX + line(t) for t in h.turns]
        self.vectors = [_unit(v) for v in client.embed(model, texts)] if texts else []

    def scores(self, question: str) -> list[float]:
        q = _unit(self.client.embed(self.model, [QUERY_PREFIX + question])[0])
        return [sum(a * b for a, b in zip(q, v, strict=True)) for v in self.vectors]


class _Indexes:
    """One dense index per History, built on first use."""

    def __init__(self, client: LLM, model: str):
        self.client, self.model = client, model
        self._by_history: WeakKeyDictionary[History, DenseIndex] = WeakKeyDictionary()

    def __call__(self, h: History) -> DenseIndex:
        if h not in self._by_history:
            self._by_history[h] = DenseIndex(self.client, h, self.model)
        return self._by_history[h]


def make_dense(client: LLM, model: str = EMBED_MODEL) -> Strategy:
    indexes = _Indexes(client, model)

    def dense(h: History, question: str, now: datetime, seed: str) -> Plan:
        s = indexes(h).scores(question)
        return Plan([[h.turns[i]] for i in sorted(range(len(s)), key=lambda i: -s[i])])

    return dense


def make_hybrid(
    client: LLM, config: RetrieverConfig | None = None, model: str = EMBED_MODEL
) -> Strategy:
    """RRF of the store retriever's ranking and the dense ranking; each hit brings its
    neighbours exactly as the store retriever's units do."""
    indexes = _Indexes(client, model)
    retriever = Retriever(config)

    def hybrid(h: History, question: str, now: datetime, seed: str) -> Plan:
        sparse = retriever.score_turns(h, question, now)
        dense = indexes(h).scores(question)
        fused: dict[int, float] = {}
        for scores in (sparse, dense):
            ranked = sorted((i for i, v in enumerate(scores) if v > 0), key=lambda i: -scores[i])
            for rank, i in enumerate(ranked):
                fused[i] = fused.get(i, 0.0) + 1 / (RRF_K + rank + 1)
        order = sorted(fused, key=lambda i: -fused[i])
        return Plan([retriever.with_neighbours(h, i) for i in order])

    return hybrid
