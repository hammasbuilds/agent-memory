"""Memory strategies: given a history, a question and a token budget, choose what goes
into the context.

Positional baselines (no search at all):
    full_head       the whole history in order, cut at the budget (naive truncation)
    sliding_window  the most recent turns that fit
    random          random turns - the chance floor for a given budget

Search over everything ever said:
    bm25_turns      BM25 over individual turns
    bm25_sessions   BM25 over whole sessions, packed session by session
    recency_bm25    BM25 over turns, multiplied by an exponential recency decay
    store           this library's retriever (`Retriever`): turn BM25 fused with its
                    session's BM25, a boost for turns inside a time window the question
                    names, mild recency decay, and each hit's neighbouring turns
"""

from __future__ import annotations

import random as _random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from datetime import datetime

from agent_memory.bm25 import BM25
from agent_memory.context import Context, TokenCache, pack
from agent_memory.datasets import Session, Turn
from agent_memory.temporal import query_window
from agent_memory.text import terms


@dataclass(frozen=True)
class RetrieverConfig:
    """Knobs of the `store` retriever. Defaults were chosen on the dev split only."""

    session_weight: float = 0.3  # weight of the session-level BM25 score (0 disables)
    half_life_days: float = 0.0  # recency half-life (0 disables decay)
    recency_floor: float = 0.5  # the oldest memory keeps this fraction of its score
    window_boost: float = 1.0  # extra multiplier inside a named time window (0 disables)
    neighbours: int = 1  # turns either side of a hit that come with it

    def without(self, component: str) -> RetrieverConfig:
        """This config with one component switched off, for ablations."""
        off = {
            "session": {"session_weight": 0.0},
            "recency": {"half_life_days": 0.0},
            "window": {"window_boost": 0.0},
            "neighbours": {"neighbours": 0},
        }
        if component not in off:
            raise ValueError(f"unknown component {component!r}; choose from {sorted(off)}")
        return replace(self, **off[component])


class History:
    """A read-only view of a conversation history with lazily built indexes."""

    def __init__(self, sessions: Sequence[Session]):
        self.sessions = tuple(sorted(sessions, key=lambda s: (s.timestamp, s.id)))
        self.turns: tuple[Turn, ...] = tuple(t for s in self.sessions for t in s.turns)
        self.position = {t.id: i for i, t in enumerate(self.turns)}
        self.session_of_turn = {t.id: k for k, s in enumerate(self.sessions) for t in s.turns}
        self.tokens = TokenCache()
        self._turn_index: BM25 | None = None
        self._session_index: BM25 | None = None

    @property
    def turn_index(self) -> BM25:
        if self._turn_index is None:
            self._turn_index = BM25([terms(f"{t.speaker} {t.text}") for t in self.turns])
        return self._turn_index

    @property
    def session_index(self) -> BM25:
        if self._session_index is None:
            self._session_index = BM25(
                [terms(" ".join(f"{t.speaker} {t.text}" for t in s.turns)) for s in self.sessions]
            )
        return self._session_index


def _normalise(scores: list[float]) -> list[float]:
    top = max(scores, default=0.0)
    return [s / top for s in scores] if top > 0 else scores


def _decay(age_days: float, half_life: float, floor: float) -> float:
    return floor + (1 - floor) * 0.5 ** (max(age_days, 0.0) / half_life)


class Retriever:
    """The store's retriever. Stateless apart from its config; indexes live on History."""

    def __init__(self, config: RetrieverConfig | None = None):
        self.config = config or RetrieverConfig()

    def score_turns(self, h: History, question: str, now: datetime) -> list[float]:
        c = self.config
        q = terms(question)
        turn = _normalise(h.turn_index.scores(q))
        if c.session_weight:
            sess = _normalise(h.session_index.scores(q))
            turn = [
                s + c.session_weight * sess[h.session_of_turn[t.id]] if s > 0 else 0.0
                for s, t in zip(turn, h.turns, strict=True)
            ]
        window = query_window(question, now) if c.window_boost else None
        out = []
        for s, t in zip(turn, h.turns, strict=True):
            if s > 0:
                if c.half_life_days:
                    age = (now - t.timestamp).total_seconds() / 86400
                    s *= _decay(age, c.half_life_days, c.recency_floor)
                if window and window[0] <= t.timestamp < window[1]:
                    s *= 1 + c.window_boost
            out.append(s)
        return out

    def with_neighbours(self, h: History, i: int) -> list[Turn]:
        """Turn i followed by its same-session neighbours, nearest first, so a unit
        that only partly fits the budget keeps the hit itself."""
        sess = h.session_of_turn[h.turns[i].id]
        unit = [h.turns[i]]
        for d in range(1, self.config.neighbours + 1):
            for j in (i - d, i + d):
                if 0 <= j < len(h.turns) and h.session_of_turn[h.turns[j].id] == sess:
                    unit.append(h.turns[j])
        return unit

    def ranked_units(self, h: History, question: str, now: datetime) -> list[list[Turn]]:
        """Hits in score order, each with its neighbours."""
        scores = self.score_turns(h, question, now)
        order = sorted((i for i, s in enumerate(scores) if s > 0), key=lambda i: -scores[i])
        return [self.with_neighbours(h, i) for i in order]

    def search(self, h: History, question: str, now: datetime, k: int = 10) -> list[Turn]:
        """The k best-scoring turns, without neighbours."""
        return [u[0] for u in _ranked(h, self.score_turns(h, question, now))[:k]]

    def context(self, h: History, question: str, now: datetime, budget: int) -> Context:
        return pack(self.ranked_units(h, question, now), budget, h.tokens)


@dataclass(frozen=True)
class Plan:
    """What a strategy wants in context, best first, before the budget is applied.

    `positional` plans (truncation, windows) stop at the first unit that does not fit;
    ranked plans skip it and keep filling.
    """

    units: list[list[Turn]]
    positional: bool = False

    def pack(self, budget: int, cache: TokenCache) -> Context:
        return pack(self.units, budget, cache, stop_at_first_misfit=self.positional)


Strategy = Callable[[History, str, datetime, str], Plan]  # (history, question, now, seed)


def _ranked(h: History, scores: list[float]) -> list[list[Turn]]:
    order = sorted((i for i, v in enumerate(scores) if v > 0), key=lambda i: -scores[i])
    return [[h.turns[i]] for i in order]


def full_head(h: History, question: str, now: datetime, seed: str) -> Plan:
    return Plan([[t] for t in h.turns], positional=True)


def sliding_window(h: History, question: str, now: datetime, seed: str) -> Plan:
    return Plan([[t] for t in reversed(h.turns)], positional=True)


def random_turns(h: History, question: str, now: datetime, seed: str) -> Plan:
    order = list(h.turns)
    _random.Random(seed).shuffle(order)
    return Plan([[t] for t in order])


def bm25_turns(h: History, question: str, now: datetime, seed: str) -> Plan:
    return Plan(_ranked(h, h.turn_index.scores(terms(question))))


def bm25_sessions(h: History, question: str, now: datetime, seed: str) -> Plan:
    s = h.session_index.scores(terms(question))
    order = sorted((i for i, v in enumerate(s) if v > 0), key=lambda i: -s[i])
    return Plan([list(h.sessions[i].turns) for i in order])


def make_recency_bm25(half_life_days: float = 30.0, floor: float = 0.0) -> Strategy:
    """BM25 over turns times 0.5 ** (age / half_life), squashed to [floor, 1]."""

    def recency_bm25(h: History, question: str, now: datetime, seed: str) -> Plan:
        s = h.turn_index.scores(terms(question))
        for i, t in enumerate(h.turns):
            if s[i] > 0:
                s[i] *= _decay((now - t.timestamp).total_seconds() / 86400, half_life_days, floor)
        return Plan(_ranked(h, s))

    return recency_bm25


def make_store(config: RetrieverConfig | None = None) -> Strategy:
    r = Retriever(config)

    def store(h: History, question: str, now: datetime, seed: str) -> Plan:
        return Plan(r.ranked_units(h, question, now))

    return store
