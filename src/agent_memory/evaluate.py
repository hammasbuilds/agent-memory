"""Evidence-recall evaluation: does the gold evidence make it into the context?

For every question and strategy the plan is packed at each token budget, and at each
top-k cut, and scored against the gold evidence turns:

    recall      fraction of the gold evidence turns that made it in
    all         every gold turn made it in (what a multi-hop question needs)
    newest      knowledge-update only: a gold turn from the *latest* evidence session
                made it in (the one holding the current value)
    stale_only  knowledge-update only: gold evidence made it in, but only the outdated
                value's session

Every history goes through `MemoryStore` (SQLite round trip) before any strategy sees
it, so the store under test is the store that ships.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

from agent_memory.context import Context, pack
from agent_memory.datasets import Question
from agent_memory.retrieval import History, Plan, Strategy
from agent_memory.store import MemoryStore
from agent_memory.temporal import query_window

BUDGETS = (256, 512, 1024, 2048, 4096, 8192)
TOP_K = (1, 3, 5, 10, 20)
HEADLINE_BUDGET = 2048


def split_of(q: Question) -> str:
    """'dev' or 'test', by a stable hash of the unit that must not straddle splits:
    the conversation for LoCoMo (its questions share one history), the question for
    LongMemEval (each has its own history). Roughly 20% dev."""
    unit = cluster_of(q)
    return "dev" if int(hashlib.sha256(unit.encode()).hexdigest(), 16) % 5 == 0 else "test"


def cluster_of(q: Question) -> str:
    """The resampling unit for confidence intervals (see `stats`)."""
    return q.qid.split(":")[0] if q.dataset == "locomo" else q.qid


@dataclass(frozen=True)
class Scored:
    recall: float
    all: bool
    newest: bool | None
    stale_only: bool | None


def score(q: Question, included: frozenset[str]) -> Scored | None:
    """Score one context against a question's gold evidence; None if the question has
    no gold evidence turns to find."""
    ev = q.evidence_turns
    if not ev:
        return None
    hit = ev & included
    newest = stale = None
    if q.qtype == "knowledge-update":
        by_session: dict[str, set[str]] = {}
        when = {s.id: s.timestamp for s in q.history}
        for s in q.history:
            for t in s.turns:
                if t.id in ev:
                    by_session.setdefault(s.id, set()).add(t.id)
        latest = max(by_session, key=lambda sid: when[sid])
        newest = bool(by_session[latest] & included)
        stale = bool(hit) and not newest
    return Scored(len(hit) / len(ev), hit == ev, newest, stale)


def history_via_store(q: Question) -> History:
    with MemoryStore(":memory:") as store:
        store.ingest(q.history)
        return store.history


def evaluate(
    questions: Iterable[Question],
    strategies: dict[str, Strategy],
    budgets: tuple[int, ...] = BUDGETS,
    top_k: tuple[int, ...] = TOP_K,
) -> Iterator[dict]:
    """One row per (question, strategy, budget-or-k). Questions sharing a history
    (LoCoMo) reuse one index."""
    cache_key: object = None
    h: History | None = None
    for q in questions:
        if q.history is not cache_key:
            cache_key, h = q.history, history_via_store(q)
        assert h is not None
        base = {
            "qid": q.qid,
            "dataset": q.dataset,
            "split": split_of(q),
            "cluster": cluster_of(q),
            "qtype": q.qtype,
            "time_expr": query_window(q.question, q.now) is not None,
        }
        for name, strategy in strategies.items():
            plan = strategy(h, q.question, q.now, q.qid)
            for budget in budgets:
                yield _row(base, name, "budget", budget, q, plan.pack(budget, h.tokens))
            for k in top_k:
                ctx = pack(plan.units[:k], 10**9, h.tokens)
                yield _row(base, name, "k", k, q, ctx)


def _row(base: dict, strategy: str, axis: str, value: int, q: Question, ctx: Context) -> dict:
    s = score(q, ctx.turn_ids)
    row = {**base, "strategy": strategy, axis: value, "tokens": ctx.tokens}
    if s is None:
        row["measurable"] = False
        return row
    row |= {"measurable": True, "recall": round(s.recall, 4), "all": s.all}
    if s.newest is not None:
        row |= {"newest": s.newest, "stale_only": s.stale_only}
    return row


def plan_recall(q: Question, h: History, plan: Plan, budget: int) -> float | None:
    """Recall of one plan at one budget - the dev-split tuning objective."""
    s = score(q, plan.pack(budget, h.tokens).turn_ids)
    return None if s is None else s.recall
