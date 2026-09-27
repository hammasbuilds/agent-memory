"""Evidence-recall evaluation: does the gold evidence make it into the context?

For every question and strategy the plan is packed at each token budget, and at each
top-k cut, and scored against two answer keys:

turn level (LoCoMo `evidence`, LongMemEval `has_answer`)
    recall      fraction of the gold evidence turns that made it in
    all         every gold turn made it in (what a multi-hop question needs)
session level (LongMemEval `answer_session_ids`; derived from the turns for LoCoMo)
    session_recall   fraction of answer sessions with at least one turn in context
    all_sessions     every answer session is represented
knowledge-update only
    newest      a gold turn from the *latest* answer session made it in (the session
                holding the current value); unmeasured if that session has no gold turn
    stale_only  gold evidence made it in, but none from the latest answer session

The two keys disagree on LongMemEval: 62 questions (30 abstention, 32 others) have an
answer session with no turn marked `has_answer`, so their turn-level key is partial.
Rows carry the question's data-quality flags so every summary can be re-cut without them.

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


def cluster_of(q: Question) -> str:
    """The unit that must not straddle the dev/test split, and the resampling unit for
    confidence intervals: the conversation for LoCoMo (its ~200 questions share one
    history); for LongMemEval the question with its `_abs` twin ("What is the name of my
    cat?" / "...my hamster?" - separate histories, but near-paraphrases)."""
    if q.dataset == "locomo":
        return q.qid.split(":")[0]
    return q.qid.removesuffix("_abs")


def history_key(q: Question) -> str:
    """Identifies a history: one per LoCoMo conversation, one per LongMemEval question."""
    return q.qid.split(":")[0] if q.dataset == "locomo" else q.qid


def split_of(q: Question) -> str:
    """'dev' or 'test', by a stable hash of `cluster_of`. Roughly 20% dev."""
    return "dev" if int(hashlib.sha256(cluster_of(q).encode()).hexdigest(), 16) % 5 == 0 else "test"


@dataclass(frozen=True)
class Scored:
    recall: float | None
    all: bool | None
    session_recall: float | None
    all_sessions: bool | None
    newest: bool | None
    stale_only: bool | None


def score(q: Question, included: frozenset[str]) -> Scored | None:
    """Score one context against a question's gold evidence; None if the question has
    neither gold turns nor gold sessions."""
    ev, ev_sessions = q.evidence_turns, q.evidence_sessions
    if not ev and not ev_sessions:
        return None
    recall = all_ = s_recall = all_s = newest = stale = None
    hit = ev & included
    if ev:
        recall, all_ = len(hit) / len(ev), hit == ev
    if ev_sessions:
        present = {s.id for s in q.history for t in s.turns if t.id in included}
        found = ev_sessions & present
        s_recall, all_s = len(found) / len(ev_sessions), found == ev_sessions
    if q.qtype == "knowledge-update" and ev_sessions:
        when = {s.id: s.timestamp for s in q.history}
        latest = max(ev_sessions, key=lambda sid: when[sid])
        latest_ev = {t.id for s in q.history if s.id == latest for t in s.turns if t.id in ev}
        if latest_ev:
            newest = bool(latest_ev & included)
            stale = bool(hit) and not newest
    return Scored(recall, all_, s_recall, all_s, newest, stale)


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
            "flags": sorted(q.flags),
        }
        positions = evidence_positions(q, h)
        for name, strategy in strategies.items():
            plan = strategy(h, q.question, q.now, q.qid)
            for budget in budgets:
                ctx = plan.pack(budget, h.tokens)
                row = _row(base, name, "budget", budget, q, ctx)
                # per evidence turn: its position in the history (0 = first turn ever,
                # 1 = last) and whether it made it in - the recency control
                row["evidence"] = [[p, tid in ctx.turn_ids] for tid, p in positions]
                yield row
            for k in top_k:
                ctx = pack(plan.units[:k], 10**9, h.tokens)
                yield _row(base, name, "k", k, q, ctx)


def evidence_positions(q: Question, h: History) -> list[tuple[str, float]]:
    """(turn id, relative position in the history) for each gold evidence turn."""
    last = max(len(h.turns) - 1, 1)
    return sorted(
        (tid, round(h.position[tid] / last, 4)) for tid in q.evidence_turns if tid in h.position
    )


def _row(base: dict, strategy: str, axis: str, value: int, q: Question, ctx: Context) -> dict:
    s = score(q, ctx.turn_ids)
    row = {**base, "strategy": strategy, axis: value, "tokens": ctx.tokens}
    if s is None:
        row["measurable"] = False
        return row
    row["measurable"] = True
    for key in ("recall", "all", "session_recall", "all_sessions", "newest", "stale_only"):
        v = getattr(s, key)
        if v is not None:
            row[key] = round(v, 4) if isinstance(v, float) else v
    return row


def plan_recall(q: Question, h: History, plan: Plan, budget: int) -> float | None:
    """Recall of one plan at one budget - the dev-split tuning objective."""
    s = score(q, plan.pack(budget, h.tokens).turn_ids)
    return None if s is None else s.recall
