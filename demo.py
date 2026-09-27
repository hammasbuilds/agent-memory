"""agent-memory on inputs whose right answers are known.

Part 1 needs nothing: a four-session chat is written to a temporary SQLite store and
queried - a lookup, a time-scoped question, a fact that changes, a forgetting pass.

Part 2 runs if the benchmarks are on disk (scripts/fetch_data.py): four real questions,
each with its gold evidence, and what three memory strategies put into a 2,048-token
context for it. The four are illustrative, one per question type; the aggregate numbers
are in results/retrieval.json.

    uv run python demo.py
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path

from agent_memory import retrieval as R
from agent_memory.datasets import Question, Session, Turn, data_dir, iter_longmemeval, load_locomo
from agent_memory.evaluate import HEADLINE_BUDGET, history_via_store, score
from agent_memory.forget import phatic
from agent_memory.retrieval import RetrieverConfig
from agent_memory.store import MemoryStore


def check(label: str, got: object, want: object) -> None:
    mark = "ok  " if got == want else "FAIL"
    print(f"  [{mark}] {label}: {got!r}" + ("" if got == want else f"  (expected {want!r})"))


def session(sid: str, when: datetime, lines: list[tuple[str, str]]) -> Session:
    return Session(sid, when, tuple(Turn(f"{sid}:{i}", sid, s, t, when, i)
                                    for i, (s, t) in enumerate(lines)))  # fmt: skip


CHAT = [
    session("jan", datetime(2024, 1, 10, 9), [
        ("user", "Hi!"),
        ("user", "I just adopted a beagle puppy called Biscuit."),
        ("assistant", "Congratulations! Beagles are lovely."),
    ]),
    session("mar", datetime(2024, 3, 5, 18), [
        ("user", "Work at Riverside Hospital is exhausting, twelve-hour nursing shifts."),
        ("assistant", "That sounds demanding - make sure you rest."),
    ]),
    session("may", datetime(2024, 5, 14, 20), [
        ("user", "Planning a walking holiday in the Lake District for September."),
        ("user", "ok thanks"),
    ]),
    session("jun", datetime(2024, 6, 20, 12), [
        ("user", "Big news: I left Riverside, I'm now a nurse at St Mary's clinic."),
    ]),
]  # fmt: skip


def part_one() -> None:
    print("PART 1 - a store with known answers")
    now = datetime(2024, 7, 1)
    with tempfile.TemporaryDirectory() as tmp, MemoryStore(Path(tmp) / "memory.db") as m:
        print(f"  ingested {m.ingest(CHAT).added} turns from {len(CHAT)} sessions")
        hit = m.search("what is my dog called?", now, k=1)[0]
        check("search 'what is my dog called?'", hit.id, "jan:1")

        ctx = m.context("where did I say I work, back in March?", budget=60, now=now)
        print("  context 'where did I say I work, back in March?' (60-token budget):")
        for line in ctx.render().splitlines():
            print(f"      {line}")
        check("the March turn is first in context", ctx.turns[0].id, "mar:0")
        check("tokens used <= budget", ctx.tokens <= 60, True)

        m.remember("user", "employer", "Riverside Hospital", CHAT[1].timestamp, "mar:0")
        m.remember("user", "employer", "St Mary's clinic", CHAT[3].timestamp, "jun:0")
        check("current employer", m.current_facts("user")[0].value, "St Mary's clinic")
        check(
            "employer as of 2024-04-01",
            m.facts_as_of(datetime(2024, 4, 1), "user")[0].value,
            "Riverside Hospital",
        )
        versions = m.history_of("user", "employer")
        check("old value kept, marked superseded", versions[0].superseded_by, versions[1].id)
        check("forget phatic turns ('Hi!', 'ok thanks')", m.forget(phatic(3), now), 2)
        print(f"  stats: {m.stats()}")


def show(q: Question, strategies: dict[str, R.Strategy]) -> None:
    h = history_via_store(q)
    gold = [t for t in h.turns if t.id in q.evidence_turns]
    print(f"\n  [{q.dataset} / {q.qtype}] {q.question}")
    print(f"  gold answer: {q.answer}")
    for t in gold:
        print(f"  gold evidence {t.id} ({t.timestamp:%d %b %Y}) {t.speaker}: {t.text[:90]}")
    for name, strat in strategies.items():
        ctx = strat(h, q.question, q.now, q.qid).pack(HEADLINE_BUDGET, h.tokens)
        s = score(q, ctx.turn_ids)
        assert s is not None
        extra = ""
        if s.newest is not None:
            extra = f", newest value found: {s.newest}"
        found = len(q.evidence_turns & ctx.turn_ids)
        sessions = round(s.session_recall * len(q.evidence_sessions)) if s.session_recall else 0
        print(
            f"    {name:15} {ctx.tokens:5} tokens, evidence turns {found}/{len(gold)},"
            f" answer sessions {sessions}/{len(q.evidence_sessions)}{extra}"
        )


def part_two() -> None:
    if not (data_dir() / "locomo10.json").exists():
        print("\nPART 2 skipped - run `uv run python scripts/fetch_data.py` for the benchmarks")
        return
    print(f"\nPART 2 - real questions, {HEADLINE_BUDGET}-token context")
    strategies = {
        "sliding_window": R.sliding_window,
        "bm25_turns": R.bm25_turns,
        "window_bm25": R.make_window_bm25(0.5),
        "store": R.make_store(RetrieverConfig()),
    }
    locomo = {q.qid: q for q in load_locomo()}
    for qid in ("conv-26:q0", "conv-26:q15", "conv-26:q152"):
        show(locomo[qid], strategies)
    lme_path = data_dir() / "longmemeval_s_cleaned.json"
    if lme_path.exists():
        ku = next(q for q in iter_longmemeval(lme_path) if q.qtype == "knowledge-update")
        show(ku, strategies)


if __name__ == "__main__":
    part_one()
    part_two()
