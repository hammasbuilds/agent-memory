"""The persistent store: SQLite tables for sessions, raw turns and extracted facts.

Raw turns are never rewritten by fact extraction - facts point back at the turn they
came from (`source_turn`), so every fact has provenance. Facts are versioned per
(subject, attribute): a newer value supersedes the older one, the older one is kept,
and `facts_as_of` answers "what was true on date X". Forgetting is a soft delete
(`forgotten_at`) until `purge()`.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from agent_memory.bm25 import BM25
from agent_memory.context import FACTS_HEADER, Context
from agent_memory.datasets import Session, Turn
from agent_memory.forget import truncate
from agent_memory.retrieval import History, Retriever, RetrieverConfig
from agent_memory.temporal import naive, utc_now
from agent_memory.text import count_tokens, terms

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    next_seq INTEGER NOT NULL DEFAULT 0  -- never reused, even after a purge
);
CREATE TABLE IF NOT EXISTS turns (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    seq INTEGER NOT NULL,
    speaker TEXT NOT NULL,
    text TEXT NOT NULL,
    ts TEXT NOT NULL,
    ingested_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),
    forgotten_at TEXT
);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, seq);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    subject TEXT NOT NULL,
    attribute TEXT NOT NULL,
    value TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    superseded_by INTEGER REFERENCES facts(id),
    source_turn TEXT REFERENCES turns(id),
    extractor TEXT NOT NULL,
    recorded_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now'))
);
CREATE INDEX IF NOT EXISTS facts_key ON facts(subject, attribute, valid_from);
"""


FACT_RELATIVE_CUTOFF = 0.5


@dataclass(frozen=True)
class Ingested:
    added: int
    already_stored: int
    empty: int  # turns with no text, skipped


@dataclass(frozen=True)
class Fact:
    id: int
    subject: str
    attribute: str
    value: str
    valid_from: datetime
    superseded_by: int | None
    source_turn: str | None
    extractor: str

    @property
    def current(self) -> bool:
        return self.superseded_by is None

    def render(self) -> str:
        return f"{self.subject} / {self.attribute}: {self.value} (since {self.valid_from:%Y-%m-%d})"


def _key(s: str) -> str:
    return " ".join(s.lower().split())


class MemoryStore:
    """A cross-session memory backed by one SQLite file (or ':memory:').

    Time is kept as naive UTC. A timezone-aware datetime passed in anywhere (turn times,
    `valid_from`, `now`) is converted to UTC first; a naive one is taken to be UTC
    already, so pass aware datetimes unless your clock really is UTC. When `now` is
    omitted it is the current UTC time.
    """

    def __init__(self, path: str | Path = ":memory:", config: RetrieverConfig | None = None):
        self.db = sqlite3.connect(str(path))
        self.db.executescript(SCHEMA)
        self.retriever = Retriever(config)
        self._history: History | None = None

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> MemoryStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---- ingestion -------------------------------------------------------------

    def add_turn(self, session_id: str, speaker: str, text: str, ts: datetime) -> str:
        """Append one turn to a session (created on first use). Returns the turn id."""
        ts = naive(ts)
        if not text.strip():
            raise ValueError("refusing to store an empty turn")
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO sessions(id, started_at) VALUES (?, ?)",
                (session_id, ts.isoformat()),
            )
            (seq,) = self.db.execute(
                "UPDATE sessions SET next_seq = next_seq + 1 WHERE id = ? RETURNING next_seq - 1",
                (session_id,),
            ).fetchone()
            tid = f"{session_id}:{seq}"
            self.db.execute(
                "INSERT INTO turns(id, session_id, seq, speaker, text, ts) VALUES (?,?,?,?,?,?)",
                (tid, session_id, seq, speaker, text, ts.isoformat()),
            )
        self._history = None
        return tid

    def ingest(self, sessions: Iterable[Session]) -> Ingested:
        """Bulk-load sessions, keeping their turn ids.

        A turn whose id is already stored is skipped, so re-ingesting is idempotent. A
        turn with no text is skipped too, as `add_turn` refuses one (LongMemEval_S has
        12). A new turn is appended after everything already in its session (its
        position comes from the session's counter, not from the input), so a second
        file that continues a session lands after the first.
        """
        added = already = empty = 0
        with self.db:
            for s in sessions:
                kept = [t for t in s.turns if t.text.strip()]
                empty += len(s.turns) - len(kept)
                if not kept:  # a session with nothing said is not stored at all
                    continue
                self.db.execute(
                    "INSERT OR IGNORE INTO sessions(id, started_at) VALUES (?, ?)",
                    (s.id, naive(s.timestamp).isoformat()),
                )
                for t in kept:
                    if self.db.execute("SELECT 1 FROM turns WHERE id=?", (t.id,)).fetchone():
                        already += 1
                        continue
                    (seq,) = self.db.execute(
                        "UPDATE sessions SET next_seq = next_seq + 1 WHERE id = ? "
                        "RETURNING next_seq - 1",
                        (s.id,),
                    ).fetchone()
                    self.db.execute(
                        "INSERT INTO turns(id, session_id, seq, speaker, text, ts) "
                        "VALUES (?,?,?,?,?,?)",
                        (t.id, s.id, seq, t.speaker, t.text, naive(t.timestamp).isoformat()),
                    )
                    added += 1
        self._history = None
        return Ingested(added, already, empty)

    # ---- turns & retrieval -------------------------------------------------------

    def sessions(self, include_forgotten: bool = False) -> list[Session]:
        where = "" if include_forgotten else "WHERE t.forgotten_at IS NULL"
        rows = self.db.execute(
            "SELECT t.id, t.session_id, t.speaker, t.text, t.ts, t.seq, s.started_at "
            f"FROM turns t JOIN sessions s ON s.id = t.session_id {where} "
            "ORDER BY s.started_at, t.session_id, t.seq"
        ).fetchall()
        grouped: dict[str, list[Turn]] = {}
        starts: dict[str, str] = {}
        for tid, sid, speaker, text, ts, seq, started in rows:
            grouped.setdefault(sid, []).append(
                Turn(tid, sid, speaker, text, datetime.fromisoformat(ts), seq)
            )
            starts[sid] = started
        return [
            Session(sid, datetime.fromisoformat(starts[sid]), tuple(ts))
            for sid, ts in grouped.items()
        ]

    @property
    def history(self) -> History:
        if self._history is None:
            self._history = History(self.sessions())
        return self._history

    def history_at(self, now: datetime) -> History:
        """The history as it stood at `now`: turns dated after it are left out."""
        full = self.history
        if all(t.timestamp <= now for t in full.turns):
            return full
        return History(
            [
                Session(s.id, s.timestamp, kept)
                for s in full.sessions
                if (kept := tuple(t for t in s.turns if t.timestamp <= now))
            ]
        )

    def search(self, query: str, now: datetime | None = None, k: int = 10) -> list[Turn]:
        """The k best turns said at or before `now` (default: the current UTC time)."""
        now = naive(now or utc_now())
        return self.retriever.search(self.history_at(now), query, now, k)

    def context(self, query: str, budget: int = 2048, now: datetime | None = None) -> Context:
        """Relevant facts as they stood at `now` first, then retrieved turns said at or
        before `now`, within `budget` tokens. `now` defaults to the current UTC time."""
        now = naive(now or utc_now())
        facts = self._relevant_facts(query, now)
        lines: list[str] = []
        used = count_tokens(FACTS_HEADER) if facts else 0
        for f in facts:
            cost = count_tokens(f.render())
            if used + cost > budget // 2:  # facts may take at most half the budget
                break
            lines.append(f.render())
            used += cost
        if not lines:
            used = 0
        ctx = self.retriever.context(self.history_at(now), query, now, budget - used)
        return replace(ctx, facts=tuple(lines), tokens=ctx.tokens + used, budget=budget)

    # ---- facts ------------------------------------------------------------------

    def remember(
        self,
        subject: str,
        attribute: str,
        value: str,
        valid_from: datetime,
        source_turn: str | None = None,
        extractor: str = "manual",
    ) -> Fact:
        """Record that `subject`'s `attribute` is `value` from `valid_from` on.

        A newer value supersedes the current one; the old version is kept. A value that
        arrives late (older `valid_from` than the current one) slots into the history
        without displacing the current value. Restating the value already in force at
        that time is a no-op that returns the existing fact.
        """
        subject, attribute, value = _key(subject), _key(attribute), value.strip()
        valid_from = naive(valid_from)
        if not (subject and attribute and value):
            raise ValueError("subject, attribute and value must all be non-empty")
        versions = self.history_of(subject, attribute)
        prior = [f for f in versions if f.valid_from <= valid_from]
        if prior and _key(prior[-1].value) == _key(value):
            return prior[-1]
        with self.db:
            cur = self.db.execute(
                "INSERT INTO facts(subject, attribute, value, valid_from, source_turn, extractor)"
                " VALUES (?,?,?,?,?,?)",
                (subject, attribute, value, valid_from.isoformat(), source_turn, extractor),
            )
            new_id = cur.lastrowid
            self._relink(subject, attribute)
        return next(f for f in self.history_of(subject, attribute) if f.id == new_id)

    def _relink(self, subject: str, attribute: str) -> None:
        """Point each version's `superseded_by` at the next-newer version."""
        ids = [
            r[0]
            for r in self.db.execute(
                "SELECT id FROM facts WHERE subject=? AND attribute=? ORDER BY valid_from, id",
                (subject, attribute),
            )
        ]
        for older, newer in zip(ids, [*ids[1:], None], strict=True):
            self.db.execute("UPDATE facts SET superseded_by=? WHERE id=?", (newer, older))

    def history_of(self, subject: str, attribute: str) -> list[Fact]:
        """Every version of one fact, oldest first."""
        return self._facts("subject=? AND attribute=?", (_key(subject), _key(attribute)))

    def current_facts(self, subject: str | None = None) -> list[Fact]:
        if subject is None:
            return self._facts("superseded_by IS NULL", ())
        return self._facts("superseded_by IS NULL AND subject=?", (_key(subject),))

    def facts_as_of(self, when: datetime, subject: str | None = None) -> list[Fact]:
        """The value of every fact as it stood at `when`."""
        rows = self._facts("valid_from <= ?", (naive(when).isoformat(),))
        latest: dict[tuple[str, str], Fact] = {}
        for f in rows:  # ordered by valid_from, so the last one wins
            if subject is None or f.subject == _key(subject):
                latest[(f.subject, f.attribute)] = f
        return list(latest.values())

    def _facts(self, where: str, args: tuple) -> list[Fact]:
        rows = self.db.execute(
            "SELECT id, subject, attribute, value, valid_from, superseded_by, source_turn, "
            f"extractor FROM facts WHERE {where} ORDER BY valid_from, id",
            args,
        ).fetchall()
        return [
            Fact(i, s, a, v, datetime.fromisoformat(vf), sb, st, ex)
            for i, s, a, v, vf, sb, st, ex in rows
        ]

    def _relevant_facts(self, query: str, now: datetime, k: int = 20) -> list[Fact]:
        facts = self.facts_as_of(now)
        if not facts:
            return []
        # A fact is indexed with the turn it came from, so "where do I work" finds an
        # 'employer' fact whose source turn says "I work at...".
        ids = [f.source_turn for f in facts if f.source_turn]
        source = dict(
            self.db.execute(
                "SELECT id, text FROM turns WHERE forgotten_at IS NULL AND id IN "
                f"({','.join('?' * len(ids))})",
                ids,
            ).fetchall()
        )
        index = BM25(
            [
                terms(f"{f.subject} {f.attribute} {f.value} {source.get(f.source_turn, '')}")
                for f in facts
            ]
        )
        # BM25's idf already down-weights a term found in every fact (usually the
        # subject, "user"), but it still scores above zero, so every fact would match.
        # Keep only facts scoring at least `FACT_RELATIVE_CUTOFF` of the best one: a
        # lone fact, or facts that all match equally, are kept; a fact matching only on
        # a common term is dropped when another matches on a rarer one.
        scores = index.scores(terms(query))
        top = max(scores, default=0.0)
        if top <= 0:
            return []
        keep = [i for i, s in enumerate(scores) if s >= FACT_RELATIVE_CUTOFF * top]
        return [facts[i] for i in sorted(keep, key=lambda i: -scores[i])[:k]]

    # ---- forgetting ---------------------------------------------------------------

    def forget(self, policy: Callable[[Turn], bool], now: datetime | None = None) -> int:
        """Soft-delete every live turn the policy selects. Returns how many."""
        doomed = [t.id for s in self.sessions() for t in s.turns if policy(t)]
        stamp = naive(now or utc_now()).isoformat()
        with self.db:
            self.db.executemany(
                "UPDATE turns SET forgotten_at=? WHERE id=?", [(stamp, i) for i in doomed]
            )
        self._history = None
        return len(doomed)

    def compact(self, max_tokens: int) -> int:
        """Truncate every live turn longer than `max_tokens`. Returns how many changed.
        Unlike `forget`, this rewrites the stored text and cannot be undone."""
        changed = [
            (short.text, t.id)
            for s in self.sessions()
            for t in s.turns
            if (short := truncate(t, max_tokens)) is not t
        ]
        with self.db:
            self.db.executemany("UPDATE turns SET text=? WHERE id=?", changed)
        self._history = None
        return len(changed)

    def purge(self) -> int:
        """Hard-delete forgotten turns. Facts keep their text; their provenance link
        is cleared rather than left dangling."""
        with self.db:
            self.db.execute(
                "UPDATE facts SET source_turn=NULL WHERE source_turn IN "
                "(SELECT id FROM turns WHERE forgotten_at IS NOT NULL)"
            )
            n = self.db.execute("DELETE FROM turns WHERE forgotten_at IS NOT NULL").rowcount
        self._history = None
        return n

    def stats(self) -> dict[str, int]:
        q = self.db.execute
        return {
            "sessions": q("SELECT COUNT(*) FROM sessions").fetchone()[0],
            "turns": q("SELECT COUNT(*) FROM turns WHERE forgotten_at IS NULL").fetchone()[0],
            "forgotten_turns": q(
                "SELECT COUNT(*) FROM turns WHERE forgotten_at IS NOT NULL"
            ).fetchone()[0],
            "facts_current": q("SELECT COUNT(*) FROM facts WHERE superseded_by IS NULL").fetchone()[
                0
            ],
            "facts_superseded": q(
                "SELECT COUNT(*) FROM facts WHERE superseded_by IS NOT NULL"
            ).fetchone()[0],
        }
