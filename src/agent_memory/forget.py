"""Forgetting and compaction policies.

A forgetting policy is a predicate over turns: True means "drop this one". Compaction
keeps the turn but shortens it. Both are pure functions over sessions so the study can
measure exactly which gold evidence each one throws away; `MemoryStore.forget` applies
the same predicates to the database.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from datetime import datetime, timedelta

from agent_memory.datasets import Session, Turn
from agent_memory.text import count_tokens, terms

Policy = Callable[[Turn], bool]


def phatic(min_terms: int = 3) -> Policy:
    """Turns with fewer than `min_terms` content words: greetings, 'haha nice', 'thanks!'."""
    return lambda t: len(terms(t.text)) < min_terms


def role(speaker: str) -> Policy:
    """Every turn by one speaker - e.g. the assistant's side of a chat."""
    return lambda t: t.speaker == speaker


def older_than(now: datetime, days: float) -> Policy:
    """Everything said more than `days` before `now`."""
    cutoff = now - timedelta(days=days)
    return lambda t: t.timestamp < cutoff


def truncate(turn: Turn, max_tokens: int) -> Turn:
    """Keep roughly the first `max_tokens` tokens of a turn, cut at a word boundary."""
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be positive, got {max_tokens}")
    if count_tokens(turn.text) <= max_tokens:
        return turn
    words = turn.text.split()
    lo, hi = 0, len(words)
    while lo < hi:  # the longest word prefix that fits, with room for the ellipsis
        mid = (lo + hi + 1) // 2
        if count_tokens(" ".join(words[:mid])) + 1 <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return replace(turn, text=" ".join(words[:lo]) + " …")


def apply(
    sessions: Sequence[Session], drop: Policy | None = None, max_tokens: int | None = None
) -> list[Session]:
    """Sessions with dropped turns removed and long turns truncated; empty sessions go."""
    out = []
    for s in sessions:
        kept = tuple(
            truncate(t, max_tokens) if max_tokens else t
            for t in s.turns
            if drop is None or not drop(t)
        )
        if kept:
            out.append(replace(s, turns=kept))
    return out


POLICY_HELP = (
    "phatic[:N] (fewer than N content words, default 3) | role:NAME | "
    "older-than:DAYS | truncate:TOKENS"
)


def parse_policy(spec: str, now: datetime) -> tuple[Policy | None, int | None]:
    """Parse a CLI policy string into (drop predicate, truncation length)."""
    name, _, arg = spec.partition(":")
    try:
        if name == "phatic":
            return phatic(int(arg) if arg else 3), None
        if name == "role" and arg:
            return role(arg), None
        if name == "older-than" and arg:
            return older_than(now, float(arg)), None
        if name == "truncate" and arg:
            n = int(arg)
            if n > 0:
                return None, n
    except ValueError:
        pass
    raise ValueError(f"bad policy {spec!r}; expected {POLICY_HELP}")
