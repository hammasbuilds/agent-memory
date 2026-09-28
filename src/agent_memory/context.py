"""The context builder: pack retrieved memory into a token budget.

Every strategy - including "just paste the whole history" - goes through `pack`, so
all of them pay for the same rendering (speaker labels, one date header per session)
and are measured with the same token counter. A retrieved turn without its session
date is useless for "when did..." questions, so the header is not optional.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime

from agent_memory.datasets import Turn, chrono
from agent_memory.forget import truncate
from agent_memory.text import count_tokens


def header(turn: Turn) -> str:
    return f"[session {turn.session_id} - {turn.timestamp:%Y-%m-%d %H:%M} ({turn.timestamp:%A})]"


def line(turn: Turn) -> str:
    return f"{turn.speaker}: {turn.text}"


FACTS_HEADER = "[known facts - newest value of each]"


@dataclass(frozen=True)
class Context:
    turns: tuple[Turn, ...]  # grouped by session, sessions by earliest turn
    tokens: int  # everything render() returns, facts included
    budget: int
    facts: tuple[str, ...] = ()  # rendered fact lines, shown before the turns
    cut: frozenset[str] = frozenset()  # ids of turns included only in part (truncated)

    @property
    def turn_ids(self) -> frozenset[str]:
        return frozenset(t.id for t in self.turns)

    def render(self) -> str:
        out: list[str] = [FACTS_HEADER, *self.facts] if self.facts else []
        current = None
        for t in self.turns:
            if t.session_id != current:
                current = t.session_id
                out.append(header(t))
            out.append(line(t))
        return "\n".join(out)


class TokenCache:
    """Per-turn token counts, computed once; a dict keyed by turn id."""

    def __init__(self) -> None:
        self._line: dict[str, int] = {}
        self._head: dict[tuple[str, datetime], int] = {}

    def line(self, t: Turn) -> int:
        n = self._line.get(t.id)
        if n is None:
            n = self._line[t.id] = count_tokens(line(t))
        return n

    def header(self, t: Turn) -> int:
        """The cost of the session header `t` would render (it shows `t`'s own time)."""
        key = (t.session_id, t.timestamp)
        n = self._head.get(key)
        if n is None:
            n = self._head[key] = count_tokens(header(t))
        return n


def pack(
    units: Iterable[Sequence[Turn]],
    budget: int,
    cache: TokenCache,
    *,
    positional: bool = False,
    start: Context | None = None,
) -> Context:
    """Greedily add units (a turn, a turn with neighbours, a whole session) in the order
    given until the budget is spent.

    Ranked strategies: a unit that does not fit whole is added turn by turn as far as it
    fits, then packing skips on to the next unit (a later, shorter one may still fit).

    Positional strategies (head truncation, sliding window) set `positional`: the first
    turn that does not fit is cut to fill the remaining budget and packing stops, which
    is what truncating a long prompt does. Without the cut, a window that meets one
    long turn stops hundreds of tokens short and loses to ranked strategies on fill,
    not on choice. A cut turn counts as present for `recall` and absent for
    `recall_strict` (see `evaluate`); both are reported.

    `start` continues from an existing context (used to split one budget between a
    recent window and search results).
    """
    if budget < 0:
        raise ValueError(f"budget must be non-negative, got {budget}")
    chosen: dict[str, Turn] = {t.id: t for t in start.turns} if start else {}
    # `render` heads each session with its earliest turn in context, so a turn earlier
    # than the current head of its session also pays the change in header cost
    heads: dict[str, Turn] = {}
    for t in sorted(chosen.values(), key=chrono):
        heads.setdefault(t.session_id, t)
    cut_ids: set[str] = set(start.cut) if start else set()
    used = start.tokens if start else 0
    for unit in units:
        full = True
        for t in unit:
            if t.id in chosen:
                continue
            old = heads.get(t.session_id)
            new_head = old is None or chrono(t) < chrono(old)
            head = cache.header(t) - (cache.header(old) if old else 0) if new_head else 0
            cost = cache.line(t) + head
            if used + cost > budget:
                full = False
                if positional and (cut := _cut_to_fit(t, budget - used - head, cache)):
                    chosen[t.id], cut_cost = cut
                    cut_ids.add(t.id)
                    if new_head:
                        heads[t.session_id] = t
                    used += cut_cost + head
                    break
                continue
            chosen[t.id] = t
            if new_head:
                heads[t.session_id] = t
            used += cost
        if not full and positional:
            break
    # one block per session (one header each, as costed), sessions in order of their
    # earliest turn, turns in order within a session
    ordered = sorted(chosen.values(), key=lambda t: (chrono(heads[t.session_id]), chrono(t)))
    return Context(tuple(ordered), used, budget, cut=frozenset(cut_ids))


def _cut_to_fit(t: Turn, room: int, cache: TokenCache) -> tuple[Turn, int] | None:
    """`t` with its text cut so its line costs at most `room` tokens, and that cost;
    None if not even a few words fit."""
    label = cache.line(t) - count_tokens(t.text)  # "speaker: "
    if room - label < MIN_CUT_TOKENS:
        return None
    short = truncate(t, room - label)
    cost = count_tokens(line(short))
    return (short, cost) if cost <= room else None


MIN_CUT_TOKENS = 8
