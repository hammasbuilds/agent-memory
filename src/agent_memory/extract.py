"""LLM fact extraction (model arm): one call per session turns what was said into
(subject, attribute, value) facts, each pointing back at the turn it came from, and
files them in the store - where a newer value supersedes an older one."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from agent_memory.datasets import Session
from agent_memory.llm import CHAT_MODEL, LLM
from agent_memory.store import MemoryStore

PROMPT = """You maintain a long-term memory about the people in a conversation.
Read the conversation session below (held on {date}) and list the durable facts it
states about its participants: what they own, do, like, plan, where they live and work,
relationships, events and when they happened. Resolve relative dates ("yesterday",
"last week") against the session date. Skip small talk and anything the assistant
merely suggests.

Reply with a JSON array only. Each item:
  {{"subject": "<person the fact is about>", "attribute": "<short, reusable attribute name,
    e.g. 'lives in', 'job', 'pet', 'went to'>", "value": "<the value>", "turn": <number of
    the turn that states it>}}
Use the same attribute name for the same kind of fact so later values can replace
earlier ones. Reply [] if there are no durable facts.

Session:
{turns}
"""


@dataclass(frozen=True)
class Extracted:
    subject: str
    attribute: str
    value: str
    turn: int


def build_prompt(session: Session) -> str:
    turns = "\n".join(f"[{i}] {t.speaker}: {t.text}" for i, t in enumerate(session.turns))
    return PROMPT.format(date=f"{session.timestamp:%Y-%m-%d %A}", turns=turns)


_ARRAY = re.compile(r"\[.*\]", re.S)


def parse(reply: str, n_turns: int) -> list[Extracted]:
    """Parse the model's JSON array, tolerating code fences and prose around it. Items
    with missing fields or an out-of-range turn number are dropped, not guessed."""
    m = _ARRAY.search(reply)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict):
            continue
        s, a, v, t = (it.get(k) for k in ("subject", "attribute", "value", "turn"))
        if not all(isinstance(x, str) and x.strip() for x in (s, a)) or v in (None, ""):
            continue
        if not isinstance(t, int) or not 0 <= t < n_turns:
            continue
        out.append(Extracted(str(s), str(a), str(v), t))
    return out


def extract_into(
    store: MemoryStore, sessions: list[Session], client: LLM, model: str = CHAT_MODEL
) -> dict[str, int]:
    """Extract facts from each session, oldest first, into the store."""

    def total() -> int:
        st = store.stats()
        return st["facts_current"] + st["facts_superseded"]

    start = total()
    counts = {"sessions": 0, "facts_parsed": 0}
    for s in sorted(sessions, key=lambda s: s.timestamp):
        found = parse(client.generate(model, build_prompt(s)), len(s.turns))
        counts["sessions"] += 1
        counts["facts_parsed"] += len(found)
        for f in found:
            store.remember(
                f.subject, f.attribute, f.value, s.timestamp, s.turns[f.turn].id, extractor=model
            )
    counts["facts_stored"] = total() - start  # restated values are not stored twice
    return counts
