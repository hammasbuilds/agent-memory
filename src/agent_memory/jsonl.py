"""Conversation turns from JSON Lines - the plain input format for the store."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

from agent_memory.datasets import Session, Turn
from agent_memory.temporal import naive

FIELDS = ("session", "time", "speaker", "text")


def read_jsonl(path: Path) -> list[Session]:
    """Sessions from a file of lines {"session": id, "time": ISO 8601, "speaker": name,
    "text": ...}, ready for `MemoryStore.ingest`. Blank lines are ignored; a malformed
    line raises ValueError naming the file and line number. Times with a UTC offset are
    converted to naive UTC, like everything else in the store."""
    turns: dict[str, list[Turn]] = {}
    starts: dict[str, datetime] = {}
    repeats: dict[tuple[str, str], int] = {}
    for n, raw in enumerate(path.read_text("utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
            if not isinstance(rec, dict):
                raise TypeError(f"expected a JSON object, got {type(rec).__name__}")
            if missing := [k for k in FIELDS if k not in rec]:
                raise KeyError(f"missing {', '.join(repr(k) for k in missing)}")
            sid, speaker, text = (_string(rec, k) for k in ("session", "speaker", "text"))
            if not isinstance(rec["time"], str):
                raise TypeError(f"'time' must be an ISO 8601 string, got {rec['time']!r}")
            ts = naive(datetime.fromisoformat(rec["time"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            why = e.args[0] if isinstance(e, KeyError) else e
            raise ValueError(
                f"{path}:{n}: {why} (each line needs session, time (ISO 8601), speaker, text)"
            ) from None
        # The id is a hash of what was said, when and by whom (plus a counter for exact
        # repeats), so re-ingesting a file is a no-op and a second file that continues
        # a session cannot collide with the first.
        key = f"{ts.isoformat()}|{speaker}|{text}"
        repeats[(sid, key)] = repeats.get((sid, key), 0) + 1
        digest = hashlib.sha1(f"{key}|{repeats[(sid, key)]}".encode()).hexdigest()[:12]
        seq = len(turns.setdefault(sid, []))
        turns[sid].append(Turn(f"{sid}:{digest}", sid, speaker, text, ts, seq))
        starts[sid] = min(starts.get(sid, ts), ts)
    return [Session(sid, starts[sid], tuple(ts)) for sid, ts in turns.items()]


def _string(rec: dict, key: str) -> str:
    """rec[key] as text. A session id may be a number; speaker and text must be strings -
    `str(None)` would store the word "None" as something someone said."""
    v = rec[key]
    if key == "session" and isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    if not isinstance(v, str):
        raise TypeError(f"{key!r} must be a string, got {type(v).__name__}")
    return v
