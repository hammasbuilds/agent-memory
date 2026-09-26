"""Load LoCoMo and LongMemEval into one shape: a question, the conversation history it
is asked against, and the gold evidence (which turns and sessions hold the answer).

Question types are mapped onto one shared vocabulary so the two benchmarks can be
reported side by side; the dataset's own label is kept in `raw_type`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from agent_memory.temporal import parse_timestamp

# LoCoMo's integer categories. The released JSON numbers them differently from the order
# the paper lists them in; this mapping follows the data itself (every category-2
# question asks "when"; category-5 questions carry an `adversarial_answer`), and is the
# one the LoCoMo evaluation code and mem0's evaluation use.
LOCOMO_TYPES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "abstention"}

LME_TYPES = {
    "single-session-user": "single-hop",
    "single-session-assistant": "single-hop-assistant",
    "single-session-preference": "preference",
    "multi-session": "multi-hop",
    "temporal-reasoning": "temporal",
    "knowledge-update": "knowledge-update",
}


@dataclass(frozen=True)
class Turn:
    id: str
    session_id: str
    speaker: str
    text: str
    timestamp: datetime
    seq: int = 0  # position within its session


def chrono(t: Turn) -> tuple[datetime, str, int]:
    """Sort key putting turns back in conversational order."""
    return (t.timestamp, t.session_id, t.seq)


@dataclass(frozen=True)
class Session:
    id: str
    timestamp: datetime
    turns: tuple[Turn, ...]


@dataclass(frozen=True)
class Question:
    qid: str
    dataset: str
    qtype: str
    raw_type: str
    question: str
    answer: str
    now: datetime
    history: tuple[Session, ...]
    evidence_turns: frozenset[str] = field(default_factory=frozenset)
    evidence_sessions: frozenset[str] = field(default_factory=frozenset)


def data_dir() -> Path:
    """`$AGENT_MEMORY_DATA` if set, else `data/raw` in the repository."""
    env = os.environ.get("AGENT_MEMORY_DATA")
    return Path(env) if env else Path(__file__).resolve().parents[2] / "data" / "raw"


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `uv run python scripts/fetch_data.py` "
            "or point AGENT_MEMORY_DATA at a directory holding the benchmark files."
        )
    return path


def load_locomo(path: Path | None = None) -> list[Question]:
    """LoCoMo: 10 long two-person conversations, ~200 questions each, turn-level evidence."""
    raw = json.loads(_require(path or data_dir() / "locomo10.json").read_text("utf-8"))
    out: list[Question] = []
    for conv in raw:
        c = conv["conversation"]
        cid = conv["sample_id"]
        sessions = []
        n = 1
        while f"session_{n}" in c:
            ts = parse_timestamp(c[f"session_{n}_date_time"])
            sid = f"{cid}:S{n}"
            turns = tuple(
                Turn(
                    id=f"{cid}:{t['dia_id']}",
                    session_id=sid,
                    speaker=t["speaker"],
                    text=_locomo_text(t),
                    timestamp=ts,
                    seq=j,
                )
                for j, t in enumerate(c[f"session_{n}"])
            )
            sessions.append(Session(sid, ts, turns))
            n += 1
        history = tuple(sessions)
        known = {t.id for s in history for t in s.turns}
        now = max(s.timestamp for s in history)
        for i, qa in enumerate(conv["qa"]):
            ev = frozenset(f"{cid}:{e}" for e in _locomo_evidence(qa.get("evidence", [])))
            ev &= known  # a handful of evidence ids point at turns that do not exist
            if qa["category"] == 5:
                # Adversarial: the question pins one speaker's experience on the other.
                # `adversarial_answer` is the trap, not the gold.
                ans = (
                    "Not answerable - the conversation does not say this about them "
                    f"(the trap answer would be: {qa['adversarial_answer']})"
                )
            else:
                ans = qa["answer"]
            out.append(
                Question(
                    qid=f"{cid}:q{i}",
                    dataset="locomo",
                    qtype=LOCOMO_TYPES[qa["category"]],
                    raw_type=str(qa["category"]),
                    question=qa["question"],
                    answer=str(ans),
                    now=now,
                    history=history,
                    evidence_turns=ev,
                    evidence_sessions=frozenset(
                        e.rsplit(":", 1)[0].replace(":D", ":S") for e in ev
                    ),
                )
            )
    return out


def _locomo_text(t: dict) -> str:
    text = t["text"]
    if t.get("blip_caption"):
        text += f" [shares a photo: {t['blip_caption']}]"
    return text


def _locomo_evidence(ev: list[str]) -> list[str]:
    """Evidence ids are 'D<session>:<turn>', occasionally several in one string
    ('D8:6; D9:17') or with stray spaces."""
    out = []
    for item in ev:
        for part in item.replace(",", ";").split(";"):
            part = part.strip()
            if part.startswith("D") and ":" in part:
                out.append(part)
    return out


def iter_json_array(path: Path) -> Iterator[dict]:
    """Decode a top-level JSON array one element at a time.

    LongMemEval_S is 277 MB; `json.load` on it builds well over a gigabyte of Python
    objects. Decoding element by element keeps only the file text plus one question
    alive at a time.
    """
    text = path.read_text("utf-8")
    dec = json.JSONDecoder()
    i = text.index("[") + 1
    n = len(text)
    while True:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n:
            raise ValueError(f"{path}: unterminated JSON array")
        if text[i] == "]":
            return
        obj, i = dec.raw_decode(text, i)
        yield obj


def iter_longmemeval(path: Path | None = None) -> Iterator[Question]:
    """LongMemEval (oracle or S): 500 questions, each with its own chat history of
    user/assistant sessions; evidence is marked per turn (`has_answer`) and per session."""
    path = _require(path or data_dir() / "longmemeval_s_cleaned.json")
    for q in iter_json_array(path):
        qid = q["question_id"]
        sessions = []
        ev_turns = set()
        for sid, date, turns in zip(
            q["haystack_session_ids"], q["haystack_dates"], q["haystack_sessions"], strict=True
        ):
            ts = parse_timestamp(date)
            built = []
            for j, t in enumerate(turns):
                tid = f"{qid}:{sid}:{j}"
                built.append(Turn(tid, sid, t["role"], t["content"], ts, j))
                if t.get("has_answer"):
                    ev_turns.add(tid)
            sessions.append(Session(sid, ts, tuple(built)))
        sessions.sort(key=lambda s: s.timestamp)  # the file lists sessions out of order
        abstention = qid.endswith("_abs")
        yield Question(
            qid=qid,
            dataset="longmemeval",
            qtype="abstention" if abstention else LME_TYPES[q["question_type"]],
            raw_type=q["question_type"] + ("_abs" if abstention else ""),
            question=q["question"],
            answer=str(q["answer"]),
            now=parse_timestamp(q["question_date"]),
            history=tuple(sessions),
            evidence_turns=frozenset(ev_turns),
            evidence_sessions=frozenset(q["answer_session_ids"]),
        )
