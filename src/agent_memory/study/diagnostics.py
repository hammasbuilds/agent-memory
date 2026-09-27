"""Mechanism diagnostics that need no strategy: how big and how long the histories
are, where evidence sits, whether the question's words appear in it, where answers
live, and the data-quality flags."""

from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean

from agent_memory.analysis import answer_location, evidence_position, lexical_visibility
from agent_memory.evaluate import history_key
from agent_memory.report import ALL
from agent_memory.study.common import Run
from agent_memory.temporal import query_window
from agent_memory.text import count_tokens


def _rate(xs: list[bool]) -> float:
    return round(mean(xs), 4) if xs else 0.0


def diagnostics_stage(run: Run) -> None:
    out = []
    for ds, qs in run.datasets():
        vis: dict[str, list[list[bool]]] = defaultdict(list)
        loc: dict[str, Counter] = defaultdict(Counter)
        timed: dict[str, list[bool]] = defaultdict(list)
        position: dict[str, list[float]] = defaultdict(list)
        flags: dict[str, Counter] = defaultdict(Counter)
        spans: list[float] = []
        sizes: list[int] = []
        seen: set[str] = set()
        for q in qs:
            if history_key(q) not in seen:
                seen.add(history_key(q))
                sizes.append(sum(count_tokens(t.text) for s in q.history for t in s.turns))
                spans.append((q.now - q.history[0].timestamp).total_seconds() / 86400)
            for f in q.flags:
                flags[f][q.qtype] += 1
                flags[f][ALL] += 1
            if (p := evidence_position(q)) is not None:
                position[q.qtype].append(p)
                position[ALL].append(p)
            timed[q.qtype].append(query_window(q.question, q.now) is not None)
            if (v := lexical_visibility(q)) is not None:
                vis[q.qtype].append(v)
                vis[ALL].append(v)
            if (where := answer_location(q)) is not None:
                loc[q.qtype][where] += 1
        out.append(
            {
                "dataset": ds,
                "histories": len(sizes),
                "history_tokens": {
                    "mean": round(mean(sizes)),
                    "min": min(sizes),
                    "max": max(sizes),
                },
                "history_span_days": {
                    "mean": round(mean(spans), 1),
                    "min": round(min(spans), 1),
                    "max": round(max(spans), 1),
                },
                "lexical_visibility": {
                    qt: {
                        "evidence_turns": sum(len(x) for x in v),
                        "evidence_turns_sharing_no_word": _rate([not b for x in v for b in x]),
                        "questions": len(v),
                        "questions_with_no_visible_evidence": _rate([not any(x) for x in v]),
                    }
                    for qt, v in sorted(vis.items())
                },
                "evidence_position": {
                    qt: {
                        "unit": "per question: mean position of its evidence turns (1 = last turn)",
                        "mean": round(mean(v), 4),
                        "questions_in_newest_10pct": _rate([x >= 0.9 for x in v]),
                        "n": len(v),
                    }
                    for qt, v in sorted(position.items())
                },
                "names_a_time": {
                    qt: {"rate": _rate(v), "n": len(v)} for qt, v in sorted(timed.items())
                },
                "answer_location": {
                    qt: {k: round(n / sum(c.values()), 4) for k, n in sorted(c.items())}
                    | {"n": sum(c.values())}
                    for qt, c in sorted(loc.items())
                },
                "data_quality_flags": {
                    f: dict(sorted(c.items())) for f, c in sorted(flags.items())
                },
            }
        )
    run.write("diagnostics.json", out)
