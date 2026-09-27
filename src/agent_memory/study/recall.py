"""The main stage: every strategy x budget x top-k on every question, both splits, and
the comparisons, controls and subsets the README quotes."""

from __future__ import annotations

import gzip
import json
import time
from dataclasses import asdict

from agent_memory import retrieval as R
from agent_memory.evaluate import HEADLINE_BUDGET, evaluate
from agent_memory.report import budget_to_reach, by_position, compare, summarise
from agent_memory.study.common import Chosen, Run, log

COMPARISONS = [
    ("store", "bm25_turns"),
    ("bm25_turns", "sliding_window"),
    ("bm25_turns", "full_head"),
    ("bm25_turns", "window_bm25"),
    ("window_bm25", "sliding_window"),
    ("bm25_sessions", "bm25_turns"),
    ("recency_bm25", "bm25_turns"),
    ("sliding_window", "random"),
]
ABLATIONS = ("session", "recency", "window", "neighbours")


def strategies(chosen: Chosen) -> dict[str, R.Strategy]:
    cfg = chosen.store
    s: dict[str, R.Strategy] = {
        "full_head": R.full_head,
        "sliding_window": R.sliding_window,
        "random": R.random_turns,
        "bm25_turns": R.bm25_turns,
        "bm25_sessions": R.bm25_sessions,
        "window_bm25": R.make_window_bm25(chosen.recent_share),
        "recency_bm25": R.make_recency_bm25(chosen.half_life_days),
        "store": R.make_store(cfg),
    }
    for comp in ABLATIONS:
        off = cfg.without(comp)
        if off != cfg:  # a component the dev sweep already switched off needs no ablation
            s[f"store-no-{comp}"] = R.make_store(off)
    return s


def _comparisons(rows: list[dict], pairs: list[tuple[str, str]], **kw: object) -> list[dict]:
    return [
        c
        for a, b in pairs
        for split in ("test", "dev")
        for c in compare(rows, a, b, value=HEADLINE_BUDGET, split=split, **kw)  # type: ignore[arg-type]
    ]


def main_stage(run: Run) -> None:
    chosen = run.chosen()
    cfg = chosen.store
    strats = strategies(chosen)
    rows_dir = run.results / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for name, qs in run.datasets():
        t0 = time.time()
        part = list(evaluate(qs, strats))
        log(f"{name}: {len(part)} rows in {time.time() - t0:.0f}s")
        # mtime=0: a rerun that reproduces the rows reproduces the file byte for byte
        with gzip.GzipFile(rows_dir / f"{name}.jsonl.gz", "wb", mtime=0) as fh:
            fh.write("".join(json.dumps(r) + "\n" for r in part).encode("utf-8"))
        rows += part
    pairs = COMPARISONS + [("store", f"store-no-{c}") for c in ABLATIONS]
    pairs = [p for p in pairs if p[1] in strats]
    headline = [r for r in rows if r.get("budget") == HEADLINE_BUDGET]
    by_budget = summarise(rows, "budget")
    out = {
        "config": {
            "store": asdict(cfg),
            "recency_half_life_days": chosen.half_life_days,
            "window_bm25_recent_share": chosen.recent_share,
        },
        "headline_budget": HEADLINE_BUDGET,
        "by_budget": by_budget,
        "by_k": summarise(rows, "k", b=1000),
        "comparisons": _comparisons(rows, pairs),
        "session_level_comparisons": _comparisons(rows, pairs[:6], metric="session_recall"),
        "knowledge_update_newest": [
            c
            for c in _comparisons(
                rows, [("recency_bm25", "bm25_turns"), ("store", "bm25_turns")], metric="newest"
            )
            if c["qtype"] == "knowledge-update" and c["split"] == "test"
        ],
        "budget_to_reach": budget_to_reach(by_budget, 0.5) + budget_to_reach(by_budget, 0.8),
        # the recency control: recall by where the evidence sits in the history
        "by_evidence_position": by_position(rows, HEADLINE_BUDGET) + by_position(rows, 512),
        # the window boost can only act on questions that name a time
        "time_expression_questions": _comparisons(
            [r for r in headline if r["time_expr"]],
            [p for p in (("store", "store-no-window"), ("store", "bm25_turns")) if p[1] in strats],
        ),
        # LongMemEval data-quality subsets: are the headline comparisons robust to them?
        "subsets": {
            "turn_key_complete": {
                "why": "drops questions with an answer session that has no has_answer turn",
                "by_budget": summarise(
                    [r for r in headline if "partial_key" not in r["flags"]], "budget"
                ),
            },
            "no_future_sessions": {
                "why": "drops questions whose history has sessions dated after the question",
                "comparisons": _comparisons(
                    [r for r in headline if "future_sessions" not in r["flags"]], pairs[:6]
                ),
            },
        },
    }
    run.write("retrieval.json", out)
