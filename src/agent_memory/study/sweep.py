"""Tuning (dev split only) and the recency sensitivity sweep (test split, not tuning)."""

from __future__ import annotations

import itertools
from collections.abc import Iterable
from dataclasses import asdict
from statistics import mean

from agent_memory import retrieval as R
from agent_memory.datasets import Question
from agent_memory.evaluate import (
    HEADLINE_BUDGET,
    evaluate,
    history_key,
    history_via_store,
    plan_recall,
    split_of,
)
from agent_memory.report import compare, summarise
from agent_memory.retrieval import History
from agent_memory.study.common import Run, log

GRID = {
    "session_weight": (0.0, 0.15, 0.3, 0.6),
    "half_life_days": (0.0, 90.0, 365.0),
    "window_boost": (0.0, 1.0),
    "neighbours": (0, 1, 2),
}
HALF_LIVES = (7.0, 30.0, 90.0, 365.0, 1825.0)
SENSITIVITY_HALF_LIVES = (7.0, 30.0, 90.0, 365.0)
WINDOW_SHARES = (0.05, 0.1, 0.25, 0.5, 0.75)


def _dev(qs: Iterable[Question]) -> list[tuple[Question, History]]:
    out, cache = [], {}
    for q in qs:
        if split_of(q) != "dev":
            continue
        key = history_key(q)
        if key not in cache:
            cache[key] = history_via_store(q)
        out.append((q, cache[key]))
    return out


def _mean_recall(dev: list[tuple[Question, History]], strategy: R.Strategy) -> float:
    """Mean turn-level recall over dev questions that have gold turns (0 if none do)."""
    vals = [
        r
        for q, h in dev
        if (r := plan_recall(q, h, strategy(h, q.question, q.now, q.qid), HEADLINE_BUDGET))
        is not None
    ]
    return mean(vals) if vals else 0.0


def dev_sweep(run: Run) -> dict:
    dev = {ds: _dev(qs) for ds, qs in run.datasets()}
    log(f"dev split: {', '.join(f'{k} {len(v)} questions' for k, v in dev.items())}")

    dev = {ds: items for ds, items in dev.items() if items}
    if not dev:
        raise ValueError("no dev-split questions to tune on")

    def objective(strategy: R.Strategy) -> dict[str, float]:
        per = {ds: round(_mean_recall(items, strategy), 4) for ds, items in dev.items()}
        return per | {"macro": round(mean(per.values()), 4)}

    store_runs = []
    for values in itertools.product(*GRID.values()):
        cfg = R.RetrieverConfig(**dict(zip(GRID, values, strict=True)))
        store_runs.append({"config": asdict(cfg), "recall": objective(R.make_store(cfg))})
    best = max(store_runs, key=lambda r: r["recall"]["macro"])
    log(f"store: best dev macro recall {best['recall']['macro']} with {best['config']}")
    recency_runs = [
        {"half_life_days": hl, "recall": objective(R.make_recency_bm25(hl))} for hl in HALF_LIVES
    ]
    share_runs = [
        {"recent_share": sh, "recall": objective(R.make_window_bm25(sh))} for sh in WINDOW_SHARES
    ]
    out = {
        "objective": f"mean turn-level evidence recall at {HEADLINE_BUDGET} tokens on the dev "
        "split, macro-averaged over the two datasets",
        "dev_questions": {ds: len(v) for ds, v in dev.items()},
        "store_grid": store_runs,
        "store_best": best,
        "recency_grid": recency_runs,
        "recency_best": max(recency_runs, key=lambda r: r["recall"]["macro"]),
        "window_share_grid": share_runs,
        "window_share_best": max(share_runs, key=lambda r: r["recall"]["macro"]),
    }
    run.write("dev_sweep.json", out)
    return out


def recency_sensitivity(run: Run) -> None:
    """Not tuning: the dev sweep already chose the half-life. This shows, on the test
    split, what stronger recency buys knowledge-update questions and what it costs every
    other type - the trade the dev sweep resolved."""
    strats: dict[str, R.Strategy] = {"bm25_turns": R.bm25_turns}
    for hl in SENSITIVITY_HALF_LIVES:
        strats[f"recency_{hl:g}d"] = R.make_recency_bm25(hl)
    budgets = (512, HEADLINE_BUDGET)
    rows = []
    for ds, qs in run.datasets():
        rows += [r for r in evaluate(qs, strats, budgets=budgets, top_k=()) if r["split"] == "test"]
        log(f"recency: {ds} done")
    run.write(
        "recency_sensitivity.json",
        {
            "split": "test",
            "half_lives_days": SENSITIVITY_HALF_LIVES,
            "summary": summarise(rows, "budget"),
            "versus_bm25_turns": [
                c
                for name in strats
                if name != "bm25_turns"
                for budget in budgets
                for c in compare(rows, name, "bm25_turns", value=budget)
                if c["qtype"] in ("all types", "knowledge-update")
            ],
            "knowledge_update_newest": [
                c
                for name in strats
                if name != "bm25_turns"
                for budget in budgets
                for c in compare(rows, name, "bm25_turns", value=budget, metric="newest")
                if c["qtype"] == "knowledge-update"
            ],
        },
    )


def window_share_sensitivity(run: Run) -> None:
    """Recent-window + search at every share on the test split, each against search
    alone. The dev sweep picks the share the main stage uses; this shows how much the
    choice matters."""
    strats: dict[str, R.Strategy] = {"bm25_turns": R.bm25_turns}
    for sh in WINDOW_SHARES:
        strats[f"window_bm25_{sh:g}"] = R.make_window_bm25(sh)
    budgets = (512, HEADLINE_BUDGET)
    rows = []
    for ds, qs in run.datasets():
        rows += [r for r in evaluate(qs, strats, budgets=budgets, top_k=()) if r["split"] == "test"]
        log(f"window share: {ds} done")
    run.write(
        "window_share.json",
        {
            "split": "test",
            "recent_shares": WINDOW_SHARES,
            "versus_bm25_turns": [
                c
                for name in strats
                if name != "bm25_turns"
                for budget in budgets
                for c in compare(rows, name, "bm25_turns", value=budget)
            ],
        },
    )
