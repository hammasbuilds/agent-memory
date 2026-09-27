"""The retrieval study (no model): evidence recall and token cost of every memory
strategy on LoCoMo and LongMemEval_S, by question type, with cluster-bootstrap CIs.

Stages (each writes results/*.json):
  1. dev sweep      choose the store retriever's knobs and the recency half-life on the
                    dev split only                             -> results/dev_sweep.json
  2. main           every strategy x budget x top-k, all questions, both splits
                                                               -> results/retrieval.json
                                                                  results/rows/*.jsonl.gz
  3. forgetting     what each forgetting / compaction policy removes, and what it costs
                    in evidence                                -> results/forgetting.json
  4. diagnostics    lexical visibility of evidence, where answers live
                                                               -> results/diagnostics.json

    uv run python scripts/run_retrieval_study.py            # everything (~20-30 min)
    uv run python scripts/run_retrieval_study.py --stage main --lme-limit 50
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from agent_memory import retrieval as R
from agent_memory.analysis import answer_location, evidence_position, lexical_visibility
from agent_memory.datasets import Question, iter_longmemeval, load_locomo
from agent_memory.evaluate import (
    HEADLINE_BUDGET,
    cluster_of,
    evaluate,
    history_via_store,
    plan_recall,
    score,
    split_of,
)
from agent_memory.forget import apply, older_than, phatic, role
from agent_memory.report import ALL, budget_to_reach, compare, summarise
from agent_memory.retrieval import History
from agent_memory.stats import bootstrap_mean
from agent_memory.temporal import query_window
from agent_memory.text import count_tokens

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write(name: str, obj: object) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1) + "\n", "utf-8")
    log(f"wrote results/{name}")


def lme(limit: int | None) -> Iterator[Question]:
    return itertools.islice(iter_longmemeval(), limit)


# ---- 1. dev sweep ------------------------------------------------------------------

GRID = {
    "session_weight": (0.0, 0.15, 0.3, 0.6),
    "half_life_days": (0.0, 90.0, 365.0),
    "window_boost": (0.0, 1.0),
    "neighbours": (0, 1, 2),
}
HALF_LIVES = (7.0, 30.0, 90.0, 365.0, 1825.0)


def _dev_histories(qs: Iterable[Question]) -> list[tuple[Question, History]]:
    out, cache = [], {}
    for q in qs:
        if split_of(q) != "dev":
            continue
        key = cluster_of(q)
        if key not in cache:
            cache[key] = history_via_store(q)
        out.append((q, cache[key]))
    return out


def _mean_recall(dev: list[tuple[Question, History]], strategy: R.Strategy) -> float:
    vals = [
        r
        for q, h in dev
        if (r := plan_recall(q, h, strategy(h, q.question, q.now, q.qid), HEADLINE_BUDGET))
        is not None
    ]
    return mean(vals)


def dev_sweep(lme_limit: int | None) -> dict:
    dev = {"locomo": _dev_histories(load_locomo()), "longmemeval": _dev_histories(lme(lme_limit))}
    log(f"dev split: {', '.join(f'{k} {len(v)} questions' for k, v in dev.items())}")

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
    best_hl = max(recency_runs, key=lambda r: r["recall"]["macro"])
    out = {
        "objective": f"mean evidence recall at {HEADLINE_BUDGET} tokens on the dev split, "
        "macro-averaged over the two datasets",
        "store_grid": store_runs,
        "store_best": best,
        "recency_grid": recency_runs,
        "recency_best": best_hl,
    }
    write("dev_sweep.json", out)
    return out


def chosen(sweep: dict) -> tuple[R.RetrieverConfig, float]:
    return R.RetrieverConfig(**sweep["store_best"]["config"]), sweep["recency_best"][
        "half_life_days"
    ]


# ---- 2. main -------------------------------------------------------------------------


def strategies(cfg: R.RetrieverConfig, half_life: float) -> dict[str, R.Strategy]:
    s: dict[str, R.Strategy] = {
        "full_head": R.full_head,
        "sliding_window": R.sliding_window,
        "random": R.random_turns,
        "bm25_turns": R.bm25_turns,
        "bm25_sessions": R.bm25_sessions,
        "recency_bm25": R.make_recency_bm25(half_life),
        "store": R.make_store(cfg),
    }
    for comp in ("session", "recency", "window", "neighbours"):
        off = cfg.without(comp)
        if off != cfg:  # a component the dev sweep already switched off needs no ablation
            s[f"store-no-{comp}"] = R.make_store(off)
    return s


COMPARISONS = [
    ("store", "bm25_turns"),
    ("bm25_turns", "sliding_window"),
    ("bm25_turns", "full_head"),
    ("bm25_sessions", "bm25_turns"),
    ("recency_bm25", "bm25_turns"),
    ("sliding_window", "random"),
]


def main_stage(cfg: R.RetrieverConfig, half_life: float, lme_limit: int | None) -> None:
    strats = strategies(cfg, half_life)
    rows_dir = RESULTS / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    for name, qs in (("locomo", load_locomo()), ("longmemeval", lme(lme_limit))):
        t0 = time.time()
        rows = list(evaluate(qs, strats))
        log(f"{name}: {len(rows)} rows in {time.time() - t0:.0f}s")
        with gzip.open(rows_dir / f"{name}.jsonl.gz", "wt", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        all_rows += rows
    comps = [("store", f"store-no-{c}") for c in ("session", "recency", "window", "neighbours")]
    comps = [c for c in COMPARISONS + comps if c[1] in strats]
    by_budget = summarise(all_rows, "budget")
    out = {
        "config": {"store": asdict(cfg), "recency_half_life_days": half_life},
        "headline_budget": HEADLINE_BUDGET,
        "by_budget": by_budget,
        "by_k": summarise(all_rows, "k", b=1000),
        "comparisons": [
            c
            for a, b in comps
            for split in ("test", "dev")
            for c in compare(all_rows, a, b, value=HEADLINE_BUDGET, split=split)
        ],
        "knowledge_update_newest": [
            c
            for a, b in (("recency_bm25", "bm25_turns"), ("store", "bm25_turns"))
            for c in compare(all_rows, a, b, value=HEADLINE_BUDGET, metric="newest", split="test")
            if c["qtype"] == "knowledge-update"
        ],
        "budget_to_reach": budget_to_reach(by_budget, 0.5) + budget_to_reach(by_budget, 0.8),
        # the window boost can only act on questions that name a time
        "time_expression_questions": [
            c
            for a, b in (("store", "store-no-window"), ("store", "bm25_turns"))
            if b in strats
            for split in ("test", "dev")
            for c in compare(
                [r for r in all_rows if r["time_expr"]], a, b, value=HEADLINE_BUDGET, split=split
            )
        ],
    }
    write("retrieval.json", out)


# ---- 3. forgetting ------------------------------------------------------------------


def _policies(ds: str) -> dict[str, tuple]:
    base = {
        "none": (None, None, None),
        "phatic:3": (phatic(3), None, None),
        "truncate:128": (None, 128, None),
        "truncate:64": (None, 64, None),
    }
    if ds == "longmemeval":
        base["role:assistant"] = (role("assistant"), None, None)
        base["older-than:30"] = (None, None, 30.0)
    else:
        base["older-than:90"] = (None, None, 90.0)
    return base


def _answer_present(q: Question, sessions: Iterable) -> bool:
    text = " ".join(t.text for s in sessions for t in s.turns if t.id in q.evidence_turns)
    return q.answer.lower() in text.lower()


def forgetting_stage(cfg: R.RetrieverConfig, lme_limit: int | None) -> None:
    store = R.make_store(cfg)
    out = []
    for ds, qs in (("locomo", load_locomo()), ("longmemeval", lme(lme_limit))):
        acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        cache: dict[str, tuple] = {}  # per history: policy -> (sessions, tokens kept, index)
        last = None
        for q in qs:
            if split_of(q) != "test":
                continue
            if q.history is not last:
                cache, last = {}, q.history
            base_tokens = sum(count_tokens(t.text) for s in q.history for t in s.turns)
            had_answer = _answer_present(q, q.history)
            for name, (drop, trunc, days) in _policies(ds).items():
                if name not in cache:
                    pol = older_than(q.now, days) if days is not None else drop
                    sessions = apply(q.history, pol, trunc)
                    kept = sum(count_tokens(t.text) for s in sessions for t in s.turns)
                    cache[name] = (sessions, kept, History(sessions))
                sessions, kept, h = cache[name]
                a = acc[name]
                a["tokens_kept"].append(kept / base_tokens)
                if q.evidence_turns:
                    live = {t.id for s in sessions for t in s.turns}
                    a["evidence_kept"].append(len(q.evidence_turns & live) / len(q.evidence_turns))
                    a["qtype"].append(q.qtype)
                    a["cluster"].append(q.qid.split(":")[0] if ds == "locomo" else q.qid)
                    ctx = store(h, q.question, q.now, q.qid).pack(HEADLINE_BUDGET, h.tokens)
                    s = score(q, ctx.turn_ids)
                    a["recall"].append(s.recall if s else 0.0)
                    if had_answer:
                        a["answer_kept"].append(float(_answer_present(q, sessions)))
        for name, a in acc.items():
            rec = {
                "dataset": ds,
                "policy": name,
                "stored_tokens_kept": round(mean(a["tokens_kept"]), 4),
                "evidence_kept": bootstrap_mean(a["evidence_kept"], a["cluster"]).as_dict(),
                f"store_recall_at_{HEADLINE_BUDGET}": bootstrap_mean(
                    a["recall"], a["cluster"]
                ).as_dict(),
                "literal_answer_kept": (
                    {"mean": round(mean(a["answer_kept"]), 4), "n": len(a["answer_kept"])}
                    if a["answer_kept"]
                    else None
                ),
                "by_type": {},
            }
            for qt in sorted(set(a["qtype"])):
                idx = [i for i, t in enumerate(a["qtype"]) if t == qt]
                rec["by_type"][qt] = {
                    "n": len(idx),
                    "evidence_kept": round(mean(a["evidence_kept"][i] for i in idx), 4),
                    "store_recall": round(mean(a["recall"][i] for i in idx), 4),
                }
            out.append(rec)
        log(f"forgetting: {ds} done")
    write("forgetting.json", {"split": "test", "budget": HEADLINE_BUDGET, "policies": out})


# ---- 3b. recency sensitivity ----------------------------------------------------------

SENSITIVITY_HALF_LIVES = (7.0, 30.0, 90.0, 365.0)


def recency_stage(lme_limit: int | None) -> None:
    """Not tuning: the dev sweep already chose the half-life. This shows, on the test
    split, what stronger recency buys knowledge-update questions and what it costs
    every other type - the trade the dev sweep resolved."""
    strats: dict[str, R.Strategy] = {"bm25_turns": R.bm25_turns}
    for hl in SENSITIVITY_HALF_LIVES:
        strats[f"recency_{hl:g}d"] = R.make_recency_bm25(hl)
    rows = []
    for ds, qs in (("locomo", load_locomo()), ("longmemeval", lme(lme_limit))):
        rows += [
            r
            for r in evaluate(qs, strats, budgets=(512, HEADLINE_BUDGET), top_k=())
            if r["split"] == "test"
        ]
        log(f"recency: {ds} done")
    write(
        "recency_sensitivity.json",
        {
            "split": "test",
            "half_lives_days": SENSITIVITY_HALF_LIVES,
            "summary": summarise(rows, "budget"),
            "knowledge_update_newest": [
                c
                for name in strats
                if name != "bm25_turns"
                for budget in (512, HEADLINE_BUDGET)
                for c in compare(rows, name, "bm25_turns", value=budget, metric="newest")
                if c["qtype"] == "knowledge-update"
            ],
        },
    )


# ---- 4. diagnostics -------------------------------------------------------------------


def diagnostics_stage(lme_limit: int | None) -> None:
    out = []
    for ds, qs in (("locomo", load_locomo()), ("longmemeval", lme(lme_limit))):
        vis: dict[str, list[float]] = defaultdict(list)
        loc: dict[str, Counter] = defaultdict(Counter)
        timed: dict[str, list[bool]] = defaultdict(list)
        position: dict[str, list[float]] = defaultdict(list)
        spans: list[float] = []
        sizes: list[int] = []
        seen: set[int] = set()
        for q in qs:
            if cluster_of(q) not in seen:
                seen.add(cluster_of(q))
                sizes.append(sum(count_tokens(t.text) for s in q.history for t in s.turns))
                spans.append((q.now - q.history[0].timestamp).total_seconds() / 86400)
            if (p := evidence_position(q)) is not None:
                position[q.qtype].append(p)
                position[ALL].append(p)
            timed[q.qtype].append(query_window(q.question, q.now) is not None)
            if (v := lexical_visibility(q)) is not None:
                vis[q.qtype].append(v)
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
                "lexical_visibility": {
                    qt: {
                        "mean": round(mean(v), 4),
                        "fully_invisible": round(mean(x == 0 for x in v), 4),
                        "n": len(v),
                    }
                    for qt, v in sorted(vis.items())
                },
                "history_span_days": {
                    "mean": round(mean(spans), 1),
                    "min": round(min(spans), 1),
                    "max": round(max(spans), 1),
                },
                "evidence_position": {
                    qt: {
                        "mean": round(mean(v), 4),
                        "in_newest_10pct": round(mean(x >= 0.9 for x in v), 4),
                        "n": len(v),
                    }
                    for qt, v in sorted(position.items())
                },
                "names_a_time": {
                    qt: {"rate": round(mean(v), 4), "n": len(v)} for qt, v in sorted(timed.items())
                },
                "answer_location": {
                    qt: {k: round(n / sum(c.values()), 4) for k, n in sorted(c.items())}
                    | {"n": sum(c.values())}
                    for qt, c in sorted(loc.items())
                },
            }
        )
    write("diagnostics.json", out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--stage",
        choices=("all", "dev", "main", "forgetting", "recency", "diagnostics"),
        default="all",
    )
    ap.add_argument("--lme-limit", type=int, default=None, help="first N LongMemEval questions")
    args = ap.parse_args()
    if args.stage in ("all", "diagnostics"):
        diagnostics_stage(args.lme_limit)
    if args.stage in ("all", "recency"):
        recency_stage(args.lme_limit)
    if args.stage in ("diagnostics", "recency"):
        return
    if args.stage in ("all", "dev"):
        sweep = dev_sweep(args.lme_limit)
    else:
        path = RESULTS / "dev_sweep.json"
        if not path.exists():
            sys.exit("results/dev_sweep.json missing - run --stage dev first")
        sweep = json.loads(path.read_text("utf-8"))
    cfg, hl = chosen(sweep)
    if args.stage in ("all", "main"):
        main_stage(cfg, hl, args.lme_limit)
    if args.stage in ("all", "forgetting"):
        forgetting_stage(cfg, args.lme_limit)


if __name__ == "__main__":
    main()
