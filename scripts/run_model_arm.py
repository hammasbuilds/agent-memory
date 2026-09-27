"""The model arm: dense/hybrid retrieval with nomic-embed-text, LLM fact extraction
into the store, and answer accuracy with qwen2.5:14b-instruct judged against the gold
answers. Every call is cached in data/model_cache.sqlite, so a killed run resumes.

    uv run python scripts/run_model_arm.py --dry-run          # job list + call counts
    uv run python scripts/run_model_arm.py                    # everything
    uv run python scripts/run_model_arm.py --stage answer --per-type 40

Stages (each writes results/*.json):
  retrieval  dense and hybrid evidence recall, same protocol as the no-model study
                                                          -> results/model_retrieval.json
  extract    LLM fact extraction over every LoCoMo session into one store per
             conversation                                 -> results/model_extraction.json
  answer     answer + judge for a per-type sample of test questions, for each memory
             strategy at the headline budget              -> results/model_answers.json
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import random
import sys
import time
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict
from pathlib import Path

from agent_memory import retrieval as R
from agent_memory.answer import answer_and_judge
from agent_memory.context import line
from agent_memory.datasets import Question, iter_longmemeval, load_locomo
from agent_memory.embedding import DOC_PREFIX, make_dense, make_hybrid
from agent_memory.evaluate import (
    HEADLINE_BUDGET,
    cluster_of,
    evaluate,
    history_via_store,
    score,
    split_of,
)
from agent_memory.extract import extract_into
from agent_memory.llm import CHAT_MODEL, EMBED_MODEL, DiskCache, Ollama
from agent_memory.report import summarise
from agent_memory.stats import bootstrap_mean
from agent_memory.store import MemoryStore

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
CACHE = ROOT / "data" / "model_cache.sqlite"
STORES = ROOT / "data" / "model_stores"
ANSWER_STRATEGIES = ("sliding_window", "bm25_turns", "store", "hybrid", "store+facts")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write(name: str, obj: object) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(obj, indent=1) + "\n", "utf-8")
    log(f"wrote results/{name}")


def load_config() -> tuple[R.RetrieverConfig, float]:
    path = RESULTS / "dev_sweep.json"
    if not path.exists():
        sys.exit("results/dev_sweep.json missing - run scripts/run_retrieval_study.py first")
    sweep = json.loads(path.read_text("utf-8"))
    return R.RetrieverConfig(**sweep["store_best"]["config"]), sweep["recency_best"][
        "half_life_days"
    ]


def questions(lme_limit: int | None) -> Iterator[tuple[str, Iterator[Question]]]:
    yield "locomo", iter(load_locomo())
    yield "longmemeval", itertools.islice(iter_longmemeval(), lme_limit)


def sample(qs: list[Question], per_type: int) -> list[Question]:
    """Up to `per_type` test-split questions of each type, seeded so reruns match."""
    by: dict[str, list[Question]] = defaultdict(list)
    for q in qs:
        if split_of(q) == "test":
            by[q.qtype].append(q)
    out = []
    for qt in sorted(by):
        pool = by[qt]
        random.Random(qt).shuffle(pool)
        out += pool[:per_type]
    return out


# ---- stages ----------------------------------------------------------------------------


def stage_retrieval(client: Ollama, cfg: R.RetrieverConfig, lme_limit: int | None) -> None:
    strats = {"dense": make_dense(client), "hybrid": make_hybrid(client, cfg)}
    rows = []
    for ds, qs in questions(lme_limit):
        t0 = time.time()
        rows += list(evaluate(qs, strats))
        log(f"retrieval: {ds} in {time.time() - t0:.0f}s")
    write(
        "model_retrieval.json",
        {
            "embed_model": EMBED_MODEL,
            "by_budget": summarise(rows, "budget"),
            "by_k": summarise(rows, "k", b=1000),
        },
    )


def _store_path(conv: str) -> Path:
    return STORES / f"locomo_{conv}.db"


def stage_extract(client: Ollama) -> None:
    STORES.mkdir(parents=True, exist_ok=True)
    per_conv = {}
    seen = set()
    for q in load_locomo():
        conv = q.qid.split(":")[0]
        if conv in seen:
            continue
        seen.add(conv)
        path = _store_path(conv)
        path.unlink(missing_ok=True)  # facts are rebuilt from cached generations
        with MemoryStore(path) as store:
            store.ingest(q.history)
            counts = extract_into(store, list(q.history), client)
            per_conv[conv] = counts | store.stats()
        log(f"extract: {conv} {per_conv[conv]}")
    write("model_extraction.json", {"model": CHAT_MODEL, "conversations": per_conv})


def stage_answer(
    client: Ollama, cfg: R.RetrieverConfig, per_type: int, lme_limit: int | None
) -> None:
    plans = {
        "sliding_window": R.sliding_window,
        "bm25_turns": R.bm25_turns,
        "store": R.make_store(cfg),
        "hybrid": make_hybrid(client, cfg),
    }
    rows = []
    for ds, qs in questions(lme_limit):
        picked = sample(list(qs), per_type)
        log(f"answer: {ds} {len(picked)} questions")
        for q in picked:
            h = history_via_store(q)
            for name in ANSWER_STRATEGIES:
                if name == "store+facts":
                    if ds != "locomo":
                        continue
                    path = _store_path(q.qid.split(":")[0])
                    if not path.exists():
                        sys.exit(f"{path} missing - run --stage extract first")
                    with MemoryStore(path, cfg) as store:
                        ctx = store.context(q.question, HEADLINE_BUDGET, q.now)
                else:
                    ctx = plans[name](h, q.question, q.now, q.qid).pack(HEADLINE_BUDGET, h.tokens)
                s = score(q, ctx.turn_ids)
                out = answer_and_judge(q, ctx, client)
                rows.append(
                    {
                        "qid": q.qid,
                        "dataset": ds,
                        "qtype": q.qtype,
                        "strategy": name,
                        "cluster": q.qid.split(":")[0] if ds == "locomo" else q.qid,
                        "tokens": ctx.tokens,
                        "recall": None if s is None else s.recall,
                        "correct": out["correct"],
                        "response": out["response"],
                    }
                )
    write(
        "model_answers.json",
        {
            "model": CHAT_MODEL,
            "budget": HEADLINE_BUDGET,
            "summary": summarise_answers(rows),
            "rows": rows,
        },
    )


def summarise_answers(rows: list[dict]) -> list[dict]:
    """Accuracy per (dataset, strategy, type) with cluster-bootstrap CIs, plus accuracy
    split by whether the gold evidence was in the context at all."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if r["correct"] is None:
            continue
        groups[(r["dataset"], r["strategy"], r["qtype"])].append(r)
        groups[(r["dataset"], r["strategy"], "all types")].append(r)
    out = []
    for (ds, strat, qt), g in sorted(groups.items()):
        rec = {
            "dataset": ds,
            "strategy": strat,
            "qtype": qt,
            "accuracy": bootstrap_mean(
                [float(r["correct"]) for r in g], [r["cluster"] for r in g]
            ).as_dict(),
        }
        for label, cond in (
            ("evidence_found", lambda r: (r["recall"] or 0) > 0),
            ("evidence_missing", lambda r: r["recall"] == 0),
        ):
            sub = [float(r["correct"]) for r in g if cond(r)]
            rec[f"accuracy_{label}"] = (
                {"mean": round(sum(sub) / len(sub), 4), "n": len(sub)} if sub else None
            )
        out.append(rec)
    return out


# ---- dry run ---------------------------------------------------------------------------


def dry_run(per_type: int, lme_limit: int | None) -> None:
    texts, n_q, sessions_locomo = set(), 0, set()
    sampled = 0
    for ds, qs in questions(lme_limit):
        qs = list(qs)
        n_q += len(qs)
        seen = set()
        for q in qs:
            if cluster_of(q) in seen:
                continue
            seen.add(cluster_of(q))
            for s in q.history:
                if ds == "locomo":
                    sessions_locomo.add(s.id)
                for t in s.turns:
                    texts.add(hashlib.sha1((DOC_PREFIX + line(t)).encode()).hexdigest())
        n = len(sample(qs, per_type))
        strategies = len(ANSWER_STRATEGIES) - (ds != "locomo")
        sampled += n * strategies
        print(f"  {ds}: {len(qs)} questions, {n} sampled for answering x {strategies} strategies")
    print("jobs:")
    print(
        f"  retrieval  {len(texts):>7} unique turn embeddings + {n_q} query embeddings "
        f"({EMBED_MODEL})"
    )
    print(
        f"  extract    {len(sessions_locomo):>7} generations ({CHAT_MODEL}, one per LoCoMo session)"
    )
    print(f"  answer     {2 * sampled:>7} generations ({sampled} answers + {sampled} judgements)")
    print(
        f"total: {len(sessions_locomo) + 2 * sampled} generations, "
        f"{len(texts) + n_q} embeddings (cached ones are free on a rerun)"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("all", "retrieval", "extract", "answer"), default="all")
    ap.add_argument("--dry-run", action="store_true", help="print jobs and call counts only")
    ap.add_argument("--per-type", type=int, default=40, help="questions per type for answering")
    ap.add_argument("--lme-limit", type=int, default=None)
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    args = ap.parse_args()
    if args.dry_run:
        dry_run(args.per_type, args.lme_limit)
        return
    cfg, _ = load_config()
    client = Ollama(args.url, DiskCache(CACHE))
    log(f"store config {asdict(cfg)}")
    if args.stage in ("all", "retrieval"):
        stage_retrieval(client, cfg, args.lme_limit)
    if args.stage in ("all", "extract"):
        stage_extract(client)
    if args.stage in ("all", "answer"):
        stage_answer(client, cfg, args.per_type, args.lme_limit)


if __name__ == "__main__":
    main()
