"""The model arm: dense and hybrid retrieval with nomic-embed-text, LLM fact
extraction into the store, and answer accuracy judged against the gold answers. Every
stage takes the client as an argument, so the whole arm runs against a fake in tests."""

from __future__ import annotations

import hashlib
import random
import time
from collections import defaultdict
from pathlib import Path

from agent_memory import retrieval as R
from agent_memory.answer import answer_and_judge
from agent_memory.context import line
from agent_memory.datasets import Question, load_locomo
from agent_memory.embedding import DOC_PREFIX, make_dense, make_hybrid
from agent_memory.evaluate import (
    HEADLINE_BUDGET,
    cluster_of,
    evaluate,
    history_key,
    history_via_store,
    score,
    split_of,
)
from agent_memory.extract import extract_into
from agent_memory.llm import CHAT_MODEL, EMBED_MODEL, LLM
from agent_memory.report import summarise, summarise_answers
from agent_memory.store import MemoryStore
from agent_memory.study.common import Run, log

ANSWER_STRATEGIES = ("sliding_window", "bm25_turns", "store", "hybrid", "store+facts")


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


def stage_retrieval(run: Run, client: LLM) -> None:
    cfg = run.chosen().store
    strats = {"dense": make_dense(client), "hybrid": make_hybrid(client, cfg)}
    rows = []
    for ds, qs in run.datasets():
        t0 = time.time()
        rows += list(evaluate(qs, strats))
        log(f"retrieval: {ds} in {time.time() - t0:.0f}s")
    run.write(
        "model_retrieval.json",
        {
            "embed_model": EMBED_MODEL,
            "by_budget": summarise(rows, "budget"),
            "by_k": summarise(rows, "k", b=1000),
        },
    )


def store_path(stores: Path, conv: str) -> Path:
    return stores / f"locomo_{conv}.db"


def stage_extract(run: Run, client: LLM, stores: Path) -> None:
    stores.mkdir(parents=True, exist_ok=True)
    per_conv = {}
    seen = set()
    for q in load_locomo():
        conv = cluster_of(q)
        if conv in seen:
            continue
        seen.add(conv)
        path = store_path(stores, conv)
        path.unlink(missing_ok=True)  # facts are rebuilt from cached generations
        with MemoryStore(path) as store:
            store.ingest(q.history)
            counts = extract_into(store, list(q.history), client)
            per_conv[conv] = counts | store.stats()
        log(f"extract: {conv} {per_conv[conv]}")
    run.write("model_extraction.json", {"model": CHAT_MODEL, "conversations": per_conv})


def stage_answer(run: Run, client: LLM, stores: Path, per_type: int, judge: str) -> None:
    cfg = run.chosen().store
    plans = {
        "sliding_window": R.sliding_window,
        "bm25_turns": R.bm25_turns,
        "store": R.make_store(cfg),
        "hybrid": make_hybrid(client, cfg),
    }
    rows = []
    for ds, qs in run.datasets():
        picked = sample(list(qs), per_type)
        log(f"answer: {ds} {len(picked)} questions")
        for q in picked:
            h = history_via_store(q)
            for name in ANSWER_STRATEGIES:
                if name == "store+facts":
                    if ds != "locomo":
                        continue
                    path = store_path(stores, cluster_of(q))
                    if not path.exists():
                        raise FileNotFoundError(f"{path} missing - run the extract stage first")
                    with MemoryStore(path, cfg) as store:
                        ctx = store.context(q.question, HEADLINE_BUDGET, q.now)
                else:
                    ctx = plans[name](h, q.question, q.now, q.qid).pack(HEADLINE_BUDGET, h.tokens)
                s = score(q, ctx.turn_ids)
                out = answer_and_judge(q, ctx, client, CHAT_MODEL, judge)
                rows.append(
                    {
                        "qid": q.qid,
                        "dataset": ds,
                        "qtype": q.qtype,
                        "strategy": name,
                        "cluster": cluster_of(q),
                        "tokens": ctx.tokens,
                        "recall": None if s is None else s.recall,
                    }
                    | out
                )
    run.write(
        "model_answers.json",
        {
            "model": CHAT_MODEL,
            "judge": judge,
            "self_judged": judge == CHAT_MODEL,
            "budget": HEADLINE_BUDGET,
            "summary": summarise_answers(rows),
            "rows": rows,
        },
    )


def dry_run(run: Run, per_type: int) -> list[str]:
    """The job list and call counts, as printable lines. Touches no model."""
    texts: set[str] = set()
    n_q, sessions_locomo, sampled = 0, set(), 0
    lines = []
    for ds, qs_iter in run.datasets():
        qs = list(qs_iter)
        n_q += len(qs)
        seen = set()
        for q in qs:
            if history_key(q) in seen:
                continue
            seen.add(history_key(q))
            for s in q.history:
                if ds == "locomo":
                    sessions_locomo.add(s.id)
                for t in s.turns:
                    texts.add(hashlib.sha1((DOC_PREFIX + line(t)).encode()).hexdigest())
        n = len(sample(qs, per_type))
        n_strats = len(ANSWER_STRATEGIES) - (ds != "locomo")
        sampled += n * n_strats
        lines.append(
            f"  {ds}: {len(qs)} questions, {n} sampled for answering x {n_strats} strategies"
        )
    gens = len(sessions_locomo) + 2 * sampled
    lines += [
        "jobs:",
        f"  retrieval  {len(texts):>7} unique turn embeddings + {n_q} query embeddings"
        f" ({EMBED_MODEL})",
        f"  extract    {len(sessions_locomo):>7} generations ({CHAT_MODEL},"
        " one per LoCoMo session)",
        f"  answer     {2 * sampled:>7} generations ({sampled} answers + {sampled} judgements)",
        f"total: {gens} generations, {len(texts) + n_q} embeddings"
        " (cached ones are free on a rerun)",
    ]
    return lines
