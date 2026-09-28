"""What each forgetting / compaction policy removes, and what that costs in evidence."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from statistics import mean

from agent_memory import retrieval as R
from agent_memory.datasets import Question, Session
from agent_memory.evaluate import HEADLINE_BUDGET, cluster_of, score, split_of
from agent_memory.forget import Policy, apply, older_than, phatic, role
from agent_memory.retrieval import History
from agent_memory.stats import cluster_mean, paired_difference
from agent_memory.study.common import Run, log
from agent_memory.text import count_tokens

# name -> (drop predicate, truncation length, age cutoff in days)
PolicySpec = tuple[Policy | None, int | None, float | None]


def policies(ds: str) -> dict[str, PolicySpec]:
    base: dict[str, PolicySpec] = {
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


def answer_present(q: Question, sessions: Iterable[Session]) -> bool:
    """Is the gold answer, verbatim, in the evidence turns' text?"""
    text = " ".join(t.text for s in sessions for t in s.turns if t.id in q.evidence_turns)
    return q.answer.lower() in text.lower()


def forgetting_stage(run: Run) -> None:
    cfg = run.chosen().store
    store = R.make_store(cfg)
    out = []
    for ds, qs in run.datasets():
        acc: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
        cache: dict[str, tuple] = {}  # per history: policy -> (sessions, tokens kept, index)
        last = None
        for q in qs:
            if split_of(q) != "test":
                continue
            if q.history is not last:
                cache, last = {}, q.history
            base_tokens = sum(count_tokens(t.text) for s in q.history for t in s.turns)
            had_answer = answer_present(q, q.history)
            for name, (drop, trunc, days) in policies(ds).items():
                if name not in cache:
                    pol = older_than(q.now, days) if days is not None else drop
                    sessions = apply(q.history, pol, trunc)
                    kept = sum(count_tokens(t.text) for s in sessions for t in s.turns)
                    cache[name] = (sessions, kept, History(sessions))
                sessions, kept, h = cache[name]
                a = acc[name]
                a["tokens_kept"].append(kept / base_tokens)
                if not q.evidence_turns:
                    continue
                live = {t.id for s in sessions for t in s.turns}
                a["evidence_kept"].append(len(q.evidence_turns & live) / len(q.evidence_turns))
                a["qtype"].append(q.qtype)
                a["cluster"].append(cluster_of(q))
                s = score(
                    q, store(h, q.question, q.now, q.qid).pack(HEADLINE_BUDGET, h.tokens).turn_ids
                )
                a["recall"].append(s.recall if s and s.recall is not None else 0.0)
                if had_answer:
                    a["answer_kept"].append(float(answer_present(q, sessions)))
        for name, a in acc.items():
            rec = {
                "dataset": ds,
                "policy": name,
                "stored_tokens_kept": round(mean(a["tokens_kept"]), 4),
                "evidence_kept": cluster_mean(a["evidence_kept"], a["cluster"]).as_dict(),
                f"store_recall_at_{HEADLINE_BUDGET}": cluster_mean(
                    a["recall"], a["cluster"]
                ).as_dict(),
                "literal_answer_kept": (
                    {"mean": round(mean(a["answer_kept"]), 4), "n": len(a["answer_kept"])}
                    if a["answer_kept"]
                    else None
                ),
                "by_type": {},
            }
            # every policy scores the same questions in the same order, so recall can be
            # compared pairwise with keeping everything ("none")
            base = acc["none"]["recall"]
            rec["recall_minus_none"] = paired_difference(a["recall"], base, a["cluster"]).as_dict()
            for qt in sorted(set(a["qtype"])):
                idx = [i for i, t in enumerate(a["qtype"]) if t == qt]
                rec["by_type"][qt] = {
                    "n": len(idx),
                    "evidence_kept": round(mean(a["evidence_kept"][i] for i in idx), 4),
                    "store_recall": round(mean(a["recall"][i] for i in idx), 4),
                    "recall_minus_none": paired_difference(
                        [a["recall"][i] for i in idx],
                        [base[i] for i in idx],
                        [a["cluster"][i] for i in idx],
                    ).as_dict(),
                }
            out.append(rec)
        log(f"forgetting: {ds} done")
    run.write("forgetting.json", {"split": "test", "budget": HEADLINE_BUDGET, "policies": out})
