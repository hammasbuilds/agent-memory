"""Turn evaluation rows into the summary tables, with cluster-bootstrap intervals."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence

from agent_memory.stats import bootstrap_mean, paired_difference

METRICS = (
    "recall",
    "recall_strict",
    "all",
    "session_recall",
    "all_sessions",
    "newest",
    "stale_only",
)
ALL = "all types"


def _group(rows: Iterable[dict], axis: str) -> dict[tuple, list[dict]]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        if axis not in r:
            continue
        key = (r["dataset"], r["split"], r["strategy"], r[axis])
        groups[(*key, r["qtype"])].append(r)
        groups[(*key, ALL)].append(r)
    return groups


def summarise(rows: Sequence[dict], axis: str = "budget", b: int = 2000) -> list[dict]:
    """One record per (dataset, split, strategy, budget-or-k, question type): mean
    tokens, and each metric with a 95% CI. Unmeasurable questions (no gold evidence)
    are counted but excluded from the metrics."""
    out = []
    for (ds, split, strat, val, qtype), group in sorted(_group(rows, axis).items(), key=str):
        rec: dict = {
            "dataset": ds,
            "split": split,
            "strategy": strat,
            axis: val,
            "qtype": qtype,
            "n_questions": len(group),
            "mean_tokens": round(sum(r["tokens"] for r in group) / len(group), 1),
        }
        measurable = [r for r in group if r["measurable"]]
        rec["n_measurable"] = len(measurable)
        for m in METRICS:
            vals = [(float(r[m]), r["cluster"]) for r in measurable if m in r]
            if vals:
                v, c = zip(*vals, strict=True)
                rec[m] = bootstrap_mean(v, c, b=b).as_dict()
        out.append(rec)
    return out


def compare(
    rows: Sequence[dict],
    a: str,
    b_: str,
    *,
    axis: str = "budget",
    value: int,
    split: str = "test",
    metric: str = "recall",
    b: int = 2000,
) -> list[dict]:
    """Paired difference a - b in `metric` on the same questions, per dataset and type."""
    by: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        if (
            r.get(axis) == value
            and r["split"] == split
            and r["strategy"] in (a, b_)
            and r["measurable"]
            and metric in r
        ):
            by[(r["dataset"], r["qid"])][r["strategy"]] = r
    per: dict[tuple, list[tuple[float, float, str]]] = defaultdict(list)
    for (ds, _), pair in by.items():
        if a in pair and b_ in pair:
            x = (float(pair[a][metric]), float(pair[b_][metric]), pair[a]["cluster"])
            per[(ds, pair[a]["qtype"])].append(x)
            per[(ds, ALL)].append(x)
    out = []
    for (ds, qtype), xs in sorted(per.items()):
        va, vb, cl = zip(*xs, strict=True)
        est = paired_difference(va, vb, cl, b=b)
        out.append(
            {
                "dataset": ds,
                "qtype": qtype,
                "a": a,
                "b": b_,
                axis: value,
                "split": split,
                "metric": metric,
                "diff": est.as_dict(),
            }
        )
    return out


def budget_to_reach(summary: Sequence[dict], target: float) -> list[dict]:
    """Smallest budget on the grid at which each strategy's mean recall reaches
    `target` (per dataset, split, type); None if no budget on the grid does."""
    best: dict[tuple, int | None] = {}
    for r in summary:
        if "budget" not in r or "recall" not in r:
            continue
        key = (r["dataset"], r["split"], r["strategy"], r["qtype"])
        best.setdefault(key, None)
        if r["recall"]["mean"] >= target and (best[key] is None or r["budget"] < best[key]):
            best[key] = r["budget"]
    return [
        {"dataset": d, "split": s, "strategy": st, "qtype": q, "target": target, "budget": v}
        for (d, s, st, q), v in sorted(best.items(), key=str)
    ]


POSITION_BUCKETS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.0001))


def by_position(
    rows: Sequence[dict], budget: int, split: str = "test", b: int = 1000
) -> list[dict]:
    """Recall of evidence turns grouped by where they sit in the history (0 = first turn,
    1 = last), per dataset and strategy. Pooled over evidence turns, resampled by
    cluster. If recency loses because evidence is old, a recency strategy should win
    in the newest bucket and lose everywhere else."""
    acc: dict[tuple, tuple[list[float], list[str]]] = defaultdict(lambda: ([], []))
    for r in rows:
        if r.get("budget") != budget or r["split"] != split:
            continue
        for pos, hit in r.get("evidence", ()):
            for lo, hi in POSITION_BUCKETS:
                if lo <= pos < hi:
                    vals, clusters = acc[(r["dataset"], r["strategy"], f"{lo:g}-{min(hi, 1):g}")]
                    vals.append(float(hit))
                    clusters.append(r["cluster"])
    return [
        {"dataset": d, "strategy": s, "position": p, "budget": budget, "split": split}
        | {"recall": bootstrap_mean(v, c, b=b).as_dict()}
        for (d, s, p), (v, c) in sorted(acc.items())
    ]


def summarise_answers(rows: Sequence[dict], b: int = 2000) -> list[dict]:
    """Answer accuracy per (dataset, strategy, type) with cluster-bootstrap CIs.

    A judge reply that is neither yes nor no (`correct` is None) counts as incorrect in
    `accuracy` and is counted in `unparsed`, so a flaky judge cannot shrink the
    denominator. Accuracy is also split by whether any gold evidence was in context.
    """
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["dataset"], r["strategy"], r["qtype"])].append(r)
        groups[(r["dataset"], r["strategy"], ALL)].append(r)
    out = []
    for (ds, strat, qt), g in sorted(groups.items()):
        correct = [float(r["correct"] is True) for r in g]
        rec: dict = {
            "dataset": ds,
            "strategy": strat,
            "qtype": qt,
            "n": len(g),
            "unparsed": sum(r["correct"] is None for r in g),
            "accuracy": bootstrap_mean(correct, [r["cluster"] for r in g], b=b).as_dict(),
        }
        for label, cond in (
            ("evidence_found", lambda r: (r.get("recall") or 0) > 0),
            ("evidence_missing", lambda r: r.get("recall") == 0),
        ):
            sub = [float(r["correct"] is True) for r in g if cond(r)]
            rec[f"accuracy_{label}"] = (
                {"mean": round(sum(sub) / len(sub), 4), "n": len(sub)} if sub else None
            )
        out.append(rec)
    return out
