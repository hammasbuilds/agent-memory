from datetime import datetime

import pytest

from agent_memory.forget import apply, older_than, parse_policy, phatic, role, truncate
from agent_memory.report import budget_to_reach, compare, summarise
from agent_memory.stats import bootstrap_mean, paired_difference
from agent_memory.text import count_tokens


def test_bootstrap_interval_brackets_the_mean():
    vals = [0, 1] * 50
    e = bootstrap_mean(vals, b=500)
    assert e.mean == 0.5 and e.lo < 0.5 < e.hi and e.n == 100


def test_constant_values_give_zero_width():
    e = bootstrap_mean([0.3] * 20, b=200)
    assert e.lo == e.hi == pytest.approx(0.3)


def test_clusters_widen_the_interval():
    # two clusters of perfectly correlated questions: really n=2, not n=200
    vals = [1.0] * 100 + [0.0] * 100
    naive = bootstrap_mean(vals, b=500)
    clustered = bootstrap_mean(vals, ["a"] * 100 + ["b"] * 100, b=500)
    assert clustered.clusters == 2
    assert clustered.hi - clustered.lo > 3 * (naive.hi - naive.lo)


def test_bootstrap_errors_and_single_cluster():
    with pytest.raises(ValueError):
        bootstrap_mean([])
    with pytest.raises(ValueError):
        bootstrap_mean([1.0], ["a", "b"])
    one = bootstrap_mean([1.0, 0.0], ["a", "a"])
    assert (one.lo, one.hi) == (0.5, 0.5)
    with pytest.raises(ValueError):
        paired_difference([1.0], [1.0, 2.0])


def _rows():
    rows = []
    for i in range(40):
        for strat, rec in (("good", 1.0 if i % 4 else 0.5), ("bad", 0.0 if i % 2 else 0.5)):
            rows.append(
                {
                    "qid": f"q{i}",
                    "dataset": "d",
                    "split": "test",
                    "cluster": f"q{i}",
                    "qtype": "t1" if i < 20 else "t2",
                    "strategy": strat,
                    "budget": 100,
                    "tokens": 90,
                    "measurable": True,
                    "recall": rec,
                    "all": rec == 1.0,
                }
            )
    rows.append(
        {
            "qid": "u",
            "dataset": "d",
            "split": "test",
            "cluster": "u",
            "qtype": "t1",
            "strategy": "good",
            "budget": 100,
            "tokens": 0,
            "measurable": False,
        }
    )
    return rows


def test_summarise_and_compare():
    summ = summarise(_rows(), b=200)
    good_all = next(r for r in summ if r["strategy"] == "good" and r["qtype"] == "all types")
    assert good_all["n_questions"] == 41 and good_all["n_measurable"] == 40
    assert good_all["recall"]["mean"] == pytest.approx((30 * 1.0 + 10 * 0.5) / 40)
    diff = compare(_rows(), "good", "bad", value=100, b=200)
    overall = next(d for d in diff if d["qtype"] == "all types")
    assert overall["diff"]["mean"] == pytest.approx(0.875 - 0.25)
    assert overall["diff"]["ci95"][0] > 0


def test_budget_to_reach():
    summ = [
        {
            "dataset": "d",
            "split": "test",
            "strategy": "s",
            "qtype": "x",
            "budget": b,
            "recall": {"mean": m},
        }
        for b, m in ((256, 0.2), (512, 0.6), (1024, 0.9))
    ]
    assert budget_to_reach(summ, 0.5)[0]["budget"] == 512
    assert budget_to_reach(summ, 0.95)[0]["budget"] is None


def test_policies(history):
    now = datetime(2024, 7, 1)
    turns = [t for s in history for t in s.turns]
    assert sum(map(phatic(3), turns)) == 5
    assert sum(map(role("assistant"), turns)) == 4
    assert sum(map(older_than(now, 30), turns)) == 7  # everything before 1 June
    kept = apply(history, older_than(now, 30))
    assert [s.id for s in kept] == ["s3"]  # emptied sessions are dropped


def test_truncate(history):
    long = history[0].turns[2]
    short = truncate(long, 5)
    assert count_tokens(short.text) <= 5 and short.text.endswith("…")
    assert truncate(long, 1000) is long
    with pytest.raises(ValueError):
        truncate(long, 0)


def test_parse_policy():
    now = datetime(2024, 7, 1)
    assert parse_policy("truncate:64", now) == (None, 64)
    assert parse_policy("phatic", now)[0] is not None
    for bad in ("truncate:0", "role:", "older-than:soon", "shred"):
        with pytest.raises(ValueError, match="bad policy"):
            parse_policy(bad, now)


def test_position_differences_pair_evidence_turns():
    from agent_memory.report import position_differences

    def row(strategy, qid, evidence):
        return {"qid": qid, "dataset": "d", "split": "test", "cluster": qid, "budget": 100,
                "strategy": strategy, "evidence": evidence}  # fmt: skip

    rows = []
    for i in range(30):
        rows.append(row("win", f"q{i}", [[0.95, True, i % 3 != 0], [0.1, False, False]]))
        rows.append(row("bm25", f"q{i}", [[0.95, i % 2 == 0, i % 2 == 0], [0.1, True, True]]))
    out = {
        (d["position"], d["credit"]): d["diff"]
        for d in position_differences(rows, "win", "bm25", 100, b=300)
    }
    assert out[("0.9-1", "lenient")]["mean"] == pytest.approx(1 - 0.5)
    assert out[("0.9-1", "strict")]["mean"] == pytest.approx(20 / 30 - 0.5, abs=1e-4)
    assert out[("0-0.25", "lenient")]["mean"] == -1.0
