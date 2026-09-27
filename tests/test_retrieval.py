from datetime import datetime

import pytest

from agent_memory import retrieval as R
from agent_memory.retrieval import History, Retriever, RetrieverConfig

NOW = datetime(2024, 7, 1)


def ids(ctx):
    return [t.id for t in ctx.turns]


def test_positional_baselines(history):
    h = History(history)
    head = R.full_head(h, "q", NOW, "seed").pack(30, h.tokens)
    window = R.sliding_window(h, "q", NOW, "seed").pack(30, h.tokens)
    assert ids(head)[0] == "s1:0"
    assert ids(window)[-1] == "s3:2"
    assert not set(ids(head)) & set(ids(window))


def test_random_is_seeded(history):
    h = History(history)
    a = R.random_turns(h, "q", NOW, "x").pack(40, h.tokens)
    b = R.random_turns(h, "q", NOW, "x").pack(40, h.tokens)
    assert ids(a) == ids(b)


def test_bm25_turns_finds_the_turn(history):
    h = History(history)
    ctx = R.bm25_turns(h, "What is my puppy called?", NOW, "s").pack(40, h.tokens)
    assert "s1:2" in ids(ctx)


def test_ranked_plans_skip_non_matching_turns(history):
    h = History(history)
    plan = R.bm25_turns(h, "xylophone", NOW, "s")
    assert plan.units == []
    assert plan.pack(1000, h.tokens).tokens == 0


def test_bm25_sessions_packs_whole_sessions(history):
    h = History(history)
    ctx = R.bm25_sessions(h, "hiking trip Lake District", NOW, "s").pack(10_000, h.tokens)
    assert ids(ctx) == ["s2:0", "s2:1", "s2:2"]


def test_recency_prefers_the_newer_of_two_matches(history):
    h = History(history)
    q = "where do I work as a nurse?"
    plain = R.bm25_turns(h, q, NOW, "s").units
    recent = R.make_recency_bm25(half_life_days=30)(h, q, NOW, "s").units
    assert recent[0][0].id == "s3:0"
    assert {plain[0][0].id, plain[1][0].id} == {"s2:0", "s3:0"}


def test_store_window_boost_follows_the_question_date(history):
    h = History(history)
    r = Retriever(RetrieverConfig(session_weight=0, neighbours=0, window_boost=5.0))
    march = r.search(h, "nurse job in March", NOW, k=1)
    june = r.search(h, "nurse job in June", NOW, k=1)
    assert march[0].id == "s2:0"
    assert june[0].id == "s3:0"


def test_store_neighbours_come_with_the_hit_and_stay_in_session(history):
    h = History(history)
    r = Retriever(RetrieverConfig(neighbours=1))
    units = r.ranked_units(h, "beagle puppy Biscuit", NOW)
    assert [t.id for t in units[0]] == ["s1:2", "s1:1", "s1:3"]
    last = r.with_neighbours(h, len(h.turns) - 1)
    assert all(t.session_id == "s3" for t in last)


def test_config_without():
    cfg = RetrieverConfig(half_life_days=90)
    assert cfg.without("recency").half_life_days == 0
    assert cfg.without("neighbours").neighbours == 0
    with pytest.raises(ValueError):
        cfg.without("magic")


def test_empty_history():
    h = History([])
    for strat in (R.full_head, R.sliding_window, R.bm25_turns, R.bm25_sessions, R.make_store()):
        assert strat(h, "anything", NOW, "s").pack(100, h.tokens).turns == ()


def test_defaults_are_what_the_dev_sweep_chose():
    import json
    from dataclasses import asdict
    from pathlib import Path

    sweep = Path(__file__).resolve().parents[1] / "results" / "dev_sweep.json"
    chosen = json.loads(sweep.read_text("utf-8"))["store_best"]["config"]
    assert asdict(RetrieverConfig()) == chosen
