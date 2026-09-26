import random
from datetime import datetime

import pytest

from agent_memory.context import Context, TokenCache, pack
from agent_memory.temporal import parse_timestamp, query_window
from agent_memory.text import count_tokens

NOW = datetime(2024, 6, 20, 12)


def test_parse_dataset_timestamps():
    assert parse_timestamp("1:56 pm on 8 May, 2023") == datetime(2023, 5, 8, 13, 56)
    assert parse_timestamp("12:05 am on 1 Jan, 2024") == datetime(2024, 1, 1, 0, 5)
    assert parse_timestamp("2023/05/20 (Sat) 02:21") == datetime(2023, 5, 20, 2, 21)
    assert parse_timestamp("2024-03-01T10:00") == datetime(2024, 3, 1, 10)
    with pytest.raises(ValueError):
        parse_timestamp("next tuesday-ish")


@pytest.mark.parametrize(
    ("question", "inside", "outside"),
    [
        ("what did I say about Biscuit last month?", datetime(2024, 5, 15), datetime(2024, 2, 1)),
        ("the thing I bought three weeks ago", datetime(2024, 5, 30), datetime(2024, 4, 1)),
        ("what happened in March?", datetime(2024, 3, 5), datetime(2024, 5, 1)),
        ("anything in December 2023", datetime(2023, 12, 24), datetime(2024, 1, 10)),
        ("what did I do yesterday", datetime(2024, 6, 19, 8), datetime(2024, 6, 10)),
        ("over the past 2 months", datetime(2024, 5, 1), datetime(2024, 3, 1)),
        ("anything in 2023?", datetime(2023, 7, 1), datetime(2024, 1, 2)),
    ],
)
def test_query_window(question, inside, outside):
    lo, hi = query_window(question, NOW)
    assert lo <= inside < hi
    assert not lo <= outside < hi


def test_month_later_than_now_means_last_year():
    window = query_window("what did we do in September", NOW)
    assert window is not None and window[0].year == 2023


def test_no_time_expression():
    assert query_window("where does Sam work?", NOW) is None


def _turns(history):
    return [t for s in history for t in s.turns]


def test_pack_respects_budget_and_counts_what_it_renders(history):
    cache = TokenCache()
    turns = _turns(history)
    rng = random.Random(0)
    for budget in (0, 5, 20, 37, 60, 1000):
        order = turns[:]
        rng.shuffle(order)
        ctx = pack(([t] for t in order), budget, cache)
        assert ctx.tokens <= budget
        assert ctx.tokens == count_tokens(ctx.render())


def test_pack_orders_chronologically_with_one_header_per_session(history):
    turns = _turns(history)
    ctx = pack(([t] for t in reversed(turns)), 10_000, TokenCache())
    assert [t.id for t in ctx.turns] == [t.id for t in turns]
    assert ctx.render().count("[session s2") == 1


def test_stop_at_first_misfit_versus_skipping(history):
    turns = _turns(history)
    cache = TokenCache()
    big = turns[2]  # the long beagle turn
    small = turns[4 + 1]  # "That sounds demanding."
    budget = cache.line(small) + cache.header(small) + 1
    positional = pack([[big], [small]], budget, cache, stop_at_first_misfit=True)
    ranked = pack([[big], [small]], budget, cache)
    assert positional.turns == ()
    assert [t.id for t in ranked.turns] == [small.id]


def test_pack_rejects_negative_budget(history):
    with pytest.raises(ValueError):
        pack([], -1, TokenCache())


def test_context_facts_render_first():
    ctx = Context((), 0, 10, facts=("sam / job: nurse (since 2024-01-01)",))
    assert ctx.render().splitlines()[1].startswith("sam / job")
