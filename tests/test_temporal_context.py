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


def test_positional_cuts_the_misfit_turn_ranked_skips_it(history):
    turns = _turns(history)
    cache = TokenCache()
    big = turns[2]  # "I just adopted a beagle puppy named Biscuit."
    small = turns[5]  # "That sounds demanding."
    budget = cache.header(big) + cache.line(big) - 1  # room for all of `big` but a word
    assert cache.header(small) + cache.line(small) <= budget
    positional = pack([[big], [small]], budget, cache, positional=True)
    ranked = pack([[big], [small]], budget, cache)
    assert [t.id for t in positional.turns] == [big.id]
    assert positional.turns[0].text.endswith("…") and positional.turns[0].text != big.text
    assert positional.tokens <= budget
    assert positional.tokens == count_tokens(positional.render())
    assert [t.id for t in ranked.turns] == [small.id]


def test_positional_fills_the_budget(history):
    turns = _turns(history)
    cache = TokenCache()
    for budget in (40, 57, 73, 91):
        ctx = pack(([t] for t in reversed(turns)), budget, cache, positional=True)
        # full, or short by less than a header plus the minimum cut
        assert budget - 30 <= ctx.tokens <= budget
        assert ctx.tokens == count_tokens(ctx.render())


def test_no_cut_when_only_a_sliver_is_left(history):
    turns = _turns(history)
    cache = TokenCache()
    budget = cache.line(turns[0]) + cache.header(turns[0]) + 3
    ctx = pack([[turns[0]], [turns[2]]], budget, cache, positional=True)
    assert [t.id for t in ctx.turns] == [turns[0].id]


def test_pack_can_continue_from_a_context(history):
    turns = _turns(history)
    cache = TokenCache()
    first = pack([[turns[-1]]], 30, cache)
    both = pack([[turns[0]], [turns[-1]]], 60, cache, start=first)
    assert {t.id for t in both.turns} == {turns[0].id, turns[-1].id}
    assert both.tokens == count_tokens(both.render())


def test_pack_rejects_negative_budget(history):
    with pytest.raises(ValueError):
        pack([], -1, TokenCache())


def test_context_facts_render_first():
    ctx = Context((), 0, 10, facts=("sam / job: nurse (since 2024-01-01)",))
    assert ctx.render().splitlines()[1].startswith("sam / job")
