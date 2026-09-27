from datetime import datetime

import pytest

from agent_memory.forget import phatic, role
from agent_memory.store import MemoryStore
from agent_memory.text import count_tokens


def test_ingest_is_idempotent_and_round_trips(history):
    with MemoryStore() as m:
        assert m.ingest(history) == 10
        assert m.ingest(history) == 0
        back = m.sessions()
        assert [s.id for s in back] == ["s1", "s2", "s3"]
        assert back[0].turns[2].text == history[0].turns[2].text
        assert back[0].turns[2].timestamp == history[0].turns[2].timestamp


def test_persists_across_reopen(tmp_path, history):
    db = tmp_path / "m.db"
    with MemoryStore(db) as m:
        m.ingest(history)
        m.remember("user", "employer", "Riverside", datetime(2024, 3, 5))
    with MemoryStore(db) as m:
        assert m.stats()["turns"] == 10
        assert m.current_facts()[0].value == "Riverside"


def test_add_turn_ids_do_not_collide_after_purge():
    with MemoryStore() as m:
        t = datetime(2024, 1, 1)
        a = m.add_turn("s", "user", "hello there friend", t)
        b = m.add_turn("s", "user", "second message here", t)
        m.forget(lambda turn: turn.id == b)
        m.purge()
        c = m.add_turn("s", "user", "third message", t)
        assert c not in (a, b)
        assert [t.id for s in m.sessions() for t in s.turns] == [a, c]
        with pytest.raises(ValueError):
            m.add_turn("s", "user", "   ", t)


def test_newer_fact_supersedes_and_history_is_kept():
    with MemoryStore() as m:
        m.remember("User", "Employer", "Riverside Hospital", datetime(2024, 3, 5), "s2:0")
        new = m.remember("user", "employer", "St Mary's", datetime(2024, 6, 20), "s3:0")
        versions = m.history_of("user", "employer")
        assert [f.value for f in versions] == ["Riverside Hospital", "St Mary's"]
        assert versions[0].superseded_by == new.id
        assert [f.value for f in m.current_facts("user")] == ["St Mary's"]
        assert m.stats()["facts_superseded"] == 1


def test_late_arriving_older_fact_does_not_displace_current():
    with MemoryStore() as m:
        m.remember("user", "city", "Leeds", datetime(2024, 1, 1))
        m.remember("user", "city", "York", datetime(2024, 6, 1))
        m.remember("user", "city", "Hull", datetime(2024, 3, 1))  # learned late
        assert [f.value for f in m.history_of("user", "city")] == ["Leeds", "Hull", "York"]
        assert m.current_facts()[0].value == "York"
        chain = {f.value: f.superseded_by for f in m.history_of("user", "city")}
        ids = {f.value: f.id for f in m.history_of("user", "city")}
        assert chain == {"Leeds": ids["Hull"], "Hull": ids["York"], "York": None}


def test_restating_a_fact_is_a_no_op():
    with MemoryStore() as m:
        a = m.remember("user", "pet", "beagle", datetime(2024, 1, 1))
        b = m.remember("user", "pet", " Beagle ", datetime(2024, 5, 1))
        assert a.id == b.id
        assert len(m.history_of("user", "pet")) == 1
        with pytest.raises(ValueError):
            m.remember("user", "", "x", datetime(2024, 1, 1))


def test_facts_as_of():
    with MemoryStore() as m:
        m.remember("user", "employer", "Riverside", datetime(2024, 3, 5))
        m.remember("user", "employer", "St Mary's", datetime(2024, 6, 20))
        assert m.facts_as_of(datetime(2024, 1, 1)) == []
        assert m.facts_as_of(datetime(2024, 4, 1))[0].value == "Riverside"
        assert m.facts_as_of(datetime(2024, 7, 1))[0].value == "St Mary's"


def test_context_puts_relevant_facts_first_within_budget(history):
    with MemoryStore() as m:
        m.ingest(history)
        m.remember("user", "employer", "St Mary's clinic", datetime(2024, 6, 20), "s3:0")
        m.remember("user", "dog", "Biscuit the beagle", datetime(2024, 1, 10), "s1:2")
        ctx = m.context("where does the user work as a nurse", budget=80, now=datetime(2024, 7, 1))
        assert ctx.facts and "St Mary" in ctx.facts[0]
        assert all("Biscuit" not in f for f in ctx.facts)
        assert ctx.tokens <= 80
        assert ctx.tokens == count_tokens(ctx.render())


def test_search_reflects_new_turns_and_forgetting(history):
    with MemoryStore() as m:
        m.ingest(history)
        assert m.search("kayak") == []
        m.add_turn("s4", "user", "Bought a red kayak today", datetime(2024, 7, 1))
        assert m.search("kayak")[0].text.startswith("Bought")
        # fewer than 3 content words: "Hi there!", "Hello! How can I help?",
        # "Congratulations on Biscuit!", "That sounds demanding.", "ok thanks"
        assert m.forget(phatic(3)) == 5
        assert m.forget(role("assistant")) == 1  # only "Congratulations on the new job!" left
        assert all(t.speaker == "user" for s in m.sessions() for t in s.turns)
        assert m.stats()["forgotten_turns"] == 6
        assert m.purge() == 6
        assert m.stats()["forgotten_turns"] == 0 and m.stats()["turns"] == 5


def test_compact_truncates_long_turns(history):
    with MemoryStore() as m:
        m.ingest(history)
        assert m.compact(6) > 0
        assert all(count_tokens(t.text) <= 6 for s in m.sessions() for t in s.turns)


def test_timezone_aware_times_are_normalised_to_utc():
    from datetime import timedelta, timezone

    karachi = timezone(timedelta(hours=5))
    with MemoryStore() as m:
        m.add_turn("s", "user", "bought a red kayak", datetime(2024, 7, 1, 9, tzinfo=karachi))
        assert m.sessions()[0].turns[0].timestamp == datetime(2024, 7, 1, 4)
        assert m.search("kayak", now=datetime(2024, 7, 2, tzinfo=karachi))
        m.remember("user", "boat", "kayak", datetime(2024, 7, 1, 9, tzinfo=karachi))
        assert m.facts_as_of(datetime(2024, 7, 1, 4, 30))[0].value == "kayak"


def test_a_single_fact_is_found():
    # the README's "As a library" example: one fact whose subject is in the query
    with MemoryStore() as m:
        m.add_turn("s1", "user", "I just adopted a beagle called Biscuit.", datetime(2024, 1, 10))
        m.remember("user", "dog", "Biscuit (beagle)", datetime(2024, 1, 10), source_turn="s1:0")
        ctx = m.context("what's my dog called?", budget=512, now=datetime(2024, 2, 1))
        assert ctx.facts == ("user / dog: Biscuit (beagle) (since 2024-01-10)",)


def test_facts_sharing_the_query_word_are_ranked_not_excluded():
    with MemoryStore() as m:
        t = datetime(2024, 1, 1)
        m.remember("user", "dog", "beagle", t)
        m.remember("user", "cat", "tabby", t)
        both = m.context("tell me about the user", budget=512, now=t)
        assert len(both.facts) == 2  # all equally relevant: all kept
        dog = m.context("what dog does the user have", budget=512, now=t)
        assert dog.facts == ("user / dog: beagle (since 2024-01-01)",)
        assert m.context("xylophone", budget=512, now=t).facts == ()


def test_ingest_appends_after_existing_turns_even_if_input_seq_restarts(history):
    from agent_memory.datasets import Session, Turn

    with MemoryStore() as m:
        m.ingest(history)
        t = datetime(2024, 6, 21)
        extra = Session(
            "s3", t, (Turn("s3:new", "s3", "user", "one more thing about kayaks", t, 0),)
        )
        assert m.ingest([extra]) == 1
        s3 = next(s for s in m.sessions() if s.id == "s3")
        assert [x.id for x in s3.turns][-1] == "s3:new" and s3.turns[-1].seq == 3
