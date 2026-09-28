import math

import pytest

from agent_memory.bm25 import BM25
from agent_memory.text import count_tokens, stem, terms


@pytest.mark.parametrize(
    ("word", "root"),
    [
        ("paintings", "paint"),
        ("painted", "paint"),
        ("painting", "paint"),
        ("running", "run"),
        ("stories", "story"),
        ("boxes", "box"),
        ("cat", "cat"),
        ("2023", "2023"),
    ],
)
def test_stem(word, root):
    assert stem(word) == root


@pytest.mark.parametrize(
    "family",
    [
        # base forms ending in "e" meet their -ed / -ing forms (the round-5 review gap)
        "move moved moves moving",
        "like liked likes liking",
        "love loved loves loving",
        "dance danced dances dancing",
        "bake baked bakes baking",
        "hike hiked hikes hiking",
        "share shared shares sharing",
        "agree agreed agrees agreeing",
        # a plural of an -ing noun reaches the verb
        "paint paints painted painting paintings",
        # doubled consonants: undoubled after -ing/-ed, except l, s, z, e, o
        "run runs running",
        "stop stopped stopping",
        "fall falls falling",
        "pass passed passes passing",
        "boss bosses",
        "study studied studies studying",
    ],
)
def test_inflections_share_one_stem(family):
    assert len({stem(w) for w in family.split()}) == 1, {w: stem(w) for w in family.split()}


@pytest.mark.parametrize("word", ["campus", "tennis", "boss", "see", "use"])
def test_words_ending_in_s_or_short_words_are_left_alone(word):
    assert stem(word) == word


def test_the_e_drop_does_not_merge_distinct_words():
    assert stem("hiking") != stem("hiring")
    assert stem("paint") != stem("pain")


def test_terms_drop_stopwords_and_possessives():
    assert terms("What did Caroline's dog eat?") == ["carolin", "dog", "eat"]  # stemmed
    assert terms("") == []


@pytest.mark.parametrize(
    ("text", "cl100k"),
    [
        ("Caroline: Hey Mel! Good to see you! How have you been?", 16),
        (
            "I just got my car serviced for the first time on March 15th, and it was a great experience.",
            23,
        ),
        (
            "Melanie: Thanks, Caroline! The painting is inspired by sunsets over the lake near our cabin.",
            21,
        ),
    ],
)
def test_count_tokens_close_to_a_real_tokeniser(text, cl100k):
    # reference counts from tiktoken's cl100k_base, recorded once
    assert abs(count_tokens(text) - cl100k) <= 0.15 * cl100k


def test_count_tokens_is_additive_over_whitespace():
    a, b = "Hello, world!", "A longer sentence with internationalisation."
    assert count_tokens(a + "\n" + b) == count_tokens(a) + count_tokens(b)
    assert count_tokens("") == 0


def test_bm25_matches_hand_computation():
    docs = [["cat", "sat"], ["dog", "sat", "sat"], ["bird"]]
    idx = BM25(docs, k1=1.2, b=0.75)
    avgdl = 2.0
    idf_cat = math.log(1 + (3 - 1 + 0.5) / (1 + 0.5))
    expected = idf_cat * 1 * 2.2 / (1 + 1.2 * (1 - 0.75 + 0.75 * 2 / avgdl))
    assert idx.scores(["cat"])[0] == pytest.approx(expected)
    assert idx.scores(["cat"])[1:] == [0.0, 0.0]


def test_bm25_rank_and_repeated_query_terms():
    idx = BM25([["apple"], ["banana", "apple"], ["banana"]])
    assert idx.rank(["banana"])[:2] == [2, 1]  # the shorter doc wins at equal tf
    assert idx.scores(["banana", "banana"]) == idx.scores(["banana"])
    assert idx.scores(["unknown"]) == [0.0, 0.0, 0.0]


def test_bm25_idf_never_negative():
    idx = BM25([["common"]] * 9 + [["rare", "common"]])
    assert all(s > 0 for s in idx.scores(["common"]))


def test_bm25_edge_cases():
    assert BM25([]).scores(["x"]) == []
    assert BM25([[]]).scores(["x"]) == [0.0]
    with pytest.raises(ValueError):
        BM25([["a"]], b=1.5)
