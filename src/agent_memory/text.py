"""Tokenisation for retrieval, and token counting for budgets.

Two different jobs, deliberately kept apart:

* `terms()` produces index terms for BM25: lowercased word tokens, stop words
  removed, a light suffix-stripping stemmer applied.
* `count_tokens()` estimates what a context costs an LLM, without shipping a
  tokeniser. It is an approximation, but it is applied identically to every
  strategy, so comparisons between strategies at the same budget are exact even
  where the absolute count is a few percent off.
"""

from __future__ import annotations

import re

_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_PIECE = re.compile(r"[A-Za-z]+|[0-9]+|[^\sA-Za-z0-9]")

_STOPWORD_TEXT = (
    "a about above after again against all am an and any are as at be because been "
    "before being below between both but by can could did do does doing down during each "
    "few for from further had has have having he her here hers herself him himself his "
    "how i if in into is it its itself just me more most my myself no nor not now of off "
    "on once only or other our ours ourselves out over own same she should so some such "
    "than that the their theirs them themselves then there these they this those through "
    "to too under until up very was we were what when where which while who whom why "
    "will with would you your yours yourself yourselves i'm i've i'd i'll you're you've "
    "it's that's what's don't didn't doesn't isn't wasn't can't won't let's hey hi oh "
    "yeah yes ok okay really also "
)
STOPWORDS = frozenset(_STOPWORD_TEXT.split())

_SUFFIXES = ("ingly", "edly", "ing", "ies", "ied", "ed", "ly", "es", "s")


def stem(word: str) -> str:
    """Strip one common English inflectional suffix, keeping at least 3 characters.

    Deliberately light: it conflates plural/tense variants ("paintings", "painted",
    "painting" -> "paint") without the over-stemming of a full Porter stemmer.
    """
    if len(word) <= 3 or word.isdigit():
        return word
    for suf in _SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            root = word[: -len(suf)]
            if suf in ("ies", "ied"):
                return root + "y"
            if suf == "es" and not root.endswith(("s", "x", "z", "ch", "sh")):
                return word[:-1]  # "shares" -> "share", not "shar"
            if suf in ("ing", "ed") and len(root) > 3 and root[-1] == root[-2]:
                return root[:-1]  # "running" -> "run"
            return root
    return word


def terms(text: str) -> list[str]:
    """Index terms for BM25: lowercase words minus stop words, stemmed."""
    out = []
    for tok in _WORD.findall(text.lower()):
        tok = tok.removesuffix("'s")
        if tok and tok not in STOPWORDS:
            out.append(stem(tok))
    return out


def count_tokens(text: str) -> int:
    """Approximate BPE token count.

    One token per punctuation mark, one per three digits, one per word of up to eight
    letters, plus one per further five letters of a longer word. Calibrated against
    OpenAI's cl100k_base on all 5,882 LoCoMo turns: 4% over in total, 5.8% mean
    absolute error per turn (see `scripts/calibrate_tokens.py`).
    """
    n = 0
    for piece in _PIECE.findall(text):
        c = piece[0]
        if c.isalpha():
            n += 1 if len(piece) <= 8 else 1 + (len(piece) - 4) // 5
        elif c.isdigit():
            n += (len(piece) + 2) // 3
        else:
            n += 1
    return n
