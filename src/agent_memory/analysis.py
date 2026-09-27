"""Two diagnostics that explain *why* a question type is lost, independent of any
retriever's tuning:

lexical_visibility   How many of a question's gold evidence turns share at least one
                     content word with the question? A turn that shares none is
                     invisible to any lexical retriever, BM25 or otherwise - it can only
                     arrive by accident (as a neighbour, or in a whole session).
answer_location      Are the gold answer's words in the evidence turns' text, or only
                     reachable once the session date is shown too? For "when did..."
                     questions the answer is usually a date the turn never says.
"""

from __future__ import annotations

import calendar

from agent_memory.datasets import Question
from agent_memory.text import terms


def _speaker_terms(q: Question) -> set[str]:
    return {w for s in q.history for t in s.turns for w in terms(t.speaker)}


def lexical_visibility(q: Question) -> list[bool] | None:
    """For each gold evidence turn: does it share a content word with the question?
    Speaker names are ignored (in LoCoMo every question names a speaker, which would make
    every one of that speaker's turns 'match'). None if there are no evidence turns."""
    if not q.evidence_turns:
        return None
    names = _speaker_terms(q)
    qt = set(terms(q.question)) - names
    ev = [t for s in q.history for t in s.turns if t.id in q.evidence_turns]
    return [bool(qt & set(terms(t.text))) for t in ev]


_DATE_WORDS = set(
    terms(
        " ".join(calendar.month_name[1:] + calendar.month_abbr[1:] + calendar.day_name[:])
        + " week weekend month year day ago last next before after early late mid around"
        + " beginning end morning evening night"
    )
)


def _is_date_word(w: str) -> bool:
    return w.isdigit() or w in _DATE_WORDS or w.rstrip("thsndr").isdigit()


def answer_location(q: Question) -> str | None:
    """Where the gold answer's content words are:

    in_text     all of them appear in the evidence turns
    needs_date  the ones missing are all date words (months, days, years, 'week',
                'last'...) - the answer is a date the turn implies ("yesterday") but
                never states, recoverable only from the session timestamp
    elsewhere   something else is missing: the answer is inferred, counted or
                aggregated rather than stated

    None when there is no gold evidence or the answer has no content words.
    """
    if not q.evidence_turns:
        return None
    ans = set(terms(q.answer))
    if not ans:
        return None
    ev = [t for s in q.history for t in s.turns if t.id in q.evidence_turns]
    missing = ans - {w for t in ev for w in terms(t.text)}
    if not missing:
        return "in_text"
    # A bare count ("3") is not a date: something beyond digits must be date-like, or
    # the number must look like a year.
    datelike = any(not w.isdigit() or len(w) == 4 for w in missing)
    if datelike and all(_is_date_word(w) for w in missing):
        return "needs_date"
    return "elsewhere"


def evidence_position(q: Question) -> float | None:
    """Mean position of the gold evidence turns in the history, 0 = the first turn ever
    said, 1 = the last before the question. Recency-based memory can only win when
    evidence sits near 1."""
    turns = [t.id for s in q.history for t in s.turns]
    if not q.evidence_turns or len(turns) < 2:
        return None
    pos = [i / (len(turns) - 1) for i, tid in enumerate(turns) if tid in q.evidence_turns]
    return sum(pos) / len(pos)
