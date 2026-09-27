"""Timestamps and relative time expressions.

`parse_timestamp` reads the two dataset formats (and ISO 8601). `query_window` turns a
phrase in a question such as "last month", "three weeks ago" or "in May 2023" into a
half-open date window relative to the time the question is asked. Windows are padded
on purpose: people say "two weeks ago" about something that happened ten days ago.
"""

from __future__ import annotations

import calendar
import re
from datetime import UTC, datetime, timedelta

MONTHS = {m.lower(): i for i, m in enumerate(calendar.month_name) if m}
MONTHS |= {m.lower(): i for i, m in enumerate(calendar.month_abbr) if m}
NUMBER_WORDS = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "couple of": 2,
    "a couple of": 2,
    "few": 3,
    "a few": 3,
}
UNIT_DAYS = {"day": 1, "week": 7, "month": 30, "year": 365}

_LOCOMO = re.compile(
    r"(\d{1,2}):(\d{2})\s*([ap]m)\s+on\s+(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", re.I
)
_LME = re.compile(r"(\d{4})/(\d{2})/(\d{2})(?:\s*\([A-Za-z]+\))?\s*(\d{2}):(\d{2})")


def naive(dt: datetime) -> datetime:
    """The store keeps naive datetimes. An aware one is converted to UTC and stripped,
    so '2024-01-01T09:00+05:00' and '2024-01-01T04:00' compare as the same moment."""
    return dt.astimezone(UTC).replace(tzinfo=None) if dt.tzinfo else dt


def utc_now() -> datetime:
    """Now, as naive UTC - the store's clock. Defaulting to local time would shift every
    relative window ("last week") by the machine's UTC offset against timestamps that
    arrived with an offset and were stored as UTC."""
    return datetime.now(UTC).replace(tzinfo=None)


def parse_timestamp(text: str) -> datetime:
    """Parse '1:56 pm on 8 May, 2023' (LoCoMo), '2023/05/20 (Sat) 02:21' (LongMemEval)
    or an ISO 8601 string. Raises ValueError on anything else."""
    text = text.strip()
    if m := _LOCOMO.fullmatch(text):
        hour, minute, ampm, day, month, year = m.groups()
        h = int(hour) % 12 + (12 if ampm.lower() == "pm" else 0)
        mon = MONTHS.get(month.lower())
        if mon is None:
            raise ValueError(f"unknown month {month!r} in {text!r}")
        return datetime(int(year), mon, int(day), h, int(minute))
    if m := _LME.fullmatch(text):
        y, mo, d, h, mi = map(int, m.groups())
        return datetime(y, mo, d, h, mi)
    try:
        return naive(datetime.fromisoformat(text))
    except ValueError:
        raise ValueError(f"unrecognised timestamp {text!r}") from None


_NUM = (
    r"(\d+|a couple of|couple of|a few|few|an|a|one|two|three|four|five|six|seven|eight|nine"
    r"|ten|eleven|twelve)"
)
_AGO = re.compile(rf"\b{_NUM}\s+(day|week|month|year)s?\s+ago\b")
_PAST = re.compile(rf"\b(?:past|last|previous)\s+{_NUM}\s+(day|week|month|year)s?\b")
_LAST = re.compile(
    r"\b(yesterday|last night|today|this morning|tonight|"
    r"(?:last|this|past) (?:week|month|year|weekend))\b"
)
_MONTH_NAMES = "|".join(sorted(MONTHS, key=len, reverse=True))
_IN_MONTH = re.compile(
    rf"\b(?:in|during|of|since)\s+(?:early |late |mid-?)?({_MONTH_NAMES})\b(?:,?\s+(\d{{4}}))?"
)
_YEAR = re.compile(r"\bin\s+((?:19|20)\d{2})\b")


def _num(tok: str) -> int:
    return int(tok) if tok.isdigit() else NUMBER_WORDS[tok]


def query_window(question: str, now: datetime) -> tuple[datetime, datetime] | None:
    """The date window a question refers to, or None if it names no time.

    The first matching phrase wins, most specific pattern first.
    """
    q = question.lower()
    day = timedelta(days=1)
    if m := _AGO.search(q):
        span = UNIT_DAYS[m.group(2)]
        centre = now - _num(m.group(1)) * span * day
        pad = max(span, 2) * day
        return centre - pad, centre + pad
    if m := _PAST.search(q):
        return now - _num(m.group(1)) * UNIT_DAYS[m.group(2)] * day - day, now + day
    if m := _LAST.search(q):
        phrase = m.group(1)
        if phrase in ("yesterday", "last night"):
            return now - 2 * day, now
        if phrase in ("today", "this morning", "tonight"):
            return now - day, now + day
        which, unit = phrase.split()
        span = 3 if unit == "weekend" else UNIT_DAYS[unit]
        if which in ("this", "past"):
            return now - span * day - day, now + day
        return now - 2 * span * day - day, now  # "last month": the previous one, padded
    if m := _IN_MONTH.search(q):
        mon = MONTHS[m.group(1)]
        year = int(m.group(2)) if m.group(2) else (now.year if mon <= now.month else now.year - 1)
        start = datetime(year, mon, 1)
        end = datetime(year + (mon == 12), mon % 12 + 1, 1)
        return start - 3 * day, end + 3 * day
    if m := _YEAR.search(q):
        y = int(m.group(1))
        return datetime(y, 1, 1), datetime(y + 1, 1, 1)
    return None
