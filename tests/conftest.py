from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pytest

from agent_memory.datasets import Session, Turn


def make_session(sid: str, when: datetime, lines: list[tuple[str, str]]) -> Session:
    return Session(
        sid,
        when,
        tuple(Turn(f"{sid}:{i}", sid, sp, tx, when, i) for i, (sp, tx) in enumerate(lines)),
    )


@pytest.fixture
def history() -> list[Session]:
    """A small three-session chat with a known answer to each kind of question."""
    return [
        make_session(
            "s1",
            datetime(2024, 1, 10, 9),
            [
                ("user", "Hi there!"),
                ("assistant", "Hello! How can I help?"),
                ("user", "I just adopted a beagle puppy named Biscuit."),
                ("assistant", "Congratulations on Biscuit!"),
            ],
        ),
        make_session(
            "s2",
            datetime(2024, 3, 5, 18),
            [
                ("user", "I work as a nurse at Riverside Hospital these days."),
                ("assistant", "That sounds demanding."),
                ("user", "Planning a hiking trip to the Lake District in May."),
            ],
        ),
        make_session(
            "s3",
            datetime(2024, 6, 20, 12),
            [
                ("user", "Big news, I moved jobs - now a nurse at St Mary's clinic."),
                ("assistant", "Congratulations on the new job!"),
                ("user", "ok thanks"),
            ],
        ),
    ]


@pytest.fixture
def locomo_file(tmp_path: Path) -> Path:
    conv = {
        "sample_id": "conv-x",
        "conversation": {
            "speaker_a": "Ann",
            "speaker_b": "Bo",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {
                    "speaker": "Ann",
                    "dia_id": "D1:1",
                    "text": "I went to a pottery class yesterday.",
                },
                {
                    "speaker": "Bo",
                    "dia_id": "D1:2",
                    "text": "Cool! I ran a charity race.",
                    "blip_caption": "a photo of runners",
                },
            ],
            "session_2_date_time": "10:00 am on 20 June, 2023",
            "session_2": [
                {"speaker": "Ann", "dia_id": "D2:1", "text": "My pottery bowl came out great."},
            ],
        },
        "qa": [
            {
                "question": "When did Ann go to pottery class?",
                "answer": "7 May 2023",
                "evidence": ["D1:1"],
                "category": 2,
            },
            {
                "question": "What did Ann make?",
                "answer": "a bowl",
                "evidence": ["D1:1; D2:1"],
                "category": 1,
            },
            {
                "question": "What race did Ann run?",
                "adversarial_answer": "a charity race",
                "evidence": ["D1:2"],
                "category": 5,
            },
            {"question": "Broken evidence?", "answer": "x", "evidence": ["D9:9"], "category": 4},
        ],
    }
    p = tmp_path / "locomo10.json"
    p.write_text(json.dumps([conv]), "utf-8")
    return p


@pytest.fixture
def lme_file(tmp_path: Path) -> Path:
    def sess(texts: list[tuple[str, str, bool]]) -> list[dict]:
        return [{"role": r, "content": c, "has_answer": h} for r, c, h in texts]

    qs = [
        {
            "question_id": "q_ku",
            "question_type": "knowledge-update",
            "question": "Where do I work now?",
            "answer": "St Mary's",
            "question_date": "2023/09/01 (Fri) 10:00",
            "haystack_session_ids": ["b", "a", "c"],
            "haystack_dates": [
                "2023/08/01 (Tue) 09:00",
                "2023/05/01 (Mon) 09:00",
                "2023/06/01 (Thu) 09:00",
            ],
            "haystack_sessions": [
                sess([("user", "I now work at St Mary's.", True), ("assistant", "Nice.", False)]),
                sess([("user", "I work at Riverside.", True), ("assistant", "Ok.", False)]),
                sess([("user", "Unrelated chat about tea.", False)]),
            ],
            "answer_session_ids": ["a", "b"],
        },
        {
            "question_id": "q_abs_abs",
            "question_type": "single-session-user",
            "question": "What is my cat's name?",
            "answer": "You never mentioned a cat.",
            "question_date": "2023/09/01 (Fri) 10:00",
            "haystack_session_ids": ["z"],
            "haystack_dates": ["2023/08/01 (Tue) 09:00"],
            "haystack_sessions": [sess([("user", "I have a dog.", False)])],
            "answer_session_ids": [],
        },
        {
            # messy on purpose: a duplicated session, a session dated after the
            # question, and an answer session with no turn marked has_answer
            "question_id": "q_messy",
            "question_type": "multi-session",
            "question": "How many hikes did I go on?",
            "answer": "2",
            "question_date": "2023/09/01 (Fri) 10:00",
            "haystack_session_ids": ["h1", "h2", "h1", "late"],
            "haystack_dates": [
                "2023/07/01 (Sat) 09:00",
                "2023/08/01 (Tue) 09:00",
                "2023/07/01 (Sat) 09:00",
                "2023/09/03 (Sun) 09:00",
            ],
            "haystack_sessions": [
                sess([("user", "Went on a hike up Snowdon.", True)]),
                sess([("user", "Another hike, this time Helvellyn.", False)]),
                sess([("user", "Went on a hike up Snowdon.", True)]),
                sess([("user", "Weather chat.", False)]),
            ],
            "answer_session_ids": ["h1", "h2"],
        },
    ]
    p = tmp_path / "longmemeval_s_cleaned.json"
    p.write_text(json.dumps(qs, indent=2), "utf-8")
    return p
