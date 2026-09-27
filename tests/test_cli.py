import json

import pytest

from agent_memory.cli import build_parser, main
from agent_memory.store import MemoryStore


def run(capsys, *argv):
    main(list(argv))
    return capsys.readouterr()


@pytest.fixture
def db(tmp_path):
    return str(tmp_path / "m.db")


@pytest.fixture
def chat(tmp_path):
    p = tmp_path / "chat.jsonl"
    lines = [
        {
            "session": "a",
            "time": "2024-01-10T09:00",
            "speaker": "user",
            "text": "I adopted a beagle called Biscuit.",
        },
        {
            "session": "a",
            "time": "2024-01-10T09:01",
            "speaker": "assistant",
            "text": "Lovely name!",
        },
        {
            "session": "b",
            "time": "2024-05-02T18:00",
            "speaker": "user",
            "text": "Work at the hospital is exhausting.",
        },
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n\n", "utf-8")
    return p


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("ingest", "search", "context", "remember", "facts", "history", "forget", "stats"):
        assert cmd in out


def test_ingest_search_context(capsys, db, chat):
    assert "ingested 3 new turns" in run(capsys, "--db", db, "ingest", str(chat)).out
    assert (
        "0 new turns from 2 sessions (3 skipped" in run(capsys, "--db", db, "ingest", str(chat)).out
    )
    out = run(capsys, "--db", db, "search", "what is my dog called beagle").out
    assert "Biscuit" in out.splitlines()[0]
    res = run(
        capsys,
        "--db",
        db,
        "context",
        "what did I say about work last month",
        "--now",
        "2024-06-01",
        "--budget",
        "40",
    )
    assert "hospital" in res.out and "[session b - 2024-05-02" in res.out
    assert "of 40 tokens" in res.err
    assert "(no matching memories)" in run(capsys, "--db", db, "search", "xylophone").out


def test_facts_and_history(capsys, db):
    run(capsys, "--db", db, "remember", "user", "employer", "Riverside", "--at", "2024-03-05")
    run(capsys, "--db", db, "remember", "user", "employer", "St Mary's", "--at", "2024-06-20")
    out = run(capsys, "--db", db, "history", "user", "employer").out.splitlines()
    assert "Riverside" in out[0] and "superseded by #2" in out[0]
    assert "St Mary's" in out[1] and "[current]" in out[1]
    assert "St Mary's" in run(capsys, "--db", db, "facts").out
    assert "Riverside" in run(capsys, "--db", db, "facts", "--as-of", "2024-04-01").out
    assert "(no such fact)" in run(capsys, "--db", db, "history", "user", "shoe size").out


def test_forget_purge_stats(capsys, db, chat):
    run(capsys, "--db", db, "ingest", str(chat))
    assert "forgot 1 turns" in run(capsys, "--db", db, "forget", "--policy", "role:assistant").out
    stats = run(capsys, "--db", db, "stats").out
    assert "forgotten_turns    1" in stats
    assert "purged 1" in run(capsys, "--db", db, "purge").out
    out = run(capsys, "--db", db, "forget", "--policy", "truncate:4").out
    assert out.startswith("truncated 2 turns")


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["forget", "--policy", "shred"], "bad policy"),
        (["ingest", "missing.jsonl"], "no such file"),
        (["remember", "user", "pet", "dog", "--at", "last tuesday"], "not an ISO"),
        (["add", "s", "user", "   "], "empty turn"),
        (["search", "anything"], "no memory at"),
    ],
)
def test_helpful_errors(db, argv, message):
    if message != "no memory at":
        main(["--db", db, "add", "s", "user", "hello there"])
    with pytest.raises(SystemExit) as e:
        main(["--db", db, *argv])
    assert message in str(e.value)


def test_bad_jsonl_line_is_located(tmp_path, db):
    p = tmp_path / "bad.jsonl"
    p.write_text(
        '{"session": "a", "time": "2024-01-01", "speaker": "u", "text": "hi"}\n{"oops": 1}\n'
    )
    with pytest.raises(SystemExit, match=r"bad.jsonl:2"):
        main(["--db", db, "ingest", str(p)])


@pytest.mark.parametrize("argv", [["context", "q", "--budget", "0"], ["search", "q", "-k", "-2"]])
def test_non_positive_budget_or_k_is_a_usage_error(db, argv, capsys):
    with pytest.raises(SystemExit) as e:
        main(["--db", db, *argv])
    assert e.value.code == 2
    assert "positive whole number" in capsys.readouterr().err


def test_a_second_file_continuing_a_session_appends(capsys, db, chat, tmp_path):
    more = tmp_path / "more.jsonl"
    lines = [
        {
            "session": "a",
            "time": "2024-01-11T08:00",
            "speaker": "user",
            "text": "Biscuit chewed my shoe.",
        },
        {"session": "a", "time": "2024-01-11T08:01", "speaker": "user", "text": "ok thanks"},
        {"session": "a", "time": "2024-01-11T08:02", "speaker": "user", "text": "ok thanks"},
    ]
    more.write_text("\n".join(json.dumps(x) for x in lines), "utf-8")
    run(capsys, "--db", db, "ingest", str(chat))
    out = run(capsys, "--db", db, "ingest", str(more)).out
    assert "ingested 3 new turns" in out and "0 skipped" in out
    with MemoryStore(db) as m:
        a = next(s for s in m.sessions() if s.id == "a")
        assert [t.text for t in a.turns] == [
            "I adopted a beagle called Biscuit.",
            "Lovely name!",
            "Biscuit chewed my shoe.",
            "ok thanks",
            "ok thanks",
        ]
        assert m.stats()["turns"] == 6
    assert "0 new turns" in run(capsys, "--db", db, "ingest", str(more)).out  # idempotent
