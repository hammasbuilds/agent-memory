"""agent-memory: a persistent memory for agents, from the command line.

agent-memory --db mem.db ingest chat.jsonl
agent-memory --db mem.db search "where does Sam work"
agent-memory --db mem.db context "what did I say about the trip last month" --budget 512
agent-memory --db mem.db remember sam employer "Acme" --at 2024-03-01
agent-memory --db mem.db history sam employer
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from agent_memory.datasets import Session, Turn, iter_longmemeval, load_locomo
from agent_memory.forget import POLICY_HELP, parse_policy
from agent_memory.jsonl import read_jsonl
from agent_memory.store import MemoryStore
from agent_memory.temporal import naive, utc_now


def _when(s: str | None) -> datetime:
    if s is None:
        return utc_now()
    try:
        return naive(datetime.fromisoformat(s))
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an ISO date/time: {s!r}") from None


def _positive(s: str) -> int:
    try:
        n = int(s)
    except ValueError:
        n = 0
    if n < 1:
        raise argparse.ArgumentTypeError(f"expected a positive whole number, got {s!r}")
    return n


def _sessions(args: argparse.Namespace) -> list[Session]:
    if args.format == "jsonl":
        return read_jsonl(args.file)
    if args.format == "locomo":
        qs = load_locomo(args.file)
        pick = args.pick or qs[0].qid.split(":")[0]
        match = [q for q in qs if q.qid.split(":")[0] == pick]
        if not match:
            raise SystemExit(f"no conversation {pick!r} in {args.file}")
        return list(match[0].history)
    for q in iter_longmemeval(args.file):
        if args.pick in (None, q.qid):
            return list(q.history)
    raise SystemExit(f"no question {args.pick!r} in {args.file}")


def cmd_ingest(store: MemoryStore, args: argparse.Namespace) -> None:
    if not args.file.exists():
        raise SystemExit(f"{args.file}: no such file")
    sessions = _sessions(args)
    got = store.ingest(sessions)
    skipped = f"{got.already_stored} skipped: already stored"
    if got.empty:
        skipped += f"; {got.empty} skipped: empty text"
    print(f"ingested {got.added} new turns from {len(sessions)} sessions ({skipped})")


def _print_turns(turns: list[Turn]) -> None:
    if not turns:
        print("(no matching memories)")
    for t in turns:
        print(f"{t.timestamp:%Y-%m-%d %H:%M}  {t.id}  {t.speaker}: {t.text}")


def cmd_search(store: MemoryStore, args: argparse.Namespace) -> None:
    _print_turns(store.search(args.query, _when(args.now), args.k))


def cmd_context(store: MemoryStore, args: argparse.Namespace) -> None:
    ctx = store.context(args.query, args.budget, _when(args.now))
    print(ctx.render() or "(nothing relevant in memory)")
    print(
        f"\n-- {ctx.tokens} of {ctx.budget} tokens, {len(ctx.turns)} turns, {len(ctx.facts)} facts",
        file=sys.stderr,
    )


def cmd_add(store: MemoryStore, args: argparse.Namespace) -> None:
    print(store.add_turn(args.session, args.speaker, args.text, _when(args.at)))


def cmd_remember(store: MemoryStore, args: argparse.Namespace) -> None:
    f = store.remember(args.subject, args.attribute, args.value, _when(args.at), args.source)
    print(f"fact #{f.id}: {f.render()}")


def cmd_facts(store: MemoryStore, args: argparse.Namespace) -> None:
    facts = (
        store.facts_as_of(_when(args.as_of), args.subject)
        if args.as_of
        else store.current_facts(args.subject)
    )
    if not facts:
        print("(no facts)")
    for f in facts:
        print(f"#{f.id}  {f.render()}")


def cmd_history(store: MemoryStore, args: argparse.Namespace) -> None:
    versions = store.history_of(args.subject, args.attribute)
    if not versions:
        print("(no such fact)")
    for f in versions:
        state = "current" if f.current else f"superseded by #{f.superseded_by}"
        src = f"  from {f.source_turn}" if f.source_turn else ""
        print(f"#{f.id}  {f.valid_from:%Y-%m-%d}  {f.value}  [{state}]{src}")


def cmd_forget(store: MemoryStore, args: argparse.Namespace) -> None:
    now = _when(args.now)
    try:
        drop, trunc = parse_policy(args.policy, now)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if trunc is not None:
        print(f"truncated {store.compact(trunc)} turns to {trunc} tokens")
    else:
        assert drop is not None
        print(f"forgot {store.forget(drop, now)} turns (soft delete; `purge` removes them)")


def cmd_purge(store: MemoryStore, args: argparse.Namespace) -> None:
    print(f"purged {store.purge()} forgotten turns")


def cmd_stats(store: MemoryStore, args: argparse.Namespace) -> None:
    for k, v in store.stats().items():
        print(f"{k:18} {v}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="agent-memory",
        description="Persistent cross-session memory for agents: SQLite store, BM25 retrieval, "
        "temporal queries, versioned facts.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n" + __doc__.split("\n", 2)[2],
    )
    p.add_argument("--db", default="memory.db", help="SQLite file (default: memory.db)")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    s = sub.add_parser("ingest", help="load conversation turns from a file")
    s.add_argument("file", type=Path)
    s.add_argument(
        "--format",
        choices=("jsonl", "locomo", "longmemeval"),
        default="jsonl",
        help="jsonl: one {session, time, speaker, text} per line (default)",
    )
    s.add_argument(
        "--pick", help="LoCoMo conversation id / LongMemEval question id (default: first)"
    )
    s.set_defaults(fn=cmd_ingest)

    s = sub.add_parser("add", help="append one turn")
    s.add_argument("session")
    s.add_argument("speaker")
    s.add_argument("text")
    s.add_argument("--at", help="ISO time (default: now)")
    s.set_defaults(fn=cmd_add)

    for name, fn, hlp in (
        ("search", cmd_search, "best-matching turns"),
        ("context", cmd_context, "a prompt-ready context within a token budget"),
    ):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("query")
        s.add_argument(
            "--now",
            help="ISO time the question is asked (default: now); "
            "'last month' etc. are relative to it; offsets are converted to UTC",
        )
        if name == "search":
            s.add_argument("-k", type=_positive, default=10, help="how many turns (default 10)")
        else:
            s.add_argument(
                "--budget", type=_positive, default=2048, help="token budget (default 2048)"
            )
        s.set_defaults(fn=fn)

    s = sub.add_parser("remember", help="record a fact; a newer value supersedes the old one")
    s.add_argument("subject")
    s.add_argument("attribute")
    s.add_argument("value")
    s.add_argument("--at", help="ISO time the value became true (default: now)")
    s.add_argument("--source", help="id of the turn that states it")
    s.set_defaults(fn=cmd_remember)

    s = sub.add_parser("facts", help="current facts (or as of a date)")
    s.add_argument("--subject")
    s.add_argument("--as-of", help="ISO date: the facts as they stood then")
    s.set_defaults(fn=cmd_facts)

    s = sub.add_parser("history", help="every version of one fact")
    s.add_argument("subject")
    s.add_argument("attribute")
    s.set_defaults(fn=cmd_history)

    s = sub.add_parser("forget", help="apply a forgetting or compaction policy")
    s.add_argument("--policy", required=True, help=POLICY_HELP)
    s.add_argument("--now", help="ISO time for older-than (default: now)")
    s.set_defaults(fn=cmd_forget)

    sub.add_parser("purge", help="hard-delete forgotten turns").set_defaults(fn=cmd_purge)
    sub.add_parser("stats", help="counts of sessions, turns and facts").set_defaults(fn=cmd_stats)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    writes = {"ingest", "add", "remember"}
    if args.cmd not in writes and args.db != ":memory:" and not Path(args.db).exists():
        raise SystemExit(f"error: no memory at {args.db} - `ingest`, `add` or `remember` first")
    try:
        with MemoryStore(args.db) as store:
            args.fn(store, args)
    except (ValueError, argparse.ArgumentTypeError, FileNotFoundError) as e:
        raise SystemExit(f"error: {e}") from None


if __name__ == "__main__":
    main()
