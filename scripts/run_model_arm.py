"""The model arm: dense/hybrid retrieval with nomic-embed-text, LLM fact extraction into
the store, and answer accuracy with qwen2.5:14b-instruct judged against the gold
answers. Every call is cached in data/model_cache.sqlite, so a killed run resumes.

    uv run python scripts/run_model_arm.py --dry-run          # job list + call counts
    uv run python scripts/run_model_arm.py                    # everything
    uv run python scripts/run_model_arm.py --stage answer --per-type 40

Stages (each writes results/*.json; the code is in agent_memory/study/models.py):
  retrieval  dense and hybrid evidence recall, same protocol as the no-model study
  extract    LLM fact extraction over every LoCoMo session, one store per conversation
  answer     answer + judge for a per-type sample of test questions, per memory strategy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_memory.llm import CHAT_MODEL, DEFAULT_URL, DiskCache, Ollama
from agent_memory.study.common import Run
from agent_memory.study.models import dry_run, stage_answer, stage_extract, stage_retrieval

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("all", "retrieval", "extract", "answer"), default="all")
    ap.add_argument("--dry-run", action="store_true", help="print jobs and call counts only")
    ap.add_argument("--per-type", type=int, default=40, help="questions per type for answering")
    ap.add_argument("--judge", default=CHAT_MODEL, help=f"judge model (default {CHAT_MODEL})")
    ap.add_argument("--lme-limit", type=int, default=None)
    ap.add_argument("--url", default=DEFAULT_URL)
    args = ap.parse_args()
    run = Run(ROOT / "results", args.lme_limit)
    if args.dry_run:
        print("\n".join(dry_run(run, args.per_type)))
        return
    client = Ollama(args.url, DiskCache(ROOT / "data" / "model_cache.sqlite"))
    stores = ROOT / "data" / "model_stores"
    if args.stage in ("all", "retrieval"):
        stage_retrieval(run, client)
    if args.stage in ("all", "extract"):
        stage_extract(run, client, stores)
    if args.stage in ("all", "answer"):
        stage_answer(run, client, stores, args.per_type, args.judge)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:  # benchmark files not downloaded
        sys.exit(str(e))
