"""The retrieval study (no model): evidence recall and token cost of every memory
strategy on LoCoMo and LongMemEval_S, by question type, with cluster-level (jackknife-t) CIs.

Stages (each writes results/*.json; the code is in agent_memory/study/):
  diagnostics  history sizes, evidence position, lexical visibility, answer location,
               data-quality flags                        -> diagnostics.json
  dev          choose the store retriever's knobs, the recency half-life and the window
               share on the dev split only               -> dev_sweep.json
  recency      every half-life of the dev grid on the test split, the dev choice marked
                                                         -> recency_sensitivity.json
  window-share every recent-window share on the test split, the dev choice marked
                                                         -> window_share.json
  main         every strategy x budget x top-k, both splits, comparisons and controls
                                                         -> retrieval.json, rows/*.jsonl.gz
  forgetting   what each forgetting / compaction policy costs in evidence
                                                         -> forgetting.json

    uv run python scripts/run_retrieval_study.py            # everything (~45 min, 1 core)
    uv run python scripts/run_retrieval_study.py --stage main --lme-limit 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agent_memory.study.common import Run
from agent_memory.study.diagnostics import diagnostics_stage
from agent_memory.study.forgetting import forgetting_stage
from agent_memory.study.recall import main_stage
from agent_memory.study.sweep import dev_sweep, recency_sensitivity, window_share_sensitivity

STAGES = {
    "diagnostics": diagnostics_stage,
    "dev": dev_sweep,
    "recency": recency_sensitivity,
    "window-share": window_share_sensitivity,
    "main": main_stage,
    "forgetting": forgetting_stage,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=("all", *STAGES), default="all")
    ap.add_argument("--lme-limit", type=int, default=None, help="first N LongMemEval questions")
    ap.add_argument("--results", type=Path, default=Path(__file__).resolve().parents[1] / "results")
    args = ap.parse_args()
    run = Run(args.results, args.lme_limit)
    for name, stage in STAGES.items():
        if args.stage in ("all", name):
            stage(run)


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:  # benchmark files not downloaded, or dev not run yet
        sys.exit(str(e))
