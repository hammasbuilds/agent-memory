# STATUS - agent-memory

**Status: READY-FOR-REVIEW** (retrieval study complete on its own; model arm built, tested
with fakes, queued - not needed for the headline)

The headline question - what evidence each memory strategy gets into a fixed context, by
question type, and whether search beats recency - is a retrieval question and is fully
answered without a model. Answer accuracy is the natural next number, and the model arm
that produces it is ready to run, but no claim in the README depends on it.

## Self-score (honest, after the hostile pass below)

| Points | Criterion | Score | Reason |
|---:|---|---:|---|
| 15 | Works from a clean clone | 15 | Fresh `git clone` into a temp dir: `uv sync --offline`, `uv run pytest -q` (99 passed) with `AGENT_MEMORY_DATA` pointed at an empty dir, `uv run python demo.py` (part 1 runs, part 2 says how to fetch the data), `ruff check` and `ruff format --check` clean. Study scripts exit with a one-line message, not a traceback, when data is missing. |
| 20 | Real data, real result | 20 | LoCoMo (1,986 questions) and LongMemEval_S (500 questions, the full 277 MB file, sha256-verified). Every README number is in `results/*.json`; per-question rows in `results/rows/`. No stubs. |
| 15 | Finding quality | 14 | Positional baselines, a random-turn chance floor, four ablations of the store retriever, a dev/test split (tuning on dev only), cluster-bootstrap CIs (LoCoMo by conversation), paired differences, a recency sensitivity sweep, three mechanism diagnostics (evidence position, lexical visibility, answer location). Surprises investigated: window worse than random (explained by evidence position), 1.00 on LongMemEval abstention (n=7, flagged), 103 histories (a bug, fixed), 77% date-shaped answers (a bug, fixed). -1: LoCoMo has only 7 test conversations, so its CIs rest on 7 clusters. |
| 15 | Correctness | 14 | 99 tests assert behaviour and failure modes (hand-computed BM25, budget invariant, supersession chains, late facts, id reuse, tz normalisation, chunk-boundary JSON, clustered CIs, CLI errors, cache resume). Three real bugs found by tests or by suspicious numbers and fixed. -1: evidence recall counts a truncated evidence turn as "found" (the forgetting table reports the literal-answer survival rate next to it for exactly this reason). |
| 10 | Usability | 9 | `agent-memory --help` with examples; every subcommand has help; clear errors for bad policy, bad time, missing file, missing db, empty turn, malformed JSONL line (with line number). -1: no `bench` subcommand - the study is a script. |
| 10 | README | 10 | House skeleton (centred title, thesis, nav, badges, mermaid through-line with the claim, findings table near the top, 6 real Input/Output samples, Quick start, Layout, Requirements, Tests, NOT-do, real problems, Keywords, MIT). British spelling. Inspiration credited in one line. |
| 10 | Code quality | 9 | ruff clean, typed, zero runtime dependencies, one job per module. -1: `scripts/run_retrieval_study.py` is ~420 lines with five stages in one file. |
| 5 | Honesty | 5 | Every number traceable to `results/`; limitations stated (approximate tokens, benchmark-uniform evidence, n=7 cell, dev winner worse on one dataset's dev split, illustrative date in the CLI example). |
| **100** | | **96** | |

## Done

- `MemoryStore` (SQLite): sessions, raw turns with timestamps, versioned facts with
  provenance (`source_turn`), supersession with history kept, late-arriving facts slotted
  in, `facts_as_of`, soft forget + purge, compaction, stats; ids never reused.
- BM25 from scratch; light stemmer; calibrated token counter (4% / 10% over cl100k).
- Temporal: dataset timestamps, relative expressions to date windows, tz normalisation.
- Eight strategies + the store retriever; one context builder for all; session date headers.
- Evaluation: recall / all / newest / stale-only per budget and per k; cluster bootstrap;
  paired comparisons; budget-to-reach; dev sweep (72 configs); recency sensitivity;
  forgetting study (6 policies); diagnostics.
- CLI (`agent-memory`), `demo.py`, resumable parallel downloader.
- Model arm (not run): Ollama client with SQLite call cache, fact extraction, dense and
  RRF-hybrid retrieval, answer + per-type judge, `scripts/run_models.sh` with RAM / GPU /
  model checks and `--dry-run`.

## Queued for the model run

`bash scripts/run_models.sh --dry-run` prints (measured on this machine):

```
  locomo: 1986 questions, 200 sampled for answering x 5 strategies
  longmemeval: 500 questions, 247 sampled for answering x 4 strategies
jobs:
  retrieval   195402 unique turn embeddings + 2486 query embeddings (nomic-embed-text)
  extract        272 generations (qwen2.5:14b-instruct, one per LoCoMo session)
  answer        3976 generations (1988 answers + 1988 judgements)
total: 4248 generations, 197888 embeddings (cached ones are free on a rerun)
```

Estimate: ~4-6 h of `qwen2.5:14b-instruct` at 3-5 s a call, plus ~1 h of embeddings.
Outputs: `results/model_retrieval.json` (dense and hybrid recall, same protocol),
`results/model_extraction.json`, `results/model_answers.json` (accuracy per strategy and
type, and accuracy split by whether the evidence was in context).

## Known weaknesses remaining

- LoCoMo's test split has 7 conversations; its intervals are wide by construction.
- Evidence recall is not answer accuracy; a found turn may still be misread.
- The dev-chosen store config is worse than dropping session fusion on LongMemEval's dev
  split (0.875 vs 0.889) though better on its test split (+0.022); chosen on the macro.
- Lexical false positives exist (a photo caption containing "june" matched a June
  question); dense retrieval is the queued answer to lexically invisible evidence.
- LongMemEval abstention: 21 of 30 questions have no evidence; the measurable cell is n=7.
- Token counts are an approximation (calibration in `results/token_calibration.json`).

## Reproduce

```bash
unset VIRTUAL_ENV
uv sync
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run python scripts/fetch_data.py                       # data/raw/*, sha256-verified
uv run python scripts/run_retrieval_study.py              # all results/*.json except token_calibration
#   or stage by stage: --stage diagnostics | recency | dev | main | forgetting
PYTHONPATH=src /path/to/python-with-tiktoken scripts/calibrate_tokens.py   # token_calibration.json
uv run python demo.py
bash scripts/run_models.sh --dry-run                      # model arm job list
bash scripts/run_models.sh                                # model arm (GPU; not run here)
```
