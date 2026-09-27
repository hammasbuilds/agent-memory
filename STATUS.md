# STATUS - agent-memory

**Status: READY-FOR-REVIEW** (second round; retrieval study complete on its own; model
arm built, tested with fakes, queued - not needed for the headline)

The first independent review scored this 80/100 (my self-score had been 96). Every one of
its 14 points and the "split the study script" note is addressed below, each with a
regression test where code changed, and every result file was regenerated.

## What changed after review

| # | Review point | Fix | Test / evidence |
|---:|---|---|---|
| 1 | `_relevant_facts` dropped every query term shared by all facts; one fact was never found | Relative BM25 cutoff (keep facts scoring ≥ 50% of the best) instead of excluding terms; README library snippet now prints its fact | `test_a_single_fact_is_found`, `test_facts_sharing_the_query_word_are_ranked_not_excluded` |
| 2 | JSONL turn ids collided across files, turns silently dropped | Content-hash ids; `ingest` appends new turns via the session counter; message says "skipped: already stored" | `test_a_second_file_continuing_a_session_appends`, `test_ingest_appends_after_existing_turns_even_if_input_seq_restarts` |
| 3 | `HTTP_PROXY` routed 127.0.0.1 through a proxy | `build_opener(ProxyHandler({}))`; `curl --noproxy '*'` | `test_proxy_environment_is_ignored_for_local_ollama`; full suite passes with `HTTP_PROXY` set |
| 4 | "Window worse than random" was a packing artefact | Positional strategies cut the boundary turn to fill the budget; claim rewritten (now −0.016 [−0.032, 0.000], "no better than random") | `test_positional_fills_the_budget`, `test_positional_cuts_the_misfit_turn_ranked_skips_it` |
| 5 | "42% of open-domain evidence turns" not in any results file | `diagnostics.json` now has `evidence_turns_sharing_no_word` (58% of 208 turns) and `questions_with_no_visible_evidence` (40% of 92 questions); README quotes both with units | fixture test of the per-turn metric |
| 6 | README I/O samples stale | All six regenerated from a fresh run and pasted unedited | - |
| 7 | Turn-level key only; 62 LongMemEval questions have a partial key | Session-level recall / all-sessions vs `answer_session_ids` for every cell; `partial_key` flag; complete-key subset reported; `newest` uses the latest answer session | `test_session_level_key_covers_what_the_turn_key_misses` |
| 8 | "Uniform evidence explains recency's loss" untested; no window + search baseline | Recall by evidence-position bucket per strategy (window 0.868 vs BM25 0.729 in the newest tenth on LoCoMo, ≤0.010 elsewhere); `window_bm25` baseline (loses 0.036 / 0.052 to BM25) | `test_window_bm25_splits_the_budget`, `by_position` in the end-to-end study test |
| 9 | Future-dated sessions and duplicate session ids not handled or disclosed | Kept and ordered by date, flagged (`future_sessions`: 76 questions; `duplicate_sessions`: 13, all identical copies, one kept; non-identical duplicates raise); headline comparisons repeated without future-session questions | `test_longmemeval_data_quality_is_flagged_not_hidden` |
| 10 | Wording: 70% not 80%, "+0.004" was LoCoMo only, 5 MB/16 vs 2 MB/32 | Split sizes stated exactly (1,401/1,986 and 406/500); both datasets' window-boost numbers given; downloader defaults now 2 MB / 32 workers, matching what was run | - |
| 11 | Unparsable judge replies left the denominator; no retries; no model digest in cache keys; self-judging | Unparsed = wrong, counted in `unparsed`; retries with backoff on 5xx and dropped connections (4xx raises); digest from `/api/tags` in every cache key; `--judge` option and self-judging noted as a limitation | `test_unparsed_verdicts_count_as_wrong_and_are_reported`, `test_server_errors_are_retried_client_errors_are_not`, `test_a_re_pulled_model_misses_the_cache` |
| 12 | No Data section | Licences (LoCoMo CC BY-NC 4.0, LongMemEval MIT), citations, how each loader treats the data | - |
| 13 | Token counter error may differ by strategy | Measured per strategy against Qwen2.5's tokeniser and cl100k_base (`token_calibration.json`): 0.95-1.03 on LoCoMo, 1.02-1.07 on LongMemEval; disclosed with numbers, too small to move findings 1-3 | - |
| 14 | `X` and `X_abs` could straddle dev/test | Split and bootstrap clusters grouped by `qid.removesuffix("_abs")` (their haystacks do not overlap - Jaccard 0.003 - but the questions are near-paraphrases) | `test_abstention_twins_share_a_split_and_a_cluster` |
| - | 441-line study script | Stages moved to `agent_memory/study/` (common, sweep, recall, forgetting, diagnostics, models); the scripts are thin CLIs | `tests/test_study.py` runs every stage and the model arm end to end on fixtures |

## Self-score (honest, after a second hostile pass)

| Points | Criterion | Score | Reason |
|---:|---|---:|---|
| 15 | Works from a clean clone | 15 | Fresh clone: `uv sync --offline`, `uv run pytest -q` (117 passed) with `AGENT_MEMORY_DATA` pointed at an empty dir and `HTTP_PROXY` set, `demo.py` (part 1 runs, part 2 says how to fetch the data), ruff check and format clean. Scripts exit with a one-line message when data is missing. |
| 20 | Real data, real result | 20 | LoCoMo (1,986 questions) and the full LongMemEval_S (500 questions, sha256-verified). Every README number is in `results/*.json`; per-question rows in `results/rows/`. |
| 15 | Finding quality | 13 | Now tested rather than asserted: the recency explanation has a per-position control and the realistic window + search baseline; both answer keys; data-quality subsets; grouped split; dev-only tuning; cluster CIs. -2: LoCoMo's test split is 7 conversations, and the session-level key is lenient enough (random turns score 0.777 on LoCoMo) that it can only check the turn key, not replace it. |
| 15 | Correctness | 13 | 117 tests including end-to-end stage tests on fixtures and a deliberately messy LongMemEval question. -2: a cut (truncated) boundary turn counts as present for positional strategies, which slightly favours them (by at most one turn per context); the review found four real bugs I had missed, so I discount my own confidence. |
| 10 | Usability | 9 | `--help` with examples on every command; clear errors; JSONL re-ingest idempotent and continuation-safe. -1: no `bench` subcommand. |
| 10 | README | 10 | House skeleton, 6 fresh Input/Output samples, Data section with licences and citations, NOT-do, real problems (including the ones review found, credited). |
| 10 | Code quality | 9 | ruff clean, typed, zero runtime deps, study split into modules. -1: `study/recall.py` builds one large results dict; some summaries are computed twice at different budgets. |
| 5 | Honesty | 5 | Every number traceable; the earlier wrong claims (window worse than random, the 42%, the 80% split) are corrected in place and listed under Problems. |
| **100** | | **94** | |

## Queued for the model run

`bash scripts/run_models.sh --dry-run` prints (measured on this machine):

```
  locomo: 1986 questions, 200 sampled for answering x 5 strategies
  longmemeval: 500 questions, 251 sampled for answering x 4 strategies
jobs:
  retrieval   195402 unique turn embeddings + 2486 query embeddings (nomic-embed-text)
  extract        272 generations (qwen2.5:14b-instruct, one per LoCoMo session)
  answer        4008 generations (2004 answers + 2004 judgements)
total: 4280 generations, 197888 embeddings (cached ones are free on a rerun)
```

Estimate: ~4-6 h of `qwen2.5:14b-instruct` at 3-5 s a call, plus ~1 h of embeddings.
Outputs: `results/model_retrieval.json`, `results/model_extraction.json`,
`results/model_answers.json` (accuracy per strategy and type, unparsed judge replies
counted as wrong and reported, accuracy split by whether evidence was in context). Pass
`--judge qwen2.5-coder:14b` to avoid self-judging.

## Known weaknesses remaining

- LoCoMo's test split has 7 conversations; its intervals are wide by construction.
- Evidence recall is not answer accuracy.
- Session-level recall is lenient; it is a check on the turn key, not a headline metric.
- The dev-chosen store config is worse than dropping session fusion on LongMemEval's dev
  split (0.873 vs 0.888), better on its test split (+0.022); chosen on the macro average.
- Token counts are approximate and differ by strategy by up to ~8% (disclosed with numbers).
- LongMemEval abstention: n=8 at turn level on the test split.
- Filler sessions recur across LongMemEval questions (mean reuse 1.24), so a few filler
  sessions appear in both splits; evidence sessions do not.

## Reproduce

```bash
unset VIRTUAL_ENV
uv sync
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run python scripts/fetch_data.py                       # data/raw/*, sha256-verified
uv run python scripts/run_retrieval_study.py              # every results/*.json except token_calibration
#   or stage by stage: --stage diagnostics | recency | dev | main | forgetting
PYTHONPATH=src /d/github/rag-forge/.venv/Scripts/python.exe scripts/calibrate_tokens.py \
    --qwen /d/github/harness-ablation/data/qwen2.5-coder-tokenizer.json.gz   # token_calibration.json
uv run python demo.py
bash scripts/run_models.sh --dry-run                      # model arm job list
bash scripts/run_models.sh                                # model arm (GPU; not run here)
```
