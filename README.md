# agent-memory

![python](https://img.shields.io/badge/python-3.11%2B-blue)
![runtime deps](https://img.shields.io/badge/runtime%20dependencies-none-success)
![data](https://img.shields.io/badge/data-LoCoMo%20%C2%B7%20LongMemEval__S-orange)
[![license](https://img.shields.io/badge/license-MIT-green)](LICENSE)

We built agent-memory, a cross-session memory layer for agents, and used it to measure what
"remember everything and search" actually finds. The library keeps every conversation turn in
SQLite with its timestamp, files facts as versioned `(subject, attribute, value)` rows that
supersede without deleting, answers "what was true on date X", retrieves with BM25 written from
scratch and builds a context that fits a token budget. The study runs nine memory strategies
through the same context builder on LoCoMo (1,986 questions) and LongMemEval_S (500 questions,
each with a ~112k-token history) and checks whether the gold evidence made it into a
2,048-token context. It is for people building agents with long-term memory who want to know,
question type by question type, what a memory strategy loses.

- In a product: [StudyLoop](https://github.com/hammasbuilds/studyloop)'s Memory page uses this
  repo's BM25 and its keep-every-raw-event design (screenshots there).
- Inspired by [claude-mem](https://github.com/thedotmack/claude-mem),
  [mem0](https://github.com/mem0ai/mem0) and [supermemory](https://github.com/supermemoryai/supermemory);
  no code from them is used.

## What it does

- **Store**: sessions, raw turns and versioned facts with provenance in one SQLite file; a
  newer fact supersedes the old one, which is kept and queryable with `--as-of`.
- **Retrieve**: Okapi BM25 over turns and over sessions, fused at session level, with time
  windows parsed from the question ("last month", "three weeks ago", "in May 2023") and an
  optional recency decay.
- **Build a context**: fills a token budget exactly, one date header per session, so "I went
  yesterday" can be resolved to a date.
- **Forget and compact**: drop phatic turns, a role, or turns older than a date; truncate long turns.
- **CLI** (`agent-memory`): `ingest` (LoCoMo, LongMemEval or JSONL), `search`, `context`,
  `remember`, `history`, `facts`.
- **Study**: positional, search and window-plus-search strategies on both benchmarks, scored
  by turn-level and session-level evidence recall, by question type, with cluster
  bootstrap / jackknife-t intervals in plain Python.
- **Model arm** (built, tested against a fake LLM, not yet run): LLM fact extraction,
  `nomic-embed-text` hybrid retrieval, and answer accuracy with `qwen2.5:14b-instruct`.

## Sample input / output

`uv run python demo.py`, part 1 needs no data:

```
PART 1 - a store with known answers
  ingested 8 turns from 4 sessions
  [ok  ] search 'what is my dog called?': 'jan:1'
  context 'where did I say I work, back in March?' (60-token budget):
      [session mar - 2024-03-05 18:00 (Tuesday)]
      user: Work at Riverside Hospital is exhausting, twelve-hour nursing shifts.
  [ok  ] the March turn is first in context: 'mar:0'
  [ok  ] tokens used <= budget: True
  [ok  ] current employer: "St Mary's clinic"
  [ok  ] employer as of 2024-04-01: 'Riverside Hospital'
  [ok  ] old value kept, marked superseded: 2
  [ok  ] forget phatic turns ('Hi!', 'ok thanks'): 2
  stats: {'sessions': 4, 'turns': 6, 'forgotten_turns': 2, 'facts_current': 1, 'facts_superseded': 1}
```

Part 2 (after `scripts/fetch_data.py`) runs real benchmark questions through four strategies.
A LoCoMo temporal question: found by search, missed by the recent-turns window, and answerable
only through the session date, since the turn says "yesterday":

```
  [locomo / temporal] When did Caroline go to the LGBTQ support group?
  gold answer: 7 May 2023
  gold evidence conv-26:D1:3 (08 May 2023) Caroline: I went to a LGBTQ support group yesterday and it was so powerful.
    sliding_window   2047 tokens, evidence turns 0/1, answer sessions 0/1
    bm25_turns       2045 tokens, evidence turns 1/1, answer sessions 1/1
    window_bm25      2044 tokens, evidence turns 1/1, answer sessions 1/1
    store            2041 tokens, evidence turns 1/1, answer sessions 1/1
```

A LoCoMo multi-hop question: four activities in four turns across three sessions, and the
question's words ("activities", "partake") appear in none of them, so every strategy finds at
most one:

```
  [locomo / multi-hop] What activities does Melanie partake in?
  gold answer: pottery, camping, painting, swimming
    sliding_window   2047 tokens, evidence turns 0/4, answer sessions 0/3
    bm25_turns       2038 tokens, evidence turns 1/4, answer sessions 3/3
    window_bm25      2029 tokens, evidence turns 1/4, answer sessions 3/3
    store            2034 tokens, evidence turns 1/4, answer sessions 3/3
```

The CLI on a real LoCoMo conversation, with versioned facts:

```
$ agent-memory --db cli.db ingest --format locomo --pick conv-30 data/raw/locomo10.json
ingested 369 new turns from 19 sessions (0 skipped: already stored)

$ agent-memory --db cli.db search "opening a dance studio" -k 2 --now 2023-07-01
2023-06-19 10:04  conv-30:D15:3  Jon: Thanks, Gina. Still working on opening a dance studio.
2023-06-19 10:04  conv-30:D15:4  Gina: When are you opening the studio?

$ agent-memory --db f.db remember jon job "banker" --at 2022-06-01
fact #1: jon / job: banker (since 2022-06-01)
$ agent-memory --db f.db remember jon job "none - lost the banker job, starting a business" --at 2023-01-19 --source conv-30:D1:2
fact #2: jon / job: none - lost the banker job, starting a business (since 2023-01-19)
$ agent-memory --db f.db history jon job
#1  2022-06-01  banker  [superseded by #2]
#2  2023-01-19  none - lost the banker job, starting a business  [current]  from conv-30:D1:2
$ agent-memory --db f.db facts --as-of 2022-12-01
#1  jon / job: banker (since 2022-06-01)
$ agent-memory --db f.db context "what is Jon's job" --budget 100
[known facts - newest value of each]
jon / job: none - lost the banker job, starting a business (since 2023-01-19)

-- 32 of 100 tokens, 0 turns, 1 facts
```

(The banker start date is typed in for the example; the conversation only says he lost the
job the day before 20 January 2023.)

## Results

Test split, 2,048-token context, 95% cluster intervals; turn-level evidence recall unless
stated. LoCoMo test = 1,401 questions from 7 of its 10 conversations; LongMemEval test = 406 of
500. Every number is in [`results/`](results/). No model was needed for any of them; answer
accuracy is not measured yet.

| | Question | Answer | Source |
|---|---|---|---|
| 1 | Does searching everything beat a recent-turns window? | Yes: BM25 over turns minus a sliding window is **+0.678** [0.641, 0.715] on LoCoMo and **+0.799** [0.763, 0.835] on LongMemEval. The window is within ±0.02 of random turns. | `retrieval.json` |
| 2 | Is that because the evidence is old? | In every position bucket older than the newest tenth the window loses by 0.65-0.81. In the newest tenth on LoCoMo it matches search (0.868 vs 0.729, paired interval includes zero). Only 9.5% (LoCoMo) and 6.7% (LongMemEval) of questions have evidence there. | `retrieval.json`, `diagnostics.json` |
| 3 | Recent turns plus search? | No share of the budget beats search alone. A 5% share (chosen on dev) ties it; on LoCoMo 50% loses 0.045 and 75% loses 0.104. | `window_share.json` |
| 4 | Which question types does retrieval lose? | Multi-hop and open-domain. Store recall 0.905 single-hop / 0.597 multi-hop / 0.446 open-domain on LoCoMo. All evidence for a multi-hop question arrives for only **34%** (LoCoMo) and **50%** (LongMemEval) of them. 58% of open-domain evidence turns share no content word with the question. | `retrieval.json`, `diagnostics.json` |
| 5 | Are temporal questions lost? | Not at retrieval (0.860 and 0.815), but 71% of LoCoMo's temporal answers are dates the evidence never states; the session date header is what makes them answerable. | `diagnostics.json` |
| 6 | Does search return the stale value after a knowledge update? | At 2,048 tokens the newest value's session is in context for 92% of update questions, only the stale one for 4.9%. At 512 tokens stale-only is 16.4%. | `recency_sensitivity.json` |
| 7 | What does the store's retriever add over plain BM25? | +0.064 [0.050, 0.077] on LoCoMo, almost all from session-level fusion; +0.009 on LongMemEval, not distinguishable from zero. Recency decay adds nothing. | `retrieval.json` |
| 8 | Does recall@k flatter coarse units? | BM25 over whole sessions reaches recall@10 0.956 on LongMemEval at a mean 29,401 tokens; at an equal 2,048-token budget it is 0.163 worse than BM25 over turns. | `retrieval.json` |
| 9 | What can be forgotten? | Truncating every turn to 128 tokens keeps 37% of LongMemEval's tokens, drops no evidence turn and raises recall 0.819 to 0.886. Dropping assistant turns keeps 13% of tokens; multi-hop rises 0.680 to 0.884 while assistant-side questions fall 0.933 to 0.089. | `forgetting.json` |

- Session-level recall (any turn of each gold session in context) is reported beside every
  turn-level cell as a check; random turns already score 0.777 on it on LoCoMo.
- Findings 1, 3, 7 and 8 hold on both LongMemEval data-quality subsets (complete turn key,
  no future-dated sessions), in `retrieval.json` under `subsets`.
- The store's settings were chosen on the dev split only (72 configurations, `dev_sweep.json`).
- Token budgets use a built-in counter that runs 4% (LoCoMo) to 8% (LongMemEval) over the
  Qwen2.5 tokeniser (`token_calibration.json`), so differences of ±0.02 between strategies
  are not meaningful.

## Tech stack

- Python 3.11+, no runtime dependencies: `sqlite3`, `json`, `math`, `urllib` from the standard library
- BM25, the light stemmer, the temporal parser, the token counter and the interval statistics
  are written in plain Python
- Model arm: a local Ollama server (`qwen2.5:14b-instruct`, `nomic-embed-text`), reached with
  `urllib` with proxies bypassed and a digest-keyed SQLite cache
- Optional `calibration` group: `tiktoken`, `tokenizers` (only for `scripts/calibrate_tokens.py`)
- pytest, ruff, hatchling, uv; GitHub Actions on Ubuntu and Windows

## Run it

```bash
git clone https://github.com/hammasbuilds/agent-memory && cd agent-memory
uv sync
uv run pytest -q                                   # 166 tests, no data, no network, no model
uv run python demo.py                              # part 1 needs nothing
uv run python scripts/fetch_data.py                # LoCoMo + LongMemEval oracle & S (~295 MB, sha256-checked)
uv run python demo.py                              # now with part 2
uv run python scripts/run_retrieval_study.py       # every results/*.json (~35 min, 1 core, < 2 GB RAM)
bash scripts/run_models.sh --dry-run               # the model arm: jobs and call counts
bash scripts/run_models.sh                         # the model arm (Ollama, GPU with ~11 GB free)
```

As a library:

```python
from datetime import datetime, timedelta, timezone

from agent_memory.store import MemoryStore

karachi = timezone(timedelta(hours=5))  # any aware datetime; stored as UTC
said = datetime(2024, 1, 10, 9, 30, tzinfo=karachi)

with MemoryStore("memory.db") as m:
    m.add_turn("s1", "user", "I just adopted a beagle called Biscuit.", said)
    m.remember("user", "dog", "Biscuit (beagle)", said, source_turn="s1:0")
    ctx = m.context("what's my dog called?", budget=512)  # now defaults to UTC
    print(ctx.render())
    print(ctx.tokens, "tokens")
```

```
[known facts - newest value of each]
user / dog: Biscuit (beagle) (since 2024-01-10)
[session s1 - 2024-01-10 04:30 (Wednesday)]
user: I just adopted a beagle called Biscuit.
55 tokens
```

- **Time is UTC.** An aware datetime is converted to UTC on the way in; a naive one is taken
  to be UTC already, so pass aware datetimes unless your clock is UTC. The header above shows
  04:30 because 09:30 at +05:00 is 04:30 UTC. `agent_memory.jsonl.read_jsonl` applies the same
  rule to the `time` field of JSONL input.
- Facts come from `remember()`, the CLI, or the model arm's extractor; no retrieval number
  above uses facts.
- `results/token_calibration.json` comes from `scripts/calibrate_tokens.py`, which needs the
  `calibration` group and the Qwen2.5-Coder tokenizer file:
  `PYTHONPATH=src python scripts/calibrate_tokens.py --qwen path/to/qwen2.5-coder-tokenizer.json.gz`.

Code layout: `src/agent_memory/` holds `store.py`, `bm25.py`, `text.py`, `temporal.py`,
`retrieval.py`, `context.py`, `forget.py`, `datasets.py`, `evaluate.py`, `stats.py`,
`report.py`, `analysis.py`, `cli.py`, the model-arm modules (`llm.py`, `embedding.py`,
`extract.py`, `answer.py`) and `study/`.

## Licence

MIT for the code, see [LICENSE](LICENSE). Neither dataset is redistributed;
`scripts/fetch_data.py` downloads both and checks their SHA-256.

- **LoCoMo**: `locomo10.json` from [snap-research/locomo](https://github.com/snap-research/locomo),
  **CC BY-NC 4.0** (non-commercial). The conversation turns quoted in this README and printed by
  the demo are from LoCoMo, under that licence. Maharana, Lee, Tulyakov, Bansal, Barbieri and
  Fang, *Evaluating Very Long-Term Conversational Memory of LLM Agents*, ACL 2024.
- **LongMemEval**: `longmemeval_s_cleaned.json` and `longmemeval_oracle.json` from
  [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned),
  **MIT**. Wu, Wang, Yu, Zhang, Chang and Yu, *LongMemEval: Benchmarking Chat Assistants on
  Long-Term Interactive Memory*, ICLR 2025.
- How the loaders treat the data (sessions put in date order, future-dated sessions kept and
  flagged, duplicate session ids, incomplete answer keys, malformed LoCoMo evidence ids) is
  counted in `results/diagnostics.json`.
