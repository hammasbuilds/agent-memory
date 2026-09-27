"""Check `count_tokens` against real BPE tokenisers, per dataset and per strategy.

The counter is used for every budget in the study, so what matters is not only its
average error but whether the error differs between strategies: a strategy whose
contexts are rich in markdown or code (LongMemEval's assistant turns) could be charged
more tokens than it really uses and look worse at a fixed budget.

Two reference tokenisers, neither a dependency of this project and neither loading
any model weights:
  cl100k_base   OpenAI's, via `tiktoken`
  qwen2.5-coder the `tokenizer.json` of Qwen2.5-Coder-14B-Instruct (plain or .gz), via
                the `tokenizers` library - the file harness-ablation ships. Qwen2.5-Coder
                uses the Qwen2.5 byte-level BPE vocabulary, so ordinary text should
                tokenise as it would for the answer arm's qwen2.5:14b-instruct; that is
                an assumption, not something this script checks.

Both libraries are in the optional `calibration` dependency group (a plain `uv sync`
does not install them):

    uv run --group calibration python scripts/calibrate_tokens.py \
        --qwen /path/to/qwen2.5-coder-tokenizer.json.gz

Writes results/token_calibration.json.
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
from collections.abc import Callable
from pathlib import Path

import tiktoken
from tokenizers import Tokenizer

from agent_memory.context import line
from agent_memory.datasets import iter_longmemeval, load_locomo
from agent_memory.evaluate import HEADLINE_BUDGET, history_via_store
from agent_memory.study.common import Run
from agent_memory.study.recall import strategies
from agent_memory.text import count_tokens

ROOT = Path(__file__).resolve().parents[1]
Counter = Callable[[str], int]


def load_qwen(path: Path) -> Counter:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        tok = Tokenizer.from_str(fh.read())
    return lambda text: len(tok.encode(text, add_special_tokens=False).ids)


def ratio(texts: list[str], ref: Counter) -> dict:
    est = [count_tokens(t) for t in texts]
    real = [ref(t) for t in texts]
    pairs = [(e, r) for e, r in zip(est, real, strict=True) if r]
    return {
        "texts": len(texts),
        "estimated_total": sum(est),
        "reference_total": sum(real),
        "ratio_total": round(sum(est) / sum(real), 4),
        "mean_abs_pct_error": round(sum(abs(e - r) / r for e, r in pairs) / len(pairs), 4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--qwen", type=Path, required=True, help="Qwen2.5-Coder tokenizer.json(.gz)")
    ap.add_argument("--every", type=int, default=10, help="use every Nth question per dataset")
    args = ap.parse_args()
    enc = tiktoken.get_encoding("cl100k_base")
    refs: dict[str, Counter] = {
        "cl100k_base": lambda t: len(enc.encode(t)),
        "qwen2.5-coder": load_qwen(args.qwen),
    }
    strats = strategies(Run(ROOT / "results").chosen())
    out: dict = {
        "budget": HEADLINE_BUDGET,
        "every_nth_question": args.every,
        "turns": [],
        "contexts": [],
    }
    datasets = {
        "locomo": load_locomo(),
        "longmemeval": list(itertools.islice(iter_longmemeval(), 0, None, args.every)),
    }
    for ds, qs in datasets.items():
        turns = {t.id: line(t) for q in qs for s in q.history for t in s.turns}
        for name, ref in refs.items():
            out["turns"].append(
                {"dataset": ds, "reference": name} | ratio(list(turns.values()), ref)
            )
        rendered: dict[str, list[str]] = {s: [] for s in strats}
        cache_key, h = None, None
        for q in qs[:: args.every if ds == "locomo" else 1]:
            if q.history is not cache_key:
                cache_key, h = q.history, history_via_store(q)
            for s, strat in strats.items():
                rendered[s].append(
                    strat(h, q.question, q.now, q.qid).pack(HEADLINE_BUDGET, h.tokens).render()
                )
        for s, texts in rendered.items():
            for name, ref in refs.items():
                out["contexts"].append(
                    {"dataset": ds, "strategy": s, "reference": name} | ratio(texts, ref)
                )
        print(f"{ds}: {len(turns)} turns, {len(next(iter(rendered.values())))} questions")
    (ROOT / "results" / "token_calibration.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
