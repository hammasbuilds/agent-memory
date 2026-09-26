"""Check `count_tokens` against a real BPE tokeniser on the benchmark text.

tiktoken is deliberately not a dependency of this project; run this with any Python
that has it installed (the numbers in results/ came from the second form):

    uv run --with tiktoken python scripts/calibrate_tokens.py
    PYTHONPATH=src /path/to/python-with-tiktoken scripts/calibrate_tokens.py

Writes results/token_calibration.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import tiktoken

from agent_memory.context import line
from agent_memory.datasets import data_dir, iter_longmemeval, load_locomo
from agent_memory.text import count_tokens

ROOT = Path(__file__).resolve().parents[1]


def calibrate(name: str, texts: list[str], enc: tiktoken.Encoding) -> dict:
    est = [count_tokens(t) for t in texts]
    real = [len(enc.encode(t)) for t in texts]
    return {
        "dataset": name,
        "turns": len(texts),
        "estimated_total": sum(est),
        "cl100k_total": sum(real),
        "ratio_total": round(sum(est) / sum(real), 4),
        "mean_abs_pct_error_per_turn": round(
            sum(abs(e - r) / r for e, r in zip(est, real, strict=True) if r) / len(texts), 4
        ),
    }


def main() -> None:
    enc = tiktoken.get_encoding("cl100k_base")
    locomo = {t.id: line(t) for q in load_locomo() for s in q.history for t in s.turns}
    lme = {
        t.id: line(t)
        for q in iter_longmemeval(data_dir() / "longmemeval_oracle.json")
        for s in q.history
        for t in s.turns
    }
    out = [
        calibrate("locomo", list(locomo.values()), enc),
        calibrate("longmemeval_oracle", list(lme.values()), enc),
    ]
    (ROOT / "results" / "token_calibration.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
