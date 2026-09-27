"""Shared plumbing for the study stages: where results go, which questions to load,
and the configuration the dev sweep chose."""

from __future__ import annotations

import itertools
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from agent_memory.datasets import Question, iter_longmemeval, load_locomo
from agent_memory.retrieval import RetrieverConfig


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@dataclass(frozen=True)
class Run:
    """Where a study writes, and how much of LongMemEval it reads."""

    results: Path
    lme_limit: int | None = None

    def write(self, name: str, obj: object) -> None:
        self.results.mkdir(parents=True, exist_ok=True)
        (self.results / name).write_text(json.dumps(obj, indent=1) + "\n", "utf-8")
        log(f"wrote results/{name}")

    def datasets(self) -> Iterator[tuple[str, Iterator[Question]]]:
        yield "locomo", iter(load_locomo())
        yield "longmemeval", itertools.islice(iter_longmemeval(), self.lme_limit)

    def chosen(self) -> Chosen:
        """The configuration picked on the dev split (see `study.sweep.dev_sweep`)."""
        path = self.results / "dev_sweep.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing - run the dev stage first")
        sweep = json.loads(path.read_text("utf-8"))
        return Chosen(
            RetrieverConfig(**sweep["store_best"]["config"]),
            float(sweep["recency_best"]["half_life_days"]),
            float(sweep["window_share_best"]["recent_share"]),
        )


@dataclass(frozen=True)
class Chosen:
    store: RetrieverConfig
    half_life_days: float
    recent_share: float
