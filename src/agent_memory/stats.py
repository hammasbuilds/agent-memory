"""Bootstrap confidence intervals, in plain Python.

LoCoMo's ~2,000 questions come from only ten conversations, so questions are not
independent: resampling questions would give intervals that are far too narrow. Every
function here therefore resamples *clusters* (a LoCoMo conversation; a LongMemEval
question, which has its own private history) and recomputes the pooled mean.
"""

from __future__ import annotations

import random
from collections.abc import Hashable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Estimate:
    mean: float
    lo: float
    hi: float
    n: int  # questions
    clusters: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "mean": round(self.mean, 4),
            "ci95": [round(self.lo, 4), round(self.hi, 4)],
            "n": self.n,
            "clusters": self.clusters,
        }


def _cluster_sums(values: Sequence[float], clusters: Sequence[Hashable]) -> list[tuple[float, int]]:
    if len(values) != len(clusters):
        raise ValueError("values and clusters must be the same length")
    acc: dict[Hashable, list[float]] = {}
    for v, c in zip(values, clusters, strict=True):
        a = acc.setdefault(c, [0.0, 0])
        a[0] += v
        a[1] += 1
    return [(s, int(n)) for s, n in acc.values()]


def _percentiles(samples: list[float]) -> tuple[float, float]:
    samples.sort()
    b = len(samples)
    return samples[int(0.025 * (b - 1))], samples[int(0.975 * (b - 1))]


def bootstrap_mean(
    values: Sequence[float],
    clusters: Sequence[Hashable] | None = None,
    b: int = 2000,
    seed: int = 0,
) -> Estimate:
    """Mean with a 95% percentile cluster-bootstrap interval."""
    if not values:
        raise ValueError("no values to estimate from")
    sums = _cluster_sums(values, clusters if clusters is not None else range(len(values)))
    mean = sum(values) / len(values)
    if len(sums) < 2:
        return Estimate(mean, mean, mean, len(values), len(sums))
    rng = random.Random(seed)
    k = len(sums)
    out = []
    for _ in range(b):
        tot = cnt = 0.0
        for s, n in rng.choices(sums, k=k):
            tot += s
            cnt += n
        out.append(tot / cnt)
    lo, hi = _percentiles(out)
    return Estimate(mean, lo, hi, len(values), k)


def paired_difference(
    a: Sequence[float],
    b_: Sequence[float],
    clusters: Sequence[Hashable] | None = None,
    b: int = 2000,
    seed: int = 0,
) -> Estimate:
    """Mean of a - b over the same questions, with a cluster-bootstrap interval."""
    if len(a) != len(b_):
        raise ValueError("paired samples must be the same length")
    return bootstrap_mean([x - y for x, y in zip(a, b_, strict=True)], clusters, b, seed)
