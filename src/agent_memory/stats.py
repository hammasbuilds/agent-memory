"""Cluster-level confidence intervals, in plain Python.

LoCoMo's ~2,000 questions come from only ten conversations, so questions are not
independent: resampling questions would give intervals that are far too narrow. Every
interval here therefore treats a *cluster* (a LoCoMo conversation; a LongMemEval
question, which has its own private history) as the unit.

Two intervals are computed for every estimate:

* `ci95` - the one quoted: a leave-one-cluster-out jackknife standard error with a
  Student t critical value on (clusters - 1) degrees of freedom. With few clusters (7
  LoCoMo test conversations: t = 2.447 on 6 df) it is honest about how little 7 units
  can tell you; with hundreds (LongMemEval) it matches the bootstrap.
* `ci95_bootstrap` - the percentile cluster bootstrap, kept for comparison. It is known
  to be too narrow with a handful of clusters (a resample of 7 cannot vary much), which
  is why it is no longer the quoted interval.

The jackknife interval is clipped to the range of the values (a mean of recalls in
[0, 1] cannot have a bound outside it).
"""

from __future__ import annotations

import math
import random
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

# Student t 0.975 quantiles for 1-30 degrees of freedom
_T975 = (
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228,
    2.201, 2.179, 2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086,
    2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042,
)  # fmt: skip
_Z975 = 1.959964


def t975(df: int) -> float:
    """The 0.975 quantile of Student's t: tabulated to 30 df, then the Cornish-Fisher
    expansion (error < 0.001 beyond 30 df)."""
    if df < 1:
        raise ValueError(f"degrees of freedom must be positive, got {df}")
    if df <= len(_T975):
        return _T975[df - 1]
    z = _Z975
    return z + (z**3 + z) / (4 * df) + (5 * z**5 + 16 * z**3 + 3 * z) / (96 * df**2)


@dataclass(frozen=True)
class Estimate:
    mean: float
    lo: float  # jackknife-t interval (quoted)
    hi: float
    n: int  # questions
    clusters: int
    boot_lo: float  # percentile cluster bootstrap (for comparison)
    boot_hi: float

    def as_dict(self) -> dict[str, object]:
        return {
            "mean": round(self.mean, 4),
            "ci95": [round(self.lo, 4), round(self.hi, 4)],
            "ci95_bootstrap": [round(self.boot_lo, 4), round(self.boot_hi, 4)],
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


def _jackknife_se(sums: list[tuple[float, int]]) -> float:
    """Leave-one-cluster-out jackknife standard error of the pooled mean."""
    total = sum(s for s, _ in sums)
    count = sum(n for _, n in sums)
    loo = [(total - s) / (count - n) for s, n in sums]
    k = len(loo)
    centre = sum(loo) / k
    return math.sqrt((k - 1) / k * sum((x - centre) ** 2 for x in loo))


def cluster_mean(
    values: Sequence[float],
    clusters: Sequence[Hashable] | None = None,
    b: int = 2000,
    seed: int = 0,
) -> Estimate:
    """Mean with a 95% jackknife-t interval and a percentile cluster-bootstrap one."""
    if not values:
        raise ValueError("no values to estimate from")
    sums = _cluster_sums(values, clusters if clusters is not None else range(len(values)))
    mean = sum(values) / len(values)
    k = len(sums)
    if k < 2:
        return Estimate(mean, mean, mean, len(values), k, mean, mean)
    half = t975(k - 1) * _jackknife_se(sums)
    lo, hi = max(mean - half, min(values)), min(mean + half, max(values))
    rng = random.Random(seed)
    out = []
    for _ in range(b):
        tot = cnt = 0.0
        for s, n in rng.choices(sums, k=k):
            tot += s
            cnt += n
        out.append(tot / cnt)
    boot_lo, boot_hi = _percentiles(out)
    return Estimate(mean, lo, hi, len(values), k, boot_lo, boot_hi)


def paired_difference(
    a: Sequence[float],
    b_: Sequence[float],
    clusters: Sequence[Hashable] | None = None,
    b: int = 2000,
    seed: int = 0,
) -> Estimate:
    """Mean of a - b over the same questions, with cluster-level intervals."""
    if len(a) != len(b_):
        raise ValueError("paired samples must be the same length")
    return cluster_mean([x - y for x, y in zip(a, b_, strict=True)], clusters, b, seed)
