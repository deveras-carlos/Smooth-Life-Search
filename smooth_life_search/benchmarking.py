"""Helpers for repeated benchmark runs and aggregate summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

import numpy as np

from .results import SearchResult


@dataclass(slots=True)
class BenchmarkSummary:
    seeds: np.ndarray
    best_values: np.ndarray
    median_best_value: float
    iqr_best_value: tuple[float, float]
    success_rate: float


class SearchRunner(Protocol):
    """Minimal runner protocol for seeded repeated trials."""

    def reset(self, seed: int | None = None) -> None: ...

    def run(self) -> SearchResult: ...


def run_seeded_trials(
    search_factory: Callable[[int], SearchRunner],
    seeds: Iterable[int],
) -> list[SearchResult]:
    """Run a sequence of seeded trials using fresh search objects."""

    results: list[SearchResult] = []
    for seed in seeds:
        resolved_seed = int(seed)
        search = search_factory(resolved_seed)
        search.reset(seed=resolved_seed)
        result = search.run()
        result.metadata["seed"] = resolved_seed
        results.append(result)
    return results


def _success_mask(best_values: np.ndarray, success_threshold: float, maximize: bool) -> np.ndarray:
    if maximize:
        return np.asarray(best_values, dtype=float) >= float(success_threshold)
    return np.asarray(best_values, dtype=float) <= float(success_threshold)


def summarize_results(results: list[SearchResult], success_threshold: float, maximize: bool = False) -> BenchmarkSummary:
    """Aggregate the per-seed results with median, IQR, and success rate."""

    if not results:
        raise ValueError("results must not be empty")
    best_values = np.asarray([result.best_value for result in results], dtype=float)
    seeds = np.asarray([int(result.metadata.get("seed", -1)) for result in results], dtype=int)
    q1, q3 = np.quantile(best_values, [0.25, 0.75])
    success_rate = float(np.mean(_success_mask(best_values, success_threshold, maximize=maximize)))
    return BenchmarkSummary(
        seeds=seeds,
        best_values=best_values,
        median_best_value=float(np.median(best_values)),
        iqr_best_value=(float(q1), float(q3)),
        success_rate=success_rate,
    )
