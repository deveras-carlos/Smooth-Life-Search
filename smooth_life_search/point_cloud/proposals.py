"""Candidate proposal and source-accounting structures."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class SourceStats:
    """Evaluation record for one generated candidate source family."""

    attempts: int = 0
    evaluations: int = 0
    improvements: int = 0
    improvement_sum: float = 0.0
    last_improvement_batch: int = 0


@dataclass(slots=True)
class CandidateProposal:
    """One generated proposal with optimizer metadata."""

    point: np.ndarray
    source: str
    family: str
    parent_key: tuple[str, ...] | None = None
    region_id: int | None = None
    axes: tuple[int, ...] | None = None
    predicted_score: float | None = None

    def __iter__(self):
        yield self.point
        yield self.source

    def __getitem__(self, index: int):
        if index == 0:
            return self.point
        if index == 1:
            return self.source
        raise IndexError(index)


@dataclass(slots=True)
class GradientResult:
    """Finite-difference gradient plus recentering diagnostics."""

    gradient: np.ndarray | None
    spent: int
    recentered: bool
    point: np.ndarray
    value: float
    improvement_count: int = 0
