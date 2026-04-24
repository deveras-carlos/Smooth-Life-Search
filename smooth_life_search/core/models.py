"""Shared models and protocols for SmoothLife Search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import numpy as np

Bounds2D = np.ndarray
Objective = Callable[[np.ndarray], float]


class SearchRunner(Protocol):
    """Object that can produce one seeded optimization run."""

    def reset(self, seed: int | None = None) -> None: ...

    def run(self) -> "SearchRun": ...


@dataclass(slots=True)
class Basin:
    """Promising connected component in the current activity field."""

    mask: np.ndarray
    centroid_grid: np.ndarray
    centroid_world: np.ndarray
    bbox_grid: tuple[int, int, int, int]
    bbox_world: np.ndarray
    support_mass: float
    objective_score: float
    stability_score: float
    alive_density: float = 0.0
    basin_best_point: np.ndarray | None = None
    basin_best_value: float | None = None
    combined_score: float = 0.0
    core_mask: np.ndarray | None = None
    core_bbox_grid: tuple[int, int, int, int] | None = None
    core_bbox_world: np.ndarray | None = None
    evaluated_count: int = 0
    best_objective_score: float = 0.0
    mean_objective_score: float = 0.0
    unexplored_fraction: float = 1.0
    incumbent_in_envelope: bool = False
    better_than_incumbent: bool = False

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask))

    @property
    def mass(self) -> float:
        return float(self.support_mass)


@dataclass(slots=True)
class SmoothLifeSnapshot:
    """State captured from the simulator for analysis or animation."""

    step_index: int
    bounds: np.ndarray
    field: np.ndarray
    inner_fill: np.ndarray
    outer_fill: np.ndarray
    objective_field: np.ndarray
    transition_field: np.ndarray
    evaluated_mask: np.ndarray
    best_point: np.ndarray
    best_value: float
    local_best_point: np.ndarray
    local_best_value: float
    box_best_point: np.ndarray
    box_best_value: float
    selected_basin_bbox: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ZoomEvent:
    """One adaptive-grid zoom operation."""

    zoom_index: int
    old_bounds: np.ndarray
    new_bounds: np.ndarray
    selected_basin_score: float
    selected_basin_bbox: np.ndarray
    evaluation_count: int
    steps_per_zoom: int
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SearchRun:
    """Final run artifact for SmoothLife Search and AGSLS."""

    best_point: np.ndarray
    best_value: float
    evaluations: int
    bounds: np.ndarray
    snapshots: list[SmoothLifeSnapshot]
    zoom_events: list[ZoomEvent]
    metadata: dict[str, Any] = field(default_factory=dict)


SearchResult = SearchRun
