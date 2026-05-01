"""Typed artifacts for point-cloud SmoothLife search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class PointCloudSample:
    """One unique objective evaluation."""

    point: np.ndarray
    value: float
    source: str
    batch_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": np.asarray(self.point, dtype=float).tolist(),
            "value": float(self.value),
            "source": self.source,
            "batch_index": int(self.batch_index),
        }


@dataclass(slots=True)
class PointCloudRegion:
    """One adaptive proposal/trust region in the archive portfolio."""

    region_id: int
    center: np.ndarray
    radius_fraction: float
    score: float
    best_value: float
    age: int = 0
    successes: int = 0
    failures: int = 0
    stall_count: int = 0
    cooldown_until: int = 0
    geometry_basis: np.ndarray = field(default_factory=lambda: np.eye(2, dtype=float))
    geometry_axis_scales: np.ndarray = field(default_factory=lambda: np.ones(2, dtype=float))
    geometry_eigenvalues: np.ndarray = field(default_factory=lambda: np.ones(2, dtype=float))
    geometry_anisotropy: float = 1.0
    geometry_sample_count: int = 0
    geometry_reason: str = "identity"

    def bounds(self, search_bounds: np.ndarray) -> np.ndarray:
        widths = search_bounds[:, 1] - search_bounds[:, 0]
        radius = float(self.radius_fraction) * widths
        lower = np.maximum(search_bounds[:, 0], self.center - radius)
        upper = np.minimum(search_bounds[:, 1], self.center + radius)
        return np.column_stack((lower, upper))

    def to_dict(self) -> dict[str, Any]:
        return {
            "region_id": int(self.region_id),
            "center": np.asarray(self.center, dtype=float).tolist(),
            "radius_fraction": float(self.radius_fraction),
            "score": float(self.score),
            "best_value": float(self.best_value),
            "age": int(self.age),
            "successes": int(self.successes),
            "failures": int(self.failures),
            "stall_count": int(self.stall_count),
            "cooldown_until": int(self.cooldown_until),
            "geometry_basis": np.asarray(self.geometry_basis, dtype=float).tolist(),
            "geometry_axis_scales": np.asarray(self.geometry_axis_scales, dtype=float).tolist(),
            "geometry_eigenvalues": np.asarray(self.geometry_eigenvalues, dtype=float).tolist(),
            "geometry_anisotropy": float(self.geometry_anisotropy),
            "geometry_sample_count": int(self.geometry_sample_count),
            "geometry_reason": self.geometry_reason,
        }


@dataclass(slots=True)
class PointCloudBatchEvent:
    """Diagnostics for one candidate batch or local refinement pass."""

    batch_index: int
    kind: str
    evaluations_before: int
    evaluations_after: int
    candidate_count: int
    source_counts: dict[str, int]
    best_before: float
    best_after: float
    best_point: np.ndarray
    improved: bool
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "batch_index": int(self.batch_index),
            "kind": self.kind,
            "evaluations_before": int(self.evaluations_before),
            "evaluations_after": int(self.evaluations_after),
            "evaluations_spent": int(self.evaluations_after - self.evaluations_before),
            "candidate_count": int(self.candidate_count),
            "source_counts": dict(self.source_counts),
            "best_before": float(self.best_before),
            "best_after": float(self.best_after),
            "best_point": np.asarray(self.best_point, dtype=float).tolist(),
            "improved": bool(self.improved),
        }
        payload.update(self.diagnostics)
        return payload


@dataclass(slots=True)
class PointCloudRegionEvent:
    """Diagnostics for one region radius/score update."""

    batch_index: int
    region_id: int
    radius_before: float
    radius_after: float
    success: bool
    improvements: int
    center: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_index": int(self.batch_index),
            "region_id": int(self.region_id),
            "radius_before": float(self.radius_before),
            "radius_after": float(self.radius_after),
            "success": bool(self.success),
            "trust_region_improvements": int(self.improvements),
            "center": np.asarray(self.center, dtype=float).tolist(),
        }


@dataclass(slots=True)
class PointCloudSnapshot:
    """Derived view of the archive for visualization and analysis."""

    step_index: int
    bounds: np.ndarray
    density_field: np.ndarray
    objective_field: np.ndarray
    evaluated_mask: np.ndarray
    best_point: np.ndarray
    best_value: float
    local_best_point: np.ndarray
    local_best_value: float
    box_best_point: np.ndarray
    box_best_value: float
    archive_points: np.ndarray
    archive_values: np.ndarray
    region_bounds: list[np.ndarray] = field(default_factory=list)
    candidate_points: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def field(self) -> np.ndarray:
        return self.density_field

    @property
    def inner_fill(self) -> np.ndarray:
        return self.density_field

    @property
    def outer_fill(self) -> np.ndarray:
        return self.density_field

    @property
    def transition_field(self) -> np.ndarray:
        return self.density_field

    @property
    def selected_basin_bbox(self) -> None:
        return None
