"""Typed artifacts for Matrix SmoothLife search."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class MatrixSmoothLifeSample:
    """One objective evaluation of a decoded matrix genome."""

    point: np.ndarray
    value: float
    source: str
    evaluation: int
    step_index: int
    field_key: tuple[str, ...]


@dataclass(slots=True)
class MatrixSmoothLifeSnapshot:
    """State captured from a Matrix SmoothLife optimization run."""

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
