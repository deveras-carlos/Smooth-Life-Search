"""Objective-cache helpers for the SmoothLife simulator."""

from __future__ import annotations

import numpy as np


def blank_objective_cache(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create objective values, normalized field, and evaluated-mask arrays."""

    return (
        np.full(shape, np.nan, dtype=float),
        np.full(shape, 0.5, dtype=float),
        np.zeros(shape, dtype=bool),
    )


def evaluation_points(
    rows: np.ndarray,
    cols: np.ndarray,
    bounds: np.ndarray,
    grid_shape: tuple[int, int],
) -> np.ndarray:
    """Map grid row/column indices to world-coordinate pixel centers."""

    height, width = grid_shape
    x = bounds[0, 0] + ((cols.astype(float) + 0.5) / width) * (bounds[0, 1] - bounds[0, 0])
    y = bounds[1, 0] + ((rows.astype(float) + 0.5) / height) * (bounds[1, 1] - bounds[1, 0])
    return np.column_stack((x, y))


def normalized_objective_field(
    objective_values: np.ndarray,
    evaluated_mask: np.ndarray,
    *,
    maximize: bool,
) -> np.ndarray:
    """Normalize explored objective values into a support field in ``[0, 1]``."""

    objective_field = np.full(objective_values.shape, 0.5, dtype=float)
    if not np.any(evaluated_mask):
        return objective_field
    explored_values = objective_values[evaluated_mask]
    min_value = float(np.min(explored_values))
    max_value = float(np.max(explored_values))
    span = max(max_value - min_value, 1e-12)
    if explored_values.size < 2 or span <= 1e-12:
        objective_field[evaluated_mask] = 1.0
        return objective_field
    if maximize:
        normalized = (explored_values - min_value) / span
    else:
        normalized = (max_value - explored_values) / span
    objective_field[evaluated_mask] = np.clip(normalized, 0.0, 1.0)
    return objective_field


def best_evaluated_flat_index(
    objective_values: np.ndarray,
    evaluated_mask: np.ndarray,
    *,
    maximize: bool,
) -> int | None:
    """Return the flat index of the best explored pixel, or ``None``."""

    if not np.any(evaluated_mask):
        return None
    flat_mask = evaluated_mask.ravel()
    flat_values = objective_values.ravel()
    explored_indices = np.flatnonzero(flat_mask)
    explored_values = flat_values[explored_indices]
    local_offset = int(np.argmax(explored_values) if maximize else np.argmin(explored_values))
    return int(explored_indices[local_offset])
