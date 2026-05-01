"""Small 2D bounds helpers for AGSLS callers."""

from __future__ import annotations

import numpy as np


def cell_widths(bounds: np.ndarray, grid_shape: tuple[int, int]) -> np.ndarray:
    """Return x/y world widths of one grid cell."""

    height, width = grid_shape
    resolved = np.asarray(bounds, dtype=float)
    widths = resolved[:, 1] - resolved[:, 0]
    return np.asarray([widths[0] / width, widths[1] / height], dtype=float)


def centered_bounds(center: np.ndarray, widths: np.ndarray, ceiling_bounds: np.ndarray) -> np.ndarray:
    """Return bounds centered on ``center`` and clipped inside ``ceiling_bounds``."""

    resolved_center = np.asarray(center, dtype=float)
    resolved_widths = np.asarray(widths, dtype=float)
    ceiling = np.asarray(ceiling_bounds, dtype=float)
    ceiling_widths = ceiling[:, 1] - ceiling[:, 0]
    resolved_widths = np.minimum(resolved_widths, ceiling_widths)
    bounds = np.empty((2, 2), dtype=float)
    for axis in range(2):
        half_width = 0.5 * float(resolved_widths[axis])
        center_axis = float(np.clip(resolved_center[axis], ceiling[axis, 0], ceiling[axis, 1]))
        lower = center_axis - half_width
        upper = center_axis + half_width
        if lower < ceiling[axis, 0]:
            upper += ceiling[axis, 0] - lower
            lower = ceiling[axis, 0]
        if upper > ceiling[axis, 1]:
            lower -= upper - ceiling[axis, 1]
            upper = ceiling[axis, 1]
        bounds[axis, 0] = max(float(ceiling[axis, 0]), lower)
        bounds[axis, 1] = min(float(ceiling[axis, 1]), upper)
    return bounds
