"""Small helpers for validating and working with search bounds."""

from __future__ import annotations

import numpy as np

BoundsLike = np.ndarray | list[tuple[float, float]] | tuple[tuple[float, float], ...]


def normalize_bounds_2d(bounds: BoundsLike, *, owner: str) -> np.ndarray:
    """Return bounds as a validated ``(2, 2)`` float array."""

    arr = np.asarray(bounds, dtype=float)
    if arr.shape != (2, 2):
        raise ValueError(f"{owner} currently supports exactly 2D bounds")
    if np.any(arr[:, 1] <= arr[:, 0]):
        raise ValueError("each bound must satisfy lower < upper")
    return arr


def normalize_bounds_nd(bounds: BoundsLike, *, owner: str, dimension: int | None = None) -> np.ndarray:
    """Return bounds as a validated ``(dimension, 2)`` float array."""

    arr = np.asarray(bounds, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{owner} bounds must have shape (dimension, 2)")
    if arr.shape[0] < 2:
        raise ValueError(f"{owner} currently requires at least 2 dimensions")
    if dimension is not None and arr.shape[0] != int(dimension):
        raise ValueError(f"{owner} expected {int(dimension)}D bounds")
    if not np.all(np.isfinite(arr)):
        raise ValueError("bounds must be finite")
    if np.any(arr[:, 1] <= arr[:, 0]):
        raise ValueError("each bound must satisfy lower < upper")
    return arr


def bounds_area(bounds: np.ndarray) -> float:
    """Return the nonzero area of a 2D box."""

    widths = np.asarray(bounds, dtype=float)[:, 1] - np.asarray(bounds, dtype=float)[:, 0]
    return float(max(widths[0], 1e-12) * max(widths[1], 1e-12))


def point_in_bounds(point: np.ndarray, bounds: np.ndarray) -> bool:
    """Return whether a finite 2D point is inside inclusive bounds."""

    resolved_point = np.asarray(point, dtype=float)
    resolved_bounds = np.asarray(bounds, dtype=float)
    return bool(
        resolved_point.shape == (2,)
        and np.all(np.isfinite(resolved_point))
        and np.all(resolved_point >= resolved_bounds[:, 0])
        and np.all(resolved_point <= resolved_bounds[:, 1])
    )
