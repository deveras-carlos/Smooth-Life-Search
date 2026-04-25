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
    gamma: float = 1.0,
) -> np.ndarray:
    """Normalize explored objective values into a support field in ``[0, 1]``.

    ``gamma`` > 1 compresses low values toward 0 so that the top of the distribution
    dominates the support field — sharpens basin signal for exploitation.
    """

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
    normalized = np.clip(normalized, 0.0, 1.0)
    if gamma != 1.0:
        normalized = np.power(normalized, float(gamma))
    objective_field[evaluated_mask] = normalized
    return objective_field


def best_evaluated_flat_index(
    objective_values: np.ndarray,
    evaluated_mask: np.ndarray,
    *,
    maximize: bool,
) -> int | None:
    """Return the flat index of the best explored pixel, or ``None``."""

    top = top_evaluated_flat_indices(
        objective_values,
        evaluated_mask,
        maximize=maximize,
        limit=1,
    )
    return None if not top else top[0]


def top_evaluated_flat_indices(
    objective_values: np.ndarray,
    evaluated_mask: np.ndarray,
    *,
    maximize: bool,
    limit: int,
) -> tuple[int, ...]:
    """Return up to ``limit`` explored flat indices ordered from best to worst."""

    if limit <= 0 or not np.any(evaluated_mask):
        return ()
    flat_mask = evaluated_mask.ravel()
    flat_values = objective_values.ravel()
    explored_indices = np.flatnonzero(flat_mask)
    if explored_indices.size == 0:
        return ()
    explored_values = flat_values[explored_indices]
    order = np.argsort(explored_values)
    if maximize:
        order = order[::-1]
    resolved_limit = min(int(limit), int(order.size))
    return tuple(int(explored_indices[int(offset)]) for offset in order[:resolved_limit])


def _axis_subpixel_offset(p0: float, p1: float, p2: float, *, maximize: bool) -> float:
    """Quadratic-fit sub-pixel offset of the extremum given three adjacent samples.

    Returns 0.0 if the fit is degenerate or the extremum lies outside ``[-0.5, 0.5]``.
    """

    curvature = p0 - 2.0 * p1 + p2
    if maximize:
        if curvature >= -1e-12:
            return 0.0
    else:
        if curvature <= 1e-12:
            return 0.0
    delta = 0.5 * (p0 - p2) / curvature
    if not np.isfinite(delta):
        return 0.0
    return float(np.clip(delta, -0.5, 0.5))


def _separable_subpixel_offset(
    objective_values: np.ndarray,
    evaluated_mask: np.ndarray,
    row: int,
    col: int,
    *,
    maximize: bool,
) -> tuple[float, float]:
    height, width = objective_values.shape
    drow = 0.0
    dcol = 0.0
    if 0 < row < height - 1 and evaluated_mask[row - 1, col] and evaluated_mask[row + 1, col]:
        drow = _axis_subpixel_offset(
            float(objective_values[row - 1, col]),
            float(objective_values[row, col]),
            float(objective_values[row + 1, col]),
            maximize=maximize,
        )
    if 0 < col < width - 1 and evaluated_mask[row, col - 1] and evaluated_mask[row, col + 1]:
        dcol = _axis_subpixel_offset(
            float(objective_values[row, col - 1]),
            float(objective_values[row, col]),
            float(objective_values[row, col + 1]),
            maximize=maximize,
        )
    return drow, dcol


def _joint_subpixel_offset(
    window: np.ndarray,
    *,
    maximize: bool,
) -> tuple[float, float] | None:
    """Fit ``f(dcol, drow) = a + b*x + h*y + d*x^2 + e*x*y + g*y^2``
    by least squares to a fully-populated 3x3 window of cached values.

    ``window[i, j]`` is the sample at (drow = i - 1, dcol = j - 1).

    Returns ``(drow, dcol)`` clamped to ``[-0.5, 0.5]^2`` or ``None`` when the
    Hessian is ill-conditioned or has the wrong definiteness for the requested
    extremum kind.
    """

    y0 = float(window.sum())
    y1 = float(window[:, 0].sum() + window[:, 2].sum())  # sum dcol^2 * f
    y2 = float(window[0, :].sum() + window[2, :].sum())  # sum drow^2 * f
    b = float((window[:, 2].sum() - window[:, 0].sum()) / 6.0)
    h = float((window[2, :].sum() - window[0, :].sum()) / 6.0)
    e = float((window[0, 0] + window[2, 2] - window[2, 0] - window[0, 2]) / 4.0)
    d = float((-2.0 * y0 + 3.0 * y1) / 6.0)
    g = float((-2.0 * y0 + 3.0 * y2) / 6.0)

    det = 4.0 * d * g - e * e
    if not np.isfinite(det) or abs(det) < 1e-12:
        return None
    if maximize:
        if d >= 0.0 or det <= 0.0:
            return None
    else:
        if d <= 0.0 or det <= 0.0:
            return None

    dcol_star = (e * h - 2.0 * g * b) / det
    drow_star = (e * b - 2.0 * d * h) / det
    if not (np.isfinite(dcol_star) and np.isfinite(drow_star)):
        return None
    return (
        float(np.clip(drow_star, -0.5, 0.5)),
        float(np.clip(dcol_star, -0.5, 0.5)),
    )


def subpixel_best_offset(
    objective_values: np.ndarray,
    evaluated_mask: np.ndarray,
    row: int,
    col: int,
    *,
    maximize: bool,
) -> tuple[float, float]:
    """Return the (drow, dcol) sub-pixel offset of the optimum near ``(row, col)``.

    When the full 3x3 neighborhood is evaluated, fit a joint 2D quadratic and
    return its vertex. This captures the cross (``dcol*drow``) term that a
    separable axis-wise fit misses on non-axis-aligned landscapes (Rosenbrock's
    banana valley is the canonical case). When any neighbor is unevaluated or
    the Hessian is ill-conditioned, fall back to the separable axis-wise fit.
    The result is always clamped to ``[-0.5, 0.5]`` along each axis.
    """

    height, width = objective_values.shape
    if (
        0 < row < height - 1
        and 0 < col < width - 1
        and bool(evaluated_mask[row - 1 : row + 2, col - 1 : col + 2].all())
    ):
        window = objective_values[row - 1 : row + 2, col - 1 : col + 2].astype(float, copy=False)
        joint = _joint_subpixel_offset(window, maximize=maximize)
        if joint is not None:
            return joint
    return _separable_subpixel_offset(
        objective_values,
        evaluated_mask,
        row,
        col,
        maximize=maximize,
    )
