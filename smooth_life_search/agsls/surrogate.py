"""Commit-phase surrogate helpers for AGSLS zoom decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core import Basin
from ..smoothlife.evaluation import evaluation_points


@dataclass(slots=True)
class CommitSurrogateResult:
    """Result of a local commit surrogate fit."""

    accepted: bool
    reason: str
    sample_count: int
    point: np.ndarray | None = None
    normalized_point: np.ndarray | None = None
    predicted_value: float | None = None
    predicted_improvement: float | None = None
    condition: float | None = None
    hessian: np.ndarray | None = None
    eigenvalues: np.ndarray | None = None
    eigenvectors: np.ndarray | None = None

    def diagnostics(self) -> dict[str, object]:
        """Return compact JSON-friendly diagnostics."""

        payload: dict[str, object] = {
            "surrogate_used": bool(self.accepted),
            "surrogate_reason": self.reason,
            "surrogate_samples": int(self.sample_count),
        }
        if self.condition is not None and np.isfinite(self.condition):
            payload["surrogate_condition"] = float(self.condition)
        if self.point is not None:
            payload["surrogate_point"] = np.asarray(self.point, dtype=float).tolist()
        if self.predicted_value is not None and np.isfinite(self.predicted_value):
            payload["surrogate_predicted_value"] = float(self.predicted_value)
        if self.predicted_improvement is not None and np.isfinite(self.predicted_improvement):
            payload["surrogate_predicted_improvement"] = float(self.predicted_improvement)
        return payload


def _rejected(reason: str, sample_count: int = 0, condition: float | None = None) -> CommitSurrogateResult:
    return CommitSurrogateResult(
        accepted=False,
        reason=reason,
        sample_count=int(sample_count),
        condition=condition,
    )


def _candidate_mask(
    basin: Basin,
    evaluated_mask: np.ndarray,
    *,
    min_samples: int,
) -> np.ndarray | None:
    if basin.mask.shape != evaluated_mask.shape:
        return None
    mask = np.asarray(basin.mask, dtype=bool).copy()
    if np.count_nonzero(mask & evaluated_mask) >= int(min_samples):
        return mask

    row_min, col_min, row_max, col_max = basin.bbox_grid
    padding = 2
    row_min = max(0, int(row_min) - padding)
    col_min = max(0, int(col_min) - padding)
    row_max = min(evaluated_mask.shape[0] - 1, int(row_max) + padding)
    col_max = min(evaluated_mask.shape[1] - 1, int(col_max) + padding)
    near = np.zeros_like(evaluated_mask, dtype=bool)
    near[row_min : row_max + 1, col_min : col_max + 1] = True
    mask |= near
    return mask


def _normalized_quality(values: np.ndarray, *, maximize: bool) -> np.ndarray:
    target = -values if maximize else values
    span = max(float(np.max(target) - np.min(target)), 1e-12)
    return 1.0 - (target - float(np.min(target))) / span


def _reference_point(basin: Basin) -> np.ndarray:
    if basin.basin_best_point is not None:
        candidate = np.asarray(basin.basin_best_point, dtype=float)
        if candidate.shape == (2,) and np.all(np.isfinite(candidate)):
            return candidate
    return np.asarray(basin.centroid_world, dtype=float)


def _limit_samples(
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    support_values: np.ndarray,
    points: np.ndarray,
    basin: Basin,
    bounds: np.ndarray,
    *,
    maximize: bool,
    max_samples: int,
    support_weight: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if values.size <= int(max_samples):
        return rows, cols, values, support_values, points

    widths = np.maximum(bounds[:, 1] - bounds[:, 0], 1e-300)
    center = _reference_point(basin)
    normalized_points = (points - bounds[:, 0]) / widths
    normalized_center = (center - bounds[:, 0]) / widths
    distances = np.linalg.norm(normalized_points - normalized_center, axis=1)
    proximity = np.exp(-0.5 * (distances / 0.25) ** 2)
    quality = _normalized_quality(values, maximize=maximize)
    support = np.asarray(support_values, dtype=float)
    if np.any(np.isfinite(support)):
        support = np.nan_to_num(support, nan=0.0, posinf=0.0, neginf=0.0)
        support_span = max(float(np.max(support) - np.min(support)), 1e-12)
        support = (support - float(np.min(support))) / support_span
    scores = quality + float(support_weight) * support + 0.25 * proximity
    limit = min(int(max_samples), int(scores.size))
    chosen = np.argpartition(scores, -limit)[-limit:]
    chosen = chosen[np.argsort(scores[chosen])[::-1]]
    return rows[chosen], cols[chosen], values[chosen], support_values[chosen], points[chosen]


def _solve_weighted_quadratic(
    normalized_points: np.ndarray,
    normalized_targets: np.ndarray,
    weights: np.ndarray,
    *,
    regularization: float,
) -> tuple[np.ndarray, float]:
    x = normalized_points[:, 0]
    y = normalized_points[:, 1]
    design = np.column_stack((np.ones_like(x), x, y, x * x, x * y, y * y))
    sqrt_weights = np.sqrt(np.maximum(weights, 1e-12))
    weighted_design = design * sqrt_weights[:, None]
    weighted_targets = normalized_targets * sqrt_weights
    condition = float(np.linalg.cond(weighted_design))
    normal = weighted_design.T @ weighted_design
    ridge = float(regularization) * np.eye(normal.shape[0], dtype=float)
    ridge[0, 0] = 0.0
    rhs = weighted_design.T @ weighted_targets
    coefficients = np.linalg.solve(normal + ridge, rhs)
    return coefficients, condition


def fit_commit_surrogate(
    *,
    objective_values: np.ndarray,
    evaluated_mask: np.ndarray,
    support_field: np.ndarray,
    basin: Basin,
    bounds: np.ndarray,
    grid_shape: tuple[int, int],
    maximize: bool,
    min_samples: int,
    max_samples: int,
    regularization: float,
    min_predicted_improvement: float,
    max_condition: float,
    support_weight: float,
) -> CommitSurrogateResult:
    """Fit a weighted local quadratic and return a trusted commit center."""

    evaluated = np.asarray(evaluated_mask, dtype=bool)
    values_grid = np.asarray(objective_values, dtype=float)
    current_bounds = np.asarray(bounds, dtype=float)
    if values_grid.shape != evaluated.shape or tuple(values_grid.shape) != tuple(grid_shape):
        return _rejected("shape_mismatch")
    if not np.all(np.isfinite(current_bounds)) or np.any(current_bounds[:, 1] <= current_bounds[:, 0]):
        return _rejected("invalid_bounds")

    mask = _candidate_mask(basin, evaluated, min_samples=int(min_samples))
    if mask is None:
        return _rejected("mask_shape_mismatch")
    finite_values = np.isfinite(values_grid)
    candidate_mask = mask & evaluated & finite_values
    rows, cols = np.nonzero(candidate_mask)
    sample_count = int(rows.size)
    if sample_count < int(min_samples):
        return _rejected("insufficient_samples", sample_count)

    values = values_grid[rows, cols]
    support = np.asarray(support_field, dtype=float)
    if support.shape != values_grid.shape:
        support_values = np.zeros_like(values, dtype=float)
    else:
        support_values = support[rows, cols]
    points = evaluation_points(rows, cols, current_bounds, grid_shape)
    rows, cols, values, support_values, points = _limit_samples(
        rows,
        cols,
        values,
        support_values,
        points,
        basin,
        current_bounds,
        maximize=maximize,
        max_samples=max_samples,
        support_weight=support_weight,
    )
    sample_count = int(values.size)
    if sample_count < int(min_samples):
        return _rejected("insufficient_samples", sample_count)

    widths = current_bounds[:, 1] - current_bounds[:, 0]
    normalized_points = (points - current_bounds[:, 0]) / widths
    target = -values if maximize else values
    target_min = float(np.min(target))
    target_span = max(float(np.max(target) - target_min), 1e-12)
    normalized_targets = (target - target_min) / target_span
    quality = 1.0 - np.clip(normalized_targets, 0.0, 1.0)
    support_clean = np.nan_to_num(support_values, nan=0.0, posinf=0.0, neginf=0.0)
    if support_clean.size:
        support_span = max(float(np.max(support_clean) - np.min(support_clean)), 1e-12)
        support_clean = (support_clean - float(np.min(support_clean))) / support_span
    weights = 0.25 + quality + float(support_weight) * support_clean
    reference = _reference_point(basin)
    normalized_reference = (reference - current_bounds[:, 0]) / widths
    distances = np.linalg.norm(normalized_points - normalized_reference, axis=1)
    weights += 0.50 * np.exp(-0.5 * (distances / 0.20) ** 2)

    try:
        coefficients, condition = _solve_weighted_quadratic(
            normalized_points,
            normalized_targets,
            weights,
            regularization=regularization,
        )
    except np.linalg.LinAlgError:
        return _rejected("singular_fit", sample_count)
    if not np.all(np.isfinite(coefficients)) or not np.isfinite(condition):
        return _rejected("nonfinite_fit", sample_count, condition)
    if condition > float(max_condition):
        return _rejected("ill_conditioned_fit", sample_count, condition)

    gradient = np.asarray([coefficients[1], coefficients[2]], dtype=float)
    hessian = np.asarray(
        [[2.0 * coefficients[3], coefficients[4]], [coefficients[4], 2.0 * coefficients[5]]],
        dtype=float,
    )
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    except np.linalg.LinAlgError:
        return _rejected("invalid_hessian", sample_count, condition)
    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
        return _rejected("invalid_hessian", sample_count, condition)
    if float(np.min(eigenvalues)) <= 1e-12:
        return _rejected("nonconvex_surrogate", sample_count, condition)
    if float(np.max(eigenvalues) / max(float(np.min(eigenvalues)), 1e-300)) > float(max_condition):
        return _rejected("ill_conditioned_hessian", sample_count, condition)

    try:
        normalized_point = -np.linalg.solve(hessian, gradient)
    except np.linalg.LinAlgError:
        return _rejected("singular_hessian", sample_count, condition)
    if not np.all(np.isfinite(normalized_point)):
        return _rejected("nonfinite_point", sample_count, condition)
    if np.any(normalized_point < -1e-12) or np.any(normalized_point > 1.0 + 1e-12):
        return _rejected("point_outside_bounds", sample_count, condition)
    normalized_point = np.clip(normalized_point, 0.0, 1.0)

    x, y = float(normalized_point[0]), float(normalized_point[1])
    predicted_normalized = float(
        coefficients[0] + coefficients[1] * x + coefficients[2] * y + coefficients[3] * x * x + coefficients[4] * x * y + coefficients[5] * y * y
    )
    if not np.isfinite(predicted_normalized):
        return _rejected("nonfinite_prediction", sample_count, condition)

    if basin.basin_best_value is not None and np.isfinite(basin.basin_best_value):
        best_target = -float(basin.basin_best_value) if maximize else float(basin.basin_best_value)
        best_normalized = (best_target - target_min) / target_span
    else:
        best_normalized = float(np.min(normalized_targets))
    predicted_improvement = float(best_normalized - predicted_normalized)
    if predicted_improvement <= float(min_predicted_improvement):
        return _rejected("no_predicted_improvement", sample_count, condition)

    point = current_bounds[:, 0] + normalized_point * widths
    predicted_target = predicted_normalized * target_span + target_min
    predicted_value = -predicted_target if maximize else predicted_target
    return CommitSurrogateResult(
        accepted=True,
        reason="accepted",
        sample_count=sample_count,
        point=np.asarray(point, dtype=float),
        normalized_point=np.asarray(normalized_point, dtype=float),
        predicted_value=float(predicted_value),
        predicted_improvement=predicted_improvement,
        condition=condition,
        hessian=hessian,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
    )


def surrogate_axis_widths(
    *,
    surrogate: CommitSurrogateResult,
    bbox_widths: np.ndarray,
    current_widths: np.ndarray,
    valley_expand: float,
    cross_shrink: float,
) -> np.ndarray | None:
    """Convert the surrogate curvature frame into conservative axis-aligned widths."""

    if not surrogate.accepted or surrogate.eigenvalues is None or surrogate.eigenvectors is None:
        return None
    eigenvalues = np.asarray(surrogate.eigenvalues, dtype=float)
    eigenvectors = np.asarray(surrogate.eigenvectors, dtype=float)
    if eigenvalues.shape != (2,) or eigenvectors.shape != (2, 2):
        return None
    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
        return None
    current_widths = np.asarray(current_widths, dtype=float)
    bbox_widths = np.asarray(bbox_widths, dtype=float)
    if np.any(current_widths <= 0.0) or np.any(bbox_widths <= 0.0):
        return None

    half_axis_normalized = 0.5 * np.minimum(current_widths, bbox_widths) / current_widths
    half_eigen = np.abs(eigenvectors).T @ half_axis_normalized
    order = np.argsort(eigenvalues)
    adjusted = half_eigen.copy()
    adjusted[int(order[0])] *= float(valley_expand)
    adjusted[int(order[-1])] *= float(cross_shrink)
    axis_half_normalized = np.abs(eigenvectors) @ adjusted
    widths = 2.0 * axis_half_normalized * current_widths
    if not np.all(np.isfinite(widths)):
        return None
    return np.maximum(widths, 0.0)


def surrogate_valley_tangent(surrogate: CommitSurrogateResult) -> np.ndarray | None:
    """Return the low-curvature unit direction in normalized surrogate coordinates."""

    if not surrogate.accepted or surrogate.eigenvalues is None or surrogate.eigenvectors is None:
        return None
    eigenvalues = np.asarray(surrogate.eigenvalues, dtype=float)
    eigenvectors = np.asarray(surrogate.eigenvectors, dtype=float)
    if eigenvalues.shape != (2,) or eigenvectors.shape != (2, 2):
        return None
    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
        return None
    positive = np.flatnonzero(eigenvalues > 0.0)
    if positive.size == 0:
        return None
    low_index = int(positive[np.argmin(eigenvalues[positive])])
    tangent = np.asarray(eigenvectors[:, low_index], dtype=float)
    norm = float(np.linalg.norm(tangent))
    if not np.isfinite(norm) or norm <= 1e-12:
        return None
    tangent = tangent / norm
    dominant = int(np.argmax(np.abs(tangent)))
    if tangent[dominant] < 0.0:
        tangent = -tangent
    return tangent
