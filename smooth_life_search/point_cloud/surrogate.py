"""Local quadratic surrogate helpers for point-cloud regions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class QuadraticSurrogate:
    """Weighted quadratic fit over one region."""

    accepted: bool
    reason: str
    sample_count: int
    point: np.ndarray | None = None
    predicted_value: float | None = None
    condition: float | None = None
    coefficients: np.ndarray | None = None
    target_min: float | None = None
    target_span: float | None = None
    region_bounds: np.ndarray | None = None

    def diagnostics(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "surrogate_used": bool(self.accepted),
            "surrogate_reason": self.reason,
            "surrogate_samples": int(self.sample_count),
        }
        if self.point is not None:
            payload["surrogate_point"] = np.asarray(self.point, dtype=float).tolist()
        if self.predicted_value is not None and np.isfinite(self.predicted_value):
            payload["surrogate_predicted_value"] = float(self.predicted_value)
        if self.condition is not None and np.isfinite(self.condition):
            payload["surrogate_condition"] = float(self.condition)
        return payload


def _rejected(reason: str, sample_count: int = 0, condition: float | None = None) -> QuadraticSurrogate:
    return QuadraticSurrogate(accepted=False, reason=reason, sample_count=int(sample_count), condition=condition)


def fit_quadratic_surrogate(
    points: np.ndarray,
    values: np.ndarray,
    *,
    region_bounds: np.ndarray,
    center: np.ndarray,
    maximize: bool,
    min_samples: int,
    max_samples: int,
    regularization: float,
    max_condition: float,
) -> QuadraticSurrogate:
    """Fit a convex local quadratic and return its optimum when accepted."""

    sample_points = np.asarray(points, dtype=float)
    sample_values = np.asarray(values, dtype=float)
    region = np.asarray(region_bounds, dtype=float)
    if sample_points.ndim != 2 or sample_points.shape[1] != 2 or sample_values.shape != (sample_points.shape[0],):
        return _rejected("shape_mismatch")
    if region.shape != (2, 2) or np.any(region[:, 1] <= region[:, 0]) or not np.all(np.isfinite(region)):
        return _rejected("invalid_region")
    inside = (
        np.all(sample_points >= region[:, 0], axis=1)
        & np.all(sample_points <= region[:, 1], axis=1)
        & np.isfinite(sample_values)
    )
    sample_points = sample_points[inside]
    sample_values = sample_values[inside]
    if sample_points.shape[0] < int(min_samples):
        return _rejected("insufficient_region_samples", sample_points.shape[0])

    widths = region[:, 1] - region[:, 0]
    normalized = (sample_points - region[:, 0]) / widths
    normalized_center = (np.asarray(center, dtype=float) - region[:, 0]) / widths
    target = -sample_values if maximize else sample_values
    target_min = float(np.min(target))
    target_span = max(float(np.max(target) - target_min), 1e-12)
    normalized_targets = (target - target_min) / target_span
    quality = 1.0 - np.clip(normalized_targets, 0.0, 1.0)
    distances = np.linalg.norm(normalized - normalized_center, axis=1)
    weights = 0.20 + quality + np.exp(-0.5 * (distances / 0.35) ** 2)

    if normalized.shape[0] > int(max_samples):
        scores = weights + quality
        chosen = np.argpartition(scores, -int(max_samples))[-int(max_samples):]
        chosen = chosen[np.argsort(scores[chosen])[::-1]]
        normalized = normalized[chosen]
        normalized_targets = normalized_targets[chosen]
        weights = weights[chosen]

    x = normalized[:, 0]
    y = normalized[:, 1]
    design = np.column_stack((np.ones_like(x), x, y, x * x, x * y, y * y))
    sqrt_weights = np.sqrt(np.maximum(weights, 1e-12))
    weighted_design = design * sqrt_weights[:, None]
    weighted_targets = normalized_targets * sqrt_weights
    sample_count = int(normalized.shape[0])
    try:
        condition = float(np.linalg.cond(weighted_design))
        normal = weighted_design.T @ weighted_design
        ridge = float(regularization) * np.eye(normal.shape[0], dtype=float)
        ridge[0, 0] = 0.0
        coefficients = np.linalg.solve(normal + ridge, weighted_design.T @ weighted_targets)
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
        eigenvalues = np.linalg.eigvalsh(hessian)
    except np.linalg.LinAlgError:
        return _rejected("invalid_hessian", sample_count, condition)
    if not np.all(np.isfinite(eigenvalues)) or float(np.min(eigenvalues)) <= 1e-12:
        return _rejected("nonconvex_surrogate", sample_count, condition)
    try:
        normalized_optimum = -np.linalg.solve(hessian, gradient)
    except np.linalg.LinAlgError:
        return _rejected("singular_optimum", sample_count, condition)
    if not np.all(np.isfinite(normalized_optimum)) or np.any(normalized_optimum < 0.0) or np.any(normalized_optimum > 1.0):
        return _rejected("optimum_outside_region", sample_count, condition)

    point = region[:, 0] + normalized_optimum * widths
    x0 = float(normalized_optimum[0])
    y0 = float(normalized_optimum[1])
    predicted_normalized = float(
        coefficients[0]
        + coefficients[1] * x0
        + coefficients[2] * y0
        + coefficients[3] * x0 * x0
        + coefficients[4] * x0 * y0
        + coefficients[5] * y0 * y0
    )
    target_value = predicted_normalized * target_span + target_min
    predicted_value = -target_value if maximize else target_value
    return QuadraticSurrogate(
        accepted=True,
        reason="accepted",
        sample_count=sample_count,
        point=point,
        predicted_value=float(predicted_value),
        condition=condition,
        coefficients=coefficients,
        target_min=target_min,
        target_span=target_span,
        region_bounds=region.copy(),
    )
