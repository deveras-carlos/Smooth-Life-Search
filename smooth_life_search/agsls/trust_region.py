"""Trust-region helpers for AGSLS acquisition batches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..smoothlife.evaluation import evaluation_points


@dataclass(slots=True)
class SampleArchive:
    """Persistent objective sample archive across AGSLS remaps."""

    points: list[np.ndarray]
    values: list[float]
    _keys: set[tuple[str, str]]

    @classmethod
    def empty(cls) -> "SampleArchive":
        return cls(points=[], values=[], _keys=set())

    def __len__(self) -> int:
        return len(self.values)

    @staticmethod
    def _key(point: np.ndarray) -> tuple[str, str]:
        resolved = np.asarray(point, dtype=float)
        return (float(resolved[0]).hex(), float(resolved[1]).hex())

    def add_point(self, point: np.ndarray, value: float) -> bool:
        resolved = np.asarray(point, dtype=float)
        resolved_value = float(value)
        if resolved.shape != (2,) or not np.all(np.isfinite(resolved)) or not np.isfinite(resolved_value):
            return False
        key = self._key(resolved)
        if key in self._keys:
            return False
        self._keys.add(key)
        self.points.append(resolved.copy())
        self.values.append(resolved_value)
        return True

    def add_points(self, points: np.ndarray, values: np.ndarray) -> int:
        resolved_points = np.asarray(points, dtype=float)
        resolved_values = np.asarray(values, dtype=float)
        if resolved_points.ndim != 2 or resolved_points.shape[1] != 2:
            return 0
        added = 0
        for point, value in zip(resolved_points, resolved_values, strict=False):
            if self.add_point(point, float(value)):
                added += 1
        return added

    def add_grid(
        self,
        objective_values: np.ndarray,
        evaluated_mask: np.ndarray,
        bounds: np.ndarray,
        grid_shape: tuple[int, int],
    ) -> int:
        evaluated = np.asarray(evaluated_mask, dtype=bool)
        values = np.asarray(objective_values, dtype=float)
        finite = evaluated & np.isfinite(values)
        rows, cols = np.nonzero(finite)
        if rows.size == 0:
            return 0
        points = evaluation_points(rows, cols, np.asarray(bounds, dtype=float), grid_shape)
        return self.add_points(points, values[rows, cols])

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.points:
            return np.empty((0, 2), dtype=float), np.empty((0,), dtype=float)
        return np.vstack(self.points).astype(float, copy=False), np.asarray(self.values, dtype=float)


@dataclass(slots=True)
class TrustRegionState:
    """Mutable trust-region center and normalized radius."""

    center: np.ndarray | None = None
    radius_fraction: float | None = None


@dataclass(slots=True)
class ArchiveQuadraticSurrogate:
    """Weighted quadratic surrogate fitted from the persistent archive."""

    accepted: bool
    reason: str
    sample_count: int
    region_bounds: np.ndarray | None = None
    point: np.ndarray | None = None
    predicted_value: float | None = None
    condition: float | None = None
    coefficients: np.ndarray | None = None
    hessian: np.ndarray | None = None
    eigenvalues: np.ndarray | None = None
    eigenvectors: np.ndarray | None = None
    target_min: float | None = None
    target_span: float | None = None

    def diagnostics(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "trust_surrogate_used": bool(self.accepted),
            "trust_surrogate_reason": self.reason,
            "trust_surrogate_samples": int(self.sample_count),
        }
        if self.condition is not None and np.isfinite(self.condition):
            payload["trust_surrogate_condition"] = float(self.condition)
        if self.point is not None:
            payload["trust_surrogate_point"] = np.asarray(self.point, dtype=float).tolist()
        if self.predicted_value is not None and np.isfinite(self.predicted_value):
            payload["trust_surrogate_predicted_value"] = float(self.predicted_value)
        return payload

    def predict(self, points: np.ndarray, *, maximize: bool) -> np.ndarray | None:
        if not self.accepted or self.coefficients is None or self.region_bounds is None:
            return None
        candidates = np.asarray(points, dtype=float)
        if candidates.ndim != 2 or candidates.shape[1] != 2:
            return None
        region = np.asarray(self.region_bounds, dtype=float)
        widths = region[:, 1] - region[:, 0]
        if np.any(widths <= 0.0):
            return None
        normalized = (candidates - region[:, 0]) / widths
        x = normalized[:, 0]
        y = normalized[:, 1]
        coefs = np.asarray(self.coefficients, dtype=float)
        predicted = coefs[0] + coefs[1] * x + coefs[2] * y + coefs[3] * x * x + coefs[4] * x * y + coefs[5] * y * y
        target_min = 0.0 if self.target_min is None else float(self.target_min)
        target_span = 1.0 if self.target_span is None else float(self.target_span)
        values = predicted * max(target_span, 1e-12) + target_min
        if maximize:
            values = -values
        return np.asarray(values, dtype=float)


def _rejected(reason: str, sample_count: int = 0, condition: float | None = None) -> ArchiveQuadraticSurrogate:
    return ArchiveQuadraticSurrogate(
        accepted=False,
        reason=reason,
        sample_count=int(sample_count),
        condition=condition,
    )


def trust_region_bounds(center: np.ndarray, radius_fraction: float, active_bounds: np.ndarray) -> np.ndarray | None:
    """Return axis-aligned trust-region bounds clipped to the active box."""

    resolved_center = np.asarray(center, dtype=float)
    active = np.asarray(active_bounds, dtype=float)
    if resolved_center.shape != (2,) or active.shape != (2, 2):
        return None
    widths = active[:, 1] - active[:, 0]
    if not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
        return None
    radius = float(radius_fraction) * widths
    lower = np.maximum(active[:, 0], resolved_center - radius)
    upper = np.minimum(active[:, 1], resolved_center + radius)
    region = np.column_stack((lower, upper))
    if not np.all(np.isfinite(region)) or np.any(region[:, 1] <= region[:, 0]):
        return None
    return region


def fit_archive_quadratic(
    *,
    archive_points: np.ndarray,
    archive_values: np.ndarray,
    region_bounds: np.ndarray,
    center: np.ndarray,
    maximize: bool,
    min_samples: int,
    max_samples: int,
    regularization: float,
    max_condition: float,
) -> ArchiveQuadraticSurrogate:
    """Fit a local quadratic from archived samples inside a trust region."""

    points = np.asarray(archive_points, dtype=float)
    values = np.asarray(archive_values, dtype=float)
    region = np.asarray(region_bounds, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or values.ndim != 1 or values.shape[0] != points.shape[0]:
        return _rejected("shape_mismatch")
    if points.shape[0] < int(min_samples):
        return _rejected("insufficient_archive", int(points.shape[0]))
    if region.shape != (2, 2) or not np.all(np.isfinite(region)) or np.any(region[:, 1] <= region[:, 0]):
        return _rejected("invalid_region")

    inside = (
        np.all(points >= region[:, 0], axis=1)
        & np.all(points <= region[:, 1], axis=1)
        & np.isfinite(values)
    )
    sample_points = points[inside]
    sample_values = values[inside]
    if sample_points.shape[0] < int(min_samples):
        return _rejected("insufficient_region_samples", int(sample_points.shape[0]))

    center = np.asarray(center, dtype=float)
    widths = region[:, 1] - region[:, 0]
    normalized_points = (sample_points - region[:, 0]) / widths
    normalized_center = (center - region[:, 0]) / widths
    target = -sample_values if maximize else sample_values
    target_min = float(np.min(target))
    target_span = max(float(np.max(target) - target_min), 1e-12)
    normalized_targets = (target - target_min) / target_span
    distances = np.linalg.norm(normalized_points - normalized_center, axis=1)
    quality = 1.0 - np.clip(normalized_targets, 0.0, 1.0)
    weights = 0.25 + quality + 0.75 * np.exp(-0.5 * (distances / 0.35) ** 2)

    if sample_points.shape[0] > int(max_samples):
        scores = weights + quality
        limit = int(max_samples)
        chosen = np.argpartition(scores, -limit)[-limit:]
        chosen = chosen[np.argsort(scores[chosen])[::-1]]
        normalized_points = normalized_points[chosen]
        normalized_targets = normalized_targets[chosen]
        weights = weights[chosen]
        sample_points = sample_points[chosen]

    sample_count = int(normalized_points.shape[0])
    if sample_count < int(min_samples):
        return _rejected("insufficient_region_samples", sample_count)

    x = normalized_points[:, 0]
    y = normalized_points[:, 1]
    design = np.column_stack((np.ones_like(x), x, y, x * x, x * y, y * y))
    sqrt_weights = np.sqrt(np.maximum(weights, 1e-12))
    weighted_design = design * sqrt_weights[:, None]
    weighted_targets = normalized_targets * sqrt_weights
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
        eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    except np.linalg.LinAlgError:
        return _rejected("invalid_hessian", sample_count, condition)
    if not np.all(np.isfinite(eigenvalues)) or not np.all(np.isfinite(eigenvectors)):
        return _rejected("invalid_hessian", sample_count, condition)

    optimum: np.ndarray | None = None
    predicted_value: float | None = None
    if float(np.min(eigenvalues)) > 1e-12:
        try:
            normalized_optimum = -np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            normalized_optimum = np.full(2, np.nan, dtype=float)
        if np.all(np.isfinite(normalized_optimum)) and np.all(normalized_optimum >= 0.0) and np.all(normalized_optimum <= 1.0):
            optimum = region[:, 0] + normalized_optimum * widths
            x0, y0 = float(normalized_optimum[0]), float(normalized_optimum[1])
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

    return ArchiveQuadraticSurrogate(
        accepted=True,
        reason="accepted",
        sample_count=sample_count,
        region_bounds=region.copy(),
        point=None if optimum is None else np.asarray(optimum, dtype=float),
        predicted_value=None if predicted_value is None else float(predicted_value),
        condition=condition,
        coefficients=np.asarray(coefficients, dtype=float),
        hessian=hessian,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        target_min=target_min,
        target_span=target_span,
    )


def deterministic_candidate_pool(
    *,
    center: np.ndarray,
    active_bounds: np.ndarray,
    radius_fraction: float,
    pool_size: int,
    surrogate: ArchiveQuadraticSurrogate | None = None,
    anchors: tuple[np.ndarray, ...] = (),
) -> np.ndarray:
    """Build deterministic trust-region candidates around a center."""

    active = np.asarray(active_bounds, dtype=float)
    resolved_center = np.asarray(center, dtype=float)
    region = trust_region_bounds(resolved_center, float(radius_fraction), active)
    if region is None:
        return np.empty((0, 2), dtype=float)
    widths = active[:, 1] - active[:, 0]
    radius = float(radius_fraction) * widths
    candidates: list[np.ndarray] = []

    def add(point: np.ndarray) -> None:
        resolved = np.asarray(point, dtype=float)
        if resolved.shape != (2,) or not np.all(np.isfinite(resolved)):
            return
        if np.any(resolved < active[:, 0]) or np.any(resolved > active[:, 1]):
            return
        if np.any(resolved < region[:, 0]) or np.any(resolved > region[:, 1]):
            return
        candidates.append(resolved.copy())

    add(resolved_center)
    for anchor in anchors:
        add(anchor)
    if surrogate is not None and surrogate.point is not None:
        surrogate_point = np.asarray(surrogate.point, dtype=float)
        delta = surrogate_point - resolved_center
        for alpha in (0.25, 0.5, 0.75, 1.0, 1.25):
            add(resolved_center + alpha * delta)

    directions = [
        np.asarray([1.0, 0.0]),
        np.asarray([-1.0, 0.0]),
        np.asarray([0.0, 1.0]),
        np.asarray([0.0, -1.0]),
        np.asarray([1.0, 1.0]) / np.sqrt(2.0),
        np.asarray([1.0, -1.0]) / np.sqrt(2.0),
        np.asarray([-1.0, 1.0]) / np.sqrt(2.0),
        np.asarray([-1.0, -1.0]) / np.sqrt(2.0),
    ]
    if surrogate is not None and surrogate.accepted and surrogate.eigenvectors is not None:
        eigenvectors = np.asarray(surrogate.eigenvectors, dtype=float)
        if eigenvectors.shape == (2, 2) and np.all(np.isfinite(eigenvectors)):
            directions.extend([eigenvectors[:, 0], -eigenvectors[:, 0], eigenvectors[:, 1], -eigenvectors[:, 1]])
    for scale in (0.25, 0.5, 1.0):
        for direction in directions:
            norm = float(np.linalg.norm(direction))
            if np.isfinite(norm) and norm > 1e-12:
                add(resolved_center + scale * radius * direction / norm)

    golden = np.pi * (3.0 - np.sqrt(5.0))
    remaining = max(int(pool_size) * 3, 32)
    for index in range(remaining):
        angle = index * golden
        radial = np.sqrt((index + 0.5) / max(remaining, 1))
        direction = np.asarray([np.cos(angle), np.sin(angle)], dtype=float)
        add(resolved_center + radial * radius * direction)
        if len(candidates) >= int(pool_size) * 2:
            break
    if not candidates:
        return np.empty((0, 2), dtype=float)

    unique: list[np.ndarray] = []
    seen: set[tuple[str, str]] = set()
    for candidate in candidates:
        key = (float(candidate[0]).hex(), float(candidate[1]).hex())
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
        if len(unique) >= int(pool_size):
            break
    return np.vstack(unique).astype(float, copy=False)


def nearest_archive_distance(
    candidates: np.ndarray,
    archive_points: np.ndarray,
    active_bounds: np.ndarray,
) -> np.ndarray:
    """Return normalized nearest-sample distances for candidate uncertainty."""

    points = np.asarray(archive_points, dtype=float)
    candidates = np.asarray(candidates, dtype=float)
    if candidates.size == 0:
        return np.empty((0,), dtype=float)
    if points.size == 0:
        return np.ones(candidates.shape[0], dtype=float)
    widths = np.maximum(np.asarray(active_bounds, dtype=float)[:, 1] - np.asarray(active_bounds, dtype=float)[:, 0], 1e-300)
    delta = (candidates[:, None, :] - points[None, :, :]) / widths
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    nearest = np.min(distances, axis=1)
    return np.asarray(np.clip(nearest / 0.25, 0.0, 1.0), dtype=float)
