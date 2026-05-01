"""Archive-derived local geometry for point-cloud proposal regions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RegionGeometry:
    """An orthonormal local basis and capped axis shape in normalized space."""

    basis: np.ndarray
    axis_scales: np.ndarray
    eigenvalues: np.ndarray
    anisotropy: float
    sample_count: int
    reason: str

    @classmethod
    def identity(cls, reason: str = "identity") -> "RegionGeometry":
        return cls(
            basis=np.eye(2, dtype=float),
            axis_scales=np.ones(2, dtype=float),
            eigenvalues=np.ones(2, dtype=float),
            anisotropy=1.0,
            sample_count=0,
            reason=reason,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "basis": np.asarray(self.basis, dtype=float).tolist(),
            "axis_scales": np.asarray(self.axis_scales, dtype=float).tolist(),
            "eigenvalues": np.asarray(self.eigenvalues, dtype=float).tolist(),
            "anisotropy": float(self.anisotropy),
            "sample_count": int(self.sample_count),
            "reason": self.reason,
        }


def _stabilize_basis(basis: np.ndarray) -> np.ndarray:
    stabilized = np.asarray(basis, dtype=float).copy()
    for axis in range(stabilized.shape[1]):
        column = stabilized[:, axis]
        dominant = int(np.argmax(np.abs(column)))
        if column[dominant] < 0.0:
            stabilized[:, axis] *= -1.0
    if np.linalg.det(stabilized) < 0.0:
        stabilized[:, -1] *= -1.0
    return stabilized


def fit_region_geometry(
    points: np.ndarray,
    values: np.ndarray,
    *,
    bounds: np.ndarray,
    center: np.ndarray,
    radius_fraction: float,
    maximize: bool,
    min_samples: int,
    anisotropy_max: float,
) -> RegionGeometry:
    """Fit a weighted local covariance geometry in normalized coordinates."""

    sample_points = np.asarray(points, dtype=float)
    sample_values = np.asarray(values, dtype=float)
    search_bounds = np.asarray(bounds, dtype=float)
    if (
        sample_points.ndim != 2
        or sample_points.shape[1] != 2
        or sample_values.shape != (sample_points.shape[0],)
        or search_bounds.shape != (2, 2)
    ):
        return RegionGeometry.identity("shape_mismatch")
    if sample_points.shape[0] < int(min_samples):
        return RegionGeometry.identity("insufficient_samples")

    widths = search_bounds[:, 1] - search_bounds[:, 0]
    if np.any(widths <= 0.0) or not np.all(np.isfinite(widths)):
        return RegionGeometry.identity("invalid_bounds")

    normalized = (sample_points - search_bounds[:, 0]) / widths
    normalized_center = (np.asarray(center, dtype=float) - search_bounds[:, 0]) / widths
    finite = np.all(np.isfinite(normalized), axis=1) & np.isfinite(sample_values)
    if int(np.count_nonzero(finite)) < int(min_samples):
        return RegionGeometry.identity("insufficient_finite_samples")

    normalized = normalized[finite]
    sample_values = sample_values[finite]
    distances = np.linalg.norm(normalized - normalized_center, axis=1)
    locality_radius = max(float(radius_fraction) * 3.0, 0.08)
    local = distances <= locality_radius
    if int(np.count_nonzero(local)) < int(min_samples):
        nearest = np.argsort(distances)[: int(min_samples)]
    else:
        nearest = np.flatnonzero(local)

    local_points = normalized[nearest]
    local_values = sample_values[nearest]
    local_distances = distances[nearest]
    if local_points.shape[0] < int(min_samples):
        return RegionGeometry.identity("insufficient_local_samples")

    target = -local_values if maximize else local_values
    target_min = float(np.min(target))
    target_span = max(float(np.max(target) - target_min), 1e-12)
    quality = 1.0 - np.clip((target - target_min) / target_span, 0.0, 1.0)
    scale = max(float(np.median(local_distances)) * 1.5, locality_radius * 0.5, 1e-6)
    locality = np.exp(-0.5 * (local_distances / scale) ** 2)
    weights = 0.05 + quality + locality
    weight_sum = float(np.sum(weights))
    if not np.isfinite(weight_sum) or weight_sum <= 0.0:
        return RegionGeometry.identity("invalid_weights")

    mean = np.average(local_points, axis=0, weights=weights)
    centered = local_points - mean
    covariance = (centered * weights[:, None]).T @ centered / weight_sum
    covariance = 0.5 * (covariance + covariance.T)
    covariance += 1e-14 * np.eye(2, dtype=float)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    except np.linalg.LinAlgError:
        return RegionGeometry.identity("invalid_covariance")
    if not np.all(np.isfinite(eigenvalues)) or float(np.max(eigenvalues)) <= 0.0:
        return RegionGeometry.identity("degenerate_covariance")

    order = np.argsort(eigenvalues)[::-1]
    ordered_values = np.maximum(eigenvalues[order], 1e-14)
    basis = _stabilize_basis(eigenvectors[:, order])
    raw_anisotropy = float(np.sqrt(ordered_values[0] / ordered_values[-1]))
    anisotropy = min(max(raw_anisotropy, 1.0), float(anisotropy_max))
    eigenvalues_capped = np.asarray([ordered_values[0], ordered_values[0] / (anisotropy * anisotropy)])
    axis_scales = np.asarray([1.0, 1.0 / anisotropy], dtype=float)
    return RegionGeometry(
        basis=basis,
        axis_scales=axis_scales,
        eigenvalues=eigenvalues_capped,
        anisotropy=anisotropy,
        sample_count=int(local_points.shape[0]),
        reason="archive_covariance",
    )
