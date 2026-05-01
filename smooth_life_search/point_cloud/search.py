"""Archive-centered point-cloud SmoothLife optimizer."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Callable

import numpy as np

from ..core import SearchRun, normalize_bounds_2d
from ..smoothlife.config import SmoothLifeConfig
from .archive import PointCloudArchive
from .config import PointCloudSearchConfig
from .models import PointCloudBatchEvent, PointCloudRegion, PointCloudRegionEvent, PointCloudSnapshot
from .surrogate import QuadraticSurrogate, fit_quadratic_surrogate

Objective = Callable[[np.ndarray], float]


class PointCloudSmoothLifeSearch:
    """Optimize a 2D objective from a persistent point-cloud archive."""

    def __init__(
        self,
        objective: Objective,
        bounds: np.ndarray | list[tuple[float, float]],
        smoothlife_config: SmoothLifeConfig | None = None,
        point_cloud_config: PointCloudSearchConfig | None = None,
    ) -> None:
        self.objective = objective
        self.original_bounds = normalize_bounds_2d(bounds, owner="PointCloudSmoothLifeSearch")
        self.smoothlife_config = smoothlife_config or SmoothLifeConfig()
        self.config = point_cloud_config or PointCloudSearchConfig()
        self.rng = np.random.default_rng()
        self.archive = PointCloudArchive.empty()
        self.best_point = np.mean(self.original_bounds, axis=1)
        self.best_value = self._worst_value()
        self.local_best_point = self.best_point.copy()
        self.local_best_value = self.best_value
        self.batch_events: list[PointCloudBatchEvent] = []
        self.region_events: list[PointCloudRegionEvent] = []
        self.trust_region_events: list[dict[str, object]] = []
        self.snapshots: list[PointCloudSnapshot] = []
        self.regions: list[PointCloudRegion] = []
        self._next_region_id = 1
        self._active_max_evaluations: int | None = None
        self._batch_index = 0
        self._last_density = np.zeros(self.config.density_grid_shape, dtype=float)
        self._last_objective_field = np.zeros(self.config.density_grid_shape, dtype=float)
        self._last_evaluated_mask = np.zeros(self.config.density_grid_shape, dtype=bool)
        self._converged = False
        self.reset()

    def reset(self, seed: int | None = None) -> None:
        """Reset the archive, portfolio, diagnostics, and RNG."""

        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.archive = PointCloudArchive.empty()
        self.best_point = np.mean(self.original_bounds, axis=1)
        self.best_value = self._worst_value()
        self.local_best_point = self.best_point.copy()
        self.local_best_value = self.best_value
        self.batch_events = []
        self.region_events = []
        self.trust_region_events = []
        self.snapshots = []
        self.regions = []
        self._next_region_id = 1
        self._active_max_evaluations = None
        self._batch_index = 0
        self._converged = False
        self._last_density = np.zeros(self.config.density_grid_shape, dtype=float)
        self._last_objective_field = np.zeros(self.config.density_grid_shape, dtype=float)
        self._last_evaluated_mask = np.zeros(self.config.density_grid_shape, dtype=bool)

    def _evaluation_limit(self) -> int:
        limit = self._active_max_evaluations if self._active_max_evaluations is not None else self.config.max_evaluations
        if limit is None:
            raise ValueError("PointCloudSmoothLifeSearch requires max_evaluations or run(evaluations=...)")
        if limit <= 0:
            raise ValueError("evaluation limit must be positive")
        return int(limit)

    def _remaining(self) -> int:
        return max(self._evaluation_limit() - len(self.archive), 0)

    def _worst_value(self) -> float:
        return -np.inf if self.smoothlife_config.maximize else np.inf

    def _target(self, value: float) -> float:
        return -float(value) if self.smoothlife_config.maximize else float(value)

    def _is_better(self, candidate: float, incumbent: float) -> bool:
        tolerance = float(self.config.best_improvement_tolerance)
        if not np.isfinite(incumbent):
            return True
        if self.smoothlife_config.maximize:
            return float(candidate) > float(incumbent) + tolerance
        return float(candidate) < float(incumbent) - tolerance

    def _clip_point(self, point: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(point, dtype=float), self.original_bounds[:, 0], self.original_bounds[:, 1])

    def _evaluate_point(self, point: np.ndarray, *, source: str) -> tuple[float | None, bool, bool]:
        """Evaluate one unique point and update the incumbent."""

        clipped = self._clip_point(point)
        existing = self.archive.get(clipped)
        if existing is not None:
            return float(existing.value), False, False
        if self._remaining() <= 0:
            return None, False, False
        value = float(self.objective(clipped))
        sample = self.archive.add(clipped, value, source=source, batch_index=self._batch_index)
        if sample is None:
            return value, False, False
        improved = self._is_better(value, self.best_value)
        if improved:
            self.best_point = clipped.copy()
            self.best_value = value
        if self._is_better(value, self.local_best_value):
            self.local_best_point = clipped.copy()
            self.local_best_value = value
        return value, True, improved

    def _evaluate_candidates(self, candidates: list[tuple[np.ndarray, str]]) -> PointCloudBatchEvent:
        before = len(self.archive)
        best_before = float(self.best_value)
        source_counts: Counter[str] = Counter()
        improvements = 0
        region_improvements: Counter[int] = Counter()
        region_attempts: Counter[int] = Counter()
        for point, source in candidates:
            if self._remaining() <= 0:
                break
            value, added, improved = self._evaluate_point(point, source=source)
            if not added:
                continue
            source_counts[source] += 1
            if source.startswith("region:"):
                try:
                    region_id = int(source.split(":", 1)[1])
                except ValueError:
                    region_id = -1
                region_attempts[region_id] += 1
                if improved:
                    region_improvements[region_id] += 1
            if improved:
                improvements += 1
        after = len(self.archive)
        self._apply_region_feedback(region_attempts, region_improvements)
        return PointCloudBatchEvent(
            batch_index=self._batch_index,
            kind="candidate_batch",
            evaluations_before=before,
            evaluations_after=after,
            candidate_count=len(candidates),
            source_counts=dict(source_counts),
            best_before=best_before,
            best_after=float(self.best_value),
            best_point=self.best_point.copy(),
            improved=bool(improvements),
            diagnostics={"improvements": int(improvements), "archive_size": int(after)},
        )

    def _apply_region_feedback(self, attempts: Counter[int], improvements: Counter[int]) -> None:
        if not attempts:
            return
        region_by_id = {region.region_id: region for region in self.regions}
        for region_id, attempt_count in attempts.items():
            region = region_by_id.get(region_id)
            if region is None:
                continue
            radius_before = float(region.radius_fraction)
            success = improvements[region_id] > 0
            if success:
                region.successes += int(improvements[region_id])
                region.radius_fraction = min(
                    float(self.config.region_max_radius_fraction),
                    region.radius_fraction * float(self.config.region_expand_factor),
                )
            else:
                region.failures += int(attempt_count)
                region.radius_fraction = max(
                    float(self.config.region_min_radius_fraction),
                    region.radius_fraction * float(self.config.region_shrink_factor),
                )
            event = PointCloudRegionEvent(
                batch_index=self._batch_index,
                region_id=region.region_id,
                radius_before=radius_before,
                radius_after=float(region.radius_fraction),
                success=success,
                improvements=int(improvements[region_id]),
                center=region.center.copy(),
            )
            self.region_events.append(event)
            self.trust_region_events.append(event.to_dict() | {"phase": "portfolio", "kind": "region_batch"})

    def _initial_design(self) -> list[tuple[np.ndarray, str]]:
        lower = self.original_bounds[:, 0]
        upper = self.original_bounds[:, 1]
        center = np.mean(self.original_bounds, axis=1)
        anchors = [
            center,
            np.asarray([lower[0], lower[1]], dtype=float),
            np.asarray([lower[0], upper[1]], dtype=float),
            np.asarray([upper[0], lower[1]], dtype=float),
            np.asarray([upper[0], upper[1]], dtype=float),
            np.asarray([center[0], lower[1]], dtype=float),
            np.asarray([center[0], upper[1]], dtype=float),
            np.asarray([lower[0], center[1]], dtype=float),
            np.asarray([upper[0], center[1]], dtype=float),
        ]
        candidates: list[tuple[np.ndarray, str]] = [(point, "anchor") for point in anchors]
        random_count = max(0, int(self.config.initial_design_size) - len(candidates))
        if random_count:
            lhs = (np.arange(random_count, dtype=float)[:, None] + self.rng.random((random_count, 2))) / max(random_count, 1)
            for axis in range(2):
                self.rng.shuffle(lhs[:, axis])
            points = lower + lhs * (upper - lower)
            candidates.extend((point, "global") for point in points)
        return candidates[: int(self.config.initial_design_size)]

    def _normalize_values_to_desirability(self, values: np.ndarray) -> np.ndarray:
        finite = np.isfinite(values)
        if not np.any(finite):
            return np.zeros_like(values, dtype=float)
        target = -values if self.smoothlife_config.maximize else values
        min_value = float(np.min(target[finite]))
        max_value = float(np.max(target[finite]))
        span = max(max_value - min_value, 1e-12)
        desirability = 1.0 - (target - min_value) / span
        return np.clip(np.where(finite, desirability, 0.0), 0.0, 1.0)

    def _density_view(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = self.config.density_grid_shape
        points, values = self.archive.arrays()
        if points.size == 0:
            empty = np.zeros(shape, dtype=float)
            return empty, empty.copy(), np.zeros(shape, dtype=bool)

        height, width = shape
        xs = (np.arange(width, dtype=float) + 0.5) / width
        ys = (np.arange(height, dtype=float) + 0.5) / height
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        density = np.ones(shape, dtype=float) * 1e-6
        objective = np.zeros(shape, dtype=float)
        evaluated = np.zeros(shape, dtype=bool)
        bounds = self.original_bounds
        widths = bounds[:, 1] - bounds[:, 0]
        normalized_points = (points - bounds[:, 0]) / widths
        desirability = self._normalize_values_to_desirability(values)
        elite = self.archive.elite_indices(
            maximize=self.smoothlife_config.maximize,
            fraction=self.config.density_elite_fraction,
            minimum=min(4, len(self.archive)),
        )
        sigma = float(self.config.density_sigma_fraction)
        for index in elite:
            px, py = normalized_points[index]
            dist2 = (grid_x - px) ** 2 + (grid_y - py) ** 2
            bump = np.exp(-0.5 * dist2 / max(sigma * sigma, 1e-12))
            density += (0.25 + desirability[index]) * bump
            objective = np.maximum(objective, desirability[index] * bump)
        cols = np.clip((normalized_points[:, 0] * width).astype(int), 0, width - 1)
        rows = np.clip((normalized_points[:, 1] * height).astype(int), 0, height - 1)
        evaluated[rows, cols] = True
        objective[rows, cols] = np.maximum(objective[rows, cols], desirability)
        for _ in range(int(self.config.density_smooth_steps)):
            neighbors = (
                np.roll(density, 1, axis=0)
                + np.roll(density, -1, axis=0)
                + np.roll(density, 1, axis=1)
                + np.roll(density, -1, axis=1)
            ) * 0.25
            lap = neighbors - density
            density = np.clip(0.85 * density + 0.15 * neighbors + 0.05 * lap, 0.0, None)
        density = density / max(float(np.max(density)), 1e-12)
        objective = objective / max(float(np.max(objective)), 1e-12)
        return density, objective, evaluated

    def _refresh_views(self) -> None:
        self._last_density, self._last_objective_field, self._last_evaluated_mask = self._density_view()

    def _update_portfolio(self) -> None:
        points, values = self.archive.arrays()
        if points.size == 0:
            self.regions = []
            return
        elite_indices = self.archive.elite_indices(
            maximize=self.smoothlife_config.maximize,
            fraction=self.config.elite_fraction,
            minimum=min(self.config.portfolio_size, len(self.archive)),
        )
        old_by_nearest = list(self.regions)
        new_regions: list[PointCloudRegion] = []
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        min_separation = float(self.config.region_initial_radius_fraction) * 0.50
        target = -values if self.smoothlife_config.maximize else values
        best_target = float(np.min(target)) if target.size else 0.0
        span = max(float(np.max(target) - best_target), 1e-12) if target.size else 1.0
        for index in elite_indices:
            point = points[index]
            normalized_point = (point - self.original_bounds[:, 0]) / widths
            if any(np.linalg.norm(normalized_point - (region.center - self.original_bounds[:, 0]) / widths) < min_separation for region in new_regions):
                continue
            existing = self._nearest_old_region(point, old_by_nearest)
            score = 1.0 - (self._target(values[index]) - best_target) / span
            if existing is None:
                region = PointCloudRegion(
                    region_id=self._next_region_id,
                    center=point.copy(),
                    radius_fraction=float(self.config.region_initial_radius_fraction),
                    score=float(score),
                    best_value=float(values[index]),
                )
                self._next_region_id += 1
            else:
                region = replace(
                    existing,
                    center=0.65 * existing.center + 0.35 * point,
                    score=float(score),
                    best_value=float(values[index]),
                    age=existing.age + 1,
                )
            new_regions.append(region)
            if len(new_regions) >= int(self.config.portfolio_size):
                break
        self.regions = new_regions

    def _nearest_old_region(self, point: np.ndarray, regions: list[PointCloudRegion]) -> PointCloudRegion | None:
        if not regions:
            return None
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        normalized_point = (point - self.original_bounds[:, 0]) / widths
        distances = [np.linalg.norm(normalized_point - (region.center - self.original_bounds[:, 0]) / widths) for region in regions]
        index = int(np.argmin(distances))
        if float(distances[index]) <= float(self.config.region_initial_radius_fraction):
            return regions[index]
        return None

    def _candidate_counts(self, limit: int) -> dict[str, int]:
        fractions = {
            "global": float(self.config.global_candidate_fraction),
            "density": float(self.config.density_candidate_fraction),
            "region": float(self.config.region_candidate_fraction if self.config.trust_regions_enabled else 0.0),
            "exploit": float(self.config.exploit_candidate_fraction),
        }
        total = sum(fractions.values())
        counts = {key: int(np.floor(limit * value / total)) for key, value in fractions.items()}
        eligible = {key: value for key, value in fractions.items() if value > 0.0}
        while sum(counts.values()) < limit:
            key = max(eligible, key=lambda item: eligible[item] - counts[item] / max(limit, 1))
            counts[key] += 1
        return counts

    def _global_candidates(self, count: int) -> list[tuple[np.ndarray, str]]:
        if count <= 0:
            return []
        lower = self.original_bounds[:, 0]
        upper = self.original_bounds[:, 1]
        points = lower + self.rng.random((count, 2)) * (upper - lower)
        return [(point, "global") for point in points]

    def _density_candidates(self, count: int) -> list[tuple[np.ndarray, str]]:
        if count <= 0:
            return []
        density = np.asarray(self._last_density, dtype=float)
        flat = density.ravel()
        if not np.any(flat > 0.0):
            return self._global_candidates(count)
        probabilities = flat / float(np.sum(flat))
        chosen = self.rng.choice(flat.size, size=count, replace=True, p=probabilities)
        height, width = density.shape
        rows, cols = np.unravel_index(chosen, density.shape)
        jitter = self.rng.random((count, 2))
        normalized_x = (cols.astype(float) + jitter[:, 0]) / width
        normalized_y = (rows.astype(float) + jitter[:, 1]) / height
        lower = self.original_bounds[:, 0]
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        points = lower + np.column_stack((normalized_x, normalized_y)) * widths
        return [(point, "density") for point in points]

    def _region_candidates(self, count: int) -> list[tuple[np.ndarray, str]]:
        if count <= 0 or not self.regions:
            return []
        candidates: list[tuple[np.ndarray, str]] = []
        per_region = max(1, int(np.ceil(count / len(self.regions))))
        for region in self.regions:
            region_bounds = region.bounds(self.original_bounds)
            lower = region_bounds[:, 0]
            upper = region_bounds[:, 1]
            random_needed = max(1, per_region - 2)
            points = lower + self.rng.random((random_needed, 2)) * (upper - lower)
            candidates.extend((point, f"region:{region.region_id}") for point in points)
            if self.config.surrogate_enabled:
                surrogate = self._fit_region_surrogate(region)
                if surrogate.accepted and surrogate.point is not None:
                    candidates.append((surrogate.point.copy(), f"region:{region.region_id}"))
            candidates.append((region.center.copy(), f"region:{region.region_id}"))
            if len(candidates) >= count:
                break
        return candidates[:count]

    def _exploit_candidates(self, count: int) -> list[tuple[np.ndarray, str]]:
        if count <= 0 or not np.isfinite(self.best_value):
            return []
        candidates: list[tuple[np.ndarray, str]] = []
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        distances = self.archive.nearest_distance(np.asarray([self.best_point], dtype=float), self.original_bounds)
        local_scale = max(float(self.config.region_min_radius_fraction), min(float(self.config.local_refinement_step_fraction), float(distances[0]) if distances.size else 0.05))
        step = np.maximum(local_scale * widths, 1e-9 * widths)
        for axis in range(2):
            for sign in (-1.0, 1.0):
                point = self.best_point.copy()
                point[axis] += sign * step[axis]
                candidates.append((self._clip_point(point), "exploit"))
                if len(candidates) >= count:
                    return candidates
        while len(candidates) < count:
            point = self.best_point + self.rng.normal(0.0, step, size=2)
            candidates.append((self._clip_point(point), "exploit"))
        return candidates

    def _fit_region_surrogate(self, region: PointCloudRegion) -> QuadraticSurrogate:
        points, values = self.archive.arrays()
        return fit_quadratic_surrogate(
            points,
            values,
            region_bounds=region.bounds(self.original_bounds),
            center=region.center,
            maximize=self.smoothlife_config.maximize,
            min_samples=self.config.surrogate_min_samples,
            max_samples=self.config.surrogate_max_samples,
            regularization=self.config.surrogate_regularization,
            max_condition=self.config.surrogate_max_condition,
        )

    def _candidate_batch(self, limit: int) -> list[tuple[np.ndarray, str]]:
        counts = self._candidate_counts(limit)
        candidates: list[tuple[np.ndarray, str]] = []
        candidates.extend(self._global_candidates(counts["global"]))
        candidates.extend(self._density_candidates(counts["density"]))
        candidates.extend(self._region_candidates(counts["region"]))
        candidates.extend(self._exploit_candidates(counts["exploit"]))
        if len(candidates) < limit:
            candidates.extend(self._global_candidates(limit - len(candidates)))
        return candidates[:limit]

    def _finite_difference_gradient(self, point: np.ndarray) -> tuple[np.ndarray | None, int]:
        if self._remaining() <= 0:
            return None, 0
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        gradient = np.zeros(2, dtype=float)
        spent_before = len(self.archive)
        for axis in range(2):
            if self._remaining() <= 1:
                return None, len(self.archive) - spent_before
            step = max(1e-6 * widths[axis], 1e-8 * max(abs(float(point[axis])), 1.0))
            plus = point.copy()
            minus = point.copy()
            plus[axis] = min(self.original_bounds[axis, 1], plus[axis] + step)
            minus[axis] = max(self.original_bounds[axis, 0], minus[axis] - step)
            plus_value, _plus_added, _plus_improved = self._evaluate_point(plus, source="local_gradient")
            minus_value, _minus_added, _minus_improved = self._evaluate_point(minus, source="local_gradient")
            if plus_value is None or minus_value is None or plus[axis] == minus[axis]:
                return None, len(self.archive) - spent_before
            gradient[axis] = (self._target(plus_value) - self._target(minus_value)) / (plus[axis] - minus[axis])
        return gradient, len(self.archive) - spent_before

    def _run_local_refinement(self) -> PointCloudBatchEvent | None:
        if not self.config.local_refinement_enabled or len(self.archive) < int(self.config.local_refinement_start_evaluations):
            return None
        if not np.isfinite(self.best_value) or self._remaining() <= 4:
            return None
        before = len(self.archive)
        best_before = float(self.best_value)
        max_new = min(int(self.config.local_refinement_max_evaluations), self._remaining())
        hessian_inverse = np.eye(2, dtype=float)
        point = self.best_point.copy()
        value = float(self.best_value)
        gradient, _spent = self._finite_difference_gradient(point)
        if gradient is None:
            return None
        improvements = 0
        gradient_norm = float(np.linalg.norm(gradient))
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        max_step_norm = float(self.config.local_refinement_step_fraction) * float(np.linalg.norm(widths))
        while len(self.archive) - before < max_new and self._remaining() > 0:
            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm <= float(self.config.local_refinement_gradient_tolerance):
                self._converged = True
                break
            direction = -hessian_inverse @ gradient
            if not np.all(np.isfinite(direction)) or float(np.dot(direction, gradient)) >= 0.0:
                hessian_inverse = np.eye(2, dtype=float)
                direction = -gradient
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-14:
                self._converged = True
                break
            if norm > max_step_norm:
                direction = direction * (max_step_norm / norm)
            accepted = False
            alpha = 1.0
            accepted_point = point.copy()
            accepted_value = value
            directional = float(np.dot(gradient, direction))
            for _ in range(32):
                if self._remaining() <= 0 or len(self.archive) - before >= max_new:
                    break
                candidate = self._clip_point(point + alpha * direction)
                candidate_value, _added, improved = self._evaluate_point(candidate, source="local_refinement")
                if candidate_value is None:
                    break
                candidate_target = self._target(candidate_value)
                current_target = self._target(value)
                if candidate_target <= current_target + 1e-4 * alpha * directional or candidate_target < current_target:
                    accepted = True
                    accepted_point = candidate
                    accepted_value = float(candidate_value)
                    if improved:
                        improvements += 1
                    break
                alpha *= 0.5
                if alpha < 1e-12:
                    break
            if not accepted:
                hessian_inverse = np.eye(2, dtype=float)
                gradient, _spent = self._finite_difference_gradient(point)
                if gradient is None:
                    break
                continue
            next_gradient, _spent = self._finite_difference_gradient(accepted_point)
            if next_gradient is None:
                point = accepted_point
                value = accepted_value
                break
            step = accepted_point - point
            gradient_delta = next_gradient - gradient
            curvature = float(np.dot(gradient_delta, step))
            if curvature > 1e-12:
                rho = 1.0 / curvature
                identity = np.eye(2, dtype=float)
                hessian_inverse = (
                    (identity - rho * np.outer(step, gradient_delta))
                    @ hessian_inverse
                    @ (identity - rho * np.outer(gradient_delta, step))
                    + rho * np.outer(step, step)
                )
            else:
                hessian_inverse = np.eye(2, dtype=float)
            point = accepted_point
            value = accepted_value
            gradient = next_gradient
        after = len(self.archive)
        if after == before:
            return None
        event = PointCloudBatchEvent(
            batch_index=self._batch_index,
            kind="local_refinement",
            evaluations_before=before,
            evaluations_after=after,
            candidate_count=after - before,
            source_counts={
                "local_gradient": sum(1 for sample in self.archive.samples[before:] if sample.source == "local_gradient"),
                "local_refinement": sum(1 for sample in self.archive.samples[before:] if sample.source == "local_refinement"),
            },
            best_before=best_before,
            best_after=float(self.best_value),
            best_point=self.best_point.copy(),
            improved=self._is_better(self.best_value, best_before),
            diagnostics={
                "improvements": int(improvements),
                "gradient_norm": float(gradient_norm),
                "archive_size": int(after),
            },
        )
        if self.config.trust_regions_enabled:
            self.trust_region_events.append(
                {
                    "batch_index": int(self._batch_index),
                    "phase": "exploitation",
                    "kind": "local_refinement",
                    "trust_region_improvements": int(improvements),
                    "evaluations_before": int(before),
                    "evaluations_after": int(after),
                    "best_before": float(best_before),
                    "best_after": float(self.best_value),
                }
            )
        return event

    def _capture_snapshot(self, *, force: bool = False, candidate_points: np.ndarray | None = None) -> None:
        if not force and self._batch_index % int(self.config.snapshot_interval_batches) != 0:
            return
        self._refresh_views()
        points, values = self.archive.arrays()
        snapshot = PointCloudSnapshot(
            step_index=int(self._batch_index),
            bounds=self.original_bounds.copy(),
            density_field=self._last_density.copy(),
            objective_field=self._last_objective_field.copy(),
            evaluated_mask=self._last_evaluated_mask.copy(),
            best_point=self.best_point.copy(),
            best_value=float(self.best_value),
            local_best_point=self.local_best_point.copy(),
            local_best_value=float(self.local_best_value),
            box_best_point=self.best_point.copy(),
            box_best_value=float(self.best_value),
            archive_points=points.copy(),
            archive_values=values.copy(),
            region_bounds=[region.bounds(self.original_bounds) for region in self.regions],
            candidate_points=None if candidate_points is None else np.asarray(candidate_points, dtype=float).copy(),
            metadata={
                "mode": "point-cloud",
                "evaluations": int(len(self.archive)),
                "archive_size": int(len(self.archive)),
                "portfolio_size": int(len(self.regions)),
                "explored_fraction": min(1.0, len(self.archive) / max(self._evaluation_limit(), 1)),
            },
        )
        self.snapshots.append(snapshot)

    def run(self, evaluations: int | None = None) -> SearchRun:
        """Run until the evaluation budget is reached or local convergence is detected."""

        eval_limit = self.config.max_evaluations if evaluations is None else int(evaluations)
        if eval_limit is None:
            raise ValueError("PointCloudSmoothLifeSearch requires max_evaluations or run(evaluations=...)")
        self._active_max_evaluations = int(eval_limit)
        self._evaluate_point(np.mean(self.original_bounds, axis=1), source="center")
        self._capture_snapshot(force=True)
        if self._remaining() > 0:
            self._batch_index = 1
            initial = self._initial_design()
            event = self._evaluate_candidates(initial)
            self.batch_events.append(event)
            self._update_portfolio()
            self._capture_snapshot(force=True, candidate_points=np.asarray([point for point, _source in initial], dtype=float))

        while self._remaining() > 0 and not self._converged:
            if self.config.max_batches is not None and self._batch_index >= int(self.config.max_batches):
                break
            self._batch_index += 1
            self._refresh_views()
            self._update_portfolio()
            candidates = self._candidate_batch(min(int(self.config.batch_size), self._remaining()))
            event = self._evaluate_candidates(candidates)
            self.batch_events.append(event)
            self._update_portfolio()
            candidate_points = np.asarray([point for point, _source in candidates], dtype=float) if candidates else None
            self._capture_snapshot(candidate_points=candidate_points)
            refinement = self._run_local_refinement()
            if refinement is not None:
                self.batch_events.append(refinement)
                self._update_portfolio()
                self._capture_snapshot(force=True)
            if event.evaluations_after == event.evaluations_before and refinement is None:
                break
            if refinement is not None and self._converged:
                break

        if not self.snapshots:
            self._capture_snapshot(force=True)
        return SearchRun(
            best_point=self.best_point.copy(),
            best_value=float(self.best_value),
            evaluations=int(len(self.archive)),
            bounds=self.original_bounds.copy(),
            snapshots=list(self.snapshots),
            zoom_events=[],
            metadata={
                "mode": "point-cloud",
                "config": {
                    "batch_size": int(self.config.batch_size),
                    "initial_design_size": int(self.config.initial_design_size),
                    "portfolio_size": int(self.config.portfolio_size),
                    "trust_regions_enabled": bool(self.config.trust_regions_enabled),
                },
                "archive_size": int(len(self.archive)),
                "portfolio": [region.to_dict() for region in self.regions],
                "batch_events": [event.to_dict() for event in self.batch_events],
                "region_events": [event.to_dict() for event in self.region_events],
                "trust_region_events": list(self.trust_region_events),
                "converged": bool(self._converged),
            },
        )

    def samples(self) -> list[dict[str, object]]:
        """Return archive samples as JSON-compatible dictionaries."""

        return [sample.to_dict() for sample in self.archive.samples]
