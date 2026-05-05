"""Archive-centered point-cloud SmoothLife optimizer."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable

import numpy as np

from ..core import SearchRun, normalize_bounds_nd
from ..smoothlife.config import SmoothLifeConfig
from .archive import PointCloudArchive
from .config import PointCloudSearchConfig
from .geometry import RegionGeometry, fit_region_geometry
from .models import PointCloudBatchEvent, PointCloudRegion, PointCloudRegionEvent, PointCloudSnapshot
from .surrogate import QuadraticSurrogate, fit_quadratic_surrogate

Objective = Callable[[np.ndarray], float]


@dataclass(slots=True)
class SourceStats:
    """Adaptive credit record for one generated candidate source family."""

    attempts: int = 0
    evaluations: int = 0
    improvements: int = 0
    improvement_sum: float = 0.0
    ema_credit: float = 0.0
    last_improvement_batch: int = 0


@dataclass(slots=True)
class CMARegionState:
    """Lightweight region-local covariance adaptation state."""

    sigma: float
    diagonal_variance: np.ndarray
    evolution_path: np.ndarray
    directions: list[np.ndarray]
    success_rate_ema: float = 0.0


@dataclass(slots=True)
class CandidateProposal:
    """One generated proposal with optimizer metadata."""

    point: np.ndarray
    source: str
    family: str
    parent_key: tuple[str, ...] | None = None
    region_id: int | None = None
    axes: tuple[int, ...] | None = None
    shade_params: tuple[float, float] | None = None
    predicted_score: float | None = None

    def __iter__(self):
        yield self.point
        yield self.source

    def __getitem__(self, index: int):
        if index == 0:
            return self.point
        if index == 1:
            return self.source
        raise IndexError(index)


@dataclass(slots=True)
class GradientResult:
    """Finite-difference gradient plus recentering diagnostics."""

    gradient: np.ndarray | None
    spent: int
    recentered: bool
    point: np.ndarray
    value: float
    improvement_count: int = 0


class PointCloudSmoothLifeSearch:
    """Optimize an N-D objective from a persistent point-cloud archive."""

    def __init__(
        self,
        objective: Objective,
        bounds: np.ndarray | list[tuple[float, float]],
        smoothlife_config: SmoothLifeConfig | None = None,
        point_cloud_config: PointCloudSearchConfig | None = None,
    ) -> None:
        self.objective = objective
        self.original_bounds = normalize_bounds_nd(bounds, owner="PointCloudSmoothLifeSearch")
        self.dimension = int(self.original_bounds.shape[0])
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
        self._local_refinement_stalled = False
        self._stop_reason = "not_started"
        self._global_sequence_index = 0
        self._stencil_step_fraction = float(self.config.region_initial_radius_fraction)
        self._last_successful_step = np.zeros(self.dimension, dtype=float)
        self._axis_activity = np.zeros(self.dimension, dtype=float)
        self._block_cursor = 0
        self._block_hessian_inverse: dict[tuple[int, ...], np.ndarray] = {}
        self._source_stats: dict[str, SourceStats] = {}
        self._shade_f_memory = np.full(int(self.config.shade_memory_size), 0.5, dtype=float)
        self._shade_cr_memory = np.full(int(self.config.shade_memory_size), 0.9, dtype=float)
        self._shade_memory_index = 0
        self._cma_states: dict[int, CMARegionState] = {}
        self._restart_lattice_index = 0
        self._restart_axis_shift = np.zeros(self.dimension, dtype=float)
        self._last_improvement_batch = 0
        self._live_population_keys: list[tuple[str, ...]] = []
        self._anchor_baseline_target: float | None = None
        self._successful_directions: list[np.ndarray] = []
        self._linkage_scores = np.zeros((self.dimension, self.dimension), dtype=float)
        self._linkage_last_update_batch = -1
        self._lbfgs_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        self._probe_recenters_total = 0
        self._axis_coverage = np.zeros(self.dimension, dtype=float)
        self._cooperative_group_cursor = 0
        self._active_set_expansions = 0
        self._active_set_last_expand_batch = -1
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
        self._local_refinement_stalled = False
        self._stop_reason = "not_started"
        self._global_sequence_index = 0
        self._stencil_step_fraction = float(self.config.region_initial_radius_fraction)
        self._last_successful_step = np.zeros(self.dimension, dtype=float)
        self._axis_activity = np.zeros(self.dimension, dtype=float)
        self._block_cursor = 0
        self._block_hessian_inverse = {}
        self._source_stats = {}
        self._shade_f_memory = np.full(int(self.config.shade_memory_size), 0.5, dtype=float)
        self._shade_cr_memory = np.full(int(self.config.shade_memory_size), 0.9, dtype=float)
        self._shade_memory_index = 0
        self._cma_states = {}
        self._restart_lattice_index = 0
        self._restart_axis_shift = self.rng.random(self.dimension)
        self._last_improvement_batch = 0
        self._live_population_keys = []
        self._anchor_baseline_target = None
        self._successful_directions = []
        self._linkage_scores = np.zeros((self.dimension, self.dimension), dtype=float)
        self._linkage_last_update_batch = -1
        self._lbfgs_pairs = []
        self._probe_recenters_total = 0
        self._axis_coverage = np.zeros(self.dimension, dtype=float)
        self._cooperative_group_cursor = 0
        self._active_set_expansions = 0
        self._active_set_last_expand_batch = -1
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

    def _high_dimensional_refinement_active(self) -> bool:
        return bool(
            self.config.high_dimensional_refinement_enabled
            and self.config.local_refinement_enabled
            and self.dimension >= int(self.config.high_dimensional_min_dimension)
            and (self.dimension >= 30 or self._basin_polishing_active())
        )

    def _dimension_scaled_batches_active(self) -> bool:
        return bool(
            self.config.dimension_scaled_batches_enabled
            and self.dimension >= int(self.config.high_dimensional_min_dimension)
        )

    def _effective_batch_size(self) -> int:
        base = int(self.config.batch_size)
        if not self._dimension_scaled_batches_active():
            return base
        return min(int(self.config.dimension_scaled_batch_max), max(base, 2 * self.dimension))

    def _evolutionary_search_active(self) -> bool:
        return self.dimension >= int(self.config.high_dimensional_min_dimension)

    def _cooperative_refinement_active(self) -> bool:
        return bool(
            self.config.cooperative_refinement_enabled
            and self._evolutionary_search_active()
            and self.dimension >= int(self.config.cooperative_min_dimension)
            and np.isfinite(self.best_value)
        )

    @staticmethod
    def _source_family(source: str) -> str:
        if source.startswith("cooperative:"):
            return "cooperative"
        if source.startswith("region:") and ":cma" in source:
            return "cma"
        if source.startswith("region:"):
            return "region"
        if source.startswith("restart:"):
            return "restart"
        if source in {"density", "smoothlife_density"}:
            return "smoothlife_density"
        if source.startswith("exploit"):
            return "exploit"
        return source

    def _proposal(
        self,
        point: np.ndarray,
        source: str,
        *,
        parent_key: tuple[str, ...] | None = None,
        region_id: int | None = None,
        axes: tuple[int, ...] | None = None,
        shade_params: tuple[float, float] | None = None,
        predicted_score: float | None = None,
    ) -> CandidateProposal:
        return CandidateProposal(
            point=np.asarray(point, dtype=float),
            source=str(source),
            family=self._source_family(str(source)),
            parent_key=parent_key,
            region_id=region_id,
            axes=None if axes is None else tuple(int(axis) for axis in axes),
            shade_params=shade_params,
            predicted_score=predicted_score,
        )

    def _as_proposal(self, candidate: CandidateProposal | tuple[np.ndarray, str]) -> CandidateProposal:
        if isinstance(candidate, CandidateProposal):
            return candidate
        point, source = candidate
        return self._proposal(point, source)

    def _nearest_archive_key(self, point: np.ndarray) -> tuple[str, ...] | None:
        points, _values = self.archive.arrays()
        if points.size == 0:
            return None
        widths = np.maximum(self.original_bounds[:, 1] - self.original_bounds[:, 0], 1e-12)
        normalized_points = (points - self.original_bounds[:, 0]) / widths
        normalized_point = (np.asarray(point, dtype=float) - self.original_bounds[:, 0]) / widths
        index = int(np.argmin(np.linalg.norm(normalized_points - normalized_point, axis=1)))
        return PointCloudArchive.key(points[index])

    def _source_stat(self, source: str) -> SourceStats:
        family = self._source_family(source)
        if family not in self._source_stats:
            self._source_stats[family] = SourceStats()
        return self._source_stats[family]

    def _record_source_results(
        self,
        attempts: Counter[str],
        evaluations: Counter[str],
        improvements: Counter[str],
        improvement_amounts: Counter[str],
    ) -> None:
        families = {
            self._source_family(source)
            for source in set(attempts) | set(evaluations) | set(improvements) | set(improvement_amounts)
        }
        for family in families:
            attempted = sum(count for source, count in attempts.items() if self._source_family(source) == family)
            evaluated = sum(count for source, count in evaluations.items() if self._source_family(source) == family)
            improved = sum(count for source, count in improvements.items() if self._source_family(source) == family)
            amount = sum(value for source, value in improvement_amounts.items() if self._source_family(source) == family)
            stats = self._source_stat(family)
            stats.attempts += int(attempted)
            stats.evaluations += int(evaluated)
            stats.improvements += int(improved)
            stats.improvement_sum += float(amount)
            if improved > 0:
                stats.last_improvement_batch = int(self._batch_index)
            batch_credit = float(np.log1p(max(float(amount), 0.0))) / max(int(evaluated), 1)
            stats.ema_credit = 0.85 * float(stats.ema_credit) + 0.15 * batch_credit

    def _source_stats_payload(self) -> dict[str, dict[str, float | int]]:
        return {
            source: {
                "attempts": int(stats.attempts),
                "evaluations": int(stats.evaluations),
                "improvements": int(stats.improvements),
                "improvement_sum": float(stats.improvement_sum),
                "ema_credit": float(stats.ema_credit),
                "last_improvement_batch": int(stats.last_improvement_batch),
            }
            for source, stats in sorted(self._source_stats.items())
        }

    def _record_successful_direction(self, before: np.ndarray, after: np.ndarray) -> None:
        widths = np.maximum(self.original_bounds[:, 1] - self.original_bounds[:, 0], 1e-12)
        delta = (np.asarray(after, dtype=float) - np.asarray(before, dtype=float)) / widths
        norm = float(np.linalg.norm(delta))
        if norm <= 1e-14 or not np.isfinite(norm):
            return
        direction = delta / norm
        self._successful_directions.append(direction.copy())
        del self._successful_directions[:-int(self.config.successful_direction_memory_size)]
        self._linkage_scores *= 0.995
        self._linkage_scores += np.outer(np.abs(direction), np.abs(direction))
        np.fill_diagonal(self._linkage_scores, 0.0)

    def _refresh_anchor_baseline_target(self) -> None:
        anchor_targets = [
            self._target(sample.value)
            for sample in self.archive.samples
            if sample.source in {"center", "anchor"} and np.isfinite(sample.value)
        ]
        self._anchor_baseline_target = None if not anchor_targets else float(min(anchor_targets))

    def _basin_polishing_active(self) -> bool:
        if not (
            self.config.basin_polishing_enabled
            and self.dimension >= int(self.config.basin_polishing_min_dimension)
            and self._evolutionary_search_active()
            and np.isfinite(self.best_value)
        ):
            return False
        baseline = self._anchor_baseline_target
        if baseline is None or not np.isfinite(baseline) or abs(float(baseline)) <= 1e-12:
            return False
        current = self._target(self.best_value)
        if not np.isfinite(current):
            return False
        if float(baseline) > 0.0:
            return current <= float(self.config.basin_polishing_activation_ratio) * float(baseline)
        return current <= float(baseline) - abs(float(baseline)) * float(self.config.basin_polishing_activation_ratio)

    def _probe_recenter_improvement_is_meaningful(self, current_value: float, candidate_value: float) -> bool:
        if not self._evolutionary_search_active():
            return False
        if not self._basin_polishing_active():
            return False
        amount = self._target(current_value) - self._target(candidate_value)
        relative_floor = 1e-3 if self._evolutionary_search_active() else 1e-6 * max(abs(self._target(current_value)), 1.0)
        threshold = max(float(self.config.best_improvement_tolerance), relative_floor)
        return bool(np.isfinite(amount) and amount >= threshold)

    def _update_linkage_scores(self, *, force: bool = False) -> None:
        if not self.config.linkage_blocks_enabled or self.dimension <= 1:
            return
        if (
            not force
            and self._linkage_last_update_batch >= 0
            and int(self._batch_index) - int(self._linkage_last_update_batch)
            < int(self.config.linkage_update_interval_batches)
        ):
            return
        self._linkage_last_update_batch = int(self._batch_index)
        if len(self.archive) < max(4, int(self.config.region_geometry_min_samples)):
            return
        points, values = self.archive.arrays()
        if points.size == 0:
            return
        target = -values if self.smoothlife_config.maximize else values
        order = np.argsort(target)
        count = min(points.shape[0], max(8, min(128, points.shape[0] // 2)))
        selected = points[order[:count]]
        widths = np.maximum(self.original_bounds[:, 1] - self.original_bounds[:, 0], 1e-12)
        normalized = (selected - self.original_bounds[:, 0]) / widths
        if normalized.shape[0] < 3:
            return
        centered = normalized - np.mean(normalized, axis=0)
        scale = np.std(centered, axis=0) + 1e-12
        corr = np.abs((centered / scale).T @ (centered / scale)) / max(normalized.shape[0] - 1, 1)
        corr = np.clip(corr, 0.0, 1e6)
        np.fill_diagonal(corr, 0.0)
        self._linkage_scores = 0.75 * self._linkage_scores + 0.25 * corr

    def _linked_block(self) -> np.ndarray | None:
        if not self.config.linkage_blocks_enabled or self.dimension <= 1:
            return None
        self._update_linkage_scores()
        scores = np.asarray(self._linkage_scores, dtype=float)
        if scores.shape != (self.dimension, self.dimension) or not np.any(scores > 0.0):
            return None
        block_size = self._block_size()
        activity = np.asarray(self._axis_activity, dtype=float)
        if np.any(activity > 0.0):
            anchor = int(np.argmax(activity))
        else:
            anchor = int(np.argmax(np.sum(scores, axis=1)))
        neighbor_order = np.argsort(scores[anchor])[::-1]
        axes = [anchor]
        for axis in neighbor_order:
            resolved = int(axis)
            if resolved == anchor:
                continue
            axes.append(resolved)
            if len(axes) >= min(block_size, 1 + int(self.config.linkage_neighbor_count)):
                break
        if len(axes) < block_size:
            for axis in np.argsort(activity)[::-1]:
                resolved = int(axis)
                if resolved not in axes:
                    axes.append(resolved)
                if len(axes) >= block_size:
                    break
        return np.asarray(sorted(set(axes))[:block_size], dtype=int)

    def _cooperative_group_size(self) -> int:
        configured = self.config.cooperative_group_size
        base = int(self.config.active_subspace_size) if configured is None else int(configured)
        return min(self.dimension, max(1, base))

    @staticmethod
    def _unit_scaled(values: np.ndarray) -> np.ndarray:
        resolved = np.asarray(values, dtype=float)
        if resolved.size == 0:
            return resolved
        resolved = np.where(np.isfinite(resolved), resolved, 0.0)
        span = max(float(np.max(resolved) - np.min(resolved)), 1e-12)
        return (resolved - float(np.min(resolved))) / span

    def _elite_axis_variance(self) -> np.ndarray:
        if len(self.archive) < 4:
            return np.zeros(self.dimension, dtype=float)
        points, values = self.archive.arrays()
        if points.size == 0:
            return np.zeros(self.dimension, dtype=float)
        target = -values if self.smoothlife_config.maximize else values
        order = np.argsort(target)
        count = min(points.shape[0], max(4, min(128, points.shape[0] // 2)))
        normalized = (points[order[:count]] - self.original_bounds[:, 0]) / (
            self.original_bounds[:, 1] - self.original_bounds[:, 0]
        )
        return np.var(normalized, axis=0)

    def _cooperative_axis_scores(self) -> np.ndarray:
        scores = np.zeros(self.dimension, dtype=float)
        scores += 0.35 * self._unit_scaled(self._axis_activity)
        if self._successful_directions:
            directions = np.asarray(
                self._successful_directions[-int(self.config.successful_direction_memory_size) :],
                dtype=float,
            )
            scores += 0.25 * self._unit_scaled(np.mean(np.abs(directions), axis=0))
        if self._linkage_scores.shape == (self.dimension, self.dimension):
            scores += 0.20 * self._unit_scaled(np.sum(np.asarray(self._linkage_scores, dtype=float), axis=1))
        scores += 0.10 * self._unit_scaled(self._elite_axis_variance())
        coverage_pressure = 1.0 / (1.0 + np.asarray(self._axis_coverage, dtype=float))
        scores += 0.10 * self._unit_scaled(coverage_pressure)
        if not np.any(scores > 0.0):
            scores = coverage_pressure
        return np.where(np.isfinite(scores), scores, 0.0)

    def _active_set_axes(self) -> np.ndarray:
        group_size = self._cooperative_group_size()
        cap = min(
            self.dimension,
            max(group_size, int(np.ceil(float(self.config.active_set_max_fraction) * self.dimension))),
        )
        scores = self._cooperative_axis_scores()
        selected: list[int] = []

        def add(axis: int) -> None:
            resolved = int(axis) % self.dimension
            if resolved not in selected:
                selected.append(resolved)

        top_count = min(cap, max(group_size, cap // 2))
        for axis in np.argsort(scores)[::-1][:top_count]:
            add(int(axis))
        coverage_count = min(cap - len(selected), max(1, cap // 4))
        for axis in np.argsort(self._axis_coverage)[:coverage_count]:
            add(int(axis))
        should_expand = (
            self._active_set_last_expand_batch < 0
            or int(self._batch_index) - int(self._active_set_last_expand_batch)
            >= int(self.config.active_set_expand_interval_batches)
        )
        if should_expand:
            self._active_set_last_expand_batch = int(self._batch_index)
            self._active_set_expansions += 1
        offset = int(self._active_set_expansions) % max(self.dimension, 1)
        spread = (np.linspace(0, self.dimension - 1, num=cap, dtype=int) + offset) % self.dimension
        for axis in spread:
            add(int(axis))
            if len(selected) >= cap:
                break
        return np.asarray(sorted(selected[:cap]), dtype=int)

    def _cooperative_group_from_seed(
        self,
        seed_axes: list[int],
        active_axes: np.ndarray,
        scores: np.ndarray,
    ) -> np.ndarray:
        group_size = self._cooperative_group_size()
        axes: list[int] = []

        def add(axis: int) -> None:
            resolved = int(axis) % self.dimension
            if resolved not in axes:
                axes.append(resolved)

        for axis in seed_axes:
            add(axis)
        anchor = axes[0] if axes else int(active_axes[0]) if active_axes.size else 0
        if self._linkage_scores.shape == (self.dimension, self.dimension):
            linkage = np.asarray(self._linkage_scores[anchor], dtype=float)
            for axis in np.argsort(linkage)[::-1]:
                if float(linkage[int(axis)]) <= 0.0:
                    break
                add(int(axis))
                if len(axes) >= min(group_size, 1 + int(self.config.linkage_neighbor_count)):
                    break
        half = group_size // 2
        start = max(0, min(anchor - half, self.dimension - group_size))
        for axis in range(start, min(self.dimension, start + group_size)):
            add(axis)
            if len(axes) >= group_size:
                break
        if len(axes) < group_size:
            for axis in active_axes[np.argsort(scores[active_axes])[::-1]]:
                add(int(axis))
                if len(axes) >= group_size:
                    break
        if len(axes) < group_size:
            for axis in np.argsort(self._axis_coverage):
                add(int(axis))
                if len(axes) >= group_size:
                    break
        return np.asarray(sorted(axes[:group_size]), dtype=int)

    def _cooperative_groups(self, count: int | None = None) -> list[np.ndarray]:
        if not self._cooperative_refinement_active():
            return []
        target_count = max(1, int(self.config.cooperative_groups_per_batch if count is None else count))
        active_axes = self._active_set_axes()
        if active_axes.size == 0:
            active_axes = np.arange(self.dimension, dtype=int)
        scores = self._cooperative_axis_scores()
        groups: list[np.ndarray] = []
        seen: set[tuple[int, ...]] = set()

        def add_group(axes: np.ndarray) -> None:
            key = tuple(int(axis) for axis in np.asarray(axes, dtype=int))
            if len(key) == 0 or key in seen:
                return
            seen.add(key)
            groups.append(np.asarray(key, dtype=int))

        top_axes = [int(axis) for axis in active_axes[np.argsort(scores[active_axes])[::-1]]]
        top_limit = 1
        for axis in top_axes[:top_limit]:
            add_group(self._cooperative_group_from_seed([axis], active_axes, scores))
            if len(groups) >= target_count:
                return groups

        if top_axes:
            active_front = min(
                self.dimension - 1,
                max(top_axes[: min(len(top_axes), self._cooperative_group_size())]) + 1,
            )
            add_group(self._cooperative_group_from_seed([active_front], active_axes, scores))
            if len(groups) >= target_count:
                return groups

        low_coverage = [int(axis) for axis in np.argsort(self._axis_coverage)[:target_count]]
        for axis in low_coverage:
            add_group(self._cooperative_group_from_seed([axis], active_axes, scores))
            if len(groups) >= target_count:
                return groups

        group_size = self._cooperative_group_size()
        while len(groups) < target_count:
            start = (int(self._cooperative_group_cursor) * group_size) % self.dimension
            self._cooperative_group_cursor += 1
            window = (np.arange(start, start + group_size, dtype=int) % self.dimension).tolist()
            add_group(self._cooperative_group_from_seed(window, active_axes, scores))
            if self._cooperative_group_cursor > target_count + self.dimension:
                break
        return groups

    def _store_lbfgs_pair(self, step: np.ndarray, gradient_delta: np.ndarray) -> None:
        if not self.config.cross_block_lbfgs_enabled:
            return
        s = np.asarray(step, dtype=float)
        y = np.asarray(gradient_delta, dtype=float)
        if s.shape != (self.dimension,) or y.shape != (self.dimension,):
            return
        curvature = float(np.dot(s, y))
        if curvature <= 1e-12 or not np.isfinite(curvature):
            return
        self._lbfgs_pairs.append((s.copy(), y.copy()))
        del self._lbfgs_pairs[:-int(self.config.cross_block_lbfgs_memory_size)]

    def _lbfgs_direction(self, gradient: np.ndarray) -> np.ndarray | None:
        if not self.config.cross_block_lbfgs_enabled or not self._lbfgs_pairs:
            return None
        q = np.asarray(gradient, dtype=float).copy()
        alphas: list[float] = []
        rhos: list[float] = []
        for s, y in reversed(self._lbfgs_pairs):
            curvature = float(np.dot(s, y))
            if curvature <= 1e-12 or not np.isfinite(curvature):
                continue
            rho = 1.0 / curvature
            alpha = rho * float(np.dot(s, q))
            q = q - alpha * y
            alphas.append(alpha)
            rhos.append(rho)
        if not alphas:
            return None
        s_last, y_last = self._lbfgs_pairs[-1]
        scale = float(np.dot(s_last, y_last)) / max(float(np.dot(y_last, y_last)), 1e-12)
        r = np.clip(scale, 1e-8, 1e8) * q
        for (s, y), alpha, rho in zip(self._lbfgs_pairs, reversed(alphas), reversed(rhos)):
            beta = rho * float(np.dot(y, r))
            r = r + s * (alpha - beta)
        direction = -r
        if not np.all(np.isfinite(direction)) or float(np.dot(direction, gradient)) >= 0.0:
            return None
        return direction

    def _population_limit(self) -> int:
        configured = self.config.evolutionary_population_size
        if configured is not None:
            return int(configured)
        return min(
            int(self.config.evolutionary_population_max),
            max(4 * self.dimension, int(self.config.initial_design_size), 64),
        )

    def _sample_by_key(self, key: tuple[str, ...] | None):
        if key is None:
            return None
        index = self.archive._keys.get(key)
        if index is None:
            return None
        return self.archive.samples[index]

    def _refresh_live_population(self) -> None:
        if not self._evolutionary_search_active() or len(self.archive) == 0:
            self._live_population_keys = []
            return
        points, values = self.archive.arrays()
        limit = min(self._population_limit(), len(self.archive))
        if limit <= 0:
            self._live_population_keys = []
            return
        target = -values if self.smoothlife_config.maximize else values
        order = np.argsort(target)
        elite_count = min(limit, max(1, min(limit // 2, 64)))
        chosen: list[int] = [int(index) for index in order[:elite_count]]
        if len(chosen) < limit:
            normalized = (points - self.original_bounds[:, 0]) / (self.original_bounds[:, 1] - self.original_bounds[:, 0])
            chosen_set = set(chosen)
            remaining = np.asarray([int(index) for index in order if int(index) not in chosen_set], dtype=int)
            if remaining.size > 0:
                selected = normalized[chosen]
                remaining_points = normalized[remaining]
                distances = np.min(
                    np.linalg.norm(remaining_points[:, None, :] - selected[None, :, :], axis=2),
                    axis=1,
                )
                diverse_order = remaining[np.argsort(distances)[::-1]]
                chosen.extend(int(index) for index in diverse_order[: limit - len(chosen)])
        self._live_population_keys = [PointCloudArchive.key(points[index]) for index in chosen[:limit]]

    def _live_population_arrays(self) -> tuple[np.ndarray, np.ndarray, list[tuple[str, ...]]]:
        if not self._live_population_keys:
            self._refresh_live_population()
        samples = [self._sample_by_key(key) for key in self._live_population_keys]
        live_samples = [sample for sample in samples if sample is not None]
        if not live_samples:
            return np.empty((0, self.dimension), dtype=float), np.empty((0,), dtype=float), []
        return (
            np.vstack([sample.point for sample in live_samples]).astype(float, copy=False),
            np.asarray([sample.value for sample in live_samples], dtype=float),
            [PointCloudArchive.key(sample.point) for sample in live_samples],
        )

    def _cma_states_payload(self) -> dict[str, dict[str, object]]:
        return {
            str(region_id): {
                "sigma": float(state.sigma),
                "success_rate_ema": float(state.success_rate_ema),
                "direction_count": int(len(state.directions)),
                "diagonal_variance_mean": float(np.mean(state.diagonal_variance)),
                "diagonal_variance_max": float(np.max(state.diagonal_variance)),
            }
            for region_id, state in sorted(self._cma_states.items())
        }

    def _projection_axes(self) -> tuple[int, int]:
        if self.config.projection_axes is None:
            return 0, 1
        first, second = int(self.config.projection_axes[0]), int(self.config.projection_axes[1])
        if first >= self.dimension or second >= self.dimension:
            raise ValueError("projection axes must be valid for the point-cloud dimension")
        return first, second

    def _coordinate_projection_basis(self, axes: tuple[int, int]) -> np.ndarray:
        basis = np.zeros((self.dimension, 2), dtype=float)
        basis[axes[0], 0] = 1.0
        basis[axes[1], 1] = 1.0
        return basis

    def _density_projection_frame(self) -> tuple[str, tuple[int, int], np.ndarray, np.ndarray]:
        axes = self._projection_axes()
        coordinate_frame = (
            "coordinate",
            axes,
            np.zeros(self.dimension, dtype=float),
            self._coordinate_projection_basis(axes),
        )
        if self.config.projection_axes is not None or self.dimension == 2:
            return coordinate_frame
        candidates = [
            region
            for region in self._active_regions()
            if region.geometry_reason == "archive_covariance"
            and np.asarray(region.geometry_basis, dtype=float).shape[0] == self.dimension
            and np.asarray(region.geometry_basis, dtype=float).shape[1] >= 2
        ]
        if not candidates:
            return coordinate_frame
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        best_normalized = self._normalized_point(self.best_point)
        region = min(
            candidates,
            key=lambda candidate: float(
                np.linalg.norm((candidate.center - self.original_bounds[:, 0]) / widths - best_normalized)
            ),
        )
        return (
            "active_subspace",
            axes,
            self._normalized_point(region.center),
            np.asarray(region.geometry_basis, dtype=float)[:, :2],
        )

    @staticmethod
    def _project_normalized_points(
        normalized_points: np.ndarray,
        frame: tuple[str, tuple[int, int], np.ndarray, np.ndarray],
    ) -> np.ndarray:
        kind, axes, origin, basis = frame
        if kind == "coordinate":
            return normalized_points[:, axes]
        return 0.5 + (normalized_points - origin) @ basis

    def _lift_density_points(
        self,
        projected_points: np.ndarray,
        frame: tuple[str, tuple[int, int], np.ndarray, np.ndarray],
    ) -> np.ndarray:
        kind, axes, origin, basis = frame
        uv = np.asarray(projected_points, dtype=float)
        if kind == "coordinate":
            base = self._normalized_point(
                self.best_point if np.all(np.isfinite(self.best_point)) else np.mean(self.original_bounds, axis=1)
            )
            normalized = np.tile(base, (uv.shape[0], 1))
            normalized[:, axes[0]] = uv[:, 0]
            normalized[:, axes[1]] = uv[:, 1]
            return normalized
        return np.clip(origin + (uv - 0.5) @ basis.T, 0.0, 1.0)

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
            previous_best = self.best_point.copy()
            self.best_point = clipped.copy()
            self.best_value = value
            self._last_successful_step = clipped - previous_best
            self._record_successful_direction(previous_best, clipped)
            self._local_refinement_stalled = False
            self._last_improvement_batch = int(self._batch_index)
        if self._is_better(value, self.local_best_value):
            self.local_best_point = clipped.copy()
            self.local_best_value = value
        return value, True, improved

    def _evaluate_candidates(
        self,
        candidates: list[CandidateProposal | tuple[np.ndarray, str]],
    ) -> PointCloudBatchEvent:
        proposals = [self._as_proposal(candidate) for candidate in candidates]
        before = len(self.archive)
        best_before = float(self.best_value)
        source_attempts: Counter[str] = Counter(proposal.source for proposal in proposals)
        source_counts: Counter[str] = Counter()
        source_improvements: Counter[str] = Counter()
        source_improvement_amounts: Counter[str] = Counter()
        source_relative_successes: Counter[str] = Counter()
        source_relative_amounts: Counter[str] = Counter()
        shade_successes: list[tuple[float, float, float]] = []
        cma_successes: list[tuple[int, np.ndarray, float]] = []
        improvements = 0
        region_improvements: Counter[int] = Counter()
        region_attempts: Counter[int] = Counter()
        for proposal in proposals:
            if self._remaining() <= 0:
                break
            point = proposal.point
            source = proposal.source
            clipped = self._clip_point(point)
            incumbent_before = float(self.best_value)
            parent_sample = self._sample_by_key(proposal.parent_key)
            parent_target = None if parent_sample is None else self._target(parent_sample.value)
            value, added, improved = self._evaluate_point(point, source=source)
            if not added:
                continue
            source_counts[source] += 1
            proposal_axes = None if proposal.axes is None else np.asarray(proposal.axes, dtype=int)
            if proposal_axes is not None and proposal_axes.size > 0:
                self._axis_coverage[proposal_axes] += 1.0
            if proposal.region_id is not None or source.startswith("region:"):
                region_id = -1 if proposal.region_id is None else int(proposal.region_id)
                if proposal.region_id is None:
                    try:
                        region_id = int(source.split(":")[1])
                    except ValueError:
                        region_id = -1
                region_attempts[region_id] += 1
                if improved:
                    region_improvements[region_id] += 1
            relative_amount = 0.0
            relative_success = False
            if parent_target is not None and value is not None:
                relative_amount = max(0.0, parent_target - self._target(float(value)))
                relative_success = relative_amount > float(self.config.best_improvement_tolerance)
            if relative_success:
                source_relative_successes[source] += 1
                weighted_amount = relative_amount * float(self.config.relative_success_credit)
                source_relative_amounts[source] += weighted_amount
                if parent_sample is not None:
                    self._record_successful_direction(parent_sample.point, clipped)
                if proposal_axes is not None and proposal_axes.size > 0:
                    self._axis_activity *= 0.999
                    self._axis_activity[proposal_axes] += max(relative_amount, 1e-12)
                if proposal.shade_params is not None:
                    shade_successes.append((float(proposal.shade_params[0]), float(proposal.shade_params[1]), weighted_amount))
                if proposal.family == "cma" and proposal.region_id is not None:
                    cma_successes.append((int(proposal.region_id), clipped.copy(), weighted_amount))
            if improved:
                improvements += 1
                source_improvements[source] += 1
                if np.isfinite(incumbent_before) and value is not None:
                    amount = max(0.0, self._target(incumbent_before) - self._target(float(value)))
                    source_improvement_amounts[source] += amount
                    if proposal_axes is not None and proposal_axes.size > 0:
                        self._axis_activity *= 0.999
                        self._axis_activity[proposal_axes] += max(amount, 1e-12)
                    if proposal.shade_params is not None and not relative_success:
                        shade_successes.append((float(proposal.shade_params[0]), float(proposal.shade_params[1]), amount))
                    if proposal.family == "cma" and proposal.region_id is not None and not relative_success:
                        cma_successes.append((int(proposal.region_id), clipped.copy(), amount))
        after = len(self.archive)
        self._apply_region_feedback(region_attempts, region_improvements)
        credit_improvements = source_improvements + source_relative_successes
        credit_amounts = source_improvement_amounts + source_relative_amounts
        self._record_source_results(source_attempts, source_counts, credit_improvements, credit_amounts)
        self._update_shade_memory(shade_successes)
        self._apply_cma_feedback(source_counts, source_improvements, cma_successes)
        self._refresh_live_population()
        return PointCloudBatchEvent(
            batch_index=self._batch_index,
            kind="candidate_batch",
            evaluations_before=before,
            evaluations_after=after,
            candidate_count=len(proposals),
            source_counts=dict(source_counts),
            best_before=best_before,
            best_after=float(self.best_value),
            best_point=self.best_point.copy(),
            improved=bool(improvements),
            diagnostics={
                "improvements": int(improvements),
                "source_improvements": dict(source_improvements),
                "source_improvement_amounts": dict(source_improvement_amounts),
                "source_relative_successes": dict(source_relative_successes),
                "source_relative_amounts": dict(source_relative_amounts),
                "source_credit": self._source_stats_payload(),
                "shade_successes": int(len(shade_successes)),
                "cma_successes": int(len(cma_successes)),
                "cooperative_refinement_active": bool(self._cooperative_refinement_active()),
                "active_set_size": int(self._active_set_axes().size) if self._cooperative_refinement_active() else 0,
                "cooperative_improvements": int(
                    sum(count for source, count in source_improvements.items() if source.startswith("cooperative:"))
                ),
                "active_set_expansions": int(self._active_set_expansions),
                "top_axis_scores": [
                    {"axis": int(axis), "score": float(self._cooperative_axis_scores()[axis])}
                    for axis in np.argsort(self._cooperative_axis_scores())[::-1][: min(8, self.dimension)]
                ]
                if self._cooperative_refinement_active()
                else [],
                "axis_coverage_max": float(np.max(self._axis_coverage)) if self._axis_coverage.size else 0.0,
                "basin_polishing_active": bool(self._basin_polishing_active()),
                "anchor_baseline_target": (
                    None if self._anchor_baseline_target is None else float(self._anchor_baseline_target)
                ),
                "successful_direction_memory_size": int(len(self._successful_directions)),
                "linkage_blocks_used": 0,
                "lbfgs_pairs": int(len(self._lbfgs_pairs)),
                "polishing_allocation": dict(source_attempts) if self._basin_polishing_active() else {},
                "surrogate_ranked_candidates": int(
                    sum(1 for proposal in proposals if proposal.predicted_score is not None)
                ),
                "rotated_stencil_improvements": int(
                    sum(count for source, count in source_improvements.items() if source.endswith(":rotated_stencil"))
                ),
                "pattern_improvements": int(source_improvements.get("exploit_pattern", 0)),
                "surrogate_improvements": int(
                    source_improvements.get("exploit_surrogate", 0)
                    + sum(count for source, count in source_improvements.items() if source.endswith(":surrogate"))
                ),
                "archive_size": int(after),
            },
        )

    def _update_shade_memory(self, successes: list[tuple[float, float, float]]) -> None:
        if not successes:
            return
        f_values = np.asarray([entry[0] for entry in successes], dtype=float)
        cr_values = np.asarray([entry[1] for entry in successes], dtype=float)
        weights = np.asarray([max(entry[2], 0.0) for entry in successes], dtype=float)
        if not np.any(weights > 0.0):
            weights = np.ones_like(f_values)
        weights = weights / max(float(np.sum(weights)), 1e-12)
        f_denominator = max(float(np.sum(weights * f_values)), 1e-12)
        mean_f = float(np.sum(weights * f_values * f_values) / f_denominator)
        mean_cr = float(np.sum(weights * cr_values))
        index = int(self._shade_memory_index % self._shade_f_memory.size)
        self._shade_f_memory[index] = float(np.clip(mean_f, 0.05, 1.0))
        self._shade_cr_memory[index] = float(np.clip(mean_cr, 0.0, 1.0))
        self._shade_memory_index += 1

    def _cma_state_for_region(self, region: PointCloudRegion) -> CMARegionState:
        state = self._cma_states.get(region.region_id)
        if state is not None and state.diagonal_variance.shape == (self.dimension,):
            return state
        state = CMARegionState(
            sigma=float(self.config.cma_sigma_init),
            diagonal_variance=np.ones(self.dimension, dtype=float),
            evolution_path=np.zeros(self.dimension, dtype=float),
            directions=[],
        )
        self._cma_states[region.region_id] = state
        return state

    def _apply_cma_feedback(
        self,
        evaluations: Counter[str],
        improvements: Counter[str],
        successes: list[tuple[int, np.ndarray, float]],
    ) -> None:
        if not self._evolutionary_search_active() or not self.config.cma_region_enabled:
            return
        regions_by_id = {region.region_id: region for region in self.regions}
        successful_region_ids: set[int] = set()
        for region_id, point, _amount in successes:
            region = regions_by_id.get(region_id)
            if region is None:
                continue
            state = self._cma_state_for_region(region)
            step = self._normalized_point(point) - self._normalized_point(region.center)
            norm = float(np.linalg.norm(step))
            if norm <= 1e-14 or not np.isfinite(norm):
                continue
            direction = step / norm
            state.evolution_path = 0.80 * state.evolution_path + 0.20 * direction
            scaled = np.square(np.clip(step / max(state.sigma, 1e-12), -5.0, 5.0))
            state.diagonal_variance = np.clip(0.90 * state.diagonal_variance + 0.10 * scaled, 0.05, 20.0)
            state.directions.append(direction.copy())
            del state.directions[:-int(self.config.cma_direction_memory_size)]
            state.success_rate_ema = 0.80 * state.success_rate_ema + 0.20
            state.sigma = float(np.clip(state.sigma * 1.15, float(self.config.region_min_radius_fraction), 0.50))
            successful_region_ids.add(region_id)
        cma_attempt_regions: set[int] = set()
        for source, count in evaluations.items():
            if count <= 0 or self._source_family(source) != "cma":
                continue
            parts = source.split(":")
            if len(parts) < 3:
                continue
            try:
                cma_attempt_regions.add(int(parts[1]))
            except ValueError:
                continue
        for region_id in cma_attempt_regions - successful_region_ids:
            region = regions_by_id.get(region_id)
            if region is None:
                continue
            state = self._cma_state_for_region(region)
            state.success_rate_ema = 0.85 * state.success_rate_ema
            state.sigma = float(
                np.clip(state.sigma * 0.92, float(self.config.region_min_radius_fraction), 0.50)
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
                region.stall_count = 0
                region.cooldown_until = 0
                region.radius_fraction = min(
                    float(self.config.region_max_radius_fraction),
                    region.radius_fraction * float(self.config.region_expand_factor),
                )
            else:
                region.failures += int(attempt_count)
                region.stall_count += 1
                region.radius_fraction = max(
                    float(self.config.region_min_radius_fraction),
                    region.radius_fraction * float(self.config.region_shrink_factor),
                )
                if region.stall_count >= int(self.config.region_stall_patience):
                    region.cooldown_until = int(self._batch_index) + int(self.config.region_cooldown_batches)
                    region.stall_count = 0
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

    def _initial_design(self) -> list[CandidateProposal]:
        lower = self.original_bounds[:, 0]
        upper = self.original_bounds[:, 1]
        center = np.mean(self.original_bounds, axis=1)
        anchors = [center]
        origin = np.zeros(self.dimension, dtype=float)
        if np.all(origin >= lower) and np.all(origin <= upper):
            anchors.append(origin)
        anchors.extend([lower.copy(), upper.copy()])
        for axis in range(self.dimension):
            low_anchor = center.copy()
            high_anchor = center.copy()
            low_anchor[axis] = lower[axis]
            high_anchor[axis] = upper[axis]
            anchors.extend([low_anchor, high_anchor])
        candidates: list[CandidateProposal] = [self._proposal(point, "anchor") for point in anchors]
        random_count = max(0, int(self.config.initial_design_size) - len(candidates))
        if random_count:
            lhs = (
                np.arange(random_count, dtype=float)[:, None]
                + self.rng.random((random_count, self.dimension))
            ) / max(random_count, 1)
            for axis in range(self.dimension):
                self.rng.shuffle(lhs[:, axis])
            points = lower + lhs * (upper - lower)
            candidates.extend(self._proposal(point, "global") for point in points)
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
        frame = self._density_projection_frame()
        projected_points = self._project_normalized_points(normalized_points, frame)
        desirability = self._normalize_values_to_desirability(values)
        elite = self.archive.elite_indices(
            maximize=self.smoothlife_config.maximize,
            fraction=self.config.density_elite_fraction,
            minimum=min(4, len(self.archive)),
        )
        if elite.size > 128:
            elite = elite[:128]
        sigma = float(self.config.density_sigma_fraction)
        for index in elite:
            px, py = projected_points[index]
            dist2 = (grid_x - px) ** 2 + (grid_y - py) ** 2
            bump = np.exp(-0.5 * dist2 / max(sigma * sigma, 1e-12))
            density += (0.25 + desirability[index]) * bump
            objective = np.maximum(objective, desirability[index] * bump)
        cols = np.clip((projected_points[:, 0] * width).astype(int), 0, width - 1)
        rows = np.clip((projected_points[:, 1] * height).astype(int), 0, height - 1)
        evaluated[rows, cols] = True
        objective[rows, cols] = np.maximum(objective[rows, cols], desirability)
        if self._evolutionary_search_active():
            crowding = np.zeros(shape, dtype=float)
            support_count = min(256, projected_points.shape[0])
            if support_count > 0:
                support_indices = np.linspace(0, projected_points.shape[0] - 1, support_count, dtype=int)
                crowding_sigma = max(float(self.config.density_sigma_fraction) * 0.70, 1e-3)
                for index in support_indices:
                    px, py = projected_points[index]
                    dist2 = (grid_x - px) ** 2 + (grid_y - py) ** 2
                    crowding += np.exp(-0.5 * dist2 / max(crowding_sigma * crowding_sigma, 1e-12))
            crowding = crowding / max(float(np.max(crowding)), 1e-12)
            novelty = np.clip(1.0 - crowding, 0.0, 1.0)
            best_projected = self._project_normalized_points(
                np.asarray([self._normalized_point(self.best_point)], dtype=float),
                frame,
            )[0]
            best_dist2 = (grid_x - best_projected[0]) ** 2 + (grid_y - best_projected[1]) ** 2
            best_sigma = max(float(self.config.density_sigma_fraction) * 1.35, 1e-3)
            improvement_velocity = np.exp(-0.5 * best_dist2 / max(best_sigma * best_sigma, 1e-12))
            uncertainty = np.sqrt(np.clip(novelty * (1.0 - np.clip(objective, 0.0, 1.0)), 0.0, 1.0))
            density = np.clip(
                0.62 * density
                + 0.14 * novelty
                + 0.12 * uncertainty
                + 0.12 * improvement_velocity
                - 0.05 * crowding,
                1e-9,
                None,
            )
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
        min_separation = float(self.config.region_initial_radius_fraction) * 0.35
        target = -values if self.smoothlife_config.maximize else values
        best_target = float(np.min(target)) if target.size else 0.0
        span = max(float(np.max(target) - best_target), 1e-12) if target.size else 1.0
        for existing in old_by_nearest:
            if len(new_regions) >= int(self.config.portfolio_size):
                break
            if any(np.linalg.norm((existing.center - region.center) / widths) < min_separation for region in new_regions):
                continue
            new_regions.append(existing)
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
            if existing is not None:
                for candidate_index, current in enumerate(new_regions):
                    if current.region_id == existing.region_id:
                        new_regions[candidate_index] = region
                        break
                else:
                    new_regions.append(region)
            else:
                new_regions.append(region)
            if len(new_regions) >= int(self.config.portfolio_size):
                break
        if len(new_regions) < int(self.config.portfolio_size):
            order = np.argsort(target)
            if self.smoothlife_config.maximize:
                order = order[::-1]
            scout_positions = np.linspace(0, max(len(order) - 1, 0), num=min(int(self.config.portfolio_size), len(order)), dtype=int)
            for position in scout_positions:
                if len(new_regions) >= int(self.config.portfolio_size):
                    break
                point = points[int(order[int(position)])]
                if any(np.linalg.norm((point - region.center) / widths) < min_separation for region in new_regions):
                    continue
                region = PointCloudRegion(
                    region_id=self._next_region_id,
                    center=point.copy(),
                    radius_fraction=float(self.config.region_initial_radius_fraction),
                    score=0.0,
                    best_value=float(values[int(order[int(position)])]),
                )
                self._next_region_id += 1
                new_regions.append(region)
        self.regions = self._with_region_geometries(new_regions[: int(self.config.portfolio_size)], points, values)

    def _with_region_geometries(
        self,
        regions: list[PointCloudRegion],
        points: np.ndarray,
        values: np.ndarray,
    ) -> list[PointCloudRegion]:
        if not regions:
            return []
        fitted: list[PointCloudRegion] = []
        for region in regions:
            if self.config.anisotropic_regions_enabled:
                geometry = fit_region_geometry(
                    points,
                    values,
                    bounds=self.original_bounds,
                    center=region.center,
                    radius_fraction=region.radius_fraction,
                    maximize=self.smoothlife_config.maximize,
                    min_samples=self.config.region_geometry_min_samples,
                    anisotropy_max=self.config.region_anisotropy_max,
                    active_subspace_size=self.config.active_subspace_size,
                )
            else:
                geometry = RegionGeometry.identity("disabled", self.dimension)
            fitted.append(self._replace_region_geometry(region, geometry))
        return fitted

    @staticmethod
    def _replace_region_geometry(region: PointCloudRegion, geometry: RegionGeometry) -> PointCloudRegion:
        return replace(
            region,
            geometry_basis=np.asarray(geometry.basis, dtype=float).copy(),
            geometry_axis_scales=np.asarray(geometry.axis_scales, dtype=float).copy(),
            geometry_eigenvalues=np.asarray(geometry.eigenvalues, dtype=float).copy(),
            geometry_anisotropy=float(geometry.anisotropy),
            geometry_sample_count=int(geometry.sample_count),
            geometry_reason=geometry.reason,
        )

    def _fit_point_geometry(self, center: np.ndarray, radius_fraction: float) -> RegionGeometry:
        if not self.config.anisotropic_regions_enabled:
            return RegionGeometry.identity("disabled", self.dimension)
        points, values = self.archive.arrays()
        return fit_region_geometry(
            points,
            values,
            bounds=self.original_bounds,
            center=center,
            radius_fraction=radius_fraction,
            maximize=self.smoothlife_config.maximize,
            min_samples=self.config.region_geometry_min_samples,
            anisotropy_max=self.config.region_anisotropy_max,
            active_subspace_size=self.config.active_subspace_size,
        )

    @staticmethod
    def _region_geometry(region: PointCloudRegion) -> RegionGeometry:
        return RegionGeometry(
            basis=np.asarray(region.geometry_basis, dtype=float),
            axis_scales=np.asarray(region.geometry_axis_scales, dtype=float),
            eigenvalues=np.asarray(region.geometry_eigenvalues, dtype=float),
            anisotropy=float(region.geometry_anisotropy),
            sample_count=int(region.geometry_sample_count),
            reason=region.geometry_reason,
        )

    def _active_regions(self) -> list[PointCloudRegion]:
        return [region for region in self.regions if int(region.cooldown_until) <= int(self._batch_index)]

    def _sleeping_region_count(self) -> int:
        return len(self.regions) - len(self._active_regions())

    def _restart_pressure_active(self) -> bool:
        if not self.config.restart_strategy_enabled or not self._evolutionary_search_active():
            return False
        if self.regions and not self._active_regions():
            return True
        return int(self._batch_index) - int(self._last_improvement_batch) >= int(self.config.restart_stall_batches)

    def _wake_incumbent_region_if_all_sleeping(self) -> None:
        if not self.regions or self._active_regions():
            return
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        best_normalized = self._normalized_point(self.best_point)
        distances = [
            np.linalg.norm((region.center - self.original_bounds[:, 0]) / widths - best_normalized)
            for region in self.regions
        ]
        index = int(np.argmin(distances))
        region = self.regions[index]
        cooldown_before = int(region.cooldown_until)
        if cooldown_before <= int(self._batch_index):
            return
        region.cooldown_until = int(self._batch_index)
        self.trust_region_events.append(
            {
                "batch_index": int(self._batch_index),
                "phase": "portfolio",
                "kind": "forced_region_wake",
                "region_id": int(region.region_id),
                "cooldown_before": cooldown_before,
                "cooldown_after": int(region.cooldown_until),
                "center": region.center.copy().tolist(),
            }
        )

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
        if limit <= 0:
            return {}
        if not self._evolutionary_search_active():
            fractions = {
                "global": float(self.config.global_candidate_fraction),
                "density": float(self.config.density_candidate_fraction),
                "region": float(self.config.region_candidate_fraction if self.config.trust_regions_enabled else 0.0),
                "exploit": float(self.config.exploit_candidate_fraction),
            }
            return self._counts_from_weights(limit, fractions)

        weights = {
            "global": max(float(self.config.global_candidate_fraction), 0.10),
            "smoothlife_density": max(float(self.config.density_candidate_fraction), 0.08),
            "region": float(self.config.region_candidate_fraction if self.config.trust_regions_enabled else 0.0),
            "exploit": max(float(self.config.exploit_candidate_fraction) * 0.50, 0.04),
            "coherent": (0.18 if self._cooperative_refinement_active() else 0.08)
            if self.config.coherent_probes_enabled
            else 0.0,
            "shade": 0.30 if self.config.shade_enabled else 0.0,
            "cma": 0.24 if self.config.cma_region_enabled and self.config.trust_regions_enabled else 0.0,
            "cooperative": 0.10 if self._cooperative_refinement_active() else 0.0,
            "restart": (
                (0.16 if self._restart_pressure_active() else max(0.03, float(self.config.source_exploration_floor)))
                if self.config.restart_strategy_enabled
                else 0.0
            ),
        }
        if self._basin_polishing_active():
            weights = {
                "global": max(float(self.config.source_exploration_floor), 0.02),
                "smoothlife_density": 0.08,
                "region": 0.22 if self.config.trust_regions_enabled else 0.0,
                "exploit": 0.22,
                "coherent": (0.18 if self._cooperative_refinement_active() else 0.10)
                if self.config.coherent_probes_enabled
                else 0.0,
                "shade": 0.16 if self.config.shade_enabled else 0.0,
                "cma": 0.24 if self.config.cma_region_enabled and self.config.trust_regions_enabled else 0.0,
                "cooperative": 0.18 if self._cooperative_refinement_active() else 0.0,
                "restart": max(float(self.config.source_exploration_floor), 0.02)
                if self.config.restart_strategy_enabled and self._restart_pressure_active()
                else 0.0,
            }
        weights = {source: weight for source, weight in weights.items() if weight > 0.0}
        if self.config.source_adaptation_enabled and len(weights) > 1 and self._source_stats:
            keys = list(weights)
            base = np.asarray([weights[key] for key in keys], dtype=float)
            base = base / max(float(np.sum(base)), 1e-12)
            scores = np.asarray(
                [
                    self._source_stats.get(key, SourceStats()).ema_credit
                    / max(float(self.config.source_credit_temperature), 1e-12)
                    for key in keys
                ],
                dtype=float,
            )
            scores -= float(np.max(scores))
            adaptive = np.exp(np.clip(scores, -60.0, 60.0))
            adaptive = adaptive / max(float(np.sum(adaptive)), 1e-12)
            blended = 0.85 * base + 0.15 * adaptive
            floor = min(float(self.config.source_exploration_floor), 0.90 / len(keys))
            blended = floor + max(0.0, 1.0 - floor * len(keys)) * blended / max(float(np.sum(blended)), 1e-12)
            weights = dict(zip(keys, blended))
        return self._counts_from_weights(limit, weights)

    @staticmethod
    def _counts_from_weights(limit: int, weights: dict[str, float]) -> dict[str, int]:
        eligible = {key: float(value) for key, value in weights.items() if float(value) > 0.0}
        if not eligible:
            return {"global": int(limit)}
        total = sum(eligible.values())
        raw = {key: limit * value / total for key, value in eligible.items()}
        counts = {key: int(np.floor(value)) for key, value in raw.items()}
        while sum(counts.values()) < limit:
            key = max(raw, key=lambda item: raw[item] - counts[item])
            counts[key] += 1
        return counts

    @staticmethod
    def _van_der_corput(index: int, base: int) -> float:
        value = 0.0
        denominator = 1.0
        n = int(index)
        while n > 0:
            n, remainder = divmod(n, base)
            denominator *= base
            value += remainder / denominator
        return value

    @staticmethod
    def _first_primes(count: int) -> list[int]:
        primes: list[int] = []
        candidate = 2
        while len(primes) < int(count):
            root = int(np.sqrt(candidate))
            if all(candidate % prime != 0 for prime in primes if prime <= root):
                primes.append(candidate)
            candidate += 1
        return primes

    def _global_candidates(self, count: int) -> list[CandidateProposal]:
        if count <= 0:
            return []
        lower = self.original_bounds[:, 0]
        upper = self.original_bounds[:, 1]
        bases = self._first_primes(self.dimension)
        normalized = []
        for _ in range(count):
            self._global_sequence_index += 1
            normalized.append([self._van_der_corput(self._global_sequence_index, base) for base in bases])
        points = lower + np.asarray(normalized, dtype=float) * (upper - lower)
        return [self._proposal(point, "global") for point in points]

    def _density_candidates(self, count: int, *, source_label: str = "density") -> list[CandidateProposal]:
        if count <= 0:
            return []
        density = np.asarray(self._last_density, dtype=float)
        global_count = min(count, int(np.ceil(count * float(self.config.global_exploration_floor))))
        density_count = count - global_count
        candidates = self._global_candidates(global_count)
        if density_count <= 0:
            return candidates
        flat = density.ravel()
        if not np.any(flat > 0.0):
            return self._global_candidates(count)
        probabilities = flat / float(np.sum(flat))
        chosen = self.rng.choice(flat.size, size=density_count, replace=True, p=probabilities)
        height, width = density.shape
        rows, cols = np.unravel_index(chosen, density.shape)
        jitter = self.rng.random((density_count, 2))
        normalized_x = (cols.astype(float) + jitter[:, 0]) / width
        normalized_y = (rows.astype(float) + jitter[:, 1]) / height
        lower = self.original_bounds[:, 0]
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        frame = self._density_projection_frame()
        normalized_points = self._lift_density_points(np.column_stack((normalized_x, normalized_y)), frame)
        points = lower + normalized_points * widths
        candidates.extend(self._proposal(point, source_label) for point in points)
        return candidates

    def _coherent_candidates(self, count: int) -> list[CandidateProposal]:
        if (
            count <= 0
            or not self.config.coherent_probes_enabled
            or not self._evolutionary_search_active()
            or not np.isfinite(self.best_value)
        ):
            return []
        normalized_best = np.clip(self._normalized_point(self.best_point), 0.0, 1.0)
        quantiles = (
            (0.80, 0.88, 0.92, 0.95, 0.98, 0.985, 0.992, 0.997)
            if self._cooperative_refinement_active()
            else (0.80, 0.88, 0.92, 0.95, 0.98)
        )
        spread = float(np.std(normalized_best))
        jitter_scale = float(
            np.clip(
                0.04 * spread if self._cooperative_refinement_active() else 0.03 * spread,
                5e-4 if self._cooperative_refinement_active() else 2e-4,
                1.5e-3 if self._cooperative_refinement_active() else 8e-4,
            )
        )
        candidates: list[CandidateProposal] = []
        seen: set[tuple[str, ...]] = set()
        parent_key = self._nearest_archive_key(self.best_point)
        attempts = 0
        while len(candidates) < count and attempts < count * 8:
            attempts += 1
            quantile = quantiles[(int(self._batch_index) + attempts - 1) % len(quantiles)]
            level = float(np.quantile(normalized_best, quantile))
            jitter = self.rng.normal(0.0, jitter_scale, size=self.dimension)
            normalized = np.clip(level + jitter, 0.02, 0.98)
            key_variance = float(np.var(normalized))
            if key_variance < 1e-10:
                axis_wave = np.linspace(-jitter_scale, jitter_scale, num=self.dimension)
                normalized = np.clip(normalized + axis_wave, 0.02, 0.98)
            point = self._point_from_normalized(normalized)
            key = PointCloudArchive.key(point)
            if key in seen or self.archive.get(point) is not None:
                continue
            seen.add(key)
            candidates.append(self._proposal(point, "coherent", parent_key=parent_key))
        return candidates

    def _stencil_points(self, center: np.ndarray, step: np.ndarray) -> list[np.ndarray]:
        directions: list[np.ndarray] = []
        for axis in range(self.dimension):
            positive = np.zeros(self.dimension, dtype=float)
            negative = np.zeros(self.dimension, dtype=float)
            positive[axis] = 1.0
            negative[axis] = -1.0
            directions.extend([positive, negative])
        max_pairs = min(self.dimension - 1, 4)
        for axis in range(max_pairs):
            other = axis + 1
            for sx, sy in ((1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)):
                direction = np.zeros(self.dimension, dtype=float)
                direction[axis] = sx
                direction[other] = sy
                directions.append(direction)
        return [self._clip_point(center + direction * step) for direction in directions]

    def _normalized_point(self, point: np.ndarray) -> np.ndarray:
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        return (np.asarray(point, dtype=float) - self.original_bounds[:, 0]) / widths

    def _point_from_normalized(self, normalized: np.ndarray) -> np.ndarray:
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        point = self.original_bounds[:, 0] + np.asarray(normalized, dtype=float) * widths
        return self._clip_point(point)

    def _rotated_stencil_points(
        self,
        center: np.ndarray,
        geometry: RegionGeometry,
        step_fraction: float,
    ) -> list[np.ndarray]:
        subspace_dimension = int(np.asarray(geometry.basis).shape[1])
        directions: list[np.ndarray] = []
        for axis in range(subspace_dimension):
            positive = np.zeros(subspace_dimension, dtype=float)
            negative = np.zeros(subspace_dimension, dtype=float)
            positive[axis] = 1.0
            negative[axis] = -1.0
            directions.extend([positive, negative])
        max_pairs = min(subspace_dimension - 1, 4)
        for axis in range(max_pairs):
            other = axis + 1
            for sx, sy in ((1.0, 1.0), (-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)):
                direction = np.zeros(subspace_dimension, dtype=float)
                direction[axis] = sx
                direction[other] = sy
                directions.append(direction)
        normalized_center = self._normalized_point(center)
        basis = np.asarray(geometry.basis, dtype=float)
        axis_steps = max(float(step_fraction), float(self.config.region_min_radius_fraction)) * np.asarray(
            geometry.axis_scales,
            dtype=float,
        )
        points: list[np.ndarray] = []
        for direction in directions:
            offset = basis @ (direction * axis_steps)
            points.append(self._point_from_normalized(normalized_center + offset))
        return points

    def _anisotropic_random_points(
        self,
        center: np.ndarray,
        geometry: RegionGeometry,
        radius_fraction: float,
        count: int,
    ) -> np.ndarray:
        if count <= 0:
            return np.empty((0, self.dimension), dtype=float)
        normalized_center = self._normalized_point(center)
        basis = np.asarray(geometry.basis, dtype=float)
        axis_lengths = max(float(radius_fraction), float(self.config.region_min_radius_fraction)) * np.asarray(
            geometry.axis_scales,
            dtype=float,
        )
        local = self.rng.normal(0.0, 1.0, size=(count, axis_lengths.size))
        norms = np.linalg.norm(local, axis=1)
        norms = np.where(norms > 1e-12, norms, 1.0)
        radii = self.rng.random(count) ** (1.0 / max(axis_lengths.size, 1))
        local = local / norms[:, None] * radii[:, None] * axis_lengths
        normalized_points = normalized_center + local @ basis.T
        return np.asarray([self._point_from_normalized(point) for point in normalized_points], dtype=float)

    def _region_candidates(self, count: int) -> list[CandidateProposal]:
        self._wake_incumbent_region_if_all_sleeping()
        active_regions = self._active_regions()
        if count <= 0 or not active_regions:
            return []
        candidates: list[CandidateProposal] = []
        per_region = max(1, int(np.ceil(count / len(active_regions))))
        for region in active_regions:
            region_bounds = region.bounds(self.original_bounds)
            lower = region_bounds[:, 0]
            upper = region_bounds[:, 1]
            widths = upper - lower
            geometry = self._region_geometry(region)
            stencil_count = min(per_region, int(np.ceil(per_region * float(self.config.region_stencil_fraction))))
            step_fraction = max(float(region.radius_fraction) * 0.50, float(self.config.region_min_radius_fraction))
            step = np.maximum(
                widths * 0.25,
                (self.original_bounds[:, 1] - self.original_bounds[:, 0])
                * float(self.config.region_min_radius_fraction),
            )
            parent_key = self._nearest_archive_key(region.center)
            stencil_candidates: list[CandidateProposal] = []
            if self.config.anisotropic_regions_enabled and geometry.reason != "disabled":
                for point in self._rotated_stencil_points(region.center, geometry, step_fraction)[:stencil_count]:
                    stencil_candidates.append(
                        self._proposal(
                            point,
                            f"region:{region.region_id}:rotated_stencil",
                            parent_key=parent_key,
                            region_id=region.region_id,
                        )
                    )
                for point in self._stencil_points(region.center, step):
                    stencil_candidates.append(
                        self._proposal(point, f"region:{region.region_id}:stencil", parent_key=parent_key, region_id=region.region_id)
                    )
            else:
                for point in self._stencil_points(region.center, step)[:stencil_count]:
                    stencil_candidates.append(
                        self._proposal(point, f"region:{region.region_id}:stencil", parent_key=parent_key, region_id=region.region_id)
                    )
            candidates.extend(stencil_candidates[:stencil_count])
            random_needed = max(1, per_region - stencil_count - 2)
            if self.config.anisotropic_regions_enabled and geometry.reason == "archive_covariance":
                points = self._anisotropic_random_points(region.center, geometry, region.radius_fraction, random_needed)
            else:
                points = lower + self.rng.random((random_needed, self.dimension)) * (upper - lower)
            candidates.extend(
                self._proposal(point, f"region:{region.region_id}:random", parent_key=parent_key, region_id=region.region_id)
                for point in points
            )
            if self.config.surrogate_enabled:
                surrogate = self._fit_region_surrogate(region)
                if surrogate.accepted and surrogate.point is not None:
                    candidates.append(
                        self._proposal(
                            surrogate.point.copy(),
                            f"region:{region.region_id}:surrogate",
                            parent_key=parent_key,
                            region_id=region.region_id,
                        )
                    )
            candidates.append(self._proposal(region.center.copy(), f"region:{region.region_id}:center", parent_key=parent_key, region_id=region.region_id))
            if len(candidates) >= count:
                break
        return candidates[:count]

    def _exploit_candidates(self, count: int) -> list[CandidateProposal]:
        if count <= 0 or not np.isfinite(self.best_value):
            return []
        candidates: list[CandidateProposal] = []
        parent_key = self._nearest_archive_key(self.best_point)
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        distances = self.archive.nearest_distance(np.asarray([self.best_point], dtype=float), self.original_bounds)
        scheduled = float(self.config.region_initial_radius_fraction) * (0.5 ** max(0, self._batch_index // 60))
        local_scale = max(
            float(self.config.region_min_radius_fraction),
            min(float(self.config.local_refinement_step_fraction), max(float(distances[0]) if distances.size else 0.05, scheduled)),
        )
        if self.config.surrogate_enabled:
            surrogate = self._fit_point_surrogate(self.best_point, max(local_scale, float(self.config.region_initial_radius_fraction)))
            if surrogate.accepted and surrogate.point is not None:
                candidates.append(self._proposal(surrogate.point.copy(), "exploit_surrogate", parent_key=parent_key))
                if len(candidates) >= count:
                    return candidates
        if np.linalg.norm(self._last_successful_step / widths) > float(self.config.region_min_radius_fraction):
            candidates.append(self._proposal(self._clip_point(self.best_point + self._last_successful_step), "exploit_pattern", parent_key=parent_key))
            if len(candidates) >= count:
                return candidates
            candidates.append(self._proposal(self._clip_point(self.best_point + 2.0 * self._last_successful_step), "exploit_pattern", parent_key=parent_key))
            if len(candidates) >= count:
                return candidates
        step = np.maximum(local_scale * widths, 1e-9 * widths)
        stencil_candidates = [self._proposal(point, "exploit_stencil", parent_key=parent_key) for point in self._stencil_points(self.best_point, step)]
        for proposal in stencil_candidates:
            candidates.append(proposal)
            if len(candidates) >= count:
                return candidates
        while len(candidates) < count:
            point = self.best_point + self.rng.normal(0.0, step, size=self.dimension)
            point = self._clip_point(point)
            candidates.append(self._proposal(point, "exploit", parent_key=parent_key))
        return candidates

    def _shade_candidates(self, count: int) -> list[CandidateProposal]:
        if count <= 0 or not self.config.shade_enabled or not self._evolutionary_search_active():
            return []
        population_points, population_values, population_keys = self._live_population_arrays()
        if population_points.shape[0] < 4:
            population_points, population_values = self.archive.arrays()
            population_keys = [PointCloudArchive.key(point) for point in population_points]
        if population_points.shape[0] < 4:
            return self._global_candidates(count)
        archive_points, archive_values = self.archive.arrays()
        population_target = -population_values if self.smoothlife_config.maximize else population_values
        population_order = np.argsort(population_target)
        elite_count = max(
            2,
            min(
                population_points.shape[0],
                int(np.ceil(float(self.config.shade_pbest_fraction) * population_points.shape[0])),
            ),
        )
        elite_indices = population_order[:elite_count]
        archive_target = -archive_values if self.smoothlife_config.maximize else archive_values
        archive_order = np.argsort(archive_target)
        archive_pool_count = max(
            4,
            min(
                archive_points.shape[0],
                int(np.ceil(float(self.config.shade_archive_fraction) * archive_points.shape[0])),
            ),
        )
        pool_parts = [population_points]
        if archive_points.shape[0] > 0:
            pool_parts.append(archive_points[archive_order[:archive_pool_count]])
        difference_pool = np.vstack(pool_parts)
        lower = self.original_bounds[:, 0]
        widths = self.original_bounds[:, 1] - lower
        candidates: list[CandidateProposal] = []
        seen: set[tuple[str, ...]] = set()
        attempts = 0
        while len(candidates) < count and attempts < count * 8:
            attempts += 1
            memory_index = int(self.rng.integers(0, self._shade_f_memory.size))
            f = float(self._shade_f_memory[memory_index] + 0.10 * self.rng.standard_cauchy())
            for _ in range(8):
                if f > 0.0:
                    break
                f = float(self._shade_f_memory[memory_index] + 0.10 * self.rng.standard_cauchy())
            f = float(np.clip(f, 0.05, 1.0))
            cr = float(np.clip(self.rng.normal(float(self._shade_cr_memory[memory_index]), 0.10), 0.0, 1.0))
            current_index = int(self.rng.integers(0, population_points.shape[0]))
            pbest_index = int(self.rng.choice(elite_indices))
            if difference_pool.shape[0] >= 2:
                difference_indices = self.rng.choice(difference_pool.shape[0], size=2, replace=False)
            else:
                difference_indices = self.rng.choice(population_points.shape[0], size=2, replace=False)
            current = population_points[current_index]
            mutant = current + f * (population_points[pbest_index] - current) + f * (
                difference_pool[int(difference_indices[0])] - difference_pool[int(difference_indices[1])]
            )
            mask = self.rng.random(self.dimension) < cr
            if not np.any(mask):
                mask[int(self.rng.integers(0, self.dimension))] = True
            trial = current.copy()
            trial[mask] = mutant[mask]
            trial = self._clip_point(trial)
            key = PointCloudArchive.key(trial)
            if key in seen or self.archive.get(trial) is not None:
                jitter = self.rng.normal(0.0, 0.0025 * widths, size=self.dimension)
                trial = self._clip_point(trial + jitter)
                key = PointCloudArchive.key(trial)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                self._proposal(
                    trial,
                    "shade",
                    parent_key=population_keys[current_index],
                    shade_params=(f, cr),
                )
            )
        if len(candidates) < count:
            candidates.extend(self._global_candidates(count - len(candidates)))
        return candidates[:count]

    def _cma_region_candidates(self, count: int) -> list[CandidateProposal]:
        if (
            count <= 0
            or not self.config.cma_region_enabled
            or not self.config.trust_regions_enabled
            or not self._evolutionary_search_active()
        ):
            return []
        self._wake_incumbent_region_if_all_sleeping()
        active_regions = self._active_regions()
        if not active_regions:
            return []
        candidates: list[CandidateProposal] = []
        per_region = max(1, int(np.ceil(count / len(active_regions))))
        for region in active_regions:
            state = self._cma_state_for_region(region)
            normalized_center = self._normalized_point(region.center)
            direction_count = len(state.directions)
            parent_key = self._nearest_archive_key(region.center)
            for _ in range(per_region):
                diagonal_noise = self.rng.normal(0.0, np.sqrt(state.diagonal_variance), size=self.dimension)
                offset = float(state.sigma) * diagonal_noise
                if direction_count:
                    coefficients = self.rng.normal(
                        0.0,
                        float(state.sigma) / max(np.sqrt(direction_count), 1.0),
                        size=direction_count,
                    )
                    low_rank = np.sum(
                        [coefficient * direction for coefficient, direction in zip(coefficients, state.directions)],
                        axis=0,
                    )
                    offset = offset + low_rank
                if np.linalg.norm(state.evolution_path) > 1e-12:
                    offset = offset + self.rng.normal(0.0, 0.35 * float(state.sigma)) * state.evolution_path
                max_radius = max(float(region.radius_fraction), float(self.config.region_min_radius_fraction))
                norm = float(np.linalg.norm(offset))
                if norm > max_radius:
                    offset = offset * (max_radius / norm)
                point = self._point_from_normalized(np.clip(normalized_center + offset, 0.0, 1.0))
                candidates.append(
                    self._proposal(
                        point,
                        f"region:{region.region_id}:cma",
                        parent_key=parent_key,
                        region_id=region.region_id,
                    )
                )
                if len(candidates) >= count:
                    return candidates
        return candidates[:count]

    def _cooperative_candidates(self, count: int) -> list[CandidateProposal]:
        if count <= 0 or not self._cooperative_refinement_active():
            return []
        groups = self._cooperative_groups(min(int(self.config.cooperative_groups_per_batch), max(1, count // 4)))
        if not groups:
            return []
        candidates: list[CandidateProposal] = []
        normalized_best = self._normalized_point(self.best_point)
        parent_key = self._nearest_archive_key(self.best_point)
        axis_scores = self._cooperative_axis_scores()
        step_base = float(
            np.clip(
                max(float(self.config.region_min_radius_fraction) * 100.0, min(0.02, float(self._stencil_step_fraction))),
                5e-4,
                0.03,
            )
        )

        def add_candidate(normalized: np.ndarray, source: str, axes: np.ndarray, parent: tuple[str, ...] | None = parent_key) -> None:
            if len(candidates) >= count:
                return
            point = self._point_from_normalized(normalized)
            candidates.append(self._proposal(point, source, parent_key=parent, axes=tuple(int(axis) for axis in axes)))

        directions: list[np.ndarray] = []
        for direction in reversed(self._successful_directions[-4:]):
            directions.append(np.asarray(direction, dtype=float))
        if np.linalg.norm(self._last_successful_step) > 1e-14:
            widths = np.maximum(self.original_bounds[:, 1] - self.original_bounds[:, 0], 1e-12)
            directions.append(self._last_successful_step / widths)
        for step, _gradient_delta in self._lbfgs_pairs[-2:]:
            directions.append(np.asarray(step, dtype=float))

        population_points, population_values, population_keys = self._live_population_arrays()
        population_target = -population_values if self.smoothlife_config.maximize else population_values
        population_order = np.argsort(population_target) if population_values.size else np.asarray([], dtype=int)

        for group in groups:
            if len(candidates) >= count:
                break
            axes = np.asarray(group, dtype=int)
            axis_order = axes[np.argsort(axis_scores[axes])[::-1]]
            for axis in axis_order[: max(1, min(3, axes.size))]:
                direction = np.zeros(self.dimension, dtype=float)
                direction[int(axis)] = 1.0
                for scale in (0.35, 1.0):
                    for sign in (1.0, -1.0):
                        add_candidate(
                            normalized_best + sign * step_base * scale * direction,
                            "cooperative:group",
                            axes,
                        )
                        if len(candidates) >= count:
                            return candidates[:count]
            for raw_direction in directions:
                projected = np.zeros(self.dimension, dtype=float)
                projected[axes] = raw_direction[axes]
                norm = float(np.linalg.norm(projected))
                if norm <= 1e-14 or not np.isfinite(norm):
                    continue
                projected = projected / norm
                for sign in (1.0, -1.0):
                    add_candidate(
                        normalized_best + sign * step_base * projected,
                        "cooperative:line_search",
                        axes,
                    )
                    if len(candidates) >= count:
                        return candidates[:count]
            if population_points.shape[0] >= 4 and population_order.size > 0:
                current_index = int(self.rng.integers(0, population_points.shape[0]))
                elite_count = max(1, min(population_points.shape[0], int(np.ceil(0.20 * population_points.shape[0]))))
                pbest_index = int(self.rng.choice(population_order[:elite_count]))
                diff_indices = self.rng.choice(population_points.shape[0], size=2, replace=False)
                current = self._normalized_point(population_points[current_index])
                pbest = self._normalized_point(population_points[pbest_index])
                first = self._normalized_point(population_points[int(diff_indices[0])])
                second = self._normalized_point(population_points[int(diff_indices[1])])
                mutant = current.copy()
                f = float(np.clip(self._shade_f_memory[self._shade_memory_index % self._shade_f_memory.size], 0.05, 1.0))
                mutant[axes] = current[axes] + f * (pbest[axes] - current[axes]) + 0.5 * f * (first[axes] - second[axes])
                trial = normalized_best.copy()
                trial[axes] = mutant[axes]
                add_candidate(
                    np.clip(trial, 0.0, 1.0),
                    "cooperative:de",
                    axes,
                    parent=population_keys[current_index],
                )
                if len(candidates) >= count:
                    return candidates[:count]
            noise = np.zeros(self.dimension, dtype=float)
            scale = max(step_base, float(self.config.cma_sigma_init) * 0.20)
            noise[axes] = self.rng.normal(0.0, scale, size=axes.size)
            add_candidate(
                normalized_best + noise,
                "cooperative:cma",
                axes,
            )
        if len(candidates) < count:
            candidates.extend(self._global_candidates(count - len(candidates)))
        return candidates[:count]

    def _restart_candidates(self, count: int) -> list[CandidateProposal]:
        if count <= 0 or not self.config.restart_strategy_enabled or not self._evolutionary_search_active():
            return []
        candidates: list[CandidateProposal] = []
        bases = self._first_primes(self.dimension)
        axis_indices = np.arange(self.dimension, dtype=float)
        golden = (np.sqrt(5.0) - 1.0) * 0.5
        attempts = 0
        while len(candidates) < count and attempts < max(count * 16, 32):
            attempts += 1
            sequence_index = self._restart_lattice_index + 1
            self._restart_lattice_index += 1
            axis_lattice = np.asarray(
                [
                    self._van_der_corput(sequence_index + 17 * (axis + 1), base)
                    for axis, base in enumerate(bases)
                ],
                dtype=float,
            )
            axis_jitter = np.mod(self._restart_axis_shift + golden * (axis_indices + sequence_index), 1.0)
            global_level = self._van_der_corput(sequence_index, 2)
            normalized = 0.55 * np.mod(axis_lattice + self._restart_axis_shift, 1.0)
            normalized += 0.30 * axis_jitter
            normalized += 0.15 * global_level
            normalized = np.clip(normalized, 0.02, 0.98)
            if self.dimension >= int(self.config.high_dimensional_min_dimension) and float(np.var(normalized)) < 1e-4:
                normalized = np.mod(normalized + golden * (axis_indices + 1.0), 1.0)
                normalized = np.clip(normalized, 0.02, 0.98)
            if self.dimension >= int(self.config.high_dimensional_min_dimension) and float(np.var(normalized)) < 1e-4:
                continue
            point = self._point_from_normalized(normalized)
            if self.archive.get(point) is not None:
                continue
            candidates.append(self._proposal(point, "restart:scout"))
        if len(candidates) < count:
            candidates.extend(self._global_candidates(count - len(candidates)))
        return candidates

    def _surrogate_ranking_active(self) -> bool:
        return bool(
            self.config.surrogate_ranking_enabled
            and self._evolutionary_search_active()
            and len(self.archive) >= max(4, min(int(self.config.surrogate_ranking_neighbor_count), 16))
        )

    def _ranking_axes(self) -> np.ndarray:
        if self.dimension <= 64:
            return np.arange(self.dimension, dtype=int)
        target = min(self.dimension, max(16, int(self.config.active_subspace_size) * 4))
        if np.any(self._axis_activity > 0.0):
            active = np.argsort(self._axis_activity)[::-1][:target]
            spread = np.linspace(0, self.dimension - 1, num=target, dtype=int)
            return np.unique(np.concatenate((active, spread))).astype(int)
        return np.linspace(0, self.dimension - 1, num=target, dtype=int)

    def _dedupe_proposals(
        self,
        candidates: list[CandidateProposal | tuple[np.ndarray, str]],
        *,
        skip_archive: bool,
    ) -> list[CandidateProposal]:
        proposals: list[CandidateProposal] = []
        seen: set[tuple[str, ...]] = set()
        for candidate in candidates:
            proposal = self._as_proposal(candidate)
            proposal.point = self._clip_point(proposal.point)
            key = PointCloudArchive.key(proposal.point)
            if key in seen:
                continue
            if skip_archive and self.archive.get(proposal.point) is not None:
                continue
            seen.add(key)
            proposals.append(proposal)
        return proposals

    def _rank_candidate_pool(
        self,
        candidates: list[CandidateProposal | tuple[np.ndarray, str]],
        limit: int,
        quotas: dict[str, int],
    ) -> list[CandidateProposal]:
        proposals = self._dedupe_proposals(candidates, skip_archive=True)
        if limit <= 0:
            return []
        if len(proposals) <= limit or not self._surrogate_ranking_active():
            return proposals[:limit]
        archive_points, archive_values = self.archive.arrays()
        if archive_points.size == 0:
            return proposals[:limit]
        target = -archive_values if self.smoothlife_config.maximize else archive_values
        max_rank_samples = min(
            archive_points.shape[0],
            max(48, min(192, int(self.config.surrogate_ranking_neighbor_count) * 4)),
        )
        if archive_points.shape[0] > max_rank_samples:
            order = np.argsort(target)
            elite_count = max(1, max_rank_samples // 2)
            recent_count = max(1, max_rank_samples // 4)
            selected: list[int] = [int(index) for index in order[:elite_count]]
            selected.extend(range(max(0, archive_points.shape[0] - recent_count), archive_points.shape[0]))
            spread_count = max_rank_samples - len(set(selected))
            if spread_count > 0:
                selected.extend(
                    int(index)
                    for index in np.linspace(0, archive_points.shape[0] - 1, num=spread_count, dtype=int)
                )
            selected = list(dict.fromkeys(selected))[:max_rank_samples]
            archive_points = archive_points[selected]
            archive_values = archive_values[selected]
            target = target[selected]

        axes = self._ranking_axes()
        widths = np.maximum(self.original_bounds[:, 1] - self.original_bounds[:, 0], 1e-12)
        archive_normalized = ((archive_points - self.original_bounds[:, 0]) / widths)[:, axes]
        neighbor_count = min(int(self.config.surrogate_ranking_neighbor_count), archive_points.shape[0])
        predictions: list[float] = []
        novelty_values: list[float] = []
        parent_gains: list[float] = []
        credits: list[float] = []

        for proposal in proposals:
            normalized = ((proposal.point - self.original_bounds[:, 0]) / widths)[axes]
            distances = np.linalg.norm(archive_normalized - normalized, axis=1)
            if neighbor_count < distances.size:
                neighbor_indices = np.argpartition(distances, neighbor_count - 1)[:neighbor_count]
            else:
                neighbor_indices = np.arange(distances.size)
            neighbor_distances = distances[neighbor_indices]
            weights = 1.0 / np.maximum(neighbor_distances, 1e-8)
            prediction = float(np.dot(weights, target[neighbor_indices]) / max(float(np.sum(weights)), 1e-12))
            novelty = float(np.min(distances)) if distances.size else 1.0
            parent_sample = self._sample_by_key(proposal.parent_key)
            parent_gain = 0.0
            if parent_sample is not None:
                parent_gain = max(0.0, self._target(parent_sample.value) - prediction)
            predictions.append(prediction)
            novelty_values.append(novelty)
            parent_gains.append(parent_gain)
            credits.append(float(self._source_stats.get(proposal.family, SourceStats()).ema_credit))

        predicted_array = np.asarray(predictions, dtype=float)
        novelty_array = np.asarray(novelty_values, dtype=float)
        parent_gain_array = np.asarray(parent_gains, dtype=float)
        credit_array = np.asarray(credits, dtype=float)

        def normalized(values: np.ndarray) -> np.ndarray:
            span = max(float(np.max(values) - np.min(values)), 1e-12)
            return (values - float(np.min(values))) / span

        score = normalized(predicted_array)
        score -= 0.12 * normalized(novelty_array)
        if np.any(parent_gain_array > 0.0):
            score -= 0.10 * normalized(parent_gain_array)
        if np.any(credit_array > 0.0):
            score -= 0.06 * normalized(credit_array)

        for proposal, value in zip(proposals, score):
            proposal.predicted_score = float(value)

        family_quotas: dict[str, int] = Counter()
        for source, quota in quotas.items():
            family_quotas[self._source_family(source)] += int(quota)
        selected: list[int] = []
        selected_set: set[int] = set()
        order = np.argsort(score)
        for family, quota in sorted(family_quotas.items(), key=lambda item: item[1], reverse=True):
            if quota <= 0 or len(selected) >= limit:
                continue
            family_indices = [int(index) for index in order if proposals[int(index)].family == family]
            for index in family_indices[:quota]:
                if index in selected_set:
                    continue
                selected.append(index)
                selected_set.add(index)
                if len(selected) >= limit:
                    break
        for index in order:
            resolved = int(index)
            if len(selected) >= limit:
                break
            if resolved in selected_set:
                continue
            selected.append(resolved)
            selected_set.add(resolved)
        return [proposals[index] for index in selected[:limit]]

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
            full_quadratic_max_dimension=self.config.surrogate_full_quadratic_max_dimension,
        )

    def _fit_point_surrogate(self, center: np.ndarray, radius_fraction: float) -> QuadraticSurrogate:
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        radius = max(float(radius_fraction), float(self.config.region_min_radius_fraction)) * widths
        lower = np.maximum(self.original_bounds[:, 0], np.asarray(center, dtype=float) - radius)
        upper = np.minimum(self.original_bounds[:, 1], np.asarray(center, dtype=float) + radius)
        region_bounds = np.column_stack((lower, upper))
        points, values = self.archive.arrays()
        return fit_quadratic_surrogate(
            points,
            values,
            region_bounds=region_bounds,
            center=np.asarray(center, dtype=float),
            maximize=self.smoothlife_config.maximize,
            min_samples=self.config.surrogate_min_samples,
            max_samples=self.config.surrogate_max_samples,
            regularization=self.config.surrogate_regularization,
            max_condition=self.config.surrogate_max_condition,
            full_quadratic_max_dimension=self.config.surrogate_full_quadratic_max_dimension,
        )

    def _generate_candidates_from_counts(self, counts: dict[str, int]) -> list[CandidateProposal]:
        candidates: list[CandidateProposal] = []
        candidates.extend(self._global_candidates(counts.get("global", 0)))
        candidates.extend(self._density_candidates(counts.get("density", 0), source_label="density"))
        candidates.extend(
            self._density_candidates(
                counts.get("smoothlife_density", 0),
                source_label="smoothlife_density",
            )
        )
        candidates.extend(self._region_candidates(counts.get("region", 0)))
        candidates.extend(self._coherent_candidates(counts.get("coherent", 0)))
        candidates.extend(self._shade_candidates(counts.get("shade", 0)))
        candidates.extend(self._cma_region_candidates(counts.get("cma", 0)))
        candidates.extend(self._cooperative_candidates(counts.get("cooperative", 0)))
        candidates.extend(self._restart_candidates(counts.get("restart", 0)))
        candidates.extend(self._exploit_candidates(counts.get("exploit", 0)))
        return candidates

    def _candidate_batch(self, limit: int) -> list[CandidateProposal]:
        if limit <= 0:
            return []
        final_counts = self._candidate_counts(limit)
        candidates = self._generate_candidates_from_counts(final_counts)
        if self._surrogate_ranking_active():
            extra_pool = min(
                limit * max(int(self.config.candidate_pool_multiplier) - 1, 0),
                max(16, limit // 2, 2 * int(self.config.active_subspace_size)),
            )
            if extra_pool > 0:
                candidates.extend(self._generate_candidates_from_counts(self._candidate_counts(extra_pool)))
            baseline = self._dedupe_proposals(candidates[:limit], skip_archive=True)
            baseline_keep = min(len(baseline), max(1, limit // 3))
            ranked = self._rank_candidate_pool(candidates, limit, final_counts)
            selected = baseline[:baseline_keep]
            selected_keys = {PointCloudArchive.key(proposal.point) for proposal in selected}
            for proposal in ranked:
                key = PointCloudArchive.key(proposal.point)
                if key in selected_keys:
                    continue
                selected.append(proposal)
                selected_keys.add(key)
                if len(selected) >= limit:
                    break
            candidates = selected
        else:
            candidates = candidates[:limit]
        fill_attempts = 0
        while len(candidates) < limit and fill_attempts < 4:
            fill_attempts += 1
            needed = limit - len(candidates)
            candidates = self._dedupe_proposals(
                [*candidates, *self._global_candidates(needed)],
                skip_archive=True,
            )[:limit]
        return candidates[:limit]

    def _finite_difference_gradient(self, point: np.ndarray) -> GradientResult:
        if self._remaining() <= 0:
            return GradientResult(None, 0, False, self._clip_point(point), float(self.best_value), 0)
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        spent_before = len(self.archive)
        current = self._clip_point(point)
        current_sample = self.archive.get(current)
        current_value = float(self.best_value if current_sample is None else current_sample.value)
        restarts = 0
        recentered = False
        improvement_count = 0
        while True:
            gradient = np.zeros(self.dimension, dtype=float)
            restarted = False
            for axis in range(self.dimension):
                if self._remaining() <= 1:
                    return GradientResult(
                        None,
                        len(self.archive) - spent_before,
                        recentered,
                        current.copy(),
                        current_value,
                        improvement_count,
                    )
                step = max(1e-6 * widths[axis], 1e-8 * max(abs(float(current[axis])), 1.0))
                plus = current.copy()
                plus[axis] = min(self.original_bounds[axis, 1], plus[axis] + step)
                plus_value, _plus_added, plus_improved = self._evaluate_point(plus, source="local_gradient")
                if plus_value is None or plus[axis] == current[axis]:
                    return GradientResult(
                        None,
                        len(self.archive) - spent_before,
                        recentered,
                        current.copy(),
                        current_value,
                        improvement_count,
                    )
                if (
                    self.config.probe_recenter_enabled
                    and plus_improved
                    and self._probe_recenter_improvement_is_meaningful(current_value, float(plus_value))
                    and restarts < int(self.config.probe_recenter_max_restarts)
                ):
                    restarts += 1
                    recentered = True
                    improvement_count += 1
                    current = self.best_point.copy()
                    current_value = float(self.best_value)
                    restarted = True
                    break
                minus = current.copy()
                minus[axis] = max(self.original_bounds[axis, 0], minus[axis] - step)
                minus_value, _minus_added, minus_improved = self._evaluate_point(minus, source="local_gradient")
                if minus_value is None or minus[axis] == current[axis] or plus[axis] == minus[axis]:
                    return GradientResult(
                        None,
                        len(self.archive) - spent_before,
                        recentered,
                        current.copy(),
                        current_value,
                        improvement_count,
                    )
                if (
                    self.config.probe_recenter_enabled
                    and minus_improved
                    and self._probe_recenter_improvement_is_meaningful(current_value, float(minus_value))
                    and restarts < int(self.config.probe_recenter_max_restarts)
                ):
                    restarts += 1
                    recentered = True
                    improvement_count += 1
                    current = self.best_point.copy()
                    current_value = float(self.best_value)
                    restarted = True
                    break
                gradient[axis] = (self._target(plus_value) - self._target(minus_value)) / (plus[axis] - minus[axis])
            if restarted:
                continue
            self._probe_recenters_total += int(improvement_count)
            return GradientResult(
                gradient,
                len(self.archive) - spent_before,
                recentered,
                current.copy(),
                current_value,
                improvement_count,
            )

    def _finite_difference_block_gradient(self, point: np.ndarray, axes: np.ndarray) -> GradientResult:
        if self._remaining() <= 0:
            return GradientResult(None, 0, False, self._clip_point(point), float(self.best_value), 0)
        resolved_axes = np.asarray(axes, dtype=int)
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        spent_before = len(self.archive)
        current = self._clip_point(point)
        current_sample = self.archive.get(current)
        current_value = float(self.best_value if current_sample is None else current_sample.value)
        restarts = 0
        recentered = False
        improvement_count = 0
        while True:
            gradient = np.zeros(resolved_axes.size, dtype=float)
            restarted = False
            for offset, axis in enumerate(resolved_axes):
                if self._remaining() <= 1:
                    return GradientResult(
                        None,
                        len(self.archive) - spent_before,
                        recentered,
                        current.copy(),
                        current_value,
                        improvement_count,
                    )
                step = max(1e-6 * widths[axis], 1e-8 * max(abs(float(current[axis])), 1.0))
                plus = current.copy()
                plus[axis] = min(self.original_bounds[axis, 1], plus[axis] + step)
                plus_value, _plus_added, plus_improved = self._evaluate_point(plus, source="block_gradient")
                if plus_value is None or plus[axis] == current[axis]:
                    return GradientResult(
                        None,
                        len(self.archive) - spent_before,
                        recentered,
                        current.copy(),
                        current_value,
                        improvement_count,
                    )
                if (
                    self.config.probe_recenter_enabled
                    and plus_improved
                    and self._probe_recenter_improvement_is_meaningful(current_value, float(plus_value))
                    and restarts < int(self.config.probe_recenter_max_restarts)
                ):
                    restarts += 1
                    recentered = True
                    improvement_count += 1
                    current = self.best_point.copy()
                    current_value = float(self.best_value)
                    restarted = True
                    break
                minus = current.copy()
                minus[axis] = max(self.original_bounds[axis, 0], minus[axis] - step)
                minus_value, _minus_added, minus_improved = self._evaluate_point(minus, source="block_gradient")
                if minus_value is None or minus[axis] == current[axis] or plus[axis] == minus[axis]:
                    return GradientResult(
                        None,
                        len(self.archive) - spent_before,
                        recentered,
                        current.copy(),
                        current_value,
                        improvement_count,
                    )
                if (
                    self.config.probe_recenter_enabled
                    and minus_improved
                    and self._probe_recenter_improvement_is_meaningful(current_value, float(minus_value))
                    and restarts < int(self.config.probe_recenter_max_restarts)
                ):
                    restarts += 1
                    recentered = True
                    improvement_count += 1
                    current = self.best_point.copy()
                    current_value = float(self.best_value)
                    restarted = True
                    break
                gradient[offset] = (self._target(plus_value) - self._target(minus_value)) / (plus[axis] - minus[axis])
            if restarted:
                continue
            self._probe_recenters_total += int(improvement_count)
            return GradientResult(
                gradient,
                len(self.archive) - spent_before,
                recentered,
                current.copy(),
                current_value,
                improvement_count,
            )

    def _finite_difference_hessian(self, point: np.ndarray, value: float) -> tuple[np.ndarray | None, int]:
        if self._remaining() <= 0:
            return None, 0
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        steps = np.asarray(
            [
                max(2e-5 * widths[axis], 1e-7 * max(abs(float(point[axis])), 1.0))
                for axis in range(self.dimension)
            ],
            dtype=float,
        )
        spent_before = len(self.archive)
        f0 = self._target(value)
        hessian = np.zeros((self.dimension, self.dimension), dtype=float)
        axis_values: list[tuple[float, float, float]] = []
        for axis in range(self.dimension):
            plus = point.copy()
            minus = point.copy()
            plus[axis] = min(self.original_bounds[axis, 1], plus[axis] + steps[axis])
            minus[axis] = max(self.original_bounds[axis, 0], minus[axis] - steps[axis])
            if plus[axis] == point[axis] or minus[axis] == point[axis] or plus[axis] == minus[axis]:
                return None, len(self.archive) - spent_before
            plus_value, _plus_added, _plus_improved = self._evaluate_point(plus, source="local_hessian")
            minus_value, _minus_added, _minus_improved = self._evaluate_point(minus, source="local_hessian")
            if plus_value is None or minus_value is None:
                return None, len(self.archive) - spent_before
            plus_target = self._target(plus_value)
            minus_target = self._target(minus_value)
            hessian[axis, axis] = (plus_target - 2.0 * f0 + minus_target) / (steps[axis] * steps[axis])
            axis_values.append((plus_target, minus_target, steps[axis]))
        for first in range(self.dimension):
            for second in range(first + 1, self.dimension):
                corner_targets: list[float] = []
                for sx, sy in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
                    corner = point.copy()
                    corner[first] = np.clip(
                        corner[first] + sx * steps[first],
                        self.original_bounds[first, 0],
                        self.original_bounds[first, 1],
                    )
                    corner[second] = np.clip(
                        corner[second] + sy * steps[second],
                        self.original_bounds[second, 0],
                        self.original_bounds[second, 1],
                    )
                    if corner[first] == point[first] or corner[second] == point[second]:
                        return None, len(self.archive) - spent_before
                    corner_value, _corner_added, _corner_improved = self._evaluate_point(
                        corner,
                        source="local_hessian",
                    )
                    if corner_value is None:
                        return None, len(self.archive) - spent_before
                    corner_targets.append(self._target(corner_value))
                cross = (corner_targets[0] - corner_targets[1] - corner_targets[2] + corner_targets[3]) / (
                    4.0 * axis_values[first][2] * axis_values[second][2]
                )
                hessian[first, second] = cross
                hessian[second, first] = cross
        if not np.all(np.isfinite(hessian)):
            return None, len(self.archive) - spent_before
        return 0.5 * (hessian + hessian.T), len(self.archive) - spent_before

    @staticmethod
    def _damped_newton_direction(
        gradient: np.ndarray,
        hessian: np.ndarray,
        damping: float,
    ) -> np.ndarray | None:
        try:
            eigenvalues = np.linalg.eigvalsh(hessian)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(eigenvalues)):
            return None
        shift = max(float(damping), -float(np.min(eigenvalues)) + float(damping))
        try:
            direction = -np.linalg.solve(hessian + shift * np.eye(hessian.shape[0], dtype=float), gradient)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(direction)) or float(np.dot(direction, gradient)) >= 0.0:
            return None
        return direction

    def _line_search_refinement(
        self,
        point: np.ndarray,
        value: float,
        gradient: np.ndarray,
        direction: np.ndarray,
        *,
        before: int,
        max_new: int,
        source: str = "local_refinement",
    ) -> tuple[bool, np.ndarray, float, bool]:
        accepted = False
        alpha = 1.0
        accepted_point = point.copy()
        accepted_value = value
        accepted_improved = False
        directional = float(np.dot(gradient, direction))
        for _ in range(32):
            if self._remaining() <= 0 or len(self.archive) - before >= max_new:
                break
            candidate = self._clip_point(point + alpha * direction)
            candidate_value, _added, improved = self._evaluate_point(candidate, source=source)
            if candidate_value is None:
                break
            candidate_target = self._target(candidate_value)
            current_target = self._target(value)
            if candidate_target <= current_target + 1e-4 * alpha * directional or candidate_target < current_target:
                accepted = True
                accepted_point = candidate
                accepted_value = float(candidate_value)
                accepted_improved = bool(improved)
                break
            alpha *= 0.5
            if alpha < 1e-12:
                break
        return accepted, accepted_point, accepted_value, accepted_improved

    def _run_local_refinement(self) -> PointCloudBatchEvent | None:
        if self._high_dimensional_refinement_active():
            return self._run_block_local_refinement()
        if self.dimension >= int(self.config.high_dimensional_min_dimension):
            return None
        if not self.config.local_refinement_enabled or len(self.archive) < int(self.config.local_refinement_start_evaluations):
            return None
        if not np.isfinite(self.best_value) or self._remaining() <= 4:
            return None
        before = len(self.archive)
        best_before = float(self.best_value)
        max_new = min(int(self.config.local_refinement_max_evaluations), self._remaining())
        hessian_inverse = np.eye(self.dimension, dtype=float)
        point = self.best_point.copy()
        value = float(self.best_value)
        gradient_result = self._finite_difference_gradient(point)
        gradient = gradient_result.gradient
        if gradient is None:
            return None
        point = gradient_result.point.copy()
        value = float(gradient_result.value)
        probe_recenters = int(gradient_result.improvement_count)
        improvements = 0
        lm_attempts = 0
        lm_accepted = 0
        bfgs_steps = 0
        gradient_fallbacks = 0
        hessian_failures = 0
        damping_increases = 0
        damping_decreases = 0
        fallback_count = 0
        damping = float(self.config.local_refinement_damping)
        method = self.config.local_refinement_method
        gradient_norm = float(np.linalg.norm(gradient))
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        max_step_norm = float(self.config.local_refinement_step_fraction) * float(np.linalg.norm(widths))
        while len(self.archive) - before < max_new and self._remaining() > 0:
            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm <= float(self.config.local_refinement_gradient_tolerance):
                self._local_refinement_stalled = True
                break
            direction: np.ndarray | None = None
            direction_method = "bfgs"
            use_lm = (method == "levenberg-marquardt" and self.dimension <= int(self.config.surrogate_full_quadratic_max_dimension)) or (
                method == "hybrid"
                and self.dimension <= int(self.config.surrogate_full_quadratic_max_dimension)
                and gradient_norm <= 1e-4
                and len(self.archive) - before >= 16
            )
            if use_lm and len(self.archive) - before < max_new:
                lm_attempts += 1
                hessian, _spent = self._finite_difference_hessian(point, value)
                if hessian is None:
                    hessian_failures += 1
                else:
                    direction = self._damped_newton_direction(gradient, hessian, damping)
                    if direction is None:
                        hessian_failures += 1
                    else:
                        direction_method = "levenberg_marquardt"
            if direction is None:
                if method == "levenberg-marquardt":
                    fallback_count += 1
                direction = -hessian_inverse @ gradient
                direction_method = "bfgs"
            if not np.all(np.isfinite(direction)) or float(np.dot(direction, gradient)) >= 0.0:
                hessian_inverse = np.eye(self.dimension, dtype=float)
                direction = -gradient
                direction_method = "gradient"
                gradient_fallbacks += 1
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-14:
                self._local_refinement_stalled = True
                break
            if norm > max_step_norm:
                direction = direction * (max_step_norm / norm)
            accepted, accepted_point, accepted_value, accepted_improved = self._line_search_refinement(
                point,
                value,
                gradient,
                direction,
                before=before,
                max_new=max_new,
            )
            if not accepted and direction_method == "levenberg_marquardt":
                damping = min(damping * 10.0, 1e12)
                damping_increases += 1
                if method == "hybrid":
                    fallback_count += 1
                    direction = -hessian_inverse @ gradient
                    direction_method = "bfgs"
                    if not np.all(np.isfinite(direction)) or float(np.dot(direction, gradient)) >= 0.0:
                        hessian_inverse = np.eye(self.dimension, dtype=float)
                        direction = -gradient
                        direction_method = "gradient"
                        gradient_fallbacks += 1
                    norm = float(np.linalg.norm(direction))
                    if norm > max_step_norm:
                        direction = direction * (max_step_norm / norm)
                    accepted, accepted_point, accepted_value, accepted_improved = self._line_search_refinement(
                        point,
                        value,
                        gradient,
                        direction,
                        before=before,
                        max_new=max_new,
                    )
            if accepted_improved:
                improvements += 1
            if not accepted:
                hessian_inverse = np.eye(self.dimension, dtype=float)
                gradient_result = self._finite_difference_gradient(point)
                gradient = gradient_result.gradient
                probe_recenters += int(gradient_result.improvement_count)
                if gradient is None:
                    break
                point = gradient_result.point.copy()
                value = float(gradient_result.value)
                continue
            if direction_method == "levenberg_marquardt":
                lm_accepted += 1
                damping = max(float(self.config.local_refinement_damping) * 1e-6, damping * 0.3)
                damping_decreases += 1
            else:
                bfgs_steps += 1
            if self._early_stop_reached():
                point = accepted_point
                value = accepted_value
                break
            next_gradient_result = self._finite_difference_gradient(accepted_point)
            next_gradient = next_gradient_result.gradient
            probe_recenters += int(next_gradient_result.improvement_count)
            if next_gradient is None:
                point = accepted_point
                value = accepted_value
                break
            next_point = next_gradient_result.point.copy()
            next_value = float(next_gradient_result.value)
            step = next_point - point
            gradient_delta = next_gradient - gradient
            curvature = float(np.dot(gradient_delta, step))
            if curvature > 1e-12:
                rho = 1.0 / curvature
                identity = np.eye(self.dimension, dtype=float)
                hessian_inverse = (
                    (identity - rho * np.outer(step, gradient_delta))
                    @ hessian_inverse
                    @ (identity - rho * np.outer(gradient_delta, step))
                    + rho * np.outer(step, step)
                )
            else:
                hessian_inverse = np.eye(self.dimension, dtype=float)
            point = next_point
            value = next_value
            gradient = next_gradient
        after = len(self.archive)
        if after == before:
            return None
        source_counts = Counter(sample.source for sample in self.archive.samples[before:])
        event = PointCloudBatchEvent(
            batch_index=self._batch_index,
            kind="local_refinement",
            evaluations_before=before,
            evaluations_after=after,
            candidate_count=after - before,
            source_counts=dict(source_counts),
            best_before=best_before,
            best_after=float(self.best_value),
            best_point=self.best_point.copy(),
            improved=self._is_better(self.best_value, best_before),
            diagnostics={
                "improvements": int(improvements),
                "gradient_norm": float(gradient_norm),
                "local_refinement_stalled": bool(self._local_refinement_stalled),
                "local_refinement_method": method,
                "lm_attempts": int(lm_attempts),
                "lm_accepted_steps": int(lm_accepted),
                "bfgs_steps": int(bfgs_steps),
                "gradient_fallbacks": int(gradient_fallbacks),
                "hessian_failures": int(hessian_failures),
                "damping_final": float(damping),
                "damping_increases": int(damping_increases),
                "damping_decreases": int(damping_decreases),
                "fallback_count": int(fallback_count),
                "probe_recenters": int(probe_recenters),
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
                    "local_refinement_stalled": bool(self._local_refinement_stalled),
                    "local_refinement_method": method,
                    "lm_accepted_steps": int(lm_accepted),
                    "fallback_count": int(fallback_count),
                    "probe_recenters": int(probe_recenters),
                }
            )
        return event

    def _block_size(self) -> int:
        return min(self.dimension, max(1, int(self.config.active_subspace_size)))

    def _block_starts(self) -> list[int]:
        block_size = self._block_size()
        overlap = min(int(self.config.block_refinement_overlap), max(0, block_size - 1))
        stride = max(1, block_size - overlap)
        starts: list[int] = []
        start = 0
        while start < self.dimension:
            starts.append(min(start, self.dimension - block_size))
            start += stride
        return sorted(set(starts))

    def _activity_centered_block(self) -> np.ndarray | None:
        if not np.any(self._axis_activity > 0.0):
            return None
        block_size = self._block_size()
        axis = int(np.argmax(self._axis_activity))
        start = min(max(axis - block_size // 2, 0), self.dimension - block_size)
        return np.arange(start, start + block_size, dtype=int)

    def _next_refinement_blocks(self) -> list[np.ndarray]:
        starts = self._block_starts()
        blocks: list[np.ndarray] = []
        target_count = int(self.config.block_refinement_blocks_per_pass)
        if self.dimension >= 100:
            target_count = max(target_count, min(4, int(np.ceil(self.dimension / max(4 * self._block_size(), 1)))))
        for _slot in range(target_count):
            start = starts[self._block_cursor % len(starts)]
            self._block_cursor += 1
            block = np.arange(start, start + self._block_size(), dtype=int)
            key = tuple(int(axis) for axis in block)
            if any(tuple(int(axis) for axis in existing) == key for existing in blocks):
                continue
            blocks.append(block)
        activity_block = self._activity_centered_block()
        if activity_block is not None and blocks:
            activity_key = tuple(int(axis) for axis in activity_block)
            if not any(tuple(int(axis) for axis in existing) == activity_key for existing in blocks):
                blocks[-1] = activity_block
        linked_block = self._linked_block()
        if linked_block is not None and blocks:
            linked_key = tuple(int(axis) for axis in linked_block)
            if not any(tuple(int(axis) for axis in existing) == linked_key for existing in blocks):
                blocks[0] = linked_block
        return blocks

    def _run_block_local_refinement(self) -> PointCloudBatchEvent | None:
        if not self.config.local_refinement_enabled or len(self.archive) < int(self.config.local_refinement_start_evaluations):
            return None
        if not np.isfinite(self.best_value) or self._remaining() <= 4:
            return None
        before = len(self.archive)
        best_before = float(self.best_value)
        max_new = min(int(self.config.local_refinement_max_evaluations), self._remaining())
        point = self.best_point.copy()
        value = float(self.best_value)
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        max_step_norm = float(self.config.local_refinement_step_fraction) * float(np.linalg.norm(widths))
        improvements = 0
        accepted_steps = 0
        block_stalls = 0
        gradient_evaluations = 0
        probe_recenters = 0
        lbfgs_direction_accepted = 0
        linkage_blocks_used = 0
        block_payloads: list[dict[str, object]] = []
        for axes in self._next_refinement_blocks():
            if self._remaining() <= 2 or len(self.archive) - before >= max_new:
                break
            key = tuple(int(axis) for axis in axes)
            if self.config.linkage_blocks_enabled and any(
                float(self._linkage_scores[int(axis), int(other)]) > 0.0
                for offset, axis in enumerate(axes)
                for other in axes[offset + 1 :]
            ):
                linkage_blocks_used += 1
            hessian_inverse = self._block_hessian_inverse.get(key)
            if hessian_inverse is None or hessian_inverse.shape != (axes.size, axes.size):
                hessian_inverse = np.eye(axes.size, dtype=float)
            gradient_result = self._finite_difference_block_gradient(point, axes)
            gradient = gradient_result.gradient
            gradient_evaluations += int(gradient_result.spent)
            probe_recenters += int(gradient_result.improvement_count)
            if gradient is None:
                block_stalls += 1
                block_payloads.append({"axes": list(key), "accepted": False, "reason": "gradient_unavailable"})
                continue
            point = gradient_result.point.copy()
            value = float(gradient_result.value)
            gradient_norm = float(np.linalg.norm(gradient))
            if gradient_norm <= float(self.config.local_refinement_gradient_tolerance):
                block_stalls += 1
                block_payloads.append({"axes": list(key), "accepted": False, "reason": "small_gradient"})
                continue
            block_direction = -hessian_inverse @ gradient
            full_gradient = np.zeros(self.dimension, dtype=float)
            full_gradient[axes] = gradient
            normalized_full_gradient = full_gradient * widths
            lbfgs_direction = self._lbfgs_direction(normalized_full_gradient)
            if lbfgs_direction is not None:
                candidate_direction = lbfgs_direction * widths
                if np.all(np.isfinite(candidate_direction)) and float(np.dot(candidate_direction, full_gradient)) < 0.0:
                    direction = candidate_direction
                    lbfgs_direction_accepted += 1
                else:
                    direction = np.zeros(self.dimension, dtype=float)
                    direction[axes] = block_direction
            else:
                direction = np.zeros(self.dimension, dtype=float)
                direction[axes] = block_direction
            if not np.all(np.isfinite(block_direction)) or float(np.dot(block_direction, gradient)) >= 0.0:
                hessian_inverse = np.eye(axes.size, dtype=float)
                block_direction = -gradient
                if lbfgs_direction is None:
                    direction = np.zeros(self.dimension, dtype=float)
                    direction[axes] = block_direction
            norm = float(np.linalg.norm(direction))
            if norm <= 1e-14:
                block_stalls += 1
                block_payloads.append({"axes": list(key), "accepted": False, "reason": "zero_direction"})
                continue
            if norm > max_step_norm:
                direction = direction * (max_step_norm / norm)
            accepted, accepted_point, accepted_value, accepted_improved = self._line_search_refinement(
                point,
                value,
                full_gradient,
                direction,
                before=before,
                max_new=max_new,
                source="block_refinement",
            )
            if not accepted:
                self._block_hessian_inverse[key] = np.eye(axes.size, dtype=float)
                block_stalls += 1
                block_payloads.append({"axes": list(key), "accepted": False, "reason": "line_search_rejected"})
                continue
            accepted_steps += 1
            if accepted_improved:
                improvements += 1
            if self._early_stop_reached():
                point = accepted_point
                value = accepted_value
                block_payloads.append({"axes": list(key), "accepted": True, "early_stop": True})
                break
            next_gradient_result = self._finite_difference_block_gradient(accepted_point, axes)
            next_gradient = next_gradient_result.gradient
            gradient_evaluations += int(next_gradient_result.spent)
            probe_recenters += int(next_gradient_result.improvement_count)
            if next_gradient is not None:
                next_point = next_gradient_result.point.copy()
                next_value = float(next_gradient_result.value)
                step = next_point[axes] - point[axes]
                gradient_delta = next_gradient - gradient
                curvature = float(np.dot(gradient_delta, step))
                if curvature > 1e-12:
                    rho = 1.0 / curvature
                    identity = np.eye(axes.size, dtype=float)
                    hessian_inverse = (
                        (identity - rho * np.outer(step, gradient_delta))
                        @ hessian_inverse
                        @ (identity - rho * np.outer(gradient_delta, step))
                        + rho * np.outer(step, step)
                    )
                else:
                    hessian_inverse = np.eye(axes.size, dtype=float)
                normalized_step = (next_point - point) / widths
                next_full_gradient = np.zeros(self.dimension, dtype=float)
                next_full_gradient[axes] = next_gradient
                self._store_lbfgs_pair(normalized_step, next_full_gradient * widths - normalized_full_gradient)
            else:
                hessian_inverse = np.eye(axes.size, dtype=float)
                next_point = accepted_point
                next_value = accepted_value
            self._block_hessian_inverse[key] = hessian_inverse
            normalized_step = np.abs((next_point - point) / widths)
            self._axis_activity *= 0.99
            self._axis_activity[axes] += normalized_step[axes]
            block_payloads.append(
                {
                    "axes": list(key),
                    "accepted": True,
                    "gradient_norm": gradient_norm,
                    "step_norm": float(np.linalg.norm(next_point - point)),
                }
            )
            point = next_point
            value = next_value
        after = len(self.archive)
        if after == before:
            return None
        source_counts = Counter(sample.source for sample in self.archive.samples[before:])
        top_axis_count = min(8, self.dimension)
        top_axes = np.argsort(self._axis_activity)[::-1][:top_axis_count]
        event = PointCloudBatchEvent(
            batch_index=self._batch_index,
            kind="local_refinement",
            evaluations_before=before,
            evaluations_after=after,
            candidate_count=after - before,
            source_counts=dict(source_counts),
            best_before=best_before,
            best_after=float(self.best_value),
            best_point=self.best_point.copy(),
            improved=self._is_better(self.best_value, best_before),
            diagnostics={
                "improvements": int(improvements),
                "local_refinement_stalled": False,
                "local_refinement_method": self.config.local_refinement_method,
                "local_refinement_variant": "block_bfgs",
                "high_dimensional_refinement_enabled": bool(self.config.high_dimensional_refinement_enabled),
                "effective_batch_size": int(self._effective_batch_size()),
                "block_size": int(self._block_size()),
                "block_axes": [payload["axes"] for payload in block_payloads],
                "block_gradient_evaluations": int(gradient_evaluations),
                "accepted_block_steps": int(accepted_steps),
                "block_stalls": int(block_stalls),
                "probe_recenters": int(probe_recenters),
                "basin_polishing_active": bool(self._basin_polishing_active()),
                "anchor_baseline_target": (
                    None if self._anchor_baseline_target is None else float(self._anchor_baseline_target)
                ),
                "linkage_blocks_used": int(linkage_blocks_used),
                "lbfgs_pairs": int(len(self._lbfgs_pairs)),
                "lbfgs_direction_accepted": int(lbfgs_direction_accepted),
                "successful_direction_memory_size": int(len(self._successful_directions)),
                "top_axis_activity": [
                    {"axis": int(axis), "score": float(self._axis_activity[axis])}
                    for axis in top_axes
                    if self._axis_activity[axis] > 0.0
                ],
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
                    "local_refinement_stalled": False,
                    "local_refinement_method": self.config.local_refinement_method,
                    "local_refinement_variant": "block_bfgs",
                    "accepted_block_steps": int(accepted_steps),
                    "block_stalls": int(block_stalls),
                    "probe_recenters": int(probe_recenters),
                    "linkage_blocks_used": int(linkage_blocks_used),
                    "lbfgs_pairs": int(len(self._lbfgs_pairs)),
                    "lbfgs_direction_accepted": int(lbfgs_direction_accepted),
                }
            )
        return event

    def _direction_set(self) -> list[np.ndarray]:
        directions: list[np.ndarray] = []

        def add(direction: np.ndarray) -> None:
            resolved = np.asarray(direction, dtype=float)
            if resolved.shape != (self.dimension,) or not np.all(np.isfinite(resolved)):
                return
            norm = float(np.linalg.norm(resolved))
            if norm <= 1e-14:
                return
            unit = resolved / norm
            if any(abs(float(np.dot(unit, existing))) > 0.995 for existing in directions):
                return
            directions.append(unit)

        for direction in reversed(self._successful_directions):
            add(direction)
            if len(directions) >= int(self.config.successful_direction_memory_size):
                break
        widths = np.maximum(self.original_bounds[:, 1] - self.original_bounds[:, 0], 1e-12)
        if np.linalg.norm(self._last_successful_step / widths) > 1e-14:
            add(self._last_successful_step / widths)
        geometry = self._fit_point_geometry(self.best_point, float(self.config.region_initial_radius_fraction))
        if geometry.basis.shape[0] == self.dimension:
            for axis in range(min(geometry.basis.shape[1], int(self.config.active_subspace_size))):
                add(np.asarray(geometry.basis[:, axis], dtype=float))
        for state in self._cma_states.values():
            add(state.evolution_path)
            for direction in state.directions[-2:]:
                add(direction)
        if self._lbfgs_pairs:
            for step, _gradient_delta in self._lbfgs_pairs[-4:]:
                add(step)
        if not directions:
            active = np.argsort(self._axis_activity)[::-1][: min(self.dimension, int(self.config.active_subspace_size))]
            for axis in active:
                direction = np.zeros(self.dimension, dtype=float)
                direction[int(axis)] = 1.0
                add(direction)
        return directions[: max(1, int(self.config.successful_direction_memory_size))]

    def _run_direction_refinement(self) -> PointCloudBatchEvent | None:
        if not (
            self.config.direction_refinement_enabled
            and self._evolutionary_search_active()
            and np.isfinite(self.best_value)
            and self._remaining() > 0
        ):
            return None
        if not self._basin_polishing_active() and self.config.local_refinement_enabled:
            return None
        if not self._basin_polishing_active() and len(self._successful_directions) < 2:
            return None
        max_new = min(int(self.config.direction_refinement_max_evaluations), self._remaining())
        if max_new <= 0:
            return None
        directions = self._direction_set()
        if not directions:
            return None
        before = len(self.archive)
        best_before = float(self.best_value)
        source_attempts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        source_improvements: Counter[str] = Counter()
        source_amounts: Counter[str] = Counter()
        improvements = 0
        widths = np.maximum(self.original_bounds[:, 1] - self.original_bounds[:, 0], 1e-12)
        step_fraction = min(
            0.08,
            max(float(self.config.region_min_radius_fraction) * 10.0, float(self._stencil_step_fraction)),
        )
        passes = 0
        while len(self.archive) - before < max_new and self._remaining() > 0:
            passes += 1
            improved_this_pass = False
            normalized_center = self._normalized_point(self.best_point)
            for direction in directions:
                if len(self.archive) - before >= max_new or self._remaining() <= 0:
                    break
                for sign in (1.0, -1.0):
                    if len(self.archive) - before >= max_new or self._remaining() <= 0:
                        break
                    current_before = float(self.best_value)
                    candidate = self._point_from_normalized(normalized_center + sign * step_fraction * direction)
                    source_attempts["direction_line_search"] += 1
                    value, added, improved = self._evaluate_point(candidate, source="direction_line_search")
                    if not added:
                        continue
                    source_counts["direction_line_search"] += 1
                    if improved and value is not None:
                        improvements += 1
                        source_improvements["direction_line_search"] += 1
                        source_amounts["direction_line_search"] += max(
                            0.0,
                            self._target(current_before) - self._target(float(value)),
                        )
                        improved_this_pass = True
                        pattern_direction = (self.best_point - self._point_from_normalized(normalized_center)) / widths
                        pattern_norm = float(np.linalg.norm(pattern_direction))
                        if pattern_norm > 1e-14 and np.isfinite(pattern_norm):
                            pattern_direction = pattern_direction / pattern_norm
                            for multiplier in (1.5, 2.25):
                                if len(self.archive) - before >= max_new or self._remaining() <= 0:
                                    break
                                pattern_before = float(self.best_value)
                                pattern = self._point_from_normalized(
                                    self._normalized_point(self.best_point)
                                    + multiplier * step_fraction * pattern_direction
                                )
                                source_attempts["direction_pattern"] += 1
                                pattern_value, pattern_added, pattern_improved = self._evaluate_point(
                                    pattern,
                                    source="direction_pattern",
                                )
                                if not pattern_added:
                                    break
                                source_counts["direction_pattern"] += 1
                                if pattern_improved and pattern_value is not None:
                                    improvements += 1
                                    source_improvements["direction_pattern"] += 1
                                    source_amounts["direction_pattern"] += max(
                                        0.0,
                                        self._target(pattern_before) - self._target(float(pattern_value)),
                                    )
                                else:
                                    break
                        break
                if improved_this_pass:
                    break
            if improved_this_pass:
                self._stencil_step_fraction = min(
                    float(self.config.region_initial_radius_fraction),
                    max(step_fraction, self._stencil_step_fraction * 1.20),
                )
                directions = self._direction_set()
                continue
            self._stencil_step_fraction = max(float(self.config.region_min_radius_fraction), self._stencil_step_fraction * 0.5)
            if passes >= 2 or self._stencil_step_fraction <= float(self.config.region_min_radius_fraction):
                break
            step_fraction = self._stencil_step_fraction
        after = len(self.archive)
        if after == before:
            return None
        self._record_source_results(source_attempts, source_counts, source_improvements, source_amounts)
        self._refresh_live_population()
        return PointCloudBatchEvent(
            batch_index=self._batch_index,
            kind="direction_refinement",
            evaluations_before=before,
            evaluations_after=after,
            candidate_count=int(sum(source_attempts.values())),
            source_counts=dict(source_counts),
            best_before=best_before,
            best_after=float(self.best_value),
            best_point=self.best_point.copy(),
            improved=self._is_better(self.best_value, best_before),
            diagnostics={
                "improvements": int(improvements),
                "source_improvements": dict(source_improvements),
                "direction_line_search_improvements": int(source_improvements.get("direction_line_search", 0)),
                "direction_pattern_improvements": int(source_improvements.get("direction_pattern", 0)),
                "successful_direction_memory_size": int(len(self._successful_directions)),
                "basin_polishing_active": bool(self._basin_polishing_active()),
                "anchor_baseline_target": (
                    None if self._anchor_baseline_target is None else float(self._anchor_baseline_target)
                ),
                "linkage_blocks_used": 0,
                "lbfgs_pairs": int(len(self._lbfgs_pairs)),
                "lbfgs_direction_accepted": 0,
                "archive_size": int(after),
            },
        )

    def _run_cooperative_refinement(self) -> PointCloudBatchEvent | None:
        if not self._cooperative_refinement_active() or self._remaining() <= 0:
            return None
        groups = self._cooperative_groups()
        if not groups:
            return None
        before = len(self.archive)
        best_before = float(self.best_value)
        group_size = self._cooperative_group_size()
        max_new = min(int(self._effective_batch_size()), self._remaining())
        source_attempts: Counter[str] = Counter()
        source_counts: Counter[str] = Counter()
        source_improvements: Counter[str] = Counter()
        source_amounts: Counter[str] = Counter()
        improvements = 0
        groups_used = 0
        widths = np.maximum(self.original_bounds[:, 1] - self.original_bounds[:, 0], 1e-12)
        step_floor = max(float(self.config.region_min_radius_fraction) * 100.0, 5e-4)
        step_base = float(np.clip(max(step_floor, min(float(self._stencil_step_fraction), 0.015)), 5e-4, 0.025))

        def evaluate(normalized: np.ndarray, source: str, axes: np.ndarray) -> bool:
            nonlocal improvements
            if self._remaining() <= 0 or len(self.archive) - before >= max_new:
                return False
            current_before = float(self.best_value)
            point = self._point_from_normalized(normalized)
            source_attempts[source] += 1
            self._axis_coverage[axes] += 1.0
            value, added, improved = self._evaluate_point(point, source=source)
            if not added:
                return False
            source_counts[source] += 1
            if improved and value is not None:
                amount = max(0.0, self._target(current_before) - self._target(float(value)))
                source_improvements[source] += 1
                source_amounts[source] += amount
                improvements += 1
                self._axis_activity *= 0.999
                self._axis_activity[axes] += max(amount, 1e-12)
                return True
            return False

        passes = 0
        while len(self.archive) - before < max_new and self._remaining() > 0 and passes < 2:
            passes += 1
            improved_this_pass = False
            axis_scores = self._cooperative_axis_scores()
            group_quota = max(4, int(np.ceil(max_new / max(len(groups), 1))))
            for axes in groups:
                if len(self.archive) - before >= max_new or self._remaining() <= 0:
                    break
                group_before = len(self.archive)
                groups_used += 1
                normalized_center = self._normalized_point(self.best_point)
                ordered_axes = axes[np.argsort(axis_scores[axes])[::-1]]
                for axis in ordered_axes:
                    if (
                        len(self.archive) - before >= max_new
                        or len(self.archive) - group_before >= group_quota
                        or self._remaining() <= 0
                    ):
                        break
                    direction = np.zeros(self.dimension, dtype=float)
                    direction[int(axis)] = 1.0
                    accepted_axis = False
                    for sign in (1.0, -1.0):
                        alpha = step_base * 0.25
                        misses = 0
                        for _trial in range(6):
                            if (
                                len(self.archive) - before >= max_new
                                or len(self.archive) - group_before >= group_quota
                                or self._remaining() <= 0
                            ):
                                break
                            if evaluate(
                                normalized_center + sign * alpha * direction,
                                "cooperative:line_search",
                                axes,
                            ):
                                step_direction = (self.best_point - self._point_from_normalized(normalized_center)) / widths
                                norm = float(np.linalg.norm(step_direction))
                                if norm > 1e-14 and np.isfinite(norm):
                                    pattern = self._normalized_point(self.best_point) + min(alpha * 1.35, 0.08) * step_direction / norm
                                    evaluate(pattern, "cooperative:group", axes)
                                normalized_center = self._normalized_point(self.best_point)
                                accepted_axis = True
                                improved_this_pass = True
                                alpha = min(alpha * 1.70, 0.08)
                                misses = 0
                                continue
                            misses += 1
                            alpha = min(alpha * 2.0, 0.08)
                            if misses >= 2:
                                break
                        if (
                            len(self.archive) - before >= max_new
                            or len(self.archive) - group_before >= group_quota
                            or self._remaining() <= 0
                        ):
                            break
                    if accepted_axis:
                        normalized_center = self._normalized_point(self.best_point)
                direction_candidates: list[np.ndarray] = []
                for direction in reversed(self._successful_directions[-4:]):
                    projected = np.zeros(self.dimension, dtype=float)
                    projected[axes] = np.asarray(direction, dtype=float)[axes]
                    norm = float(np.linalg.norm(projected))
                    if norm > 1e-14 and np.isfinite(norm):
                        direction_candidates.append(projected / norm)
                if self._lbfgs_pairs:
                    for step, _gradient_delta in self._lbfgs_pairs[-2:]:
                        projected = np.zeros(self.dimension, dtype=float)
                        projected[axes] = step[axes]
                        norm = float(np.linalg.norm(projected))
                        if norm > 1e-14 and np.isfinite(norm):
                            direction_candidates.append(projected / norm)
                for direction in direction_candidates[:3]:
                    if (
                        len(self.archive) - before >= max_new
                        or len(self.archive) - group_before >= group_quota
                        or self._remaining() <= 0
                    ):
                        break
                    normalized_center = self._normalized_point(self.best_point)
                    for sign in (1.0, -1.0):
                        if evaluate(
                            normalized_center + sign * step_base * direction,
                            "cooperative:line_search",
                            axes,
                        ):
                            improved_this_pass = True
                            break
                    if improved_this_pass:
                        break
            if improved_this_pass:
                self._stencil_step_fraction = min(
                    float(self.config.region_initial_radius_fraction),
                    max(float(self._stencil_step_fraction), step_base * 1.15),
                )
                groups = self._cooperative_groups()
                continue
            self._stencil_step_fraction = max(float(self.config.region_min_radius_fraction), self._stencil_step_fraction * 0.75)
            break
        after = len(self.archive)
        if after == before:
            return None
        self._record_source_results(source_attempts, source_counts, source_improvements, source_amounts)
        self._refresh_live_population()
        top_axes = np.argsort(self._cooperative_axis_scores())[::-1][: min(8, self.dimension)]
        return PointCloudBatchEvent(
            batch_index=self._batch_index,
            kind="cooperative_refinement",
            evaluations_before=before,
            evaluations_after=after,
            candidate_count=int(sum(source_attempts.values())),
            source_counts=dict(source_counts),
            best_before=best_before,
            best_after=float(self.best_value),
            best_point=self.best_point.copy(),
            improved=self._is_better(self.best_value, best_before),
            diagnostics={
                "improvements": int(improvements),
                "source_improvements": dict(source_improvements),
                "source_improvement_amounts": dict(source_amounts),
                "cooperative_refinement_active": bool(self._cooperative_refinement_active()),
                "active_set_size": int(self._active_set_axes().size),
                "cooperative_groups_used": int(groups_used),
                "cooperative_improvements": int(improvements),
                "active_set_expansions": int(self._active_set_expansions),
                "axis_coverage_max": float(np.max(self._axis_coverage)) if self._axis_coverage.size else 0.0,
                "top_axis_scores": [
                    {"axis": int(axis), "score": float(self._cooperative_axis_scores()[axis])}
                    for axis in top_axes
                ],
                "archive_size": int(after),
            },
        )

    def _run_stencil_refinement(self) -> PointCloudBatchEvent | None:
        if not np.isfinite(self.best_value) or self._remaining() <= 0:
            return None
        before = len(self.archive)
        best_before = float(self.best_value)
        max_new = min(int(self._effective_batch_size()), self._remaining())
        improvements = 0
        source_counts: Counter[str] = Counter()
        source_improvements: Counter[str] = Counter()
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        min_step = float(self.config.region_min_radius_fraction)
        max_step = float(self.config.region_initial_radius_fraction)
        while len(self.archive) - before < max_new and self._remaining() > 0:
            step = np.maximum(self._stencil_step_fraction * widths, min_step * widths)
            improved_this_round = False
            if self.config.surrogate_enabled:
                surrogate = self._fit_point_surrogate(self.best_point, max(self._stencil_step_fraction, max_step))
                if surrogate.accepted and surrogate.point is not None:
                    _value, added, improved = self._evaluate_point(surrogate.point, source="exploit_surrogate")
                    if added:
                        source_counts["exploit_surrogate"] += 1
                        if improved:
                            improvements += 1
                            source_improvements["exploit_surrogate"] += 1
                            improved_this_round = True
                    if len(self.archive) - before >= max_new or self._remaining() <= 0:
                        break
            stencil_candidates = [(point, "exploit_stencil") for point in self._stencil_points(self.best_point, step)]
            for point, source in stencil_candidates:
                if len(self.archive) - before >= max_new or self._remaining() <= 0:
                    break
                previous_best = self.best_point.copy()
                _value, added, improved = self._evaluate_point(point, source=source)
                if not added:
                    continue
                source_counts[source] += 1
                if improved:
                    improvements += 1
                    source_improvements[source] += 1
                    improved_this_round = True
                    pattern_step = self.best_point - previous_best
                    for multiplier in (1.0, 2.0, 3.0):
                        if len(self.archive) - before >= max_new or self._remaining() <= 0:
                            break
                        pattern_point = self._clip_point(self.best_point + multiplier * pattern_step)
                        _pattern_value, pattern_added, pattern_improved = self._evaluate_point(
                            pattern_point,
                            source="exploit_pattern",
                        )
                        if not pattern_added:
                            break
                        source_counts["exploit_pattern"] += 1
                        if pattern_improved:
                            improvements += 1
                            source_improvements["exploit_pattern"] += 1
                            improved_this_round = True
                            pattern_step = self._last_successful_step.copy()
                        else:
                            break
            if improved_this_round:
                self._stencil_step_fraction = min(max_step, self._stencil_step_fraction * 1.25)
            else:
                self._stencil_step_fraction = max(min_step, self._stencil_step_fraction * 0.5)
            if self._stencil_step_fraction <= min_step and not improved_this_round:
                break
        after = len(self.archive)
        if after == before:
            return None
        return PointCloudBatchEvent(
            batch_index=self._batch_index,
            kind="stencil_refinement",
            evaluations_before=before,
            evaluations_after=after,
            candidate_count=after - before,
            source_counts=dict(source_counts),
            best_before=best_before,
            best_after=float(self.best_value),
            best_point=self.best_point.copy(),
            improved=self._is_better(self.best_value, best_before),
            diagnostics={
                "improvements": int(improvements),
                "source_improvements": dict(source_improvements),
                "stencil_step_fraction": float(self._stencil_step_fraction),
                "rotated_stencil_improvements": int(source_improvements.get("exploit_rotated_stencil", 0)),
                "pattern_improvements": int(source_improvements.get("exploit_pattern", 0)),
                "surrogate_improvements": int(source_improvements.get("exploit_surrogate", 0)),
                "archive_size": int(after),
            },
        )

    def _early_stop_reached(self) -> bool:
        if not self.config.early_stop_enabled or self.config.early_stop_value is None:
            return False
        if self.smoothlife_config.maximize:
            return float(self.best_value) >= float(self.config.early_stop_value)
        return float(self.best_value) <= float(self.config.early_stop_value)

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
                "active_region_count": int(len(self._active_regions())),
                "sleeping_region_count": int(self._sleeping_region_count()),
                "explored_fraction": min(1.0, len(self.archive) / max(self._evaluation_limit(), 1)),
                "basin_polishing_active": bool(self._basin_polishing_active()),
                "anchor_baseline_target": (
                    None if self._anchor_baseline_target is None else float(self._anchor_baseline_target)
                ),
                "cooperative_refinement_active": bool(self._cooperative_refinement_active()),
                "active_set_size": int(self._active_set_axes().size) if self._cooperative_refinement_active() else 0,
            },
        )
        self.snapshots.append(snapshot)

    def run(self, evaluations: int | None = None) -> SearchRun:
        """Run until the evaluation budget, max-batch cap, or explicit early stop is reached."""

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
            self._refresh_anchor_baseline_target()
            self._update_portfolio()
            self._capture_snapshot(force=True, candidate_points=np.asarray([point for point, _source in initial], dtype=float))

        self._stop_reason = "budget_exhausted" if self._remaining() <= 0 else "running"

        while self._remaining() > 0:
            if self._early_stop_reached():
                self._stop_reason = "early_stop_value"
                break
            if self.config.max_batches is not None and self._batch_index >= int(self.config.max_batches):
                self._stop_reason = "max_batches"
                break
            self._batch_index += 1
            self._refresh_views()
            self._update_portfolio()
            self._update_linkage_scores()
            pre_stencil_refinement = None
            if not self.config.local_refinement_enabled and self.config.early_stop_enabled:
                pre_stencil_refinement = self._run_stencil_refinement()
                if pre_stencil_refinement is not None:
                    self.batch_events.append(pre_stencil_refinement)
                    self._update_portfolio()
                    self._capture_snapshot(force=True)
                    if self._early_stop_reached():
                        self._stop_reason = "early_stop_value"
                        break
            candidates = self._candidate_batch(min(int(self._effective_batch_size()), self._remaining()))
            event = self._evaluate_candidates(candidates)
            self.batch_events.append(event)
            self._update_portfolio()
            candidate_points = np.asarray([point for point, _source in candidates], dtype=float) if candidates else None
            self._capture_snapshot(candidate_points=candidate_points)
            cooperative_refinement = self._run_cooperative_refinement()
            if cooperative_refinement is not None:
                self.batch_events.append(cooperative_refinement)
                self._update_portfolio()
                self._capture_snapshot(force=True)
            refinement = None
            if event.improved or not self._local_refinement_stalled:
                refinement = self._run_local_refinement()
            if refinement is not None:
                self.batch_events.append(refinement)
                self._update_portfolio()
                self._capture_snapshot(force=True)
            direction_refinement = self._run_direction_refinement()
            if direction_refinement is not None:
                self.batch_events.append(direction_refinement)
                self._update_portfolio()
                self._capture_snapshot(force=True)
            stencil_refinement = None
            finite_difference_suppressed = (
                self.config.local_refinement_enabled
                and self.dimension >= int(self.config.high_dimensional_min_dimension)
                and not self._high_dimensional_refinement_active()
            )
            if (not self.config.local_refinement_enabled or finite_difference_suppressed) and pre_stencil_refinement is None:
                stencil_refinement = self._run_stencil_refinement()
            if stencil_refinement is not None:
                self.batch_events.append(stencil_refinement)
                self._update_portfolio()
                self._capture_snapshot(force=True)
            if (
                event.evaluations_after == event.evaluations_before
                and cooperative_refinement is None
                and refinement is None
                and direction_refinement is None
                and stencil_refinement is None
                and pre_stencil_refinement is None
            ):
                self._stop_reason = "no_progress"
                break
            self._stop_reason = "budget_exhausted" if self._remaining() <= 0 else "running"

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
                    "effective_batch_size": int(self._effective_batch_size()),
                    "initial_design_size": int(self.config.initial_design_size),
                    "portfolio_size": int(self.config.portfolio_size),
                    "trust_regions_enabled": bool(self.config.trust_regions_enabled),
                    "anisotropic_regions_enabled": bool(self.config.anisotropic_regions_enabled),
                    "region_anisotropy_max": float(self.config.region_anisotropy_max),
                    "local_refinement_method": self.config.local_refinement_method,
                    "high_dimensional_refinement_enabled": bool(self.config.high_dimensional_refinement_enabled),
                    "dimension_scaled_batches_enabled": bool(self.config.dimension_scaled_batches_enabled),
                    "source_adaptation_enabled": bool(self.config.source_adaptation_enabled),
                    "coherent_probes_enabled": bool(self.config.coherent_probes_enabled),
                    "shade_enabled": bool(self.config.shade_enabled),
                    "cma_region_enabled": bool(self.config.cma_region_enabled),
                    "restart_strategy_enabled": bool(self.config.restart_strategy_enabled),
                    "evolutionary_population_size": (
                        None
                        if self.config.evolutionary_population_size is None
                        else int(self.config.evolutionary_population_size)
                    ),
                    "evolutionary_population_max": int(self.config.evolutionary_population_max),
                    "relative_success_credit": float(self.config.relative_success_credit),
                    "surrogate_ranking_enabled": bool(self.config.surrogate_ranking_enabled),
                    "candidate_pool_multiplier": int(self.config.candidate_pool_multiplier),
                    "surrogate_ranking_neighbor_count": int(self.config.surrogate_ranking_neighbor_count),
                    "probe_recenter_enabled": bool(self.config.probe_recenter_enabled),
                    "probe_recenter_max_restarts": int(self.config.probe_recenter_max_restarts),
                    "basin_polishing_enabled": bool(self.config.basin_polishing_enabled),
                    "basin_polishing_min_dimension": int(self.config.basin_polishing_min_dimension),
                    "basin_polishing_activation_ratio": float(self.config.basin_polishing_activation_ratio),
                    "successful_direction_memory_size": int(self.config.successful_direction_memory_size),
                    "direction_refinement_enabled": bool(self.config.direction_refinement_enabled),
                    "direction_refinement_max_evaluations": int(self.config.direction_refinement_max_evaluations),
                    "linkage_blocks_enabled": bool(self.config.linkage_blocks_enabled),
                    "linkage_update_interval_batches": int(self.config.linkage_update_interval_batches),
                    "linkage_neighbor_count": int(self.config.linkage_neighbor_count),
                    "cross_block_lbfgs_enabled": bool(self.config.cross_block_lbfgs_enabled),
                    "cross_block_lbfgs_memory_size": int(self.config.cross_block_lbfgs_memory_size),
                    "cooperative_refinement_enabled": bool(self.config.cooperative_refinement_enabled),
                    "cooperative_min_dimension": int(self.config.cooperative_min_dimension),
                    "cooperative_group_size": (
                        None if self.config.cooperative_group_size is None else int(self.config.cooperative_group_size)
                    ),
                    "cooperative_groups_per_batch": int(self.config.cooperative_groups_per_batch),
                    "active_set_max_fraction": float(self.config.active_set_max_fraction),
                    "active_set_expand_interval_batches": int(self.config.active_set_expand_interval_batches),
                    "early_stop_enabled": bool(self.config.early_stop_enabled),
                    "early_stop_value": (
                        None
                        if self.config.early_stop_value is None
                        else float(self.config.early_stop_value)
                    ),
                },
                "archive_size": int(len(self.archive)),
                "portfolio": [region.to_dict() for region in self.regions],
                "batch_events": [event.to_dict() for event in self.batch_events],
                "region_events": [event.to_dict() for event in self.region_events],
                "trust_region_events": list(self.trust_region_events),
                "source_stats": self._source_stats_payload(),
                "shade_memory": {
                    "f": self._shade_f_memory.astype(float).tolist(),
                    "cr": self._shade_cr_memory.astype(float).tolist(),
                    "index": int(self._shade_memory_index),
                },
                "cma_region_states": self._cma_states_payload(),
                "anchor_baseline_target": (
                    None if self._anchor_baseline_target is None else float(self._anchor_baseline_target)
                ),
                "basin_polishing_active": bool(self._basin_polishing_active()),
                "successful_direction_memory_size": int(len(self._successful_directions)),
                "probe_recenters": int(self._probe_recenters_total),
                "linkage_score_max": float(np.max(self._linkage_scores)) if self._linkage_scores.size else 0.0,
                "lbfgs_pairs": int(len(self._lbfgs_pairs)),
                "cooperative_refinement_active": bool(self._cooperative_refinement_active()),
                "active_set_size": int(self._active_set_axes().size) if self._cooperative_refinement_active() else 0,
                "active_set_expansions": int(self._active_set_expansions),
                "axis_coverage_max": float(np.max(self._axis_coverage)) if self._axis_coverage.size else 0.0,
                "stop_reason": self._stop_reason,
                "local_refinement_stalled": bool(self._local_refinement_stalled),
                "active_region_count": int(len(self._active_regions())),
                "sleeping_region_count": int(self._sleeping_region_count()),
                "converged": bool(self._early_stop_reached()),
            },
        )

    def samples(self) -> list[dict[str, object]]:
        """Return archive samples as JSON-compatible dictionaries."""

        return [sample.to_dict() for sample in self.archive.samples]
