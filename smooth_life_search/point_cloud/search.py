"""Archive-centered point-cloud SmoothLife optimizer."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from typing import Callable

import numpy as np

from ..core import SearchRun, normalize_bounds_nd
from ..smoothlife.config import SmoothLifeConfig
from .allocation import StageAllocationState, lean_stage_weights
from .archive import PointCloudArchive
from .config import PointCloudSearchConfig
from .geometry import RegionGeometry, fit_region_geometry
from .models import PointCloudBatchEvent, PointCloudRegion, PointCloudRegionEvent, PointCloudSnapshot
from .proposals import CandidateProposal, GradientResult, SourceStats
from .surrogate import QuadraticSurrogate, fit_quadratic_surrogate

Objective = Callable[[np.ndarray], float]


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
        self._last_density_frames: list[tuple[tuple[str, tuple[int, int], np.ndarray, np.ndarray], np.ndarray]] = []
        self._last_projection_frame_count = 0
        self._last_surrogate_reliability = 1.0
        self._last_surrogate_rank_weight = 1.0
        self._local_refinement_stalled = False
        self._stop_reason = "not_started"
        self._global_sequence_index = 0
        self._stencil_step_fraction = float(self.config.region_initial_radius_fraction)
        self._last_successful_step = np.zeros(self.dimension, dtype=float)
        self._axis_activity = np.zeros(self.dimension, dtype=float)
        self._block_cursor = 0
        self._block_hessian_inverse: dict[tuple[int, ...], np.ndarray] = {}
        self._source_stats: dict[str, SourceStats] = {}
        self._last_improvement_batch = 0
        self._anchor_baseline_target: float | None = None
        self._successful_directions: list[np.ndarray] = []
        self._linkage_scores = np.zeros((self.dimension, self.dimension), dtype=float)
        self._linkage_last_update_batch = -1
        self._lbfgs_pairs: list[tuple[np.ndarray, np.ndarray]] = []
        self._probe_recenters_total = 0
        self._axis_coverage = np.zeros(self.dimension, dtype=float)
        self._last_cooperative_improved_axes: tuple[int, ...] = ()
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
        self._last_improvement_batch = 0
        self._anchor_baseline_target = None
        self._successful_directions = []
        self._linkage_scores = np.zeros((self.dimension, self.dimension), dtype=float)
        self._linkage_last_update_batch = -1
        self._lbfgs_pairs = []
        self._probe_recenters_total = 0
        self._axis_coverage = np.zeros(self.dimension, dtype=float)
        self._last_cooperative_improved_axes = ()
        self._cooperative_group_cursor = 0
        self._active_set_expansions = 0
        self._active_set_last_expand_batch = -1
        self._last_density = np.zeros(self.config.density_grid_shape, dtype=float)
        self._last_objective_field = np.zeros(self.config.density_grid_shape, dtype=float)
        self._last_evaluated_mask = np.zeros(self.config.density_grid_shape, dtype=bool)
        self._last_density_frames = []
        self._last_projection_frame_count = 0
        self._last_surrogate_reliability = 1.0
        self._last_surrogate_rank_weight = 1.0

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
        )

    def _dimension_scaled_batches_active(self) -> bool:
        return bool(
            self.config.dimension_scaled_batches_enabled
            and self.dimension >= int(self.config.high_dimensional_min_dimension)
        )

    def _effective_batch_size(self) -> int:
        base = int(self.config.batch_size)
        if not self.config.local_refinement_enabled and self.config.direction_refinement_enabled:
            return max(base, min(int(self.config.dimension_scaled_batch_max), 48))
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

    def _cooperative_heavy_allocation_active(self) -> bool:
        return bool(
            self._cooperative_refinement_active()
            and not (
                self.config.local_refinement_enabled
                and self._basin_polishing_active()
                and 100 <= self.dimension < 200
            )
        )

    def _cooperative_refinement_due(self) -> bool:
        if not self._cooperative_refinement_active():
            return False
        if not self._cooperative_heavy_allocation_active():
            return False
        if self._deep_basin_polishing_active():
            return self._batch_index % 2 == 0
        if self.dimension >= 200 or not self._basin_polishing_active():
            return True
        return self._batch_index % 4 == 0

    @staticmethod
    def _source_family(source: str) -> str:
        if source.startswith("cooperative:"):
            return "cooperative"
        if source.startswith("region:"):
            return "region"
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
        predicted_score: float | None = None,
    ) -> CandidateProposal:
        return CandidateProposal(
            point=np.asarray(point, dtype=float),
            source=str(source),
            family=self._source_family(str(source)),
            parent_key=parent_key,
            region_id=region_id,
            axes=None if axes is None else tuple(int(axis) for axis in axes),
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
        incumbent_improvements: Counter[str],
        incumbent_amounts: Counter[str],
    ) -> None:
        families = {
            self._source_family(source)
            for source in (set(attempts) | set(evaluations) | set(incumbent_improvements) | set(incumbent_amounts))
        }
        for family in families:
            attempted = sum(count for source, count in attempts.items() if self._source_family(source) == family)
            evaluated = sum(count for source, count in evaluations.items() if self._source_family(source) == family)
            improved = sum(
                count for source, count in incumbent_improvements.items() if self._source_family(source) == family
            )
            amount = sum(
                value for source, value in incumbent_amounts.items() if self._source_family(source) == family
            )
            stats = self._source_stat(family)
            stats.attempts += int(attempted)
            stats.evaluations += int(evaluated)
            stats.improvements += int(improved)
            stats.improvement_sum += float(amount)
            if improved > 0:
                stats.last_improvement_batch = int(self._batch_index)

    def _source_stats_payload(self) -> dict[str, dict[str, float | int]]:
        total_evaluations = max(sum(int(stats.evaluations) for stats in self._source_stats.values()), 1)
        return {
            source: {
                "attempts": int(stats.attempts),
                "evaluations": int(stats.evaluations),
                "improvements": int(stats.improvements),
                "improvement_sum": float(stats.improvement_sum),
                "improvement_rate": float(stats.improvements) / max(int(stats.evaluations), 1),
                "evaluation_share": float(stats.evaluations) / total_evaluations,
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

    def _basin_progress_ratio(self) -> float | None:
        baseline = self._anchor_baseline_target
        if baseline is None or not np.isfinite(baseline) or float(baseline) <= 0.0:
            return None
        current = self._target(self.best_value)
        if not np.isfinite(current):
            return None
        return float(current) / float(baseline)

    def _deep_basin_polishing_active(self) -> bool:
        ratio = self._basin_progress_ratio()
        return bool(self._basin_polishing_active() and self.dimension >= 200 and ratio is not None and ratio <= 0.08)

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
        coverage_weight = float(self.config.axis_coverage_pressure)
        coverage_pressure = 1.0 / (1.0 + np.asarray(self._axis_coverage, dtype=float))
        scores += coverage_weight * self._unit_scaled(coverage_pressure)
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

    def _cooperative_frontier_seeds(
        self,
        count: int,
        active_axes: np.ndarray,
        scores: np.ndarray,
    ) -> list[int]:
        if not self.config.cooperative_frontier_enabled or count <= 0:
            return []
        seeds: list[int] = []

        def add(axis: int) -> None:
            resolved = int(axis) % self.dimension
            if resolved not in seeds:
                seeds.append(resolved)

        recent_axes = list(self._last_cooperative_improved_axes)
        if not recent_axes and self._successful_directions:
            direction = np.asarray(self._successful_directions[-1], dtype=float)
            recent_axes = [int(axis) for axis in np.argsort(np.abs(direction))[::-1][: self._cooperative_group_size()]]
        if not recent_axes and np.any(self._axis_activity > 0.0):
            recent_axes = [int(np.argmax(self._axis_activity))]
        for axis in recent_axes:
            add(axis + 1)
            add(axis - 1)
            if self._linkage_scores.shape == (self.dimension, self.dimension):
                linkage = np.asarray(self._linkage_scores[int(axis) % self.dimension], dtype=float)
                linked = int(np.argmax(linkage))
                if float(linkage[linked]) > 0.0:
                    add(linked)
            if len(seeds) >= count:
                return seeds[:count]
        coverage_order = [int(axis) for axis in np.argsort(self._axis_coverage) if int(axis) in set(active_axes.tolist())]
        for axis in coverage_order:
            add(axis)
            if len(seeds) >= count:
                return seeds[:count]
        for axis in active_axes[np.argsort(scores[active_axes])[::-1]]:
            add(int(axis))
            if len(seeds) >= count:
                break
        return seeds[:count]

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
        frontier_count = (
            max(1, int(np.ceil(target_count * float(self.config.cooperative_frontier_fraction))))
            if self.config.cooperative_frontier_enabled
            else 0
        )
        for axis in self._cooperative_frontier_seeds(frontier_count, active_axes, scores):
            add_group(self._cooperative_group_from_seed([axis], active_axes, scores))
            if len(groups) >= target_count:
                return groups

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

    @staticmethod
    def _frame_key(frame: tuple[str, tuple[int, int], np.ndarray, np.ndarray]) -> tuple[object, ...]:
        kind, axes, _origin, basis = frame
        if kind == "coordinate":
            return kind, tuple(int(axis) for axis in axes)
        dominant = tuple(int(axis) for axis in np.argmax(np.abs(np.asarray(basis, dtype=float)), axis=0))
        signs = tuple(int(np.sign(np.asarray(basis, dtype=float)[axis, column])) for column, axis in enumerate(dominant))
        return kind, dominant, signs

    def _density_projection_frame(self) -> tuple[str, tuple[int, int], np.ndarray, np.ndarray]:
        return self._density_projection_frames()[0]

    def _active_subspace_density_frame(
        self,
        axes: tuple[int, int],
    ) -> tuple[str, tuple[int, int], np.ndarray, np.ndarray] | None:
        candidates = [
            region
            for region in self._active_regions()
            if region.geometry_reason == "archive_covariance"
            and np.asarray(region.geometry_basis, dtype=float).shape[0] == self.dimension
            and np.asarray(region.geometry_basis, dtype=float).shape[1] >= 2
        ]
        if not candidates:
            return None
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

    def _density_projection_frames(self) -> list[tuple[str, tuple[int, int], np.ndarray, np.ndarray]]:
        axes = self._projection_axes()
        coordinate_frame = (
            "coordinate",
            axes,
            np.zeros(self.dimension, dtype=float),
            self._coordinate_projection_basis(axes),
        )
        if (
            not self.config.projection_ensemble_enabled
            or self.dimension == 2
            or int(self.config.projection_ensemble_size) <= 1
        ):
            return [coordinate_frame]
        frames: list[tuple[str, tuple[int, int], np.ndarray, np.ndarray]] = []
        seen: set[tuple[object, ...]] = set()

        def add(frame: tuple[str, tuple[int, int], np.ndarray, np.ndarray]) -> None:
            if len(frames) >= int(self.config.projection_ensemble_size):
                return
            basis = np.asarray(frame[3], dtype=float)
            if basis.shape != (self.dimension, 2) or not np.all(np.isfinite(basis)):
                return
            key = self._frame_key(frame)
            if key in seen:
                return
            seen.add(key)
            frames.append((frame[0], frame[1], np.asarray(frame[2], dtype=float).copy(), basis.copy()))

        if self.config.projection_axes is not None:
            add(coordinate_frame)
        else:
            active_frame = self._active_subspace_density_frame(axes)
            active_first_dimension = max(16, int(self.config.active_subspace_size) * 2)
            if active_frame is not None and self.dimension <= active_first_dimension:
                add(active_frame)
                add(coordinate_frame)
            else:
                add(coordinate_frame)
                if active_frame is not None:
                    add(active_frame)
        if self._successful_directions:
            directions = np.asarray(self._successful_directions[-int(self.config.successful_direction_memory_size) :], dtype=float)
            activity = np.mean(np.abs(directions), axis=0)
            ranked = [int(axis) for axis in np.argsort(activity)[::-1] if int(axis) < self.dimension]
            if len(ranked) >= 2 and float(activity[ranked[1]]) > 0.0:
                success_axes = (ranked[0], ranked[1])
                add(
                    (
                        "successful_axes",
                        success_axes,
                        np.zeros(self.dimension, dtype=float),
                        self._coordinate_projection_basis(success_axes),
                    )
                )
        coverage_order = [int(axis) for axis in np.argsort(self._axis_coverage) if int(axis) < self.dimension]
        if len(coverage_order) >= 2:
            coverage_axes = (coverage_order[0], coverage_order[1])
            add(
                (
                    "coverage_axes",
                    coverage_axes,
                    np.zeros(self.dimension, dtype=float),
                    self._coordinate_projection_basis(coverage_axes),
                )
            )
        if len(frames) < int(self.config.projection_ensemble_size) and np.any(self._axis_activity > 0.0):
            activity_order = [int(axis) for axis in np.argsort(self._axis_activity)[::-1] if int(axis) < self.dimension]
            for first in activity_order:
                for second in coverage_order:
                    if first == second:
                        continue
                    add(
                        (
                            "activity_coverage_axes",
                            (first, second),
                            np.zeros(self.dimension, dtype=float),
                            self._coordinate_projection_basis((first, second)),
                        )
                    )
                    break
                if len(frames) >= int(self.config.projection_ensemble_size):
                    break
        return frames or [coordinate_frame]

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
        improvements = 0
        region_improvements: Counter[int] = Counter()
        region_attempts: Counter[int] = Counter()
        for proposal in proposals:
            if self._remaining() <= 0:
                break
            point = proposal.point
            source = proposal.source
            incumbent_before = float(self.best_value)
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
            if improved:
                improvements += 1
                source_improvements[source] += 1
                if np.isfinite(incumbent_before) and value is not None:
                    amount = max(0.0, self._target(incumbent_before) - self._target(float(value)))
                    source_improvement_amounts[source] += amount
                    if proposal_axes is not None and proposal_axes.size > 0:
                        self._axis_activity *= 0.999
                        self._axis_activity[proposal_axes] += max(amount, 1e-12)
                        if proposal.source.startswith("cooperative:"):
                            self._last_cooperative_improved_axes = tuple(int(axis) for axis in proposal_axes)
        after = len(self.archive)
        self._apply_region_feedback(region_attempts, region_improvements)
        self._record_source_results(
            source_attempts,
            source_counts,
            source_improvements,
            source_improvement_amounts,
        )
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
                "source_stats": self._source_stats_payload(),
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
                "projection_frame_count": int(self._last_projection_frame_count),
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
                "surrogate_reliability": float(self._last_surrogate_reliability),
                "surrogate_rank_weight": float(self._last_surrogate_rank_weight),
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

    def _density_view_for_frame(
        self,
        frame: tuple[str, tuple[int, int], np.ndarray, np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    def _density_view(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        shape = self.config.density_grid_shape
        frames = self._density_projection_frames()
        if not frames:
            empty = np.zeros(shape, dtype=float)
            self._last_density_frames = []
            self._last_projection_frame_count = 0
            return empty, empty.copy(), np.zeros(shape, dtype=bool)
        frame_views = [(frame, *self._density_view_for_frame(frame)) for frame in frames]
        weights = np.ones(len(frame_views), dtype=float)
        if len(weights) > 1:
            weights[0] = 1.25
        weights = weights / max(float(np.sum(weights)), 1e-12)
        density = np.zeros(shape, dtype=float)
        objective = np.zeros(shape, dtype=float)
        evaluated = np.zeros(shape, dtype=bool)
        self._last_density_frames = []
        for weight, (frame, frame_density, frame_objective, frame_evaluated) in zip(weights, frame_views):
            density += float(weight) * np.asarray(frame_density, dtype=float)
            objective = np.maximum(objective, np.asarray(frame_objective, dtype=float))
            evaluated |= np.asarray(frame_evaluated, dtype=bool)
            self._last_density_frames.append((frame, np.asarray(frame_density, dtype=float).copy()))
        density = density / max(float(np.max(density)), 1e-12)
        objective = objective / max(float(np.max(objective)), 1e-12)
        self._last_projection_frame_count = len(self._last_density_frames)
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

        weights = lean_stage_weights(
            StageAllocationState(
                dimension=self.dimension,
                evolutionary_active=True,
                basin_polishing_active=self._basin_polishing_active(),
                deep_basin_polishing_active=self._deep_basin_polishing_active(),
                cooperative_active=self._cooperative_refinement_active(),
                cooperative_heavy=self._cooperative_heavy_allocation_active(),
                trust_regions_enabled=bool(self.config.trust_regions_enabled),
                coherent_probes_enabled=bool(self.config.coherent_probes_enabled),
            )
        )
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
        global_count = min(count, int(np.ceil(count * float(self.config.global_exploration_floor))))
        density_count = count - global_count
        candidates = self._global_candidates(global_count)
        if density_count <= 0:
            return candidates
        if not self._last_density_frames:
            self._refresh_views()
        frame_densities = [
            (frame, np.asarray(density, dtype=float))
            for frame, density in self._last_density_frames
            if np.any(np.asarray(density, dtype=float) > 0.0)
        ]
        if not frame_densities:
            return self._global_candidates(count)
        lower = self.original_bounds[:, 0]
        widths = self.original_bounds[:, 1] - self.original_bounds[:, 0]
        frame_count = len(frame_densities)
        slots = [density_count // frame_count] * frame_count
        for index in range(density_count % frame_count):
            slots[index] += 1
        for slot_count, (frame, density) in zip(slots, frame_densities):
            if slot_count <= 0:
                continue
            flat = density.ravel()
            probabilities = flat / float(np.sum(flat))
            chosen = self.rng.choice(flat.size, size=slot_count, replace=True, p=probabilities)
            height, width = density.shape
            rows, cols = np.unravel_index(chosen, density.shape)
            jitter = self.rng.random((slot_count, 2))
            normalized_x = (cols.astype(float) + jitter[:, 0]) / width
            normalized_y = (rows.astype(float) + jitter[:, 1]) / height
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
        heavy_cooperative = self._cooperative_heavy_allocation_active()
        quantiles = (
            (0.80, 0.88, 0.92, 0.95, 0.98, 0.985, 0.992, 0.997)
            if heavy_cooperative
            else (0.80, 0.88, 0.92, 0.95, 0.98)
        )
        spread = float(np.std(normalized_best))
        jitter_scale = float(
            np.clip(
                0.04 * spread if heavy_cooperative else 0.03 * spread,
                5e-4 if heavy_cooperative else 2e-4,
                1.5e-3 if heavy_cooperative else 8e-4,
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
            noise = np.zeros(self.dimension, dtype=float)
            noise[axes] = self.rng.normal(0.0, step_base, size=axes.size)
            add_candidate(
                normalized_best + noise,
                "cooperative:group",
                axes,
            )
        if len(candidates) < count:
            candidates.extend(self._global_candidates(count - len(candidates)))
        return candidates[:count]

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

    def _surrogate_reliability_score(
        self,
        archive_normalized: np.ndarray,
        target: np.ndarray,
        neighbor_count: int,
    ) -> float:
        if not self.config.surrogate_reliability_enabled:
            return 1.0
        sample_count = int(archive_normalized.shape[0])
        if sample_count < max(8, min(int(neighbor_count), 8)):
            return 0.5
        selected_count = min(sample_count, 48)
        if selected_count < sample_count:
            order = np.argsort(target)
            elite_count = max(1, selected_count // 2)
            spread = np.linspace(0, sample_count - 1, num=selected_count - elite_count, dtype=int)
            selected = np.asarray(list(dict.fromkeys([*order[:elite_count].tolist(), *spread.tolist()])), dtype=int)
            selected = selected[:selected_count]
        else:
            selected = np.arange(sample_count, dtype=int)
        predictions: list[float] = []
        actual: list[float] = []
        k = max(3, min(int(neighbor_count), sample_count - 1))
        for index in selected:
            distances = np.linalg.norm(archive_normalized - archive_normalized[int(index)], axis=1)
            distances[int(index)] = np.inf
            if k < distances.size:
                neighbor_indices = np.argpartition(distances, k - 1)[:k]
            else:
                neighbor_indices = np.flatnonzero(np.isfinite(distances))
            if neighbor_indices.size == 0:
                continue
            weights = 1.0 / np.maximum(distances[neighbor_indices], 1e-8)
            predictions.append(float(np.dot(weights, target[neighbor_indices]) / max(float(np.sum(weights)), 1e-12)))
            actual.append(float(target[int(index)]))
        if len(predictions) < 4:
            return 0.5
        predicted = np.asarray(predictions, dtype=float)
        observed = np.asarray(actual, dtype=float)
        if float(np.std(predicted)) <= 1e-12 or float(np.std(observed)) <= 1e-12:
            return 0.5
        correlation = float(np.corrcoef(predicted, observed)[0, 1])
        if not np.isfinite(correlation):
            return 0.5
        return float(np.clip(correlation, 0.0, 1.0))

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
        reliability = self._surrogate_reliability_score(archive_normalized, target, neighbor_count)
        if self.config.surrogate_reliability_enabled:
            prediction_weight = float(
                self.config.surrogate_rank_weight_min
                + reliability
                * (float(self.config.surrogate_rank_weight_max) - float(self.config.surrogate_rank_weight_min))
            )
            novelty_weight = 0.12 + 0.20 * (1.0 - reliability)
        else:
            prediction_weight = 1.0
            novelty_weight = 0.12
        self._last_surrogate_reliability = float(reliability)
        self._last_surrogate_rank_weight = float(prediction_weight)

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
            predictions.append(prediction)
            novelty_values.append(novelty)

        predicted_array = np.asarray(predictions, dtype=float)
        novelty_array = np.asarray(novelty_values, dtype=float)

        def normalized(values: np.ndarray) -> np.ndarray:
            span = max(float(np.max(values) - np.min(values)), 1e-12)
            return (values - float(np.min(values))) / span

        score = prediction_weight * normalized(predicted_array)
        score -= novelty_weight * normalized(novelty_array)

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
        candidates.extend(self._cooperative_candidates(counts.get("cooperative", 0)))
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
        if self.dimension <= max(12, int(self.config.active_subspace_size) * 2):
            return self.dimension
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
            and np.isfinite(self.best_value)
            and self._remaining() > 0
        ):
            return None
        if (
            not self._basin_polishing_active()
            and self.dimension > 2
            and self.dimension < int(self.config.high_dimensional_min_dimension)
        ):
            return None
        if (
            not self._basin_polishing_active()
            and self.config.local_refinement_enabled
            and not self._local_refinement_stalled
        ):
            return None
        if (
            not self._basin_polishing_active()
            and self.config.local_refinement_enabled
            and len(self._successful_directions) < 2
        ):
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
        min_step = float(self.config.direction_line_search_min_step_fraction)
        bracketed = self.config.direction_line_search_mode == "bracketed"
        step_scales = (
            [0.5**index for index in range(int(self.config.direction_line_search_max_steps))]
            if bracketed
            else [1.0]
        )
        quadratic_steps = 0
        bracket_evaluations = 0

        def evaluate_direction(
            base_normalized: np.ndarray,
            direction: np.ndarray,
            signed_step: float,
            source: str,
        ) -> tuple[float | None, bool, bool]:
            if len(self.archive) - before >= max_new or self._remaining() <= 0:
                return None, False, False
            current_before = float(self.best_value)
            candidate = self._point_from_normalized(base_normalized + signed_step * direction)
            source_attempts[source] += 1
            value, added, improved = self._evaluate_point(candidate, source=source)
            if not added:
                return value, False, False
            source_counts[source] += 1
            if improved and value is not None:
                nonlocal_improvement = max(0.0, self._target(current_before) - self._target(float(value)))
                source_improvements[source] += 1
                source_amounts[source] += nonlocal_improvement
            return value, added, improved

        def quadratic_candidate_step(samples: list[tuple[float, float]]) -> float | None:
            unique: dict[float, float] = {}
            for step, value in samples:
                unique[round(float(step), 14)] = float(value)
            if len(unique) < 3:
                return None
            steps = np.asarray(list(unique.keys()), dtype=float)
            targets = np.asarray(list(unique.values()), dtype=float)
            if not np.all(np.isfinite(steps)) or not np.all(np.isfinite(targets)):
                return None
            matrix = np.column_stack((steps * steps, steps, np.ones_like(steps)))
            try:
                a, b, _c = np.linalg.lstsq(matrix, targets, rcond=None)[0]
            except np.linalg.LinAlgError:
                return None
            if not np.isfinite(a) or not np.isfinite(b) or float(a) <= 1e-14:
                return None
            candidate = float(-b / (2.0 * a))
            low = float(np.min(steps))
            high = float(np.max(steps))
            if not low <= candidate <= high:
                return None
            if abs(candidate) < min_step:
                return None
            if np.any(np.isclose(candidate, steps, atol=min_step * 0.25, rtol=0.0)):
                return None
            return candidate

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
                    samples = [(0.0, self._target(float(self.best_value)))]
                    for scale in step_scales:
                        if len(self.archive) - before >= max_new or self._remaining() <= 0:
                            break
                        signed_step = sign * max(min_step, step_fraction * float(scale))
                        value, added, improved = evaluate_direction(
                            normalized_center,
                            direction,
                            signed_step,
                            "direction_line_search",
                        )
                        if added and value is not None:
                            samples.append((signed_step, self._target(float(value))))
                        if not improved or value is None:
                            continue
                        improvements += 1
                        improved_this_pass = True
                        if bracketed and len(self.archive) - before < max_new and self._remaining() > 0:
                            bracket_step = sign * min(0.25, max(abs(signed_step) * 1.6, abs(signed_step) + min_step))
                            if abs(bracket_step) > abs(signed_step) + 1e-14:
                                bracket_value, bracket_added, bracket_improved = evaluate_direction(
                                    normalized_center,
                                    direction,
                                    bracket_step,
                                    "direction_line_search",
                                )
                                if bracket_added and bracket_value is not None:
                                    bracket_evaluations += 1
                                    samples.append((bracket_step, self._target(float(bracket_value))))
                                    if bracket_improved:
                                        improvements += 1
                        if bracketed and len(self.archive) - before < max_new and self._remaining() > 0:
                            quadratic_step = quadratic_candidate_step(samples)
                            if quadratic_step is not None:
                                quadratic_value, quadratic_added, quadratic_improved = evaluate_direction(
                                    normalized_center,
                                    direction,
                                    quadratic_step,
                                    "direction_line_search",
                                )
                                if quadratic_added:
                                    quadratic_steps += 1
                                    if quadratic_improved and quadratic_value is not None:
                                        improvements += 1
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
                "direction_line_search_mode": self.config.direction_line_search_mode,
                "direction_quadratic_steps": int(quadratic_steps),
                "direction_bracket_evaluations": int(bracket_evaluations),
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
                self._last_cooperative_improved_axes = tuple(int(axis) for axis in axes)
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
                "cooperative_frontier_enabled": bool(self.config.cooperative_frontier_enabled),
                "cooperative_frontier_fraction": float(self.config.cooperative_frontier_fraction),
                "active_set_size": int(self._active_set_axes().size),
                "cooperative_groups_used": int(groups_used),
                "cooperative_improvements": int(improvements),
                "active_set_expansions": int(self._active_set_expansions),
                "last_cooperative_improved_axes": [int(axis) for axis in self._last_cooperative_improved_axes],
                "axis_coverage_max": float(np.max(self._axis_coverage)) if self._axis_coverage.size else 0.0,
                "top_axis_scores": [
                    {"axis": int(axis), "score": float(self._cooperative_axis_scores()[axis])}
                    for axis in top_axes
                ],
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
                "projection_frame_count": int(self._last_projection_frame_count),
                "surrogate_reliability": float(self._last_surrogate_reliability),
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
            candidates = self._candidate_batch(min(int(self._effective_batch_size()), self._remaining()))
            event = self._evaluate_candidates(candidates)
            self.batch_events.append(event)
            self._update_portfolio()
            candidate_points = np.asarray([point for point, _source in candidates], dtype=float) if candidates else None
            self._capture_snapshot(candidate_points=candidate_points)
            polishing = None
            prefer_cooperative = bool(
                self._cooperative_refinement_due()
                and (
                    self._local_refinement_stalled
                    or (len(self._successful_directions) >= 2 and self._batch_index % 3 == 0)
                )
            )
            if prefer_cooperative:
                polishing = self._run_cooperative_refinement()
            if polishing is None and (event.improved or not self._local_refinement_stalled):
                local_before = len(self.archive)
                polishing = self._run_local_refinement()
                if polishing is None and len(self.archive) == local_before:
                    self._local_refinement_stalled = True
            if polishing is None and not prefer_cooperative and self._cooperative_refinement_due():
                polishing = self._run_cooperative_refinement()
            if polishing is None:
                polishing = self._run_direction_refinement()
            if polishing is not None:
                self.batch_events.append(polishing)
                self._update_portfolio()
                self._capture_snapshot(force=True)
            if (
                event.evaluations_after == event.evaluations_before
                and polishing is None
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
                    "coherent_probes_enabled": bool(self.config.coherent_probes_enabled),
                    "projection_ensemble_enabled": bool(self.config.projection_ensemble_enabled),
                    "projection_ensemble_size": int(self.config.projection_ensemble_size),
                    "projection_ensemble_refresh_batches": int(self.config.projection_ensemble_refresh_batches),
                    "surrogate_ranking_enabled": bool(self.config.surrogate_ranking_enabled),
                    "candidate_pool_multiplier": int(self.config.candidate_pool_multiplier),
                    "surrogate_ranking_neighbor_count": int(self.config.surrogate_ranking_neighbor_count),
                    "surrogate_reliability_enabled": bool(self.config.surrogate_reliability_enabled),
                    "surrogate_rank_weight_min": float(self.config.surrogate_rank_weight_min),
                    "surrogate_rank_weight_max": float(self.config.surrogate_rank_weight_max),
                    "probe_recenter_enabled": bool(self.config.probe_recenter_enabled),
                    "probe_recenter_max_restarts": int(self.config.probe_recenter_max_restarts),
                    "basin_polishing_enabled": bool(self.config.basin_polishing_enabled),
                    "basin_polishing_min_dimension": int(self.config.basin_polishing_min_dimension),
                    "basin_polishing_activation_ratio": float(self.config.basin_polishing_activation_ratio),
                    "successful_direction_memory_size": int(self.config.successful_direction_memory_size),
                    "direction_refinement_enabled": bool(self.config.direction_refinement_enabled),
                    "direction_refinement_max_evaluations": int(self.config.direction_refinement_max_evaluations),
                    "direction_line_search_mode": self.config.direction_line_search_mode,
                    "direction_line_search_max_steps": int(self.config.direction_line_search_max_steps),
                    "direction_line_search_min_step_fraction": float(self.config.direction_line_search_min_step_fraction),
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
                    "cooperative_frontier_enabled": bool(self.config.cooperative_frontier_enabled),
                    "cooperative_frontier_fraction": float(self.config.cooperative_frontier_fraction),
                    "axis_coverage_pressure": float(self.config.axis_coverage_pressure),
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
                "anchor_baseline_target": (
                    None if self._anchor_baseline_target is None else float(self._anchor_baseline_target)
                ),
                "basin_polishing_active": bool(self._basin_polishing_active()),
                "successful_direction_memory_size": int(len(self._successful_directions)),
                "projection_frame_count": int(self._last_projection_frame_count),
                "surrogate_reliability": float(self._last_surrogate_reliability),
                "surrogate_rank_weight": float(self._last_surrogate_rank_weight),
                "probe_recenters": int(self._probe_recenters_total),
                "linkage_score_max": float(np.max(self._linkage_scores)) if self._linkage_scores.size else 0.0,
                "lbfgs_pairs": int(len(self._lbfgs_pairs)),
                "cooperative_refinement_active": bool(self._cooperative_refinement_active()),
                "active_set_size": int(self._active_set_axes().size) if self._cooperative_refinement_active() else 0,
                "active_set_expansions": int(self._active_set_expansions),
                "last_cooperative_improved_axes": [int(axis) for axis in self._last_cooperative_improved_axes],
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
