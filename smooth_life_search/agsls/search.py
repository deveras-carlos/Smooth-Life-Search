"""Three-phase Adaptive Grid Smooth Life Search controller."""

from __future__ import annotations

from collections import Counter
from typing import Callable, Literal

import numpy as np

from ..core import Basin, RuntimeSignals, SearchRun, ZoomEvent, normalize_bounds_2d, point_in_bounds
from ..smoothlife.config import SmoothLifeConfig
from ..smoothlife.search import SmoothLifeSearch
from .config import AGSLSConfig
from .surrogate import CommitSurrogateResult, fit_commit_surrogate, surrogate_axis_widths, surrogate_valley_tangent
from .trust_region import (
    ArchiveQuadraticSurrogate,
    SampleArchive,
    TrustRegionState,
    deterministic_candidate_pool,
    fit_archive_quadratic,
    nearest_archive_distance,
    trust_region_bounds,
)

Objective = Callable[[np.ndarray], float]
PhaseName = Literal["exploration", "commit", "exploitation"]


class AdaptiveGridSmoothLifeSearch:
    """Drive SmoothLife, then zoom through commit and exploitation phases."""

    def __init__(
        self,
        objective: Objective,
        bounds: np.ndarray | list[tuple[float, float]],
        smoothlife_config: SmoothLifeConfig | None = None,
        agsls_config: AGSLSConfig | None = None,
    ) -> None:
        self.objective = objective
        self.original_bounds = normalize_bounds_2d(bounds, owner="AGSLS")
        self.smoothlife_config = smoothlife_config or SmoothLifeConfig()
        self.agsls_config = agsls_config or AGSLSConfig()
        self.engine = SmoothLifeSearch(objective, self.original_bounds, self.smoothlife_config)
        self.zoom_events: list[ZoomEvent] = []
        self.active_max_evaluations: int | None = None
        self._active_zoom_limit: int | None = None
        self._decision_trace: list[dict[str, object]] = []
        self._phase_counts: Counter[str] = Counter()
        self._last_basin_count = 0
        self._last_top_basin_score_gap = 0.0
        self.sample_archive = SampleArchive.empty()
        self._trust_region_states: dict[PhaseName, TrustRegionState] = {
            "commit": TrustRegionState(),
            "exploitation": TrustRegionState(),
        }
        self._trust_region_events: list[dict[str, object]] = []
        self._archive_current_grid_samples()

    def reset(self, seed: int | None = None) -> None:
        """Reset SmoothLife state and AGSLS history."""

        self.engine.reset(seed=seed, bounds=self.original_bounds.copy())
        self.engine.set_zoom_index(0, max_zoom_cycles=self.agsls_config.max_zoom_cycles)
        self.zoom_events = []
        self.active_max_evaluations = None
        self._active_zoom_limit = None
        self._decision_trace = []
        self._phase_counts = Counter()
        self._last_basin_count = 0
        self._last_top_basin_score_gap = 0.0
        self.sample_archive = SampleArchive.empty()
        self._trust_region_states = {
            "commit": TrustRegionState(),
            "exploitation": TrustRegionState(),
        }
        self._trust_region_events = []
        self._archive_current_grid_samples()

    def runtime_signals(self) -> RuntimeSignals:
        base = self.engine.runtime_signals()
        return RuntimeSignals(
            step_index=base.step_index,
            zoom_index=len(self.zoom_events),
            evaluations=base.evaluations,
            remaining_budget=base.remaining_budget,
            total_budget=base.total_budget,
            explored_fraction=base.explored_fraction,
            current_box_widths=base.current_box_widths,
            best_value=base.best_value,
            local_best_value=base.local_best_value,
            global_improvement=base.global_improvement,
            stage_improvement=base.stage_improvement,
            basin_count=self._last_basin_count,
            top_basin_score_gap=self._last_top_basin_score_gap,
            max_zoom_cycles=self._active_zoom_limit or self.agsls_config.max_zoom_cycles,
        )

    def _require_state(self):
        state = self.engine.state
        if state is None:
            raise RuntimeError("engine state missing")
        return state

    def _evaluation_limit(self) -> int:
        limit = self.active_max_evaluations if self.active_max_evaluations is not None else self.agsls_config.max_evaluations
        if limit is None:
            raise ValueError("AGSLS runs require max_evaluations or run(evaluations=...)")
        if limit <= 0:
            raise ValueError("evaluation limit must be positive")
        return int(limit)

    def _remaining_evaluations(self) -> int:
        state = self._require_state()
        return max(self._evaluation_limit() - int(state.evaluations), 0)

    def _archive_current_grid_samples(self) -> int:
        state = self._require_state()
        return self.sample_archive.add_grid(
            state.objective_values,
            state.evaluated_mask,
            state.bounds,
            self.engine.config.grid_shape,
        )

    def _max_evaluations_reached(self) -> bool:
        return self._remaining_evaluations() <= 0

    def _budget_fraction(self) -> float:
        state = self._require_state()
        return min(1.0, max(0.0, float(state.evaluations) / float(self._evaluation_limit())))

    def _phase_for_budget(self) -> PhaseName:
        fraction = self._budget_fraction()
        if fraction < self.agsls_config.exploration_fraction:
            return "exploration"
        if fraction < self.agsls_config.commit_fraction:
            return "commit"
        return "exploitation"

    def _apply_phase_parameters(self, phase: PhaseName) -> None:
        if phase == "exploration":
            overrides = {
                "objective_gamma": self.agsls_config.exploration_objective_gamma,
                "support_ema_alpha": self.agsls_config.exploration_support_ema_alpha,
                "objective_guidance_mode": "sampled",
                "objective_uncertainty_weight": 0.0,
                "objective_drift_strength": 0.0,
            }
        elif phase == "commit":
            overrides = {
                "objective_gamma": self.agsls_config.commit_objective_gamma,
                "support_ema_alpha": self.agsls_config.commit_support_ema_alpha,
                "objective_guidance_mode": "rbf",
                "objective_rbf_top_k": self.agsls_config.commit_guidance_top_k,
                "objective_rbf_sigma": self.agsls_config.commit_guidance_sigma,
                "objective_rbf_temperature": self.agsls_config.commit_guidance_temperature,
                "objective_uncertainty_weight": self.agsls_config.commit_uncertainty_weight,
                "objective_drift_strength": self.agsls_config.commit_drift_strength,
            }
        else:
            overrides = {
                "objective_gamma": self.agsls_config.exploitation_objective_gamma,
                "support_ema_alpha": self.agsls_config.exploitation_support_ema_alpha,
                "objective_guidance_mode": "rbf",
                "objective_rbf_top_k": self.agsls_config.exploitation_guidance_top_k,
                "objective_rbf_sigma": self.agsls_config.exploitation_guidance_sigma,
                "objective_rbf_temperature": self.agsls_config.exploitation_guidance_temperature,
                "objective_uncertainty_weight": self.agsls_config.exploitation_uncertainty_weight,
                "objective_drift_strength": self.agsls_config.exploitation_drift_strength,
            }
        self.engine.apply_runtime_overrides(overrides, rebuild_kernels=False)

    def _support_field(self) -> np.ndarray:
        smoothed_support = getattr(self.engine, "smoothed_support_field", None)
        if callable(smoothed_support):
            return np.asarray(smoothed_support(), dtype=float)
        return np.asarray(self.engine.support_field(), dtype=float)

    def _run_engine_steps(self, steps: int) -> int:
        gained = 0
        for _ in range(max(int(steps), 0)):
            if self._max_evaluations_reached():
                break
            before = int(self._require_state().evaluations)
            self.engine.step(1)
            after = int(self._require_state().evaluations)
            gained += max(after - before, 0)
            if after <= before:
                break
        self._archive_current_grid_samples()
        return gained

    def _persistence_map(
        self,
        steps: int,
        *,
        min_explored_fraction: float = 0.0,
        reserve_evaluations: int = 0,
    ) -> np.ndarray:
        persistence = np.zeros(self.engine.config.grid_shape, dtype=float)
        executed_steps = 0
        min_steps = max(int(steps), 1)
        reserve = max(int(reserve_evaluations), 0)
        while True:
            state = self._require_state()
            explored_fraction = float(np.mean(state.evaluated_mask))
            if executed_steps >= min_steps and explored_fraction >= float(min_explored_fraction):
                break
            if self._remaining_evaluations() <= reserve:
                break
            before_evaluations = int(state.evaluations)
            self.engine.step(1)
            state = self._require_state()
            support = self._support_field()
            alive = self.engine.alive_mask(self.agsls_config.alive_core_threshold)
            threshold = float(np.quantile(support, self.agsls_config.basin_quantile))
            persistence += (alive & (support >= threshold)).astype(float)
            executed_steps += 1
            if int(state.evaluations) <= before_evaluations:
                break
        if executed_steps == 0:
            return persistence
        self._archive_current_grid_samples()
        return persistence / max(executed_steps, 1)

    def _rank_basins_for_phase(self, phase: PhaseName, persistence: np.ndarray | None = None) -> list[Basin]:
        state = self._require_state()
        if persistence is None:
            persistence = np.zeros(self.engine.config.grid_shape, dtype=float)
        basins = self.engine.basin_candidates(
            threshold_quantile=self.agsls_config.basin_quantile,
            min_cells=self.agsls_config.min_basin_cells,
            alive_core_threshold=self.agsls_config.alive_core_threshold,
            cluster_eps_pixels=self.agsls_config.cluster_eps_pixels,
            cluster_min_samples=self.agsls_config.cluster_min_samples,
            basin_envelope_quantile_offset=self.agsls_config.basin_envelope_quantile_offset,
            basin_envelope_growth_pixels=self.agsls_config.basin_envelope_growth_pixels,
            support_field=self._support_field(),
        )
        if not basins:
            self._last_basin_count = 0
            self._last_top_basin_score_gap = 0.0
            return []

        max_mass = max(float(basin.support_mass) for basin in basins)
        max_area = max(max(int(basin.area), 1) for basin in basins)
        for basin in basins:
            scoring_mask = basin.core_mask if basin.core_mask is not None else basin.mask
            basin.stability_score = float(np.mean(persistence[scoring_mask])) if np.any(scoring_mask) else 0.0
            norm_mass = float(basin.support_mass) / max(max_mass, 1e-12)
            norm_area = float(basin.area) / max(max_area, 1)
            if phase == "commit":
                basin.combined_score = (
                    self.agsls_config.commit_mass_weight * norm_mass
                    + self.agsls_config.commit_density_weight * float(basin.alive_density)
                    + self.agsls_config.commit_stability_weight * float(basin.stability_score)
                    + self.agsls_config.commit_objective_weight * float(basin.objective_score)
                    - self.agsls_config.commit_area_penalty * norm_area
                )
            else:
                incumbent_bonus = 2.0 if basin.incumbent_in_envelope else 0.0
                best_score = float(basin.best_objective_score) if basin.evaluated_count > 0 else 0.0
                basin.combined_score = incumbent_bonus + best_score + 0.25 * norm_mass + 0.25 * float(basin.stability_score)

        eligible = [
            basin
            for basin in basins
            if basin.area >= self.agsls_config.min_basin_cells and basin.alive_density >= self.agsls_config.min_alive_density
        ]
        eligible.sort(key=lambda basin: float(basin.combined_score), reverse=True)
        self._last_basin_count = len(eligible)
        self._last_top_basin_score_gap = (
            float(eligible[0].combined_score - eligible[1].combined_score) if len(eligible) >= 2 else 0.0
        )
        return eligible

    @staticmethod
    def _centered_bounds(center: np.ndarray, widths: np.ndarray, ceiling_bounds: np.ndarray) -> np.ndarray:
        resolved_center = np.asarray(center, dtype=float)
        resolved_widths = np.asarray(widths, dtype=float)
        ceiling = np.asarray(ceiling_bounds, dtype=float)
        ceiling_widths = ceiling[:, 1] - ceiling[:, 0]
        resolved_widths = np.minimum(resolved_widths, ceiling_widths)
        bounds = np.empty((2, 2), dtype=float)
        for axis in range(2):
            width = float(resolved_widths[axis])
            center_axis = float(np.clip(resolved_center[axis], ceiling[axis, 0], ceiling[axis, 1]))
            half = 0.5 * width
            lower = center_axis - half
            upper = center_axis + half
            if lower < ceiling[axis, 0]:
                upper += ceiling[axis, 0] - lower
                lower = ceiling[axis, 0]
            if upper > ceiling[axis, 1]:
                lower -= upper - ceiling[axis, 1]
                upper = ceiling[axis, 1]
            lower = max(float(ceiling[axis, 0]), lower)
            upper = min(float(ceiling[axis, 1]), upper)
            if not upper > lower:
                lower, upper = AdaptiveGridSmoothLifeSearch._nearest_representable_interval(center_axis, ceiling[axis])
            bounds[axis, 0] = lower
            bounds[axis, 1] = upper
        return bounds

    @staticmethod
    def _nearest_representable_interval(center: float, ceiling: np.ndarray) -> tuple[float, float]:
        lower_limit = float(ceiling[0])
        upper_limit = float(ceiling[1])
        center = float(np.clip(center, lower_limit, upper_limit))
        lower = float(np.nextafter(center, -np.inf))
        upper = float(np.nextafter(center, np.inf))
        if lower < lower_limit:
            lower = center
        if upper > upper_limit:
            upper = center
        if upper > lower:
            return lower, upper
        if center > lower_limit:
            lower = float(np.nextafter(center, -np.inf))
            if center > lower:
                return max(lower_limit, lower), center
        if center < upper_limit:
            upper = float(np.nextafter(center, np.inf))
            if upper > center:
                return center, min(upper_limit, upper)
        return center, center

    @staticmethod
    def _bounds_area_for_progress(bounds: np.ndarray) -> float:
        widths = np.asarray(bounds, dtype=float)[:, 1] - np.asarray(bounds, dtype=float)[:, 0]
        if not np.all(np.isfinite(widths)) or np.any(widths <= 0.0):
            return float("nan")
        return float(widths[0] * widths[1])

    @staticmethod
    def _strict_progress(old_bounds: np.ndarray, new_bounds: np.ndarray) -> bool:
        old_area = AdaptiveGridSmoothLifeSearch._bounds_area_for_progress(old_bounds)
        new_area = AdaptiveGridSmoothLifeSearch._bounds_area_for_progress(new_bounds)
        return (
            np.all(np.isfinite(new_bounds))
            and np.all(new_bounds[:, 1] > new_bounds[:, 0])
            and np.isfinite(old_area)
            and np.isfinite(new_area)
            and new_area < old_area
            and not np.allclose(old_bounds, new_bounds, rtol=0.0, atol=0.0)
        )

    def _expand_bounds_to_retain_incumbent(
        self,
        bounds: np.ndarray,
        current_bounds: np.ndarray,
        cell_widths: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        state = self._require_state()
        best_point = np.asarray(state.best_point, dtype=float)
        if best_point.shape != (2,) or not np.all(np.isfinite(best_point)):
            return bounds, False
        if not point_in_bounds(best_point, current_bounds):
            return bounds, False
        retained = point_in_bounds(best_point, bounds)
        expanded = np.asarray(bounds, dtype=float).copy()
        current_widths = np.asarray(current_bounds, dtype=float)[:, 1] - np.asarray(current_bounds, dtype=float)[:, 0]
        padding = np.maximum(
            np.asarray(cell_widths, dtype=float),
            float(self.agsls_config.commit_incumbent_padding_fraction) * current_widths,
        )
        for axis in range(2):
            lower_limit = float(current_bounds[axis, 0])
            upper_limit = float(current_bounds[axis, 1])
            if best_point[axis] < expanded[axis, 0] + padding[axis]:
                expanded[axis, 0] = max(lower_limit, float(best_point[axis] - padding[axis]))
            if best_point[axis] > expanded[axis, 1] - padding[axis]:
                expanded[axis, 1] = min(upper_limit, float(best_point[axis] + padding[axis]))
        return expanded, retained or point_in_bounds(best_point, expanded)

    def _fit_surrogate_for_basin(
        self,
        basin: Basin,
        current_bounds: np.ndarray,
        *,
        min_samples: int,
        max_samples: int,
    ) -> CommitSurrogateResult:
        state = self._require_state()
        return fit_commit_surrogate(
            objective_values=state.objective_values,
            evaluated_mask=state.evaluated_mask,
            support_field=self._support_field(),
            basin=basin,
            bounds=current_bounds,
            grid_shape=self.engine.config.grid_shape,
            maximize=self.engine.config.maximize,
            min_samples=int(min_samples),
            max_samples=int(max_samples),
            regularization=self.agsls_config.commit_surrogate_regularization,
            min_predicted_improvement=self.agsls_config.commit_surrogate_min_predicted_improvement,
            max_condition=self.agsls_config.commit_surrogate_max_condition,
            support_weight=self.agsls_config.commit_surrogate_support_weight,
        )

    def _commit_surrogate(self, basin: Basin, current_bounds: np.ndarray) -> CommitSurrogateResult:
        if not self.agsls_config.commit_surrogate_enabled:
            return CommitSurrogateResult(accepted=False, reason="disabled", sample_count=0)
        return self._fit_surrogate_for_basin(
            basin,
            current_bounds,
            min_samples=self.agsls_config.commit_surrogate_min_samples,
            max_samples=self.agsls_config.commit_surrogate_max_samples,
        )

    def _is_better_value(self, candidate_value: float | None, incumbent_value: float | None) -> bool:
        if candidate_value is None or incumbent_value is None:
            return False
        if not np.isfinite(candidate_value) or not np.isfinite(incumbent_value):
            return False
        if self.engine.config.maximize:
            return float(candidate_value) > float(incumbent_value)
        return float(candidate_value) < float(incumbent_value)

    def _evaluate_direct_sample(self, point: np.ndarray) -> tuple[float, bool]:
        """Evaluate an off-grid sample and update best records without touching grid cache."""

        if self._remaining_evaluations() <= 0:
            return float("nan"), False
        state = self._require_state()
        resolved = np.asarray(point, dtype=float)
        if resolved.shape != (2,) or not np.all(np.isfinite(resolved)):
            return float("nan"), False
        value = float(self.objective(resolved))
        state.evaluations += 1
        self.sample_archive.add_point(resolved, value)
        improved_global = not np.isfinite(state.best_value) or self.engine._is_better(value, float(state.best_value))
        if improved_global:
            state.best_point = resolved.copy()
            state.best_value = float(value)
        if not np.isfinite(state.local_best_value) or self.engine._is_better(value, float(state.local_best_value)):
            state.local_best_point = resolved.copy()
            state.local_best_value = float(value)
        if not np.isfinite(state.box_best_value) or self.engine._is_better(value, float(state.box_best_value)):
            state.box_best_point = resolved.copy()
            state.box_best_value = float(value)
        return float(value), bool(improved_global)

    def _incumbent_surrogate_basin(self, current_bounds: np.ndarray, *, min_samples: int | None = None) -> Basin | None:
        state = self._require_state()
        best_point = np.asarray(state.best_point, dtype=float)
        if best_point.shape != (2,) or not np.all(np.isfinite(best_point)):
            return None
        if not point_in_bounds(best_point, current_bounds):
            return None
        height, width = self.engine.config.grid_shape
        current_widths = current_bounds[:, 1] - current_bounds[:, 0]
        if np.any(current_widths <= 0.0):
            return None
        col = int(np.clip(np.floor(((best_point[0] - current_bounds[0, 0]) / current_widths[0]) * width), 0, width - 1))
        row = int(np.clip(np.floor(((best_point[1] - current_bounds[1, 0]) / current_widths[1]) * height), 0, height - 1))
        sample_floor = self.agsls_config.commit_surrogate_min_samples if min_samples is None else int(min_samples)
        radius = max(4, int(np.ceil(np.sqrt(float(sample_floor)))))
        row_min = max(0, row - radius)
        row_max = min(height - 1, row + radius)
        col_min = max(0, col - radius)
        col_max = min(width - 1, col + radius)
        mask = np.zeros(self.engine.config.grid_shape, dtype=bool)
        mask[row_min : row_max + 1, col_min : col_max + 1] = True
        bbox_world = np.asarray(
            [
                [
                    current_bounds[0, 0] + (col_min / width) * current_widths[0],
                    current_bounds[0, 0] + ((col_max + 1) / width) * current_widths[0],
                ],
                [
                    current_bounds[1, 0] + (row_min / height) * current_widths[1],
                    current_bounds[1, 0] + ((row_max + 1) / height) * current_widths[1],
                ],
            ],
            dtype=float,
        )
        support = self._support_field()
        objective_field = np.asarray(state.objective_field, dtype=float)
        evaluated_count = int(np.count_nonzero(mask & state.evaluated_mask))
        objective_score = float(np.max(objective_field[mask])) if np.any(mask) else 0.0
        return Basin(
            mask=mask,
            centroid_grid=np.asarray([row, col], dtype=float),
            centroid_world=best_point.copy(),
            bbox_grid=(row_min, col_min, row_max, col_max),
            bbox_world=bbox_world,
            support_mass=float(np.sum(support[mask])) if support.shape == mask.shape else 0.0,
            objective_score=objective_score,
            stability_score=0.0,
            alive_density=1.0,
            basin_best_point=best_point.copy(),
            basin_best_value=float(state.best_value),
            evaluated_count=evaluated_count,
            best_objective_score=objective_score,
            mean_objective_score=objective_score,
            incumbent_in_envelope=True,
        )

    def _should_try_incumbent_surrogate(self, basin: Basin, current_bounds: np.ndarray) -> bool:
        if not self.agsls_config.commit_surrogate_enabled:
            return False
        if basin.incumbent_in_envelope:
            return False
        state = self._require_state()
        if not point_in_bounds(np.asarray(state.best_point, dtype=float), current_bounds):
            return False
        if basin.basin_best_value is None:
            return np.isfinite(state.best_value)
        return self._is_better_value(float(state.best_value), float(basin.basin_best_value))

    def _empty_valley_tracking_diagnostics(self, reason: str) -> dict[str, object]:
        state = self._require_state()
        return {
            "valley_tracking_used": False,
            "valley_tracking_reason": reason,
            "valley_tracking_probes": 0,
            "valley_tracking_improvements": 0,
            "valley_tracking_best_before": float(state.best_value),
            "valley_tracking_best_after": float(state.best_value),
        }

    def _exploitation_valley_surrogate(
        self,
        basin: Basin | None,
        current_bounds: np.ndarray,
    ) -> tuple[CommitSurrogateResult, str]:
        state = self._require_state()
        best_point = np.asarray(state.best_point, dtype=float)
        min_samples = int(self.agsls_config.exploitation_valley_surrogate_min_samples)
        max_samples = int(self.agsls_config.exploitation_valley_surrogate_max_samples)
        if point_in_bounds(best_point, current_bounds):
            incumbent_basin = self._incumbent_surrogate_basin(current_bounds, min_samples=min_samples)
            if incumbent_basin is not None:
                incumbent = self._fit_surrogate_for_basin(
                    incumbent_basin,
                    current_bounds,
                    min_samples=min_samples,
                    max_samples=max_samples,
                )
                if incumbent.accepted:
                    return incumbent, "incumbent"
        if basin is not None:
            fallback = self._fit_surrogate_for_basin(
                basin,
                current_bounds,
                min_samples=min_samples,
                max_samples=max_samples,
            )
            return fallback, "basin"
        return CommitSurrogateResult(accepted=False, reason="no_basin", sample_count=0), "none"

    def _evaluate_valley_probe(self, point: np.ndarray) -> tuple[float, bool]:
        return self._evaluate_direct_sample(point)

    def _track_exploitation_valley(self, basin: Basin | None, current_bounds: np.ndarray) -> dict[str, object]:
        if not self.agsls_config.exploitation_valley_tracking_enabled:
            return self._empty_valley_tracking_diagnostics("disabled")
        probe_limit = min(
            int(self.agsls_config.exploitation_valley_probe_evaluations),
            max(self._remaining_evaluations() - int(self.engine.config.evaluations_per_step), 0),
        )
        if probe_limit <= 0:
            return self._empty_valley_tracking_diagnostics("no_probe_budget")

        state = self._require_state()
        best_before = float(state.best_value)
        surrogate, source = self._exploitation_valley_surrogate(basin, current_bounds)
        tangent = surrogate_valley_tangent(surrogate)
        if tangent is None:
            details = self._empty_valley_tracking_diagnostics(f"surrogate_{surrogate.reason}")
            details["valley_tracking_surrogate_source"] = source
            details["valley_tracking_surrogate_samples"] = int(surrogate.sample_count)
            return details

        current_widths = np.asarray(current_bounds, dtype=float)[:, 1] - np.asarray(current_bounds, dtype=float)[:, 0]
        if not np.all(np.isfinite(current_widths)) or np.any(current_widths <= 0.0):
            return self._empty_valley_tracking_diagnostics("invalid_bounds")

        probes = 0
        improvements = 0
        step_fraction = float(self.agsls_config.exploitation_valley_step_fraction)
        min_step = float(self.agsls_config.exploitation_valley_min_step_fraction)
        decay = float(self.agsls_config.exploitation_valley_step_decay)
        reason = "no_improvement"
        while probes < probe_limit and step_fraction >= min_step and not self._max_evaluations_reached():
            base_point = np.asarray(state.best_point, dtype=float)
            if not point_in_bounds(base_point, current_bounds):
                reason = "best_outside_bounds"
                break
            candidates: list[tuple[float, np.ndarray, float, bool]] = []
            for direction in (1.0, -1.0):
                if probes >= probe_limit:
                    break
                candidate = base_point + direction * step_fraction * tangent * current_widths
                if not np.all(np.isfinite(candidate)) or not point_in_bounds(candidate, current_bounds):
                    continue
                value, improved = self._evaluate_valley_probe(candidate)
                probes += 1
                candidates.append((value, candidate, direction, improved))
            if not candidates:
                reason = "no_valid_candidates"
                break

            best_value, best_candidate, _direction, improved_pair = candidates[0]
            for value, candidate, direction, improved in candidates[1:]:
                if self.engine._is_better(value, best_value):
                    best_value, best_candidate, _direction, improved_pair = value, candidate, direction, improved
            if improved_pair:
                state.best_point = best_candidate.copy()
                state.best_value = float(best_value)
                if self.engine._is_better(best_value, float(state.local_best_value)):
                    state.local_best_point = best_candidate.copy()
                    state.local_best_value = float(best_value)
                if self.engine._is_better(best_value, float(state.box_best_value)):
                    state.box_best_point = best_candidate.copy()
                    state.box_best_value = float(best_value)
                improvements += 1
                reason = "improved"
            elif step_fraction <= min_step:
                reason = "no_improvement_at_min_step"
                break
            step_fraction *= decay

        if probes > 0:
            self.engine._update_improvement_trackers()
        if improvements > 0:
            reason = "improved"
        return {
            "valley_tracking_used": probes > 0,
            "valley_tracking_reason": reason,
            "valley_tracking_probes": int(probes),
            "valley_tracking_improvements": int(improvements),
            "valley_tracking_best_before": best_before,
            "valley_tracking_best_after": float(state.best_value),
            "valley_tracking_tangent": np.asarray(tangent, dtype=float).tolist(),
            "valley_tracking_surrogate_source": source,
            "valley_tracking_surrogate_samples": int(surrogate.sample_count),
        }

    def _support_at_points(self, points: np.ndarray, bounds: np.ndarray) -> np.ndarray:
        candidates = np.asarray(points, dtype=float)
        if candidates.size == 0:
            return np.empty((0,), dtype=float)
        support = self._support_field()
        height, width = self.engine.config.grid_shape
        current_widths = np.asarray(bounds, dtype=float)[:, 1] - np.asarray(bounds, dtype=float)[:, 0]
        if support.shape != (height, width) or np.any(current_widths <= 0.0):
            return np.zeros(candidates.shape[0], dtype=float)
        cols = np.floor(((candidates[:, 0] - bounds[0, 0]) / current_widths[0]) * width).astype(int)
        rows = np.floor(((candidates[:, 1] - bounds[1, 0]) / current_widths[1]) * height).astype(int)
        cols = np.clip(cols, 0, width - 1)
        rows = np.clip(rows, 0, height - 1)
        values = np.asarray(support[rows, cols], dtype=float)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    def _trust_region_preferred_center(
        self,
        phase: PhaseName,
        basin: Basin | None,
        current_bounds: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, object], tuple[np.ndarray, ...]]:
        state = self._require_state()
        anchors: list[np.ndarray] = []
        best_point = np.asarray(state.best_point, dtype=float)
        details: dict[str, object] = {"trust_region_center_source": "box_center"}
        if point_in_bounds(best_point, current_bounds):
            anchors.append(best_point.copy())

        if phase == "exploitation" and point_in_bounds(best_point, current_bounds):
            details["trust_region_center_source"] = "global_best"
            return best_point.copy(), details, tuple(anchors)

        if basin is not None:
            if basin.basin_best_point is not None and point_in_bounds(basin.basin_best_point, current_bounds):
                anchors.append(np.asarray(basin.basin_best_point, dtype=float).copy())
            if point_in_bounds(basin.centroid_world, current_bounds):
                anchors.append(np.asarray(basin.centroid_world, dtype=float).copy())
            if phase == "commit" and self.agsls_config.commit_surrogate_enabled:
                surrogate = self._commit_surrogate(basin, current_bounds)
                details.update(
                    {
                        "trust_region_commit_surrogate_used": bool(surrogate.accepted),
                        "trust_region_commit_surrogate_reason": surrogate.reason,
                        "trust_region_commit_surrogate_samples": int(surrogate.sample_count),
                    }
                )
                if surrogate.accepted and surrogate.point is not None and point_in_bounds(surrogate.point, current_bounds):
                    anchors.append(np.asarray(surrogate.point, dtype=float).copy())
                    details["trust_region_center_source"] = "commit_surrogate"
                    return np.asarray(surrogate.point, dtype=float).copy(), details, tuple(anchors)
            if basin.basin_best_point is not None and point_in_bounds(basin.basin_best_point, current_bounds):
                details["trust_region_center_source"] = "basin_best"
                return np.asarray(basin.basin_best_point, dtype=float).copy(), details, tuple(anchors)
            if point_in_bounds(basin.centroid_world, current_bounds):
                details["trust_region_center_source"] = "basin_centroid"
                return np.asarray(basin.centroid_world, dtype=float).copy(), details, tuple(anchors)

        if point_in_bounds(best_point, current_bounds):
            details["trust_region_center_source"] = "global_best"
            return best_point.copy(), details, tuple(anchors)
        return np.mean(current_bounds, axis=1), details, tuple(anchors)

    def _initial_trust_radius_fraction(self, basin: Basin | None, current_bounds: np.ndarray) -> float:
        initial = float(self.agsls_config.trust_region_initial_radius_fraction)
        if basin is None:
            return initial
        current_widths = np.asarray(current_bounds, dtype=float)[:, 1] - np.asarray(current_bounds, dtype=float)[:, 0]
        bbox = np.asarray(basin.bbox_world, dtype=float)
        bbox_widths = bbox[:, 1] - bbox[:, 0]
        if np.any(current_widths <= 0.0) or np.any(bbox_widths <= 0.0):
            return initial
        basin_fraction = float(np.max(0.5 * bbox_widths / current_widths))
        radius = min(initial, max(float(self.agsls_config.trust_region_min_radius_fraction), 1.25 * basin_fraction))
        return float(np.clip(radius, self.agsls_config.trust_region_min_radius_fraction, initial))

    def _trust_region_state(self, phase: PhaseName) -> TrustRegionState:
        if phase not in self._trust_region_states:
            self._trust_region_states[phase] = TrustRegionState()
        return self._trust_region_states[phase]

    def _trust_region_surrogate(
        self,
        phase: PhaseName,
        center: np.ndarray,
        current_bounds: np.ndarray,
        radius_fraction: float,
    ) -> tuple[ArchiveQuadraticSurrogate, np.ndarray | None]:
        region = trust_region_bounds(center, radius_fraction, current_bounds)
        if region is None:
            return ArchiveQuadraticSurrogate(accepted=False, reason="invalid_region", sample_count=0), None
        points, values = self.sample_archive.arrays()
        min_samples = (
            self.agsls_config.commit_surrogate_min_samples
            if phase == "commit"
            else self.agsls_config.exploitation_valley_surrogate_min_samples
        )
        max_samples = (
            self.agsls_config.commit_surrogate_max_samples
            if phase == "commit"
            else self.agsls_config.exploitation_valley_surrogate_max_samples
        )
        surrogate = fit_archive_quadratic(
            archive_points=points,
            archive_values=values,
            region_bounds=region,
            center=center,
            maximize=self.engine.config.maximize,
            min_samples=min_samples,
            max_samples=max_samples,
            regularization=self.agsls_config.commit_surrogate_regularization,
            max_condition=self.agsls_config.commit_surrogate_max_condition,
        )
        return surrogate, region

    def _rank_trust_region_candidates(
        self,
        *,
        phase: PhaseName,
        candidates: np.ndarray,
        surrogate: ArchiveQuadraticSurrogate,
        current_bounds: np.ndarray,
    ) -> np.ndarray:
        if candidates.size == 0:
            return np.empty((0,), dtype=int)
        points, _values = self.sample_archive.arrays()
        uncertainty = nearest_archive_distance(candidates, points, current_bounds)
        duplicate_mask = uncertainty <= 1e-12
        support = self._support_at_points(candidates, current_bounds)
        support_span = max(float(np.max(support) - np.min(support)), 1e-12) if support.size else 1.0
        support_score = (support - float(np.min(support))) / support_span if support.size else np.zeros(candidates.shape[0])

        predictions = surrogate.predict(candidates, maximize=self.engine.config.maximize)
        if predictions is None or not np.any(np.isfinite(predictions)):
            predicted_score = np.zeros(candidates.shape[0], dtype=float)
            improvement_bonus = np.zeros(candidates.shape[0], dtype=float)
        else:
            finite_predictions = np.where(np.isfinite(predictions), predictions, np.nan)
            finite_values = finite_predictions[np.isfinite(finite_predictions)]
            span = max(float(np.max(finite_values) - np.min(finite_values)), 1e-12)
            if self.engine.config.maximize:
                predicted_score = (np.nan_to_num(finite_predictions, nan=float(np.min(finite_values))) - float(np.min(finite_values))) / span
                improvement_bonus = (finite_predictions > float(self._require_state().best_value)).astype(float)
            else:
                predicted_score = (float(np.max(finite_values)) - np.nan_to_num(finite_predictions, nan=float(np.max(finite_values)))) / span
                improvement_bonus = (finite_predictions < float(self._require_state().best_value)).astype(float)

        uncertainty_weight = (
            self.agsls_config.commit_acquisition_uncertainty_weight
            if phase == "commit"
            else self.agsls_config.exploitation_acquisition_uncertainty_weight
        )
        score = (
            predicted_score
            + 0.35 * improvement_bonus
            + float(self.agsls_config.trust_region_support_weight) * support_score
            + float(uncertainty_weight) * uncertainty
        )
        score = np.asarray(score, dtype=float)
        score[duplicate_mask] = -np.inf
        score = np.where(np.isfinite(score), score, -np.inf)
        return np.argsort(score)[::-1]

    def _run_trust_region_batch(
        self,
        phase: PhaseName,
        basin: Basin | None,
        current_bounds: np.ndarray,
    ) -> dict[str, object]:
        if phase == "exploration":
            return {"trust_region_used": False, "trust_region_reason": "exploration"}
        if not self.agsls_config.trust_region_enabled:
            return {"trust_region_used": False, "trust_region_reason": "disabled"}
        eval_limit = (
            self.agsls_config.commit_trust_region_evaluations
            if phase == "commit"
            else self.agsls_config.exploitation_trust_region_evaluations
        )
        eval_limit = min(int(eval_limit), self._remaining_evaluations())
        if eval_limit <= 0:
            return {"trust_region_used": False, "trust_region_reason": "no_budget"}

        self._archive_current_grid_samples()
        state = self._require_state()
        best_before = float(state.best_value)
        center, center_details, anchors = self._trust_region_preferred_center(phase, basin, current_bounds)
        trust_state = self._trust_region_state(phase)
        radius_before = (
            self._initial_trust_radius_fraction(basin, current_bounds)
            if trust_state.radius_fraction is None
            else float(trust_state.radius_fraction)
        )
        radius_before = float(
            np.clip(
                radius_before,
                self.agsls_config.trust_region_min_radius_fraction,
                1.0,
            )
        )
        trust_state.center = np.asarray(center, dtype=float).copy()
        trust_state.radius_fraction = radius_before

        surrogate, region = self._trust_region_surrogate(phase, center, current_bounds, radius_before)
        candidates = deterministic_candidate_pool(
            center=center,
            active_bounds=current_bounds,
            radius_fraction=radius_before,
            pool_size=self.agsls_config.trust_region_candidate_pool_size,
            surrogate=surrogate,
            anchors=anchors,
        )
        ranked = self._rank_trust_region_candidates(
            phase=phase,
            candidates=candidates,
            surrogate=surrogate,
            current_bounds=current_bounds,
        )
        if phase == "exploitation" and candidates.shape[0] > 0:
            priority = np.arange(min(candidates.shape[0], max(32, eval_limit)), dtype=int)
            priority_set = {int(index) for index in priority.tolist()}
            ordered_indices = [int(index) for index in priority.tolist()] + [
                int(index) for index in ranked if int(index) not in priority_set
            ]
        else:
            ordered_indices = [int(index) for index in ranked]

        evaluations_spent = 0
        improvements = 0
        accepted_candidates = 0
        best_candidate_value = float(state.best_value)
        best_candidate_point = np.asarray(state.best_point, dtype=float).copy()
        for candidate_index in ordered_indices:
            if evaluations_spent >= eval_limit or self._remaining_evaluations() <= 0:
                break
            candidate = np.asarray(candidates[int(candidate_index)], dtype=float)
            if not point_in_bounds(candidate, current_bounds):
                continue
            archive_points, _archive_values = self.sample_archive.arrays()
            if archive_points.size > 0:
                duplicate_distance = nearest_archive_distance(candidate.reshape(1, 2), archive_points, current_bounds)
                if duplicate_distance.size and float(duplicate_distance[0]) <= 1e-12:
                    continue
            value, improved = self._evaluate_direct_sample(candidate)
            if not np.isfinite(value):
                continue
            evaluations_spent += 1
            accepted_candidates += 1
            if self.engine._is_better(value, best_candidate_value):
                best_candidate_value = float(value)
                best_candidate_point = candidate.copy()
            if improved:
                improvements += 1

        if evaluations_spent > 0:
            self.engine._update_improvement_trackers()

        success = bool(improvements > 0)
        if success:
            center_after = np.asarray(state.best_point, dtype=float).copy()
            radius_after = min(
                1.0,
                radius_before * float(self.agsls_config.trust_region_expand_factor),
            )
            reason = "improved"
        else:
            center_after = np.asarray(center, dtype=float).copy()
            radius_after = max(
                float(self.agsls_config.trust_region_min_radius_fraction),
                radius_before * float(self.agsls_config.trust_region_shrink_factor),
            )
            reason = "no_improvement" if accepted_candidates > 0 else "no_candidates"

        trust_state.center = center_after.copy()
        trust_state.radius_fraction = float(radius_after)
        event: dict[str, object] = {
            "phase": phase,
            "trust_region_used": evaluations_spent > 0,
            "trust_region_reason": reason,
            "trust_region_success": success,
            "trust_region_candidates": int(candidates.shape[0]),
            "trust_region_evaluations": int(evaluations_spent),
            "trust_region_improvements": int(improvements),
            "trust_region_center_before": np.asarray(center, dtype=float).tolist(),
            "trust_region_center_after": center_after.tolist(),
            "trust_region_radius_before": float(radius_before),
            "trust_region_radius_after": float(radius_after),
            "trust_region_best_before": best_before,
            "trust_region_best_after": float(state.best_value),
            "trust_region_remap": False,
            "trust_region_region_bounds": None if region is None else np.asarray(region, dtype=float).tolist(),
            **center_details,
            **surrogate.diagnostics(),
        }
        if best_candidate_point is not None and np.all(np.isfinite(best_candidate_point)):
            event["trust_region_best_candidate"] = best_candidate_point.tolist()
            event["trust_region_best_candidate_value"] = float(best_candidate_value)
        self._trust_region_events.append(event)
        return event

    def _commit_bounds(
        self,
        basin: Basin,
        current_bounds: np.ndarray,
        trust_region_details: dict[str, object] | None = None,
    ) -> tuple[np.ndarray | None, dict[str, object]]:
        current_widths = current_bounds[:, 1] - current_bounds[:, 0]
        height, width = self.engine.config.grid_shape
        cell_widths = np.asarray([current_widths[0] / width, current_widths[1] / height], dtype=float)
        min_widths = np.maximum(self.agsls_config.commit_min_shrink_fraction * current_widths, cell_widths)
        bbox_widths = basin.bbox_world[:, 1] - basin.bbox_world[:, 0]
        padded_widths = bbox_widths * (1.0 + 2.0 * self.agsls_config.commit_zoom_padding)
        surrogate = self._commit_surrogate(basin, current_bounds)
        surrogate_source = "basin"
        if self._should_try_incumbent_surrogate(basin, current_bounds):
            incumbent_basin = self._incumbent_surrogate_basin(current_bounds)
            if incumbent_basin is not None:
                incumbent_surrogate = self._commit_surrogate(incumbent_basin, current_bounds)
                if incumbent_surrogate.accepted:
                    surrogate = incumbent_surrogate
                    surrogate_source = "incumbent"
        center = basin.centroid_world if basin.basin_best_point is None else basin.basin_best_point
        if surrogate.accepted and surrogate.point is not None:
            center = surrogate.point
            surrogate_widths = surrogate_axis_widths(
                surrogate=surrogate,
                bbox_widths=padded_widths,
                current_widths=current_widths,
                valley_expand=self.agsls_config.commit_surrogate_valley_expand,
                cross_shrink=self.agsls_config.commit_surrogate_cross_shrink,
            )
            if surrogate_widths is not None:
                padded_widths = np.minimum(padded_widths, surrogate_widths)
        trust_center = None
        trust_radius = None
        if (
            trust_region_details is not None
            and bool(trust_region_details.get("trust_region_used"))
            and bool(trust_region_details.get("trust_region_success"))
            and bool(trust_region_details.get("trust_region_zoom_override"))
        ):
            raw_center = trust_region_details.get("trust_region_center_after")
            raw_radius = trust_region_details.get("trust_region_radius_after")
            try:
                candidate_center = np.asarray(raw_center, dtype=float)
                candidate_radius = float(raw_radius)
            except (TypeError, ValueError):
                candidate_center = np.empty((0,), dtype=float)
                candidate_radius = float("nan")
            if (
                candidate_center.shape == (2,)
                and np.all(np.isfinite(candidate_center))
                and np.isfinite(candidate_radius)
                and candidate_radius > 0.0
                and point_in_bounds(candidate_center, current_bounds)
            ):
                trust_center = candidate_center
                trust_radius = candidate_radius
                center = candidate_center
                trust_widths = 2.0 * candidate_radius * current_widths
                padded_widths = np.minimum(padded_widths, trust_widths)
        target_widths = np.minimum(current_widths, np.maximum(padded_widths, min_widths))
        new_bounds = self._centered_bounds(np.asarray(center, dtype=float), target_widths, current_bounds)
        new_bounds, retained_incumbent = self._expand_bounds_to_retain_incumbent(new_bounds, current_bounds, cell_widths)
        surrogate_diagnostics = {
            **surrogate.diagnostics(),
            "surrogate_source": surrogate_source,
            "trust_region_zoom_center_used": trust_center is not None,
            "trust_region_zoom_radius": None if trust_radius is None else float(trust_radius),
        }
        if not self._strict_progress(current_bounds, new_bounds):
            details = {
                "zoom_reason": "no_commit_shrink",
                **surrogate_diagnostics,
            }
            return None, details
        return new_bounds, {
            "zoom_reason": "commit_basin",
            "zoom_center": np.asarray(center, dtype=float).tolist(),
            "min_widths": min_widths.tolist(),
            "retained_global_best": bool(retained_incumbent),
            **surrogate_diagnostics,
        }

    def _exploitation_center(self, basin: Basin | None, current_bounds: np.ndarray) -> tuple[np.ndarray, bool, str]:
        state = self._require_state()
        best_point = np.asarray(state.best_point, dtype=float)
        if point_in_bounds(best_point, current_bounds):
            return best_point.copy(), True, "global_best"
        if basin is not None and basin.basin_best_point is not None and point_in_bounds(basin.basin_best_point, current_bounds):
            return np.asarray(basin.basin_best_point, dtype=float).copy(), False, "basin_best"
        if basin is not None and point_in_bounds(basin.centroid_world, current_bounds):
            return np.asarray(basin.centroid_world, dtype=float).copy(), False, "basin_centroid"
        return np.mean(current_bounds, axis=1), False, "box_center"

    def _exploitation_bounds(self, basin: Basin | None, current_bounds: np.ndarray) -> tuple[np.ndarray | None, dict[str, object]]:
        current_widths = current_bounds[:, 1] - current_bounds[:, 0]
        target_widths = current_widths * float(self.agsls_config.exploitation_shrink_fraction)
        center, anchored, center_kind = self._exploitation_center(basin, current_bounds)
        new_bounds = self._centered_bounds(center, target_widths, current_bounds)
        if not self._strict_progress(current_bounds, new_bounds):
            return None, {"zoom_reason": "no_exploitation_shrink", "zoom_center_kind": center_kind}
        used_representable_floor = bool(np.any((new_bounds[:, 1] - new_bounds[:, 0]) > np.maximum(target_widths, 0.0) * (1.0 + 1e-12)))
        return new_bounds, {
            "zoom_reason": "exploitation_global_best" if anchored else "exploitation_context",
            "zoom_center": center.tolist(),
            "zoom_center_kind": center_kind,
            "anchored_on_global_best": anchored,
            "representable_floor": used_representable_floor,
        }

    def _zoom_bounds_for_phase(
        self,
        phase: PhaseName,
        basin: Basin | None,
        trust_region_details: dict[str, object] | None = None,
    ) -> tuple[np.ndarray | None, dict[str, object]]:
        current_bounds = self._require_state().bounds.copy()
        if phase == "commit":
            if basin is None:
                return None, {"zoom_reason": "no_commit_basin"}
            return self._commit_bounds(basin, current_bounds, trust_region_details=trust_region_details)
        if phase == "exploitation":
            return self._exploitation_bounds(basin, current_bounds)
        return None, {"zoom_reason": "exploration_no_zoom"}

    @staticmethod
    def _basin_diagnostics(basin: Basin | None) -> dict[str, object]:
        if basin is None:
            return {}
        return {
            "score": float(basin.combined_score),
            "bbox": basin.bbox_world.tolist(),
            "area": int(basin.area),
            "support_mass": float(basin.support_mass),
            "alive_density": float(basin.alive_density),
            "stability_score": float(basin.stability_score),
            "objective_score": float(basin.objective_score),
            "evaluated_count": int(basin.evaluated_count),
            "incumbent_in_envelope": bool(basin.incumbent_in_envelope),
        }

    def _record_zoom(
        self,
        *,
        phase: PhaseName,
        selected: Basin | None,
        old_bounds: np.ndarray,
        new_bounds: np.ndarray,
        steps: int,
        evaluations_before: int,
        best_value_before: float,
        details: dict[str, object],
    ) -> None:
        state = self._require_state()
        diagnostics = self._basin_diagnostics(selected)
        old_area = self._bounds_area_for_progress(old_bounds)
        new_area = self._bounds_area_for_progress(new_bounds)
        shrink_ratio = new_area / old_area if np.isfinite(old_area) and old_area > 0.0 and np.isfinite(new_area) else 0.0
        diagnostics.update(
            {
                "phase": phase,
                "zoom_reason": str(details.get("zoom_reason", phase)),
                "evaluations_before": int(evaluations_before),
                "evaluations_after": int(state.evaluations),
                "best_value_before": float(best_value_before),
                "best_value_after": float(state.best_value),
                "budget_fraction": self._budget_fraction(),
                "shrink_ratio": float(shrink_ratio),
                **details,
            }
        )
        bbox = selected.bbox_world.copy() if selected is not None else new_bounds.copy()
        self.zoom_events.append(
            ZoomEvent(
                zoom_index=len(self.zoom_events),
                old_bounds=old_bounds.copy(),
                new_bounds=new_bounds.copy(),
                selected_basin_score=float(selected.combined_score) if selected is not None else 0.0,
                selected_basin_bbox=bbox,
                evaluation_count=int(state.evaluations),
                steps_per_zoom=int(steps),
                diagnostics=diagnostics,
            )
        )
        if self.engine.snapshots:
            snapshot = self.engine.snapshots[-1]
            snapshot.selected_basin_bbox = bbox
            snapshot.metadata["phase"] = phase
            snapshot.metadata["zoom_decision"] = "accepted"
            snapshot.metadata["zoom_reason"] = diagnostics["zoom_reason"]
            snapshot.metadata["selected_basin_diagnostics"] = diagnostics

    def _record_decision(
        self,
        *,
        phase: PhaseName,
        accepted: bool,
        reason: str,
        evaluations_before: int,
        bounds_before: np.ndarray,
        selected: Basin | None,
        details: dict[str, object] | None = None,
    ) -> None:
        state = self._require_state()
        self._decision_trace.append(
            {
                "phase": phase,
                "accepted": bool(accepted),
                "reason": str(reason),
                "evaluations_before": int(evaluations_before),
                "evaluations_after": int(state.evaluations),
                "bounds_before": np.asarray(bounds_before, dtype=float).tolist(),
                "bounds_after": np.asarray(state.bounds, dtype=float).tolist(),
                "best_value_after": float(state.best_value),
                "selected_basin": self._basin_diagnostics(selected),
                **(details or {}),
            }
        )

    def _decision_round(self, phase: PhaseName) -> bool:
        state = self._require_state()
        evaluations_before = int(state.evaluations)
        best_value_before = float(state.best_value)
        bounds_before = state.bounds.copy()
        self._phase_counts[phase] += 1
        self.engine.set_zoom_index(len(self.zoom_events), self._active_zoom_limit or self.agsls_config.max_zoom_cycles)
        self._apply_phase_parameters(phase)
        steps = self.agsls_config.commit_steps_per_zoom if phase == "commit" else self.agsls_config.exploitation_steps_per_zoom
        min_explored_fraction = self.agsls_config.commit_min_explored_fraction if phase == "commit" else 0.0
        reserve_evaluations = self.engine.config.evaluations_per_step if phase == "exploitation" else 0
        persistence = self._persistence_map(
            steps,
            min_explored_fraction=min_explored_fraction,
            reserve_evaluations=reserve_evaluations,
        )
        self._archive_current_grid_samples()
        explored_fraction_before_zoom = float(np.mean(self._require_state().evaluated_mask))
        ranked = self._rank_basins_for_phase(phase, persistence)
        selected = ranked[0] if ranked else None
        trust_region_details = (
            self._run_trust_region_batch(phase, selected, bounds_before)
            if phase in ("commit", "exploitation")
            else {}
        )
        valley_details = (
            self._track_exploitation_valley(selected, bounds_before)
            if phase == "exploitation" and not self.agsls_config.trust_region_enabled
            else {}
        )
        new_bounds, details = self._zoom_bounds_for_phase(phase, selected, trust_region_details=trust_region_details)
        details.update(trust_region_details)
        details.update(valley_details)
        details.setdefault("explored_fraction_before_zoom", explored_fraction_before_zoom)
        reason = str(details.get("zoom_reason", phase))
        if new_bounds is None or self._remaining_evaluations() <= 0:
            trust_progressed = int(trust_region_details.get("trust_region_evaluations", 0) or 0) > 0
            if self.engine.snapshots:
                self.engine.snapshots[-1].metadata["phase"] = phase
                self.engine.snapshots[-1].metadata["zoom_decision"] = "deferred"
                self.engine.snapshots[-1].metadata["zoom_reason"] = reason
            self._record_decision(
                phase=phase,
                accepted=False,
                reason=reason,
                evaluations_before=evaluations_before,
                bounds_before=bounds_before,
                selected=selected,
                details=details,
            )
            return bool(trust_progressed)
        self.engine.remap_to_bounds(new_bounds)
        if trust_region_details and self._trust_region_events:
            self._trust_region_events[-1]["trust_region_remap"] = True
            trust_region_details["trust_region_remap"] = True
        self._archive_current_grid_samples()
        self._record_zoom(
            phase=phase,
            selected=selected,
            old_bounds=bounds_before,
            new_bounds=new_bounds,
            steps=steps,
            evaluations_before=evaluations_before,
            best_value_before=best_value_before,
            details=details,
        )
        self._record_decision(
            phase=phase,
            accepted=True,
            reason=reason,
            evaluations_before=evaluations_before,
            bounds_before=bounds_before,
            selected=selected,
            details=details,
        )
        return True

    def _run_exploration_phase(self, eval_limit: int) -> None:
        target = int(round(float(eval_limit) * self.agsls_config.exploration_fraction))
        self._apply_phase_parameters("exploration")
        previous_engine_limit = self.engine.active_max_evaluations
        self.engine.active_max_evaluations = min(int(eval_limit), max(int(target), int(self._require_state().evaluations)))
        try:
            while int(self._require_state().evaluations) < target and not self._max_evaluations_reached():
                self._phase_counts["exploration"] += 1
                before = int(self._require_state().evaluations)
                self._run_engine_steps(self.agsls_config.exploration_steps_per_tick)
                if self.engine.snapshots:
                    self.engine.snapshots[-1].metadata["phase"] = "exploration"
                    self.engine.snapshots[-1].metadata["zoom_decision"] = "none"
                if int(self._require_state().evaluations) <= before:
                    break
        finally:
            self.engine.active_max_evaluations = previous_engine_limit

    def _run_commit_phase(self, eval_limit: int, zoom_limit: int) -> None:
        target = int(round(float(eval_limit) * self.agsls_config.commit_fraction))
        previous_engine_limit = self.engine.active_max_evaluations
        self.engine.active_max_evaluations = min(int(eval_limit), max(int(target), int(self._require_state().evaluations)))
        try:
            while (
                int(self._require_state().evaluations) < target
                and len(self.zoom_events) < zoom_limit
                and not self._max_evaluations_reached()
            ):
                before_evaluations = int(self._require_state().evaluations)
                before_zooms = len(self.zoom_events)
                self._decision_round("commit")
                progressed = int(self._require_state().evaluations) > before_evaluations or len(self.zoom_events) > before_zooms
                if not progressed:
                    break
        finally:
            self.engine.active_max_evaluations = previous_engine_limit

    def _run_exploitation_phase(self, zoom_limit: int) -> None:
        while len(self.zoom_events) < zoom_limit and not self._max_evaluations_reached():
            accepted = self._decision_round("exploitation")
            if not accepted:
                break

    def step(self) -> bool:
        """Run one phase-appropriate AGSLS action."""

        phase = self._phase_for_budget()
        if phase == "exploration":
            before = int(self._require_state().evaluations)
            self._apply_phase_parameters("exploration")
            self._run_engine_steps(self.agsls_config.exploration_steps_per_tick)
            self._phase_counts["exploration"] += 1
            return int(self._require_state().evaluations) > before
        return self._decision_round(phase)

    def run(self, zoom_cycles: int | None = None, evaluations: int | None = None) -> SearchRun:
        """Run AGSLS through exploration, commit, and exploitation phases."""

        eval_limit = self.agsls_config.max_evaluations if evaluations is None else int(evaluations)
        if eval_limit is None:
            raise ValueError("AGSLS runs require max_evaluations or run(evaluations=...)")
        if eval_limit <= 0:
            raise ValueError("evaluation limit must be positive")
        zoom_limit = self.agsls_config.max_zoom_cycles if zoom_cycles is None else int(zoom_cycles)
        if zoom_limit <= 0:
            raise ValueError("zoom_cycles must be positive")
        self.active_max_evaluations = int(eval_limit)
        self.engine.active_max_evaluations = int(eval_limit)
        self._active_zoom_limit = int(zoom_limit)
        try:
            self._run_exploration_phase(int(eval_limit))
            self._run_commit_phase(int(eval_limit), int(zoom_limit))
            self._run_exploitation_phase(int(zoom_limit))
        finally:
            self.active_max_evaluations = None
            self.engine.active_max_evaluations = None
            self._active_zoom_limit = None
        state = self._require_state()
        return SearchRun(
            best_point=state.best_point.copy(),
            best_value=float(state.best_value),
            evaluations=int(state.evaluations),
            bounds=state.bounds.copy(),
            snapshots=list(self.engine.snapshots),
            zoom_events=list(self.zoom_events),
            metadata={
                "mode": "agsls",
                "phases": {
                    "exploration_fraction": float(self.agsls_config.exploration_fraction),
                    "commit_fraction": float(self.agsls_config.commit_fraction),
                },
                "phase_counts": dict(sorted(self._phase_counts.items())),
                "zoom_cycles": len(self.zoom_events),
                "decision_trace": list(self._decision_trace),
                "trust_region_events": list(self._trust_region_events),
                "sample_archive_size": len(self.sample_archive),
            },
        )

    def zoom_history(self) -> list[ZoomEvent]:
        """Return recorded zoom events."""

        return list(self.zoom_events)

    def snapshot(self):
        """Return the current SmoothLife snapshot."""

        return self.engine.snapshot()
