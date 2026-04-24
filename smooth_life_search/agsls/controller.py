"""Adaptive Grid Smooth Life Search controller."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..core import (
    AGSLS_PER_DECISION_FIELDS,
    SMOOTHLIFE_PER_STEP_FIELDS,
    ZOOM_BOUNDARY_FIELDS,
    Basin,
    RuntimeSignals,
    SchedulePolicy,
    SearchRun,
    ZoomEvent,
    bounds_area,
    normalize_bounds_2d,
    point_in_bounds,
)
from ..smoothlife.config import SmoothLifeConfig
from ..smoothlife.simulator import SmoothLifeSearch
from .config import AGSLSConfig
from .scheduling import steps_for_zoom_cycle
from .scoring import score_basins

Objective = Callable[[ np.ndarray ], float]


class AdaptiveGridSmoothLifeSearch:
    """AGSLS: use SmoothLifeSearch locally, then zoom into promising dense groups."""

    def __init__(
        self,
        objective: Objective,
        bounds: np.ndarray | list[ tuple[ float, float ] ],
        smoothlife_config: SmoothLifeConfig | None = None,
        agsls_config: AGSLSConfig | None = None,
        runtime_policy: SchedulePolicy | None = None,
    ) -> None:
        self.objective = objective
        self.original_bounds = self._normalize_bounds( bounds )
        self.smoothlife_config = smoothlife_config or SmoothLifeConfig()
        self.agsls_config = agsls_config or AGSLSConfig()
        self.runtime_policy = runtime_policy
        self.engine = SmoothLifeSearch( objective, self.original_bounds, self.smoothlife_config, runtime_policy=runtime_policy )
        self.zoom_events: list[ ZoomEvent ] = [ ]
        self.active_max_evaluations: int | None = None
        self._active_zoom_limit: int | None = None
        self._last_basin_count = 0
        self._last_top_basin_score_gap = 0.0
        self._decision_reason_counts: dict[str, int] = { }
        self._accepted_zoom_count = 0
        self._decision_trace: list[ dict[str, object] ] = [ ]
        self._box_round_counts: dict[int, int] = { }
        self._late_stage_round_counts: dict[int, int] = { }
        self._late_stage_step_counts: dict[int, int] = { }
        self._periodic_local_search_summaries: dict[int, dict[str, object]] = { }

    @staticmethod
    def _normalize_bounds( bounds: np.ndarray | list[ tuple[ float, float ] ] ) -> np.ndarray:
        return normalize_bounds_2d( bounds, owner="AGSLS" )

    def reset( self, seed: int | None = None ) -> None:
        """Reset the underlying SmoothLife engine and zoom history."""

        self.engine.reset( seed=seed, bounds=self.original_bounds.copy() )
        self.engine.set_zoom_index( 0, max_zoom_cycles=self.agsls_config.max_zoom_cycles )
        self.zoom_events = [ ]
        self._active_zoom_limit = None
        self._last_basin_count = 0
        self._last_top_basin_score_gap = 0.0
        self._decision_reason_counts = { }
        self._accepted_zoom_count = 0
        self._decision_trace = [ ]
        self._box_round_counts = { }
        self._late_stage_round_counts = { }
        self._late_stage_step_counts = { }
        self._periodic_local_search_summaries = { }

    def _max_evaluations_reached( self ) -> bool:
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        limit = self.active_max_evaluations if self.active_max_evaluations is not None else self.agsls_config.max_evaluations
        return limit is not None and state.evaluations >= limit

    def _remaining_evaluations( self ) -> int | None:
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        limit = self.active_max_evaluations if self.active_max_evaluations is not None else self.agsls_config.max_evaluations
        if limit is None:
            return None
        return max( int( limit ) - int( state.evaluations ), 0 )

    def _bounded_evaluation_batch( self, requested: int ) -> int:
        remaining = self._remaining_evaluations()
        if remaining is None:
            return int( requested )
        return max( 0, min( int( requested ), remaining ) )

    def _effective_zoom_limit( self, eval_limit: int | None, configured_limit: int ) -> int:
        baseline = self.agsls_config.zoom_cycles_budget_baseline
        if eval_limit is None or baseline is None or baseline <= 0 or eval_limit < baseline:
            return int( configured_limit )
        ratio = float( eval_limit ) / float( baseline )
        increment = int( math.floor( math.log2( ratio ) ) )
        return int( configured_limit ) + max( 0, increment )

    @staticmethod
    def _bounds_area( bounds: np.ndarray ) -> float:
        return bounds_area( bounds )

    def _box_area_ratio( self, bounds: np.ndarray ) -> float:
        return self._bounds_area( bounds ) / max( self._bounds_area( self.original_bounds ), 1e-12 )

    @staticmethod
    def _point_in_bounds( point: np.ndarray, bounds: np.ndarray ) -> bool:
        return point_in_bounds( point, bounds )

    def _objective_value_key( self, value: float ) -> float:
        return float( value ) if self.engine.config.maximize else -float( value )

    def _minimum_zoom_widths( self, current_bounds: np.ndarray ) -> np.ndarray:
        height, width = self.engine.config.grid_shape
        current_widths = np.asarray( current_bounds, dtype=float )[ :, 1 ] - np.asarray( current_bounds, dtype=float )[ :, 0 ]
        cell_widths = np.asarray( [ current_widths[ 0 ] / width, current_widths[ 1 ] / height ], dtype=float )
        return self.agsls_config.min_zoom_cells * cell_widths

    def _enforce_min_side_fraction( self, bounds: np.ndarray, current_bounds: np.ndarray ) -> np.ndarray:
        new_bounds = np.asarray( bounds, dtype=float ).copy()
        min_widths = self.agsls_config.min_side_fraction * ( self.original_bounds[ :, 1 ] - self.original_bounds[ :, 0 ] )
        widths = new_bounds[ :, 1 ] - new_bounds[ :, 0 ]
        for idx in range( 2 ):
            if widths[ idx ] >= min_widths[ idx ]:
                continue
            center = 0.5 * ( new_bounds[ idx, 0 ] + new_bounds[ idx, 1 ] )
            half_width = 0.5 * min_widths[ idx ]
            lower = max( current_bounds[ idx, 0 ], center - half_width )
            upper = min( current_bounds[ idx, 1 ], center + half_width )
            if upper - lower < min_widths[ idx ]:
                if lower <= current_bounds[ idx, 0 ]:
                    upper = min( current_bounds[ idx, 1 ], lower + min_widths[ idx ] )
                else:
                    lower = max( current_bounds[ idx, 0 ], upper - min_widths[ idx ] )
            new_bounds[ idx, 0 ] = lower
            new_bounds[ idx, 1 ] = upper
        return new_bounds

    def _centered_bounds( self, center: np.ndarray, widths: np.ndarray, current_bounds: np.ndarray ) -> np.ndarray:
        resolved_center = np.asarray( center, dtype=float )
        resolved_widths = np.asarray( widths, dtype=float )
        bounds = np.column_stack( ( resolved_center - 0.5 * resolved_widths, resolved_center + 0.5 * resolved_widths ) )
        bounds[ :, 0 ] = np.maximum( bounds[ :, 0 ], current_bounds[ :, 0 ] )
        bounds[ :, 1 ] = np.minimum( bounds[ :, 1 ], current_bounds[ :, 1 ] )
        return self._expand_bounds_to_min_widths( bounds, resolved_center, resolved_widths, current_bounds )

    def _finalize_zoom_bounds(
        self,
        bounds: np.ndarray,
        center: np.ndarray,
        current_bounds: np.ndarray,
        *,
        anchor_point: np.ndarray | None = None,
        incumbent_point: np.ndarray | None = None,
    ) -> np.ndarray:
        min_zoom_widths = self._minimum_zoom_widths( current_bounds )
        resolved_center = np.asarray( center, dtype=float )
        new_bounds = self._expand_bounds_to_min_widths(
            np.asarray( bounds, dtype=float ),
            resolved_center,
            min_zoom_widths,
            current_bounds,
        )
        anchor = resolved_center if anchor_point is None else np.asarray( anchor_point, dtype=float )
        widths = new_bounds[ :, 1 ] - new_bounds[ :, 0 ]
        directional_padding = self.agsls_config.zoom_padding * widths
        for idx in range( 2 ):
            edge_band = self.agsls_config.edge_risk_fraction * max( widths[ idx ], 1e-12 )
            if anchor[ idx ] - new_bounds[ idx, 0 ] <= edge_band:
                new_bounds[ idx, 0 ] -= directional_padding[ idx ]
            if new_bounds[ idx, 1 ] - anchor[ idx ] <= edge_band:
                new_bounds[ idx, 1 ] += directional_padding[ idx ]
        if incumbent_point is not None and np.all( np.isfinite( incumbent_point ) ):
            resolved_incumbent = np.asarray( incumbent_point, dtype=float )
            for idx in range( 2 ):
                if resolved_incumbent[ idx ] < current_bounds[ idx, 0 ] or resolved_incumbent[ idx ] > current_bounds[ idx, 1 ]:
                    continue
                width = max( float( new_bounds[ idx, 1 ] - new_bounds[ idx, 0 ] ), float( min_zoom_widths[ idx ] ) )
                edge_band = self.agsls_config.edge_risk_fraction * width
                if resolved_incumbent[ idx ] < new_bounds[ idx, 0 ]:
                    new_bounds[ idx, 0 ] = max( current_bounds[ idx, 0 ], resolved_incumbent[ idx ] - edge_band )
                elif resolved_incumbent[ idx ] > new_bounds[ idx, 1 ]:
                    new_bounds[ idx, 1 ] = min( current_bounds[ idx, 1 ], resolved_incumbent[ idx ] + edge_band )
        widths = new_bounds[ :, 1 ] - new_bounds[ :, 0 ]
        padding = self.agsls_config.zoom_padding * widths
        new_bounds[ :, 0 ] -= padding
        new_bounds[ :, 1 ] += padding
        new_bounds[ :, 0 ] = np.maximum( new_bounds[ :, 0 ], current_bounds[ :, 0 ] )
        new_bounds[ :, 1 ] = np.minimum( new_bounds[ :, 1 ], current_bounds[ :, 1 ] )
        new_bounds = self._expand_bounds_to_min_widths( new_bounds, resolved_center, min_zoom_widths, current_bounds )
        return self._enforce_min_side_fraction( new_bounds, current_bounds )

    @staticmethod
    def _empty_microgrid_summary() -> dict[ str, object ]:
        return {
            "microgrid_ran": False,
            "microgrid_candidate_centers": [ ],
            "microgrid_sample_count_total": 0,
            "microgrid_sample_count_per_center": 0,
            "microgrid_winning_center_kind": "",
            "microgrid_winning_point": [ ],
            "microgrid_winning_value": None,
            "microgrid_refined_bounds": [ ],
        }

    def _microgrid_summary_payload( self, summary: dict[ str, object ] | None ) -> dict[ str, object ]:
        payload = self._empty_microgrid_summary()
        if summary is not None:
            payload.update( summary )
        return payload

    @staticmethod
    def _empty_translation_summary() -> dict[ str, object ]:
        return {
            "translation_ran": False,
            "translation_trigger_reason": "",
            "translation_target_kind": "",
            "translation_source_point": [ ],
            "translation_target_point": [ ],
            "translation_applied_vector": [ ],
            "translation_refined_bounds": [ ],
        }

    def _translation_summary_payload( self, summary: dict[ str, object ] | None ) -> dict[ str, object ]:
        payload = self._empty_translation_summary()
        if summary is not None:
            payload.update( summary )
        return payload

    @staticmethod
    def _empty_pattern_search_summary() -> dict[ str, object ]:
        return {
            "pattern_search_ran": False,
            "pattern_search_seed_kind": "",
            "pattern_search_iterations": 0,
            "pattern_search_evaluations_spent": 0,
            "pattern_search_evaluations_reused": 0,
            "pattern_search_final_step": [ ],
            "pattern_search_final_point": [ ],
            "pattern_search_final_value": None,
            "pattern_search_refined_bounds": [ ],
            "pattern_search_improved": False,
        }

    def _pattern_search_summary_payload( self, summary: dict[ str, object ] | None ) -> dict[ str, object ]:
        payload = self._empty_pattern_search_summary()
        if summary is not None:
            payload.update( summary )
        return payload

    @staticmethod
    def _empty_periodic_local_search_summary() -> dict[ str, object ]:
        return {
            "periodic_local_search_ran": False,
            "periodic_local_search_runs": 0,
            "periodic_local_search_total_evaluations_spent": 0,
            "periodic_local_search_last_step_index": None,
            "periodic_local_search_last_seed_count": 0,
            "periodic_local_search_last_best_point": [ ],
            "periodic_local_search_last_best_value": None,
            "periodic_local_search_last_improved": False,
            "periodic_local_search_last_evaluations_spent": 0,
        }

    def _periodic_local_search_summary_payload(
        self,
        *,
        box_id: int | None = None,
        summary: dict[ str, object ] | None = None,
    ) -> dict[ str, object ]:
        payload = self._empty_periodic_local_search_summary()
        if box_id is not None:
            payload.update( self._periodic_local_search_summaries.get( int( box_id ), {} ) )
        if summary is not None:
            payload.update( summary )
        return payload

    def _late_stage_summary_payload(
        self,
        *,
        box_id: int | None = None,
        microgrid_summary: dict[ str, object ] | None = None,
        translation_summary: dict[ str, object ] | None = None,
        pattern_search_summary: dict[ str, object ] | None = None,
        periodic_local_search_summary: dict[ str, object ] | None = None,
    ) -> dict[ str, object ]:
        payload = self._microgrid_summary_payload( microgrid_summary )
        payload.update( self._translation_summary_payload( translation_summary ) )
        payload.update( self._pattern_search_summary_payload( pattern_search_summary ) )
        payload.update(
            self._periodic_local_search_summary_payload(
                box_id=box_id,
                summary=periodic_local_search_summary,
            )
        )
        return payload

    def _sync_latest_snapshot_state( self ) -> None:
        state = self.engine.state
        if state is None or not self.engine.snapshots:
            return
        snapshot = self.engine.snapshots[ -1 ]
        snapshot.best_point = state.best_point.copy()
        snapshot.best_value = float( state.best_value )
        snapshot.local_best_point = state.local_best_point.copy()
        snapshot.local_best_value = float( state.local_best_value )
        snapshot.box_best_point = state.box_best_point.copy()
        snapshot.box_best_value = float( state.box_best_value )

    def runtime_signals( self ) -> RuntimeSignals:
        base = self.engine.runtime_signals()
        return RuntimeSignals(
            step_index=base.step_index,
            zoom_index=base.zoom_index,
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
            max_zoom_cycles=base.max_zoom_cycles,
        )

    def apply_runtime_overrides( self, overrides: dict[ str, float | int ] ) -> dict[ str, float | int ]:
        changed: dict[ str, float | int ] = { }
        if not overrides:
            return changed
        for field_name, value in overrides.items():
            if not hasattr( self.agsls_config, field_name ):
                continue
            current = getattr( self.agsls_config, field_name )
            if current == value:
                continue
            setattr( self.agsls_config, field_name, value )
            changed[ field_name ] = value
        if changed:
            self.agsls_config.__post_init__()
        return changed

    def apply_runtime_policy( self, allowed_fields: set[ str ] | frozenset[ str ] ) -> dict[ str, float | int ]:
        if self.runtime_policy is None:
            return { }
        overrides = self.runtime_policy.resolve( self.runtime_signals(), allowed_fields )
        return self.apply_runtime_overrides( overrides )

    def _update_basin_runtime_state( self, basins: list[ Basin ] ) -> None:
        self._last_basin_count = len( basins )
        if len( basins ) >= 2:
            self._last_top_basin_score_gap = float( basins[ 0 ].combined_score - basins[ 1 ].combined_score )
        else:
            self._last_top_basin_score_gap = 0.0

    def _persistence_map(
        self,
        steps: int,
        *,
        box_id: int,
        late_stage_state: dict[ str, float | int | bool ],
    ) -> np.ndarray:
        height, width = self.engine.config.grid_shape
        persistence = np.zeros( ( height, width ), dtype=float )
        executed_steps = 0
        for _ in range( steps ):
            if self._max_evaluations_reached():
                break
            remaining = self._remaining_evaluations()
            if remaining is not None and remaining < self.engine.config.evaluations_per_step:
                break
            self.engine.step( 1 )
            self._maybe_run_periodic_local_search( box_id=box_id, late_stage_state=late_stage_state )
            state = self.engine.state
            if state is None:
                raise RuntimeError( "engine state missing after step" )
            support = self.engine.support_field()
            alive = self.engine.alive_mask( self.agsls_config.alive_core_threshold )
            threshold = float( np.quantile( support, self.agsls_config.basin_quantile ) )
            persistence += ( alive & ( support >= threshold ) ).astype( float )
            executed_steps += 1
        return persistence / max( executed_steps, 1 )

    def _rank_basins( self, persistence: np.ndarray ) -> list[ Basin ]:
        basins = self.engine.basin_candidates(
            threshold_quantile=self.agsls_config.basin_quantile,
            min_cells=self.agsls_config.min_basin_cells,
            alive_core_threshold=self.agsls_config.alive_core_threshold,
            cluster_eps_pixels=self.agsls_config.cluster_eps_pixels,
            cluster_min_samples=self.agsls_config.cluster_min_samples,
            basin_envelope_quantile_offset=self.agsls_config.basin_envelope_quantile_offset,
            basin_envelope_growth_pixels=self.agsls_config.basin_envelope_growth_pixels,
        )
        for basin in basins:
            scoring_mask = basin.core_mask if basin.core_mask is not None else basin.mask
            basin.stability_score = float( np.mean( persistence[ scoring_mask ] ) ) if np.any( scoring_mask ) else 0.0
        return score_basins( basins, self.agsls_config )

    def _basin_metadata( self, basins: list[ Basin ] ) -> list[ dict[str, float | int | bool | list[ list[ float ] ]] ]:
        payload: list[ dict[str, float | int | bool | list[ list[ float ] ]] ] = [ ]
        for rank, basin in enumerate( basins, start=1 ):
            payload.append(
                {
                    "rank": float( rank ),
                    "score": float( basin.combined_score ),
                    "alive_density": float( basin.alive_density ),
                    "support_mass": float( basin.support_mass ),
                    "bbox": basin.bbox_world.tolist(),
                    "core_bbox": [] if basin.core_bbox_world is None else basin.core_bbox_world.tolist(),
                    "evaluated_count": int( basin.evaluated_count ),
                    "best_objective_score": float( basin.best_objective_score ),
                    "mean_objective_score": float( basin.mean_objective_score ),
                    "unexplored_fraction": float( basin.unexplored_fraction ),
                    "incumbent_in_envelope": bool( basin.incumbent_in_envelope ),
                    "better_than_incumbent": bool( basin.better_than_incumbent ),
                }
            )
        return payload

    def _basin_diagnostics( self, basin: Basin ) -> dict[ str, float | int | bool | list[ list[ float ] ] ]:
        return {
            "score": float( basin.combined_score ),
            "bbox": basin.bbox_world.tolist(),
            "core_bbox": [] if basin.core_bbox_world is None else basin.core_bbox_world.tolist(),
            "area": int( basin.area ),
            "support_mass": float( basin.support_mass ),
            "alive_density": float( basin.alive_density ),
            "stability_score": float( basin.stability_score ),
            "objective_score": float( basin.objective_score ),
            "evaluated_count": int( basin.evaluated_count ),
            "best_objective_score": float( basin.best_objective_score ),
            "mean_objective_score": float( basin.mean_objective_score ),
            "unexplored_fraction": float( basin.unexplored_fraction ),
            "incumbent_in_envelope": bool( basin.incumbent_in_envelope ),
            "better_than_incumbent": bool( basin.better_than_incumbent ),
        }

    def _record_decision_trace(
        self,
        *,
        box_id: int,
        accepted: bool,
        decision_reason: str,
        evaluations_before: int,
        evaluations_after: int,
        best_value_before: float,
        best_value_after: float,
        bounds_before: np.ndarray,
        bounds_after: np.ndarray,
        selected_basin: Basin | None,
        incumbent_point_before_zoom: np.ndarray,
        late_stage_mode: bool = False,
        intensification_rounds: int = 0,
        projected_shrink_ratio: float = 1.0,
        focus_mask_coverage: float = 0.0,
        late_stage_exit_reason: str = "none",
        microgrid_summary: dict[ str, object ] | None = None,
        translation_summary: dict[ str, object ] | None = None,
        pattern_search_summary: dict[ str, object ] | None = None,
    ) -> None:
        box_round_index = int( self._box_round_counts.get( box_id, 0 ) ) + 1
        self._box_round_counts[ box_id ] = box_round_index
        late_stage_payload = self._late_stage_summary_payload(
            box_id=box_id,
            microgrid_summary=microgrid_summary,
            translation_summary=translation_summary,
            pattern_search_summary=pattern_search_summary,
        )
        self._decision_trace.append(
            {
                "zoom_index": int( box_id ),
                "box_id": int( box_id ),
                "box_round_index": box_round_index,
                "accepted": bool( accepted ),
                "decision_reason": str( decision_reason ),
                "evaluations_before": int( evaluations_before ),
                "evaluations_after": int( evaluations_after ),
                "best_value_before": float( best_value_before ),
                "best_value_after": float( best_value_after ),
                "bounds_before": np.asarray( bounds_before, dtype=float ).tolist(),
                "bounds_after": np.asarray( bounds_after, dtype=float ).tolist(),
                "incumbent_point_before_zoom": np.asarray( incumbent_point_before_zoom, dtype=float ).tolist(),
                "selected_basin_diagnostics": None if selected_basin is None else self._basin_diagnostics( selected_basin ),
                "basin_count": int( self._last_basin_count ),
                "top_basin_score_gap": float( self._last_top_basin_score_gap ),
                "late_stage_mode": bool( late_stage_mode ),
                "intensification_rounds": int( intensification_rounds ),
                "projected_shrink_ratio": float( projected_shrink_ratio ),
                "focus_mask_coverage": float( focus_mask_coverage ),
                "late_stage_exit_reason": str( late_stage_exit_reason ),
                **late_stage_payload,
            }
        )

    def _late_stage_state( self, *, box_id: int, active_limit: int, bounds: np.ndarray ) -> dict[ str, float | int | bool ]:
        signals = self.runtime_signals()
        zoom_fraction = signals.zoom_fraction()
        prior_box_rounds = int( self._box_round_counts.get( box_id, 0 ) )
        plateau = prior_box_rounds > 0 and max( float( signals.global_improvement ), float( signals.stage_improvement ) ) <= self.agsls_config.late_stage_plateau_threshold
        small_box = self._box_area_ratio( bounds ) <= 0.1
        rounds = int( self._late_stage_round_counts.get( box_id, 0 ) )
        late = zoom_fraction >= self.agsls_config.late_stage_zoom_fraction_threshold
        return {
            "zoom_fraction": float( zoom_fraction ),
            "plateau": bool( plateau ),
            "small_box": bool( small_box ),
            "late": bool( late ),
            "intensification_rounds": rounds,
            "late_stage_mode": bool( late or plateau or small_box or rounds > 0 ),
        }

    def _late_stage_step_batch( self, base_batch: int, *, late_stage_state: dict[ str, float | int | bool ] ) -> int:
        if not bool( late_stage_state.get( "late_stage_mode", False ) ):
            return int( base_batch )
        return max( 1, min( int( base_batch ), int( self.agsls_config.late_stage_eval_batch ) ) )

    def _projected_shrink_ratio( self, old_bounds: np.ndarray, new_bounds: np.ndarray ) -> float:
        return self._bounds_area( new_bounds ) / max( self._bounds_area( old_bounds ), 1e-12 )

    def _late_stage_focus_mask( self, basin: Basin ) -> tuple[ np.ndarray, np.ndarray, float ]:
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        base_mask = np.asarray( basin.mask, dtype=bool )
        explored_mask = base_mask & np.asarray( state.evaluated_mask, dtype=bool )
        support = self.engine.support_field()
        if not np.any( explored_mask ):
            coverage = 1.0 if np.any( base_mask ) else 0.0
            return base_mask.copy(), np.asarray( support, dtype=float ), float( coverage )
        objective_scores = np.asarray( state.objective_field, dtype=float )[ explored_mask ]
        candidate_rows, candidate_cols = np.nonzero( explored_mask )
        elite_k = min( int( self.agsls_config.late_stage_elite_k ), int( candidate_rows.size ) )
        top_order = np.argsort( objective_scores )[ -elite_k: ]
        elite_rows = candidate_rows[ top_order ]
        elite_cols = candidate_cols[ top_order ]
        row_grid, col_grid = np.indices( base_mask.shape )
        radius = max( int( self.agsls_config.late_stage_focus_radius_cells ), 1 )
        elite_proximity = np.zeros( base_mask.shape, dtype=float )
        focus_mask = np.zeros( base_mask.shape, dtype=bool )
        for row, col in zip( elite_rows, elite_cols ):
            distance = np.sqrt( ( row_grid - int( row ) ) ** 2 + ( col_grid - int( col ) ) ** 2 )
            elite_proximity = np.maximum( elite_proximity, np.clip( 1.0 - ( distance / float( radius ) ), 0.0, 1.0 ) )
        focus_mask |= elite_proximity > 0.0
        if basin.core_mask is not None:
            focus_mask |= np.asarray( basin.core_mask, dtype=bool )
        focus_mask &= base_mask
        if not np.any( focus_mask ):
            focus_mask = base_mask.copy()
        coverage = float( np.count_nonzero( focus_mask ) ) / max( int( np.count_nonzero( base_mask ) ), 1 )
        return focus_mask, elite_proximity, coverage

    def _late_stage_score_field( self, basin: Basin ) -> tuple[ np.ndarray, np.ndarray, float ]:
        focus_mask, elite_proximity, coverage = self._late_stage_focus_mask( basin )
        support = np.asarray( self.engine.support_field(), dtype=float )
        score_field = 0.50 * self.engine.exploration_score_field() + 0.25 * support + 0.25 * elite_proximity
        return focus_mask, score_field, coverage

    def _late_stage_translation_target( self, basin: Basin ) -> tuple[ np.ndarray, str ]:
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        mask = np.asarray( basin.core_mask if basin.core_mask is not None else basin.mask, dtype=bool )
        evaluated_mask = mask & np.asarray( state.evaluated_mask, dtype=bool )
        if np.any( evaluated_mask ):
            rows, cols = np.nonzero( evaluated_mask )
            values = np.asarray( state.objective_values[ rows, cols ], dtype=float )
            primary = -values if self.engine.config.maximize else values
            order = np.lexsort( ( cols, rows, primary ) )
            elite_k = min( int( self.agsls_config.late_stage_elite_k ), int( order.size ) )
            selected_offsets = order[ :elite_k ]
            weights = np.arange( elite_k, 0, -1, dtype=float )
            elite_points = np.asarray(
                [
                    self.engine._pixel_center( int( rows[ offset ] ), int( cols[ offset ] ), state.bounds )
                    for offset in selected_offsets
                ],
                dtype=float,
            )
            target = np.average( elite_points, axis=0, weights=weights )
            return np.asarray( target, dtype=float ), "elite_weighted"
        if basin.basin_best_point is not None:
            return np.asarray( basin.basin_best_point, dtype=float ), "basin_best"
        return np.asarray( basin.centroid_world, dtype=float ), "centroid"

    def _run_late_stage_translation(
        self,
        basin: Basin,
        old_bounds: np.ndarray,
        standard_shrink_bounds: np.ndarray,
        trigger_reason: str,
    ) -> tuple[ np.ndarray, dict[ str, object ] ] | None:
        if not self.agsls_config.late_stage_translation_enabled:
            return None
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        old_widths = np.asarray( old_bounds, dtype=float )[ :, 1 ] - np.asarray( old_bounds, dtype=float )[ :, 0 ]
        if np.any( old_widths <= 0.0 ):
            return None
        target_point, target_kind = self._late_stage_translation_target( basin )
        source_point = np.mean( np.asarray( old_bounds, dtype=float ), axis=1 )
        target_offset = np.asarray( target_point, dtype=float ) - np.asarray( source_point, dtype=float )
        normalized_offset = np.abs( target_offset ) / np.maximum( old_widths, 1e-12 )
        if float( np.max( normalized_offset ) ) < float( self.agsls_config.late_stage_translation_min_offset_fraction ):
            return None
        movement_cap = float( self.agsls_config.late_stage_translation_step_fraction ) * old_widths
        applied_step = np.clip( target_offset, -movement_cap, movement_cap )
        if np.allclose( applied_step, 0.0 ):
            return None
        translated_center = np.asarray( source_point, dtype=float ) + applied_step
        standard_widths = np.asarray( standard_shrink_bounds, dtype=float )[ :, 1 ] - np.asarray( standard_shrink_bounds, dtype=float )[ :, 0 ]
        candidate_bounds = self._centered_bounds( translated_center, standard_widths, old_bounds )
        incumbent_point = None
        if not basin.better_than_incumbent and self._point_in_bounds( state.best_point, old_bounds ):
            incumbent_point = np.asarray( state.best_point, dtype=float )
        refined_bounds = self._finalize_zoom_bounds(
            candidate_bounds,
            translated_center,
            old_bounds,
            anchor_point=np.asarray( target_point, dtype=float ),
            incumbent_point=incumbent_point,
        )
        if np.allclose( refined_bounds, old_bounds ):
            return None
        summary = {
            "translation_ran": True,
            "translation_trigger_reason": str( trigger_reason ),
            "translation_target_kind": str( target_kind ),
            "translation_source_point": np.asarray( source_point, dtype=float ).tolist(),
            "translation_target_point": np.asarray( target_point, dtype=float ).tolist(),
            "translation_applied_vector": np.asarray( applied_step, dtype=float ).tolist(),
            "translation_refined_bounds": np.asarray( refined_bounds, dtype=float ).tolist(),
        }
        return refined_bounds, summary

    def _evaluate_arbitrary_points( self, points: np.ndarray ) -> np.ndarray:
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        resolved_points = np.asarray( points, dtype=float )
        if resolved_points.size == 0:
            return np.asarray( [ ], dtype=float )
        resolved_points = np.reshape( resolved_points, ( -1, 2 ) )
        values = np.asarray( [ float( self.objective( point ) ) for point in resolved_points ], dtype=float )
        state.evaluations += int( values.size )
        for point, value in zip( resolved_points, values ):
            candidate = np.asarray( point, dtype=float )
            candidate_value = float( value )
            if not np.isfinite( state.box_best_value ) or self.engine._is_better( candidate_value, float( state.box_best_value ) ):
                state.box_best_point = candidate.copy()
                state.box_best_value = candidate_value
            if not np.isfinite( state.local_best_value ) or self.engine._is_better( candidate_value, float( state.local_best_value ) ):
                state.local_best_point = candidate.copy()
                state.local_best_value = candidate_value
            if not np.isfinite( state.best_value ) or self.engine._is_better( candidate_value, float( state.best_value ) ):
                state.best_point = candidate.copy()
                state.best_value = candidate_value
        self.engine._update_improvement_trackers()
        self._sync_latest_snapshot_state()
        return values

    @staticmethod
    def _periodic_local_search_budget_split( seed_count: int, total_budget: int ) -> tuple[ int, ... ]:
        if seed_count <= 0 or total_budget <= 0:
            return tuple()
        base, remainder = divmod( int( total_budget ), int( seed_count ) )
        return tuple( base + ( 1 if index < remainder else 0 ) for index in range( int( seed_count ) ) )

    def _late_stage_periodic_elite_seeds( self, current_bounds: np.ndarray ) -> list[ np.ndarray ]:
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        if not np.any( state.evaluated_mask ):
            return [ ]
        rows, cols = np.nonzero( state.evaluated_mask )
        values = np.asarray( state.objective_values[ rows, cols ], dtype=float )
        primary = -values if self.engine.config.maximize else values
        order = np.lexsort( ( cols, rows, primary ) )
        elite_k = min( int( self.agsls_config.late_stage_elite_k ), int( order.size ) )
        return [
            self.engine._pixel_center( int( rows[ offset ] ), int( cols[ offset ] ), current_bounds )
            for offset in order[ :elite_k ]
        ]

    def _run_periodic_local_search_from_seed(
        self,
        seed: np.ndarray,
        current_bounds: np.ndarray,
        *,
        initial_step: np.ndarray,
        min_step: np.ndarray,
        shrink: float,
        max_iterations: int,
        evaluation_budget: int,
    ) -> dict[ str, object ] | None:
        seed_value, reused = self._reuse_evaluated_value( seed, current_bounds )
        if not reused:
            return None
        current_point = np.asarray( seed, dtype=float ).copy()
        current_value = float( seed_value )
        starting_value = float( seed_value )
        step = np.asarray( initial_step, dtype=float ).copy()
        fresh_spent = 0
        iterations = 0
        while iterations < int( max_iterations ) and fresh_spent < int( evaluation_budget ):
            if np.all( step <= min_step ):
                break
            iterations += 1
            best_trial_point: np.ndarray | None = None
            best_trial_value = float( current_value )
            terminated_on_budget = False
            offsets = np.asarray(
                [
                    [ step[ 0 ], 0.0 ],
                    [ -step[ 0 ], 0.0 ],
                    [ 0.0, step[ 1 ] ],
                    [ 0.0, -step[ 1 ] ],
                ],
                dtype=float,
            )
            for offset in offsets:
                if fresh_spent >= int( evaluation_budget ):
                    break
                if self._bounded_evaluation_batch( 1 ) < 1:
                    terminated_on_budget = True
                    break
                candidate = current_point + offset
                if not self._point_in_bounds( candidate, current_bounds ):
                    continue
                values = self._evaluate_arbitrary_points( np.asarray( candidate, dtype=float ).reshape( 1, 2 ) )
                if values.size == 0:
                    terminated_on_budget = True
                    break
                fresh_spent += 1
                candidate_value = float( values[ 0 ] )
                if self.engine._is_better( candidate_value, best_trial_value ):
                    best_trial_value = candidate_value
                    best_trial_point = np.asarray( candidate, dtype=float ).copy()
            if terminated_on_budget:
                break
            if best_trial_point is not None:
                current_point = best_trial_point
                current_value = float( best_trial_value )
            else:
                step = step * float( shrink )
        return {
            "point": current_point.copy(),
            "value": float( current_value ),
            "fresh_spent": int( fresh_spent ),
            "improved": bool( self.engine._is_better( current_value, starting_value ) ),
        }

    def _run_periodic_downhill_local_search(
        self,
        current_bounds: np.ndarray,
        seeds: list[ np.ndarray ],
    ) -> dict[ str, object ] | None:
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        if not seeds:
            return None
        current_widths = np.asarray( current_bounds, dtype=float )[ :, 1 ] - np.asarray( current_bounds, dtype=float )[ :, 0 ]
        if np.any( current_widths <= 0.0 ):
            return None
        total_budget = self._bounded_evaluation_batch( self.agsls_config.late_stage_periodic_local_search_max_evaluations )
        if total_budget <= 0:
            return None
        budgets = self._periodic_local_search_budget_split( len( seeds ), total_budget )
        initial_step = float( self.agsls_config.late_stage_periodic_local_search_initial_step_fraction ) * current_widths
        min_step = float( self.agsls_config.late_stage_periodic_local_search_min_step_fraction ) * current_widths
        shrink = float( self.agsls_config.late_stage_periodic_local_search_shrink )
        max_iterations = int( self.agsls_config.late_stage_periodic_local_search_max_iterations )
        best_value_before = float( state.best_value )
        best_result: dict[ str, object ] | None = None
        total_fresh_spent = 0
        for seed, seed_budget in zip( seeds, budgets ):
            result = self._run_periodic_local_search_from_seed(
                np.asarray( seed, dtype=float ),
                current_bounds,
                initial_step=initial_step,
                min_step=min_step,
                shrink=shrink,
                max_iterations=max_iterations,
                evaluation_budget=int( seed_budget ),
            )
            if result is None:
                continue
            total_fresh_spent += int( result[ "fresh_spent" ] )
            if best_result is None or self.engine._is_better( float( result[ "value" ] ), float( best_result[ "value" ] ) ):
                best_result = result
            if self._bounded_evaluation_batch( 1 ) < 1:
                break
        if best_result is None:
            return None
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        return {
            "periodic_local_search_last_step_index": int( state.step_index ),
            "periodic_local_search_last_seed_count": int( len( seeds ) ),
            "periodic_local_search_last_best_point": np.asarray( best_result[ "point" ], dtype=float ).tolist(),
            "periodic_local_search_last_best_value": float( best_result[ "value" ] ),
            "periodic_local_search_last_improved": bool( self.engine._is_better( float( state.best_value ), best_value_before ) ),
            "periodic_local_search_last_evaluations_spent": int( total_fresh_spent ),
        }

    def _store_periodic_local_search_summary( self, box_id: int, summary: dict[ str, object ] ) -> dict[ str, object ]:
        payload = self._periodic_local_search_summary_payload( box_id=box_id )
        payload[ "periodic_local_search_ran" ] = True
        payload[ "periodic_local_search_runs" ] = int( payload.get( "periodic_local_search_runs", 0 ) ) + 1
        payload[ "periodic_local_search_total_evaluations_spent" ] = (
            int( payload.get( "periodic_local_search_total_evaluations_spent", 0 ) )
            + int( summary.get( "periodic_local_search_last_evaluations_spent", 0 ) )
        )
        payload.update( summary )
        self._periodic_local_search_summaries[ int( box_id ) ] = payload
        return payload

    def _mark_latest_snapshot_periodic_local_search( self, box_id: int ) -> None:
        if not self.engine.snapshots:
            return
        self._sync_latest_snapshot_state()
        snapshot = self.engine.snapshots[ -1 ]
        snapshot.metadata[ "late_stage_mode" ] = True
        snapshot.metadata.update( self._late_stage_summary_payload( box_id=box_id ) )

    def _maybe_run_periodic_local_search(
        self,
        *,
        box_id: int,
        late_stage_state: dict[ str, float | int | bool ],
    ) -> dict[ str, object ] | None:
        if not self.agsls_config.late_stage_periodic_local_search_enabled:
            return None
        if not bool( late_stage_state.get( "late_stage_mode", False ) ):
            return None
        step_count = int( self._late_stage_step_counts.get( box_id, 0 ) ) + 1
        self._late_stage_step_counts[ int( box_id ) ] = step_count
        interval = int( self.agsls_config.late_stage_periodic_local_search_interval_steps )
        if interval <= 0 or step_count % interval != 0:
            return None
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        current_bounds = state.bounds.copy()
        seeds = self._late_stage_periodic_elite_seeds( current_bounds )
        summary = self._run_periodic_downhill_local_search( current_bounds, seeds )
        if summary is not None:
            summary = self._store_periodic_local_search_summary( box_id, summary )
        self._mark_latest_snapshot_periodic_local_search( box_id )
        return summary

    def _late_stage_microgrid_candidates( self, basin: Basin, current_bounds: np.ndarray ) -> list[ dict[ str, object ] ]:
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        limit = max( int( self.agsls_config.late_stage_microgrid_centers ), 1 )
        candidates: list[ dict[ str, object ] ] = [ ]

        def add_candidate( kind: str, point: np.ndarray | None ) -> bool:
            if point is None:
                return False
            resolved_point = np.asarray( point, dtype=float )
            if not self._point_in_bounds( resolved_point, current_bounds ):
                return False
            for existing in candidates:
                if np.allclose( np.asarray( existing[ "point" ], dtype=float ), resolved_point, atol=1e-12, rtol=0.0 ):
                    return False
            candidates.append( {"kind": str( kind ), "point": resolved_point.copy()} )
            return True

        add_candidate( "basin_best", basin.basin_best_point )
        if len( candidates ) < limit:
            add_candidate( "incumbent", state.best_point )
        mask = basin.core_mask if basin.core_mask is not None else basin.mask
        evaluated_mask = np.asarray( mask, dtype=bool ) & np.asarray( state.evaluated_mask, dtype=bool )
        if np.any( evaluated_mask ):
            rows, cols = np.nonzero( evaluated_mask )
            values = np.asarray( state.objective_values[ rows, cols ], dtype=float )
            order = np.argsort( values )
            if self.engine.config.maximize:
                order = order[ ::-1 ]
            elite_rank = 1
            for offset in order:
                if len( candidates ) >= limit:
                    break
                point = self.engine._pixel_center( int( rows[ offset ] ), int( cols[ offset ] ), state.bounds )
                added = add_candidate( f"elite_{ elite_rank }", point )
                if added:
                    elite_rank += 1
        return candidates[ :limit ]

    def _late_stage_microgrid_sample_bounds( self, center: np.ndarray, current_bounds: np.ndarray ) -> np.ndarray:
        current_widths = np.asarray( current_bounds, dtype=float )[ :, 1 ] - np.asarray( current_bounds, dtype=float )[ :, 0 ]
        min_zoom_widths = self._minimum_zoom_widths( current_bounds )
        sample_widths = np.maximum(
            self.agsls_config.late_stage_microgrid_side_fraction * current_widths,
            min_zoom_widths,
        )
        return self._centered_bounds( center, sample_widths, current_bounds )

    def _late_stage_microgrid_lattice( self, bounds: np.ndarray ) -> np.ndarray:
        resolution = int( self.agsls_config.late_stage_microgrid_resolution )
        x_values = np.linspace( float( bounds[ 0, 0 ] ), float( bounds[ 0, 1 ] ), resolution, dtype=float )
        y_values = np.linspace( float( bounds[ 1, 0 ] ), float( bounds[ 1, 1 ] ), resolution, dtype=float )
        grid_x, grid_y = np.meshgrid( x_values, y_values, indexing="xy" )
        return np.column_stack( ( grid_x.ravel(), grid_y.ravel() ) )

    def _run_late_stage_microgrid( self, basin: Basin, current_bounds: np.ndarray ) -> tuple[ np.ndarray, dict[ str, object ] ] | None:
        if not self.agsls_config.late_stage_microgrid_enabled:
            return None
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        candidates = self._late_stage_microgrid_candidates( basin, current_bounds )
        if not candidates:
            return None
        samples_per_center = int( self.agsls_config.late_stage_microgrid_resolution ) ** 2
        total_samples = int( len( candidates ) * samples_per_center )
        if self._bounded_evaluation_batch( total_samples ) < total_samples:
            return None
        incumbent_point = np.asarray( state.best_point, dtype=float )
        baseline_best_value = float( state.best_value )
        best_candidate: dict[ str, object ] | None = None
        best_key: tuple[ float, float, float, float ] | None = None
        for candidate_index, candidate in enumerate( candidates ):
            center = np.asarray( candidate[ "point" ], dtype=float )
            sample_bounds = self._late_stage_microgrid_sample_bounds( center, current_bounds )
            lattice = self._late_stage_microgrid_lattice( sample_bounds )
            values = self._evaluate_arbitrary_points( lattice )
            order = np.argsort( values )
            if self.engine.config.maximize:
                order = order[ ::-1 ]
            top_count = min( 4, len( order ) )
            best_offset = int( order[ 0 ] )
            top_values = values[ order[ :top_count ] ]
            best_point = np.asarray( lattice[ best_offset ], dtype=float )
            best_value = float( values[ best_offset ] )
            mean_top_value = float( np.mean( top_values ) )
            incumbent_distance = (
                float( np.linalg.norm( best_point - incumbent_point ) )
                if self._point_in_bounds( incumbent_point, current_bounds )
                else 0.0
            )
            candidate_key = (
                self._objective_value_key( best_value ),
                self._objective_value_key( mean_top_value ),
                -incumbent_distance,
                -float( candidate_index ),
            )
            if best_key is None or candidate_key > best_key:
                best_key = candidate_key
                best_candidate = {
                    "kind": str( candidate[ "kind" ] ),
                    "sample_bounds": sample_bounds.copy(),
                    "best_point": best_point.copy(),
                    "best_value": best_value,
                }
        if best_candidate is None:
            return None
        sample_bounds = np.asarray( best_candidate[ "sample_bounds" ], dtype=float )
        refined_widths = 0.5 * ( sample_bounds[ :, 1 ] - sample_bounds[ :, 0 ] )
        best_point = np.asarray( best_candidate[ "best_point" ], dtype=float )
        raw_refined_bounds = self._centered_bounds( best_point, refined_widths, current_bounds )
        preserve_incumbent = (
            self._point_in_bounds( incumbent_point, current_bounds )
            and not self.engine._is_better( float( best_candidate[ "best_value" ] ), baseline_best_value )
        )
        refined_bounds = self._finalize_zoom_bounds(
            raw_refined_bounds,
            best_point,
            current_bounds,
            anchor_point=best_point,
            incumbent_point=incumbent_point if preserve_incumbent else None,
        )
        summary = {
            "microgrid_ran": True,
            "microgrid_candidate_centers": [ str( candidate[ "kind" ] ) for candidate in candidates ],
            "microgrid_sample_count_total": total_samples,
            "microgrid_sample_count_per_center": samples_per_center,
            "microgrid_winning_center_kind": str( best_candidate[ "kind" ] ),
            "microgrid_winning_point": best_point.tolist(),
            "microgrid_winning_value": float( best_candidate[ "best_value" ] ),
            "microgrid_refined_bounds": refined_bounds.tolist(),
        }
        return refined_bounds, summary

    def _reuse_evaluated_value( self, point: np.ndarray, bounds: np.ndarray ) -> tuple[ float, bool ]:
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        resolved_point = np.asarray( point, dtype=float )
        resolved_bounds = np.asarray( bounds, dtype=float )
        if resolved_point.shape != ( 2, ) or not np.all( np.isfinite( resolved_point ) ):
            return float( "nan" ), False
        if not np.all( resolved_point >= resolved_bounds[ :, 0 ] ) or not np.all( resolved_point <= resolved_bounds[ :, 1 ] ):
            return float( "nan" ), False
        height, width = self.engine.config.grid_shape
        widths = resolved_bounds[ :, 1 ] - resolved_bounds[ :, 0 ]
        if np.any( widths <= 0.0 ):
            return float( "nan" ), False
        col_float = ( resolved_point[ 0 ] - resolved_bounds[ 0, 0 ] ) / widths[ 0 ] * width - 0.5
        row_float = ( resolved_point[ 1 ] - resolved_bounds[ 1, 0 ] ) / widths[ 1 ] * height - 0.5
        col = int( np.clip( np.round( col_float ), 0, width - 1 ) )
        row = int( np.clip( np.round( row_float ), 0, height - 1 ) )
        if not bool( state.evaluated_mask[ row, col ] ):
            return float( "nan" ), False
        pixel_center = self.engine._pixel_center( row, col, resolved_bounds )
        cell_widths = np.asarray( [ widths[ 0 ] / width, widths[ 1 ] / height ], dtype=float )
        tolerance = float( self.agsls_config.late_stage_pattern_search_reuse_tolerance_cells ) * cell_widths
        if np.any( np.abs( resolved_point - pixel_center ) > tolerance ):
            return float( "nan" ), False
        return float( state.objective_values[ row, col ] ), True

    def _pattern_search_seed( self, basin: Basin, current_bounds: np.ndarray ) -> tuple[ np.ndarray, str ] | None:
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        if np.all( np.isfinite( state.best_point ) ) and self._point_in_bounds( state.best_point, current_bounds ):
            return np.asarray( state.best_point, dtype=float ).copy(), "incumbent"
        if basin.basin_best_point is not None:
            seed = np.asarray( basin.basin_best_point, dtype=float )
            if self._point_in_bounds( seed, current_bounds ):
                return seed.copy(), "basin_best"
        if basin.centroid_world is not None:
            seed = np.asarray( basin.centroid_world, dtype=float )
            if self._point_in_bounds( seed, current_bounds ):
                return seed.copy(), "centroid"
        return None

    def _pattern_search_evaluate( self, point: np.ndarray, current_bounds: np.ndarray ) -> tuple[ float, bool ] | None:
        """Return (value, reused). Returns None when no fresh budget and no reuse."""
        cached_value, reused = self._reuse_evaluated_value( point, current_bounds )
        if reused:
            return cached_value, True
        if self._bounded_evaluation_batch( 1 ) < 1:
            return None
        values = self._evaluate_arbitrary_points( np.asarray( point, dtype=float ).reshape( 1, 2 ) )
        if values.size == 0:
            return None
        return float( values[ 0 ] ), False

    def _run_late_stage_pattern_search( self, basin: Basin, current_bounds: np.ndarray ) -> tuple[ np.ndarray, dict[ str, object ] ] | None:
        if self.agsls_config.late_stage_exploiter != "pattern_search":
            return None
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        current_widths = np.asarray( current_bounds, dtype=float )[ :, 1 ] - np.asarray( current_bounds, dtype=float )[ :, 0 ]
        if np.any( current_widths <= 0.0 ):
            return None
        seed_result = self._pattern_search_seed( basin, current_bounds )
        if seed_result is None:
            return None
        seed, seed_kind = seed_result
        seed_evaluation = self._pattern_search_evaluate( seed, current_bounds )
        if seed_evaluation is None:
            return None
        seed_value, seed_reused = seed_evaluation
        starting_value = seed_value
        step = float( self.agsls_config.late_stage_pattern_search_initial_step_fraction ) * current_widths
        min_step = float( self.agsls_config.late_stage_pattern_search_min_step_fraction ) * current_widths
        shrink = float( self.agsls_config.late_stage_pattern_search_shrink )
        max_iterations = int( self.agsls_config.late_stage_pattern_search_max_iterations )
        max_fresh = int( self.agsls_config.late_stage_pattern_search_max_evaluations )
        fresh_spent = 0 if seed_reused else 1
        reused_hits = 1 if seed_reused else 0
        iterations = 0
        improved = False
        terminated_on_budget = False
        while iterations < max_iterations:
            if np.all( step <= min_step ):
                break
            iterations += 1
            trial_offsets = np.asarray(
                [ [ step[ 0 ], 0.0 ], [ -step[ 0 ], 0.0 ], [ 0.0, step[ 1 ] ], [ 0.0, -step[ 1 ] ] ],
                dtype=float,
            )
            best_trial_point = None
            best_trial_value = seed_value
            for offset in trial_offsets:
                trial_point = seed + offset
                if not self._point_in_bounds( trial_point, current_bounds ):
                    continue
                if fresh_spent >= max_fresh:
                    # only attempt reused lookups once the fresh budget is gone
                    cached_value, reused = self._reuse_evaluated_value( trial_point, current_bounds )
                    if not reused:
                        continue
                    value = cached_value
                    reused_hits += 1
                else:
                    outcome = self._pattern_search_evaluate( trial_point, current_bounds )
                    if outcome is None:
                        terminated_on_budget = True
                        break
                    value, was_reused = outcome
                    if was_reused:
                        reused_hits += 1
                    else:
                        fresh_spent += 1
                if self.engine._is_better( value, best_trial_value ):
                    best_trial_value = value
                    best_trial_point = trial_point
            if terminated_on_budget:
                break
            if best_trial_point is not None:
                seed = np.asarray( best_trial_point, dtype=float )
                seed_value = best_trial_value
                improved = True
            else:
                step = step * shrink
            if self._bounded_evaluation_batch( 1 ) < 1 and fresh_spent >= max_fresh:
                break
        if fresh_spent == 0 and reused_hits == 0 and not improved:
            return None
        final_widths = np.maximum( 2.0 * step, self._minimum_zoom_widths( current_bounds ) )
        raw_refined_bounds = self._centered_bounds( seed, final_widths, current_bounds )
        preserve_incumbent = not improved and self._point_in_bounds( state.best_point, current_bounds )
        refined_bounds = self._finalize_zoom_bounds(
            raw_refined_bounds,
            seed,
            current_bounds,
            anchor_point=seed,
            incumbent_point=np.asarray( state.best_point, dtype=float ) if preserve_incumbent else None,
        )
        summary = {
            "pattern_search_ran": True,
            "pattern_search_seed_kind": str( seed_kind ),
            "pattern_search_iterations": int( iterations ),
            "pattern_search_evaluations_spent": int( fresh_spent ),
            "pattern_search_evaluations_reused": int( reused_hits ),
            "pattern_search_final_step": step.tolist(),
            "pattern_search_final_point": np.asarray( seed, dtype=float ).tolist(),
            "pattern_search_final_value": float( seed_value ),
            "pattern_search_refined_bounds": refined_bounds.tolist(),
            "pattern_search_improved": bool( improved and self.engine._is_better( seed_value, starting_value ) ),
        }
        return refined_bounds, summary

    def _run_late_stage_exploiter( self, basin: Basin, current_bounds: np.ndarray ) -> tuple[ str, tuple[ np.ndarray, dict[ str, object ] ] | None ]:
        exploiter = str( self.agsls_config.late_stage_exploiter )
        if exploiter == "pattern_search":
            return "pattern_search", self._run_late_stage_pattern_search( basin, current_bounds )
        if exploiter == "microgrid":
            return "microgrid", self._run_late_stage_microgrid( basin, current_bounds )
        return "none", None

    def _should_intensify(
        self,
        *,
        box_id: int,
        basin: Basin,
        old_bounds: np.ndarray,
        new_bounds: np.ndarray,
        late_stage_state: dict[ str, float | int | bool ],
    ) -> tuple[ bool, float, str ]:
        projected_shrink_ratio = self._projected_shrink_ratio( old_bounds, new_bounds )
        rounds = int( late_stage_state.get( "intensification_rounds", 0 ) )
        if rounds >= int( self.agsls_config.late_stage_max_rounds ):
            return False, float( projected_shrink_ratio ), "max_rounds"
        if self._bounded_evaluation_batch( self.agsls_config.late_stage_eval_batch ) <= 0:
            return False, float( projected_shrink_ratio ), "budget_limit"
        if np.allclose( new_bounds, old_bounds ):
            return True, float( projected_shrink_ratio ), "no_shrink"
        meaningful_better_zoom = bool( basin.better_than_incumbent ) and projected_shrink_ratio < float( self.agsls_config.late_stage_min_shrink_ratio )
        if meaningful_better_zoom:
            return False, float( projected_shrink_ratio ), "meaningful_shrink"
        weak_shrink = projected_shrink_ratio >= float( self.agsls_config.late_stage_min_shrink_ratio )
        if weak_shrink and bool( basin.incumbent_in_envelope ):
            return True, float( projected_shrink_ratio ), "weak_shrink"
        if bool( late_stage_state.get( "late_stage_mode", False ) ):
            return True, float( projected_shrink_ratio ), "late_box"
        return False, float( projected_shrink_ratio ), "direct_zoom"

    def _run_late_stage_intensification( self, basin: Basin ) -> tuple[ int, float ]:
        focus_mask, score_field, coverage = self._late_stage_score_field( basin )
        batch_size = self._bounded_evaluation_batch( self.agsls_config.late_stage_eval_batch )
        if batch_size <= 0:
            return 0, float( coverage )
        gained = int( self.engine.explore_top_pixels( batch_size, mask=focus_mask, score_field=score_field ) )
        if gained == 0 and np.any( basin.mask ):
            gained = int( self.engine.explore_top_pixels( batch_size, mask=basin.mask, score_field=score_field ) )
            coverage = 1.0
        return gained, float( coverage )

    def _mark_latest_snapshot(
        self,
        *,
        box_id: int | None = None,
        decision: str,
        basins: list[ Basin ] | None = None,
        microgrid_summary: dict[ str, object ] | None = None,
        translation_summary: dict[ str, object ] | None = None,
        pattern_search_summary: dict[ str, object ] | None = None,
    ) -> None:
        if not self.engine.snapshots:
            return
        snapshot = self.engine.snapshots[ -1 ]
        snapshot.metadata[ "zoom_decision" ] = decision
        if basins is not None:
            snapshot.metadata[ "dense_groups" ] = self._basin_metadata( basins )
        snapshot.metadata.update(
            self._late_stage_summary_payload(
                box_id=box_id,
                microgrid_summary=microgrid_summary,
                translation_summary=translation_summary,
                pattern_search_summary=pattern_search_summary,
            )
        )

    def _record_decision_reason( self, reason: str, *, accepted: bool = False ) -> None:
        self._decision_reason_counts[ reason ] = int( self._decision_reason_counts.get( reason, 0 ) ) + 1
        if accepted:
            self._accepted_zoom_count += 1

    def _probe_score_field( self ) -> np.ndarray:
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        support = self.engine.support_field()
        uncertainty = 1.0 - np.abs( state.objective_field - 0.5 ) * 2.0
        return self.engine.exploration_score_field() + 0.35 * support + 0.15 * np.clip( uncertainty, 0.0, 1.0 )

    def _probe_basins( self, basins: list[ Basin ] ) -> None:
        if self.agsls_config.candidate_probe_evaluations <= 0:
            return
        for basin in basins:
            if self._max_evaluations_reached():
                break
            batch_size = self._bounded_evaluation_batch( self.agsls_config.candidate_probe_evaluations )
            if batch_size <= 0:
                break
            self.engine.explore_top_pixels(
                batch_size,
                mask=basin.mask,
                score_field=self._probe_score_field(),
            )

    def _eligible_basins( self, basins: list[ Basin ] ) -> list[ Basin ]:
        eligible: list[ Basin ] = [ ]
        for basin in basins:
            if basin.area < self.agsls_config.min_basin_cells:
                continue
            if basin.alive_density < self.agsls_config.min_alive_density:
                continue
            eligible.append( basin )
        return eligible

    def _should_choose_leader( self, basins: list[ Basin ], explored_in_stage: int ) -> tuple[ bool, str ]:
        if not basins:
            return False, "no_group"
        if len( basins ) == 1:
            return True, "single_group"
        score_gap = float( basins[ 0 ].combined_score - basins[ 1 ].combined_score )
        if score_gap >= self.agsls_config.dominance_margin:
            return True, "dominant_group"
        if score_gap <= self.agsls_config.similarity_margin:
            if explored_in_stage <= 0 and self.agsls_config.candidate_probe_evaluations > 0:
                return False, "similar_groups_probe"
            return True, "similar_groups"
        if explored_in_stage >= self.agsls_config.undecided_stage_max_evaluations:
            return True, "exploration_cap"
        union_mask = np.zeros_like( basins[ 0 ].mask, dtype=bool )
        for basin in basins:
            union_mask |= basin.mask
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        if not np.any( union_mask & ~state.evaluated_mask ):
            return True, "no_unexplored_cells"
        return False, "continue_exploring"

    def _select_basin( self, basins: list[ Basin ] ) -> Basin:
        if len( basins ) <= 1:
            return basins[ 0 ]
        top_score = float( basins[ 0 ].combined_score )
        contenders = [
            basin
            for basin in basins
            if top_score - float( basin.combined_score ) <= self.agsls_config.similarity_margin
        ]
        if len( contenders ) <= 1:
            return basins[ 0 ]
        return max(
            contenders,
            key=lambda basin: (
                float( basin.best_objective_score ) if basin.evaluated_count > 0 else -np.inf,
                float( basin.stability_score ),
                -int( basin.area ),
                float( basin.combined_score ),
            ),
        )

    def _expand_bounds_to_min_widths(
        self,
        bounds: np.ndarray,
        center: np.ndarray,
        min_widths: np.ndarray,
        current_bounds: np.ndarray,
    ) -> np.ndarray:
        expanded = np.asarray( bounds, dtype=float ).copy()
        center = np.asarray( center, dtype=float )
        for idx in range( 2 ):
            width = float( expanded[ idx, 1 ] - expanded[ idx, 0 ] )
            if width >= float( min_widths[ idx ] ):
                continue
            half_width = 0.5 * float( min_widths[ idx ] )
            resolved_center = float( np.clip( center[ idx ], current_bounds[ idx, 0 ], current_bounds[ idx, 1 ] ) )
            lower = resolved_center - half_width
            upper = resolved_center + half_width
            if lower < current_bounds[ idx, 0 ]:
                upper += current_bounds[ idx, 0 ] - lower
                lower = current_bounds[ idx, 0 ]
            if upper > current_bounds[ idx, 1 ]:
                lower -= upper - current_bounds[ idx, 1 ]
                upper = current_bounds[ idx, 1 ]
            expanded[ idx, 0 ] = max( current_bounds[ idx, 0 ], lower )
            expanded[ idx, 1 ] = min( current_bounds[ idx, 1 ], upper )
        return expanded

    def _padded_bounds( self, basin: Basin ) -> np.ndarray:
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        current_bounds = state.bounds
        new_bounds = basin.bbox_world.copy()
        center = basin.basin_best_point if basin.basin_best_point is not None else basin.centroid_world
        incumbent_point = None
        if not basin.incumbent_in_envelope and not basin.better_than_incumbent and np.all( np.isfinite( state.best_point ) ):
            incumbent_point = np.asarray( state.best_point, dtype=float )
        anchor_point = basin.basin_best_point if basin.basin_best_point is not None else center
        return self._finalize_zoom_bounds(
            new_bounds,
            np.asarray( center, dtype=float ),
            current_bounds,
            anchor_point=np.asarray( anchor_point, dtype=float ),
            incumbent_point=incumbent_point,
        )

    def _record_zoom(
        self,
        selected: Basin,
        old_bounds: np.ndarray,
        new_bounds: np.ndarray,
        steps: int,
        reason: str,
        *,
        evaluations_before: int,
        best_value_before: float,
        incumbent_point_before_zoom: np.ndarray,
        late_stage_mode: bool = False,
        intensification_rounds: int = 0,
        projected_shrink_ratio: float = 1.0,
        focus_mask_coverage: float = 0.0,
        late_stage_exit_reason: str = "none",
        selection_reason: str | None = None,
        microgrid_summary: dict[ str, object ] | None = None,
        translation_summary: dict[ str, object ] | None = None,
        pattern_search_summary: dict[ str, object ] | None = None,
    ) -> None:
        box_id = int( len( self.zoom_events ) )
        diagnostics = self._basin_diagnostics( selected )
        diagnostics.update(
            {
                "accepted": True,
                "decision_reason": str( reason ),
                "selection_reason": str( selection_reason or reason ),
                "box_id": box_id,
                "evaluations_before": int( evaluations_before ),
                "evaluations_after": int( self.engine.state.evaluations if self.engine.state is not None else evaluations_before ),
                "best_value_before": float( best_value_before ),
                "best_value_after": float( self.engine.state.best_value if self.engine.state is not None else best_value_before ),
                "incumbent_point_before_zoom": np.asarray( incumbent_point_before_zoom, dtype=float ).tolist(),
                "late_stage_mode": bool( late_stage_mode ),
                "intensification_rounds": int( intensification_rounds ),
                "projected_shrink_ratio": float( projected_shrink_ratio ),
                "focus_mask_coverage": float( focus_mask_coverage ),
                "late_stage_exit_reason": str( late_stage_exit_reason ),
            }
        )
        diagnostics.update(
            self._late_stage_summary_payload(
                box_id=box_id,
                microgrid_summary=microgrid_summary,
                translation_summary=translation_summary,
                pattern_search_summary=pattern_search_summary,
            )
        )
        self.zoom_events.append(
            ZoomEvent(
                zoom_index=len( self.zoom_events ),
                old_bounds=old_bounds,
                new_bounds=new_bounds.copy(),
                selected_basin_score=float( selected.combined_score ),
                selected_basin_bbox=selected.bbox_world.copy(),
                evaluation_count=int( self.engine.state.evaluations if self.engine.state is not None else 0 ),
                steps_per_zoom=steps,
                diagnostics=diagnostics,
            )
        )
        if self.engine.snapshots:
            snapshot = self.engine.snapshots[ -1 ]
            snapshot.selected_basin_bbox = selected.bbox_world.copy()
            snapshot.metadata[ "zoom_decision" ] = "accepted"
            snapshot.metadata[ "zoom_reason" ] = reason
            snapshot.metadata[ "selected_basin_diagnostics" ] = diagnostics
            snapshot.metadata[ "late_stage_mode" ] = bool( late_stage_mode )
            snapshot.metadata[ "intensification_rounds" ] = int( intensification_rounds )
            snapshot.metadata[ "projected_shrink_ratio" ] = float( projected_shrink_ratio )
            snapshot.metadata[ "focus_mask_coverage" ] = float( focus_mask_coverage )
            snapshot.metadata[ "late_stage_exit_reason" ] = str( late_stage_exit_reason )
            snapshot.metadata.update(
                self._late_stage_summary_payload(
                    box_id=box_id,
                    microgrid_summary=microgrid_summary,
                    translation_summary=translation_summary,
                    pattern_search_summary=pattern_search_summary,
                )
            )

    def step( self ) -> bool:
        """Run one AGSLS decision round. Returns True when a zoom is accepted."""

        zoom_index = len( self.zoom_events )
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        evaluations_before = int( state.evaluations )
        best_value_before = float( state.best_value )
        bounds_before = state.bounds.copy()
        incumbent_point_before_zoom = state.best_point.copy()
        active_limit = self._active_zoom_limit if self._active_zoom_limit is not None else self.agsls_config.max_zoom_cycles
        self.engine.set_zoom_index( zoom_index, max_zoom_cycles=active_limit )
        self.apply_runtime_policy( AGSLS_PER_DECISION_FIELDS )
        self.apply_runtime_policy( ZOOM_BOUNDARY_FIELDS )
        self.engine.apply_runtime_policy( ZOOM_BOUNDARY_FIELDS, rebuild_kernels=True )
        self.apply_runtime_policy( SMOOTHLIFE_PER_STEP_FIELDS )
        self.engine.apply_runtime_policy( SMOOTHLIFE_PER_STEP_FIELDS, rebuild_kernels=False )
        late_stage_state = self._late_stage_state( box_id=zoom_index, active_limit=active_limit, bounds=bounds_before )
        microgrid_summary = self._empty_microgrid_summary()
        translation_summary = self._empty_translation_summary()
        pattern_search_summary = self._empty_pattern_search_summary()
        scheduled_step_batch = int( self.engine.config.evaluations_per_step )
        self.engine.config.evaluations_per_step = self._late_stage_step_batch( scheduled_step_batch, late_stage_state=late_stage_state )
        steps = steps_for_zoom_cycle( self.agsls_config, zoom_index )
        persistence = self._persistence_map( steps, box_id=zoom_index, late_stage_state=late_stage_state )
        ranked = self._rank_basins( persistence )
        self._update_basin_runtime_state( ranked )
        self._probe_basins( ranked )
        ranked = self._rank_basins( persistence )
        self._update_basin_runtime_state( ranked )
        eligible = self._eligible_basins( ranked )
        if not eligible:
            self.engine.config.evaluations_per_step = scheduled_step_batch
            self._mark_latest_snapshot(
                box_id=zoom_index,
                decision="deferred_no_eligible_group",
                basins=ranked,
                microgrid_summary=microgrid_summary,
                translation_summary=translation_summary,
                pattern_search_summary=pattern_search_summary,
            )
            self._record_decision_reason( "deferred_no_eligible_group" )
            state = self.engine.state
            if state is None:
                raise RuntimeError( "engine state missing" )
            self._record_decision_trace(
                box_id=zoom_index,
                accepted=False,
                decision_reason="deferred_no_eligible_group",
                evaluations_before=evaluations_before,
                evaluations_after=int( state.evaluations ),
                best_value_before=best_value_before,
                best_value_after=float( state.best_value ),
                bounds_before=bounds_before,
                bounds_after=state.bounds.copy(),
                selected_basin=ranked[ 0 ] if ranked else None,
                incumbent_point_before_zoom=incumbent_point_before_zoom,
                late_stage_mode=bool( late_stage_state.get( "late_stage_mode", False ) ),
                intensification_rounds=int( late_stage_state.get( "intensification_rounds", 0 ) ),
                projected_shrink_ratio=1.0,
                focus_mask_coverage=0.0,
                late_stage_exit_reason="no_eligible_group",
                microgrid_summary=microgrid_summary,
                translation_summary=translation_summary,
                pattern_search_summary=pattern_search_summary,
            )
            return False

        explored_in_stage = 0
        decision_ready, decision_reason = self._should_choose_leader( eligible, explored_in_stage )
        while not decision_ready and not self._max_evaluations_reached():
            top_groups = eligible[ : min( 3, len( eligible ) ) ]
            union_mask = np.zeros_like( top_groups[ 0 ].mask, dtype=bool )
            for basin in top_groups:
                union_mask |= basin.mask
            remaining = self.agsls_config.undecided_stage_max_evaluations - explored_in_stage
            if remaining <= 0:
                decision_reason = "exploration_cap"
                break
            batch_size = self._bounded_evaluation_batch( min( self.agsls_config.candidate_probe_evaluations, remaining ) )
            if batch_size <= 0:
                decision_ready = True
                decision_reason = "evaluation_limit"
                break
            gained = self.engine.explore_top_pixels(
                batch_size,
                mask=union_mask,
                score_field=self._probe_score_field(),
            )
            explored_in_stage += int( gained )
            ranked = self._rank_basins( persistence )
            eligible = self._eligible_basins( ranked )
            if not eligible:
                self.engine.config.evaluations_per_step = scheduled_step_batch
                self._mark_latest_snapshot(
                    box_id=zoom_index,
                    decision="deferred_groups_lost",
                    basins=ranked,
                    microgrid_summary=microgrid_summary,
                    translation_summary=translation_summary,
                    pattern_search_summary=pattern_search_summary,
                )
                self._record_decision_reason( "deferred_groups_lost" )
                state = self.engine.state
                if state is None:
                    raise RuntimeError( "engine state missing" )
                self._record_decision_trace(
                    box_id=zoom_index,
                    accepted=False,
                    decision_reason="deferred_groups_lost",
                    evaluations_before=evaluations_before,
                    evaluations_after=int( state.evaluations ),
                    best_value_before=best_value_before,
                    best_value_after=float( state.best_value ),
                    bounds_before=bounds_before,
                    bounds_after=state.bounds.copy(),
                    selected_basin=ranked[ 0 ] if ranked else None,
                    incumbent_point_before_zoom=incumbent_point_before_zoom,
                    late_stage_mode=bool( late_stage_state.get( "late_stage_mode", False ) ),
                    intensification_rounds=int( self._late_stage_round_counts.get( zoom_index, 0 ) ),
                    projected_shrink_ratio=1.0,
                    focus_mask_coverage=0.0,
                    late_stage_exit_reason="groups_lost",
                    microgrid_summary=microgrid_summary,
                    translation_summary=translation_summary,
                    pattern_search_summary=pattern_search_summary,
                )
                return False
            decision_ready, decision_reason = self._should_choose_leader( eligible, explored_in_stage )
            if gained == 0:
                decision_ready = True
                decision_reason = "no_unexplored_cells"

        selected = self._select_basin( eligible )
        new_bounds = self._padded_bounds( selected )
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        old_bounds = state.bounds.copy()
        should_intensify, projected_shrink_ratio, late_stage_exit_reason = self._should_intensify(
            box_id=zoom_index,
            basin=selected,
            old_bounds=old_bounds,
            new_bounds=new_bounds,
            late_stage_state=late_stage_state,
        )
        translation_accepted = False
        if should_intensify:
            translation_result = self._run_late_stage_translation(
                selected,
                old_bounds,
                new_bounds,
                late_stage_exit_reason,
            )
            if translation_result is not None:
                new_bounds, translation_summary = translation_result
                translation_accepted = True
                should_intensify = False
                projected_shrink_ratio = self._projected_shrink_ratio( old_bounds, new_bounds )
                late_stage_exit_reason = "translated_zoom"
        exploiter_kind = "none"
        exploiter_result: tuple[ np.ndarray, dict[ str, object ] ] | None = None
        if not translation_accepted and bool( late_stage_state.get( "late_stage_mode", False ) ):
            exploiter_kind, exploiter_result = self._run_late_stage_exploiter( selected, old_bounds )
        if exploiter_result is not None:
            exploiter_bounds, exploiter_summary = exploiter_result
            if exploiter_kind == "microgrid":
                microgrid_summary = exploiter_summary
                late_stage_exit_reason = "microgrid_refine"
            elif exploiter_kind == "pattern_search":
                pattern_search_summary = exploiter_summary
                late_stage_exit_reason = "pattern_search_refine"
            new_bounds = exploiter_bounds
            should_intensify = False
            projected_shrink_ratio = self._projected_shrink_ratio( old_bounds, new_bounds )
        late_stage_mode = (
            bool( late_stage_state.get( "late_stage_mode", False ) )
            or should_intensify
            or translation_accepted
            or np.allclose( new_bounds, old_bounds )
        )
        if should_intensify:
            gained, focus_mask_coverage = self._run_late_stage_intensification( selected )
            intensification_rounds = int( self._late_stage_round_counts.get( zoom_index, 0 ) ) + 1
            self._late_stage_round_counts[ zoom_index ] = intensification_rounds
            self.engine.config.evaluations_per_step = scheduled_step_batch
            self._mark_latest_snapshot(
                box_id=zoom_index,
                decision="late_stage_intensify",
                basins=eligible,
                microgrid_summary=microgrid_summary,
                translation_summary=translation_summary,
                pattern_search_summary=pattern_search_summary,
            )
            if self.engine.snapshots:
                snapshot = self.engine.snapshots[ -1 ]
                snapshot.metadata[ "zoom_reason" ] = "late_stage_intensify"
                snapshot.metadata[ "late_stage_mode" ] = True
                snapshot.metadata[ "intensification_rounds" ] = int( intensification_rounds )
                snapshot.metadata[ "projected_shrink_ratio" ] = float( projected_shrink_ratio )
                snapshot.metadata[ "focus_mask_coverage" ] = float( focus_mask_coverage )
                snapshot.metadata[ "late_stage_exit_reason" ] = str( late_stage_exit_reason )
                snapshot.metadata.update(
                    self._late_stage_summary_payload(
                        box_id=zoom_index,
                        microgrid_summary=microgrid_summary,
                        translation_summary=translation_summary,
                        pattern_search_summary=pattern_search_summary,
                    )
                )
            self._record_decision_reason( "late_stage_intensify" )
            state = self.engine.state
            if state is None:
                raise RuntimeError( "engine state missing" )
            self._record_decision_trace(
                box_id=zoom_index,
                accepted=False,
                decision_reason="late_stage_intensify",
                evaluations_before=evaluations_before,
                evaluations_after=int( state.evaluations ),
                best_value_before=best_value_before,
                best_value_after=float( state.best_value ),
                bounds_before=bounds_before,
                bounds_after=state.bounds.copy(),
                selected_basin=selected,
                incumbent_point_before_zoom=incumbent_point_before_zoom,
                late_stage_mode=True,
                intensification_rounds=intensification_rounds,
                projected_shrink_ratio=projected_shrink_ratio,
                focus_mask_coverage=focus_mask_coverage,
                late_stage_exit_reason=late_stage_exit_reason,
                microgrid_summary=microgrid_summary,
                translation_summary=translation_summary,
                pattern_search_summary=pattern_search_summary,
            )
            return False
        if np.allclose( new_bounds, old_bounds ):
            self.engine.config.evaluations_per_step = scheduled_step_batch
            self._mark_latest_snapshot(
                box_id=zoom_index,
                decision="deferred_no_shrink",
                basins=eligible,
                microgrid_summary=microgrid_summary,
                translation_summary=translation_summary,
                pattern_search_summary=pattern_search_summary,
            )
            self._record_decision_reason( "deferred_no_shrink" )
            self._record_decision_trace(
                box_id=zoom_index,
                accepted=False,
                decision_reason="deferred_no_shrink",
                evaluations_before=evaluations_before,
                evaluations_after=int( state.evaluations ),
                best_value_before=best_value_before,
                best_value_after=float( state.best_value ),
                bounds_before=bounds_before,
                bounds_after=state.bounds.copy(),
                selected_basin=selected,
                incumbent_point_before_zoom=incumbent_point_before_zoom,
                late_stage_mode=late_stage_mode,
                intensification_rounds=int( self._late_stage_round_counts.get( zoom_index, 0 ) ),
                projected_shrink_ratio=projected_shrink_ratio,
                focus_mask_coverage=0.0,
                late_stage_exit_reason="no_shrink",
                microgrid_summary=microgrid_summary,
                translation_summary=translation_summary,
                pattern_search_summary=pattern_search_summary,
            )
            return False
        remaining = self._remaining_evaluations()
        if remaining is not None and remaining < self.engine.config.evaluations_per_step:
            self.engine.config.evaluations_per_step = scheduled_step_batch
            self._mark_latest_snapshot(
                box_id=zoom_index,
                decision="deferred_evaluation_limit",
                basins=eligible,
                microgrid_summary=microgrid_summary,
                translation_summary=translation_summary,
                pattern_search_summary=pattern_search_summary,
            )
            self._record_decision_reason( "deferred_evaluation_limit" )
            self._record_decision_trace(
                box_id=zoom_index,
                accepted=False,
                decision_reason="deferred_evaluation_limit",
                evaluations_before=evaluations_before,
                evaluations_after=int( state.evaluations ),
                best_value_before=best_value_before,
                best_value_after=float( state.best_value ),
                bounds_before=bounds_before,
                bounds_after=state.bounds.copy(),
                selected_basin=selected,
                incumbent_point_before_zoom=incumbent_point_before_zoom,
                late_stage_mode=late_stage_mode,
                intensification_rounds=int( self._late_stage_round_counts.get( zoom_index, 0 ) ),
                projected_shrink_ratio=projected_shrink_ratio,
                focus_mask_coverage=0.0,
                late_stage_exit_reason="evaluation_limit",
                microgrid_summary=microgrid_summary,
                translation_summary=translation_summary,
                pattern_search_summary=pattern_search_summary,
            )
            return False
        self.engine.config.evaluations_per_step = scheduled_step_batch
        accepted_reason = decision_reason
        intensification_rounds = int( self._late_stage_round_counts.get( zoom_index, 0 ) )
        accepted_exit_reason = late_stage_exit_reason
        if translation_accepted:
            accepted_reason = "late_stage_translate_zoom"
        elif intensification_rounds > 0:
            accepted_reason = "late_stage_resume_zoom"
            accepted_exit_reason = "resume_zoom"
        self.engine.remap_to_bounds( new_bounds )
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        self._record_zoom(
            selected,
            old_bounds,
            new_bounds,
            steps,
            accepted_reason,
            evaluations_before=evaluations_before,
            best_value_before=best_value_before,
            incumbent_point_before_zoom=incumbent_point_before_zoom,
            late_stage_mode=late_stage_mode,
            intensification_rounds=intensification_rounds,
            projected_shrink_ratio=projected_shrink_ratio,
            focus_mask_coverage=0.0,
            late_stage_exit_reason=accepted_exit_reason,
            selection_reason=decision_reason,
            microgrid_summary=microgrid_summary,
            translation_summary=translation_summary,
            pattern_search_summary=pattern_search_summary,
        )
        self._record_decision_reason( accepted_reason, accepted=True )
        self._record_decision_trace(
            box_id=zoom_index,
            accepted=True,
            decision_reason=accepted_reason,
            evaluations_before=evaluations_before,
            evaluations_after=int( state.evaluations ),
            best_value_before=best_value_before,
            best_value_after=float( state.best_value ),
            bounds_before=bounds_before,
            bounds_after=state.bounds.copy(),
            selected_basin=selected,
            incumbent_point_before_zoom=incumbent_point_before_zoom,
            late_stage_mode=late_stage_mode,
            intensification_rounds=intensification_rounds,
            projected_shrink_ratio=projected_shrink_ratio,
            focus_mask_coverage=0.0,
            late_stage_exit_reason=accepted_exit_reason,
            microgrid_summary=microgrid_summary,
            translation_summary=translation_summary,
            pattern_search_summary=pattern_search_summary,
        )
        return True

    def run( self, zoom_cycles: int | None = None, evaluations: int | None = None ) -> SearchRun:
        """Run AGSLS until the zoom or evaluation limit is reached."""

        configured_limit = self.agsls_config.max_zoom_cycles if zoom_cycles is None else int( zoom_cycles )
        eval_limit = self.agsls_config.max_evaluations if evaluations is None else int( evaluations )
        limit = self._effective_zoom_limit( eval_limit, configured_limit )
        self.active_max_evaluations = eval_limit
        self.engine.active_max_evaluations = eval_limit
        self._active_zoom_limit = limit
        self.engine.set_zoom_index( len( self.zoom_events ), max_zoom_cycles=limit )
        decision_rounds = 0
        max_rounds = max( limit * 4, limit )
        try:
            while len( self.zoom_events ) < limit:
                state = self.engine.state
                if state is None:
                    raise RuntimeError( "engine state missing" )
                if eval_limit is not None and state.evaluations >= eval_limit:
                    break
                remaining = self._remaining_evaluations()
                if remaining is not None and remaining < self.engine.config.evaluations_per_step:
                    break
                widths = state.bounds[ :, 1 ] - state.bounds[ :, 0 ]
                min_widths = self.agsls_config.min_side_fraction * ( self.original_bounds[ :, 1 ] - self.original_bounds[ :, 0 ] )
                if np.all( widths <= min_widths ):
                    break
                previous_evaluations = int( state.evaluations )
                previous_zoom_count = len( self.zoom_events )
                accepted = self.step()
                decision_rounds += 1
                state = self.engine.state
                if state is None:
                    raise RuntimeError( "engine state missing" )
                made_progress = int( state.evaluations ) > previous_evaluations or len( self.zoom_events ) > previous_zoom_count
                if not accepted and not made_progress:
                    break
                if eval_limit is None and decision_rounds >= max_rounds:
                    break
        finally:
            self.active_max_evaluations = None
            self.engine.active_max_evaluations = None
            self._active_zoom_limit = None
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        return SearchRun(
            best_point=state.best_point.copy(),
            best_value=float( state.best_value ),
            evaluations=int( state.evaluations ),
            bounds=state.bounds.copy(),
            snapshots=list( self.engine.snapshots ),
            zoom_events=list( self.zoom_events ),
            metadata={
                "mode": "agsls",
                "zoom_cycles": len( self.zoom_events ),
                "decision_rounds": len( self._decision_trace ),
                "schedule_summary": None if self.runtime_policy is None else self.runtime_policy.summary(),
                "decision_reason_counts": dict( sorted( self._decision_reason_counts.items() ) ),
                "zoom_acceptance_count": int( self._accepted_zoom_count ),
                "decision_trace": list( self._decision_trace ),
            },
        )

    def zoom_history( self ) -> list[ ZoomEvent ]:
        """Return recorded zoom events."""

        return list( self.zoom_events )

    def snapshot( self ):
        """Return the current SmoothLife snapshot."""

        return self.engine.snapshot()
