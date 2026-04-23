"""Adaptive Grid Smooth Life Search controller."""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

from ..adaptive import AGSLS_PER_DECISION_FIELDS, RuntimeSignals, SchedulePolicy, SMOOTHLIFE_PER_STEP_FIELDS, ZOOM_BOUNDARY_FIELDS
from ..results import Basin, SearchRun, ZoomEvent
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

    @staticmethod
    def _normalize_bounds( bounds: np.ndarray | list[ tuple[ float, float ] ] ) -> np.ndarray:
        arr = np.asarray( bounds, dtype=float )
        if arr.shape != ( 2, 2 ):
            raise ValueError( "AGSLS currently supports exactly 2D bounds" )
        if np.any( arr[ :, 1 ] <= arr[ :, 0 ] ):
            raise ValueError( "each bound must satisfy lower < upper" )
        return arr

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
        widths = np.asarray( bounds, dtype=float )[ :, 1 ] - np.asarray( bounds, dtype=float )[ :, 0 ]
        return float( max( widths[ 0 ], 1e-12 ) * max( widths[ 1 ], 1e-12 ) )

    def _box_area_ratio( self, bounds: np.ndarray ) -> float:
        return self._bounds_area( bounds ) / max( self._bounds_area( self.original_bounds ), 1e-12 )

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

    def _persistence_map( self, steps: int ) -> np.ndarray:
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
    ) -> None:
        box_round_index = int( self._box_round_counts.get( box_id, 0 ) ) + 1
        self._box_round_counts[ box_id ] = box_round_index
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
            }
        )

    def _late_stage_state( self, *, box_id: int, active_limit: int, bounds: np.ndarray ) -> dict[ str, float | int | bool ]:
        signals = self.runtime_signals()
        zoom_fraction = signals.zoom_fraction()
        prior_box_rounds = int( self._box_round_counts.get( box_id, 0 ) )
        plateau = prior_box_rounds > 0 and max( float( signals.global_improvement ), float( signals.stage_improvement ) ) <= self.agsls_config.late_stage_plateau_threshold
        small_box = self._box_area_ratio( bounds ) <= 0.25
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

    def _mark_latest_snapshot( self, *, decision: str, basins: list[ Basin ] | None = None ) -> None:
        if not self.engine.snapshots:
            return
        snapshot = self.engine.snapshots[ -1 ]
        snapshot.metadata[ "zoom_decision" ] = decision
        if basins is not None:
            snapshot.metadata[ "dense_groups" ] = self._basin_metadata( basins )

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
        height, width = self.engine.config.grid_shape
        current_widths = current_bounds[ :, 1 ] - current_bounds[ :, 0 ]
        cell_widths = np.asarray( [ current_widths[ 0 ] / width, current_widths[ 1 ] / height ], dtype=float )
        min_zoom_widths = self.agsls_config.min_zoom_cells * cell_widths
        center = basin.basin_best_point if basin.basin_best_point is not None else basin.centroid_world
        new_bounds = self._expand_bounds_to_min_widths( new_bounds, center, min_zoom_widths, current_bounds )
        widths = new_bounds[ :, 1 ] - new_bounds[ :, 0 ]
        if basin.basin_best_point is not None:
            best_point = np.asarray( basin.basin_best_point, dtype=float )
            directional_padding = self.agsls_config.zoom_padding * widths
            for idx in range( 2 ):
                edge_band = self.agsls_config.edge_risk_fraction * max( widths[ idx ], 1e-12 )
                if best_point[ idx ] - new_bounds[ idx, 0 ] <= edge_band:
                    new_bounds[ idx, 0 ] -= directional_padding[ idx ]
                if new_bounds[ idx, 1 ] - best_point[ idx ] <= edge_band:
                    new_bounds[ idx, 1 ] += directional_padding[ idx ]
        if not basin.incumbent_in_envelope and not basin.better_than_incumbent and np.all( np.isfinite( state.best_point ) ):
            incumbent_point = np.asarray( state.best_point, dtype=float )
            for idx in range( 2 ):
                if incumbent_point[ idx ] < current_bounds[ idx, 0 ] or incumbent_point[ idx ] > current_bounds[ idx, 1 ]:
                    continue
                width = max( float( new_bounds[ idx, 1 ] - new_bounds[ idx, 0 ] ), float( min_zoom_widths[ idx ] ) )
                edge_band = self.agsls_config.edge_risk_fraction * width
                if incumbent_point[ idx ] < new_bounds[ idx, 0 ]:
                    new_bounds[ idx, 0 ] = max( current_bounds[ idx, 0 ], incumbent_point[ idx ] - edge_band )
                elif incumbent_point[ idx ] > new_bounds[ idx, 1 ]:
                    new_bounds[ idx, 1 ] = min( current_bounds[ idx, 1 ], incumbent_point[ idx ] + edge_band )
        widths = new_bounds[ :, 1 ] - new_bounds[ :, 0 ]
        padding = self.agsls_config.zoom_padding * widths
        new_bounds[ :, 0 ] -= padding
        new_bounds[ :, 1 ] += padding
        new_bounds[ :, 0 ] = np.maximum( new_bounds[ :, 0 ], current_bounds[ :, 0 ] )
        new_bounds[ :, 1 ] = np.minimum( new_bounds[ :, 1 ], current_bounds[ :, 1 ] )
        new_bounds = self._expand_bounds_to_min_widths( new_bounds, center, min_zoom_widths, current_bounds )
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
    ) -> None:
        diagnostics = self._basin_diagnostics( selected )
        diagnostics.update(
            {
                "accepted": True,
                "decision_reason": str( reason ),
                "selection_reason": str( selection_reason or reason ),
                "box_id": int( len( self.zoom_events ) ),
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
        scheduled_step_batch = int( self.engine.config.evaluations_per_step )
        self.engine.config.evaluations_per_step = self._late_stage_step_batch( scheduled_step_batch, late_stage_state=late_stage_state )
        steps = steps_for_zoom_cycle( self.agsls_config, zoom_index )
        persistence = self._persistence_map( steps )
        ranked = self._rank_basins( persistence )
        self._update_basin_runtime_state( ranked )
        self._probe_basins( ranked )
        ranked = self._rank_basins( persistence )
        self._update_basin_runtime_state( ranked )
        eligible = self._eligible_basins( ranked )
        if not eligible:
            self.engine.config.evaluations_per_step = scheduled_step_batch
            self._mark_latest_snapshot( decision="deferred_no_eligible_group", basins=ranked )
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
                self._mark_latest_snapshot( decision="deferred_groups_lost", basins=ranked )
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
        late_stage_mode = bool( late_stage_state.get( "late_stage_mode", False ) ) or should_intensify or np.allclose( new_bounds, old_bounds )
        if should_intensify:
            gained, focus_mask_coverage = self._run_late_stage_intensification( selected )
            intensification_rounds = int( self._late_stage_round_counts.get( zoom_index, 0 ) ) + 1
            self._late_stage_round_counts[ zoom_index ] = intensification_rounds
            self.engine.config.evaluations_per_step = scheduled_step_batch
            self._mark_latest_snapshot( decision="late_stage_intensify", basins=eligible )
            if self.engine.snapshots:
                snapshot = self.engine.snapshots[ -1 ]
                snapshot.metadata[ "zoom_reason" ] = "late_stage_intensify"
                snapshot.metadata[ "late_stage_mode" ] = True
                snapshot.metadata[ "intensification_rounds" ] = int( intensification_rounds )
                snapshot.metadata[ "projected_shrink_ratio" ] = float( projected_shrink_ratio )
                snapshot.metadata[ "focus_mask_coverage" ] = float( focus_mask_coverage )
                snapshot.metadata[ "late_stage_exit_reason" ] = str( late_stage_exit_reason )
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
            )
            return False
        if np.allclose( new_bounds, old_bounds ):
            self.engine.config.evaluations_per_step = scheduled_step_batch
            self._mark_latest_snapshot( decision="deferred_no_shrink", basins=eligible )
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
            )
            return False
        remaining = self._remaining_evaluations()
        if remaining is not None and remaining < self.engine.config.evaluations_per_step:
            self.engine.config.evaluations_per_step = scheduled_step_batch
            self._mark_latest_snapshot( decision="deferred_evaluation_limit", basins=eligible )
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
            )
            return False
        self.engine.config.evaluations_per_step = scheduled_step_batch
        accepted_reason = decision_reason
        intensification_rounds = int( self._late_stage_round_counts.get( zoom_index, 0 ) )
        if intensification_rounds > 0:
            accepted_reason = "late_stage_resume_zoom"
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
            late_stage_exit_reason="resume_zoom" if intensification_rounds > 0 else late_stage_exit_reason,
            selection_reason=decision_reason,
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
            late_stage_exit_reason="resume_zoom" if intensification_rounds > 0 else late_stage_exit_reason,
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
