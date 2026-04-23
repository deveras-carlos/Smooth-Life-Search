"""Adaptive Grid Smooth Life Search controller."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..adaptive import AGSLS_PER_DECISION_FIELDS, RuntimeSignals, SchedulePolicy, ZOOM_BOUNDARY_FIELDS
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
        self._last_basin_count = 0
        self._last_top_basin_score_gap = 0.0
        self._decision_reason_counts: dict[str, int] = { }
        self._accepted_zoom_count = 0

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
        self._last_basin_count = 0
        self._last_top_basin_score_gap = 0.0
        self._decision_reason_counts = { }
        self._accepted_zoom_count = 0

    def _max_evaluations_reached( self ) -> bool:
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        limit = self.active_max_evaluations if self.active_max_evaluations is not None else self.agsls_config.max_evaluations
        return limit is not None and state.evaluations >= limit

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
        )
        for basin in basins:
            basin.stability_score = float( np.mean( persistence[ basin.mask ] ) ) if np.any( basin.mask ) else 0.0
        return score_basins( basins, self.agsls_config )

    def _basin_metadata( self, basins: list[ Basin ] ) -> list[ dict[str, float | list[ list[ float ] ]] ]:
        payload: list[ dict[str, float | list[ list[ float ] ]] ] = [ ]
        for rank, basin in enumerate( basins, start=1 ):
            payload.append(
                {
                    "rank": float( rank ),
                    "score": float( basin.combined_score ),
                    "alive_density": float( basin.alive_density ),
                    "support_mass": float( basin.support_mass ),
                    "bbox": basin.bbox_world.tolist(),
                }
            )
        return payload

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
            self.engine.explore_top_pixels(
                self.agsls_config.candidate_probe_evaluations,
                mask=basin.mask,
                score_field=self._probe_score_field(),
            )

    def _eligible_basins( self, basins: list[ Basin ] ) -> list[ Basin ]:
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        eligible: list[ Basin ] = [ ]
        for basin in basins:
            if basin.area < self.agsls_config.min_basin_cells:
                continue
            if basin.alive_density < self.agsls_config.min_alive_density:
                continue
            if basin.basin_best_value is None or basin.basin_best_point is None:
                continue
            if not self.engine._is_better_or_equal( float( basin.basin_best_value ), state.local_best_value ):
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

    def _padded_bounds( self, basin: Basin ) -> np.ndarray:
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        current_bounds = state.bounds
        new_bounds = basin.bbox_world.copy()
        widths = new_bounds[ :, 1 ] - new_bounds[ :, 0 ]
        padding = self.agsls_config.zoom_padding * widths
        new_bounds[ :, 0 ] -= padding
        new_bounds[ :, 1 ] += padding
        new_bounds[ :, 0 ] = np.maximum( new_bounds[ :, 0 ], current_bounds[ :, 0 ] )
        new_bounds[ :, 1 ] = np.minimum( new_bounds[ :, 1 ], current_bounds[ :, 1 ] )
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

    def _record_zoom( self, selected: Basin, old_bounds: np.ndarray, new_bounds: np.ndarray, steps: int, reason: str ) -> None:
        self.zoom_events.append(
            ZoomEvent(
                zoom_index=len( self.zoom_events ),
                old_bounds=old_bounds,
                new_bounds=new_bounds.copy(),
                selected_basin_score=float( selected.combined_score ),
                selected_basin_bbox=selected.bbox_world.copy(),
                evaluation_count=int( self.engine.state.evaluations if self.engine.state is not None else 0 ),
                steps_per_zoom=steps,
            )
        )
        if self.engine.snapshots:
            snapshot = self.engine.snapshots[ -1 ]
            snapshot.selected_basin_bbox = selected.bbox_world.copy()
            snapshot.metadata[ "zoom_decision" ] = "accepted"
            snapshot.metadata[ "zoom_reason" ] = reason

    def step( self ) -> bool:
        """Run one AGSLS decision round. Returns True when a zoom is accepted."""

        zoom_index = len( self.zoom_events )
        self.engine.set_zoom_index( zoom_index, max_zoom_cycles=self.agsls_config.max_zoom_cycles )
        self.apply_runtime_policy( AGSLS_PER_DECISION_FIELDS )
        self.apply_runtime_policy( ZOOM_BOUNDARY_FIELDS )
        self.engine.apply_runtime_policy( ZOOM_BOUNDARY_FIELDS, rebuild_kernels=True )
        steps = steps_for_zoom_cycle( self.agsls_config, zoom_index )
        persistence = self._persistence_map( steps )
        ranked = self._rank_basins( persistence )
        self._update_basin_runtime_state( ranked )
        self._probe_basins( ranked )
        ranked = self._rank_basins( persistence )
        self._update_basin_runtime_state( ranked )
        eligible = self._eligible_basins( ranked )
        if not eligible:
            self._mark_latest_snapshot( decision="deferred_no_eligible_group", basins=ranked )
            self._record_decision_reason( "deferred_no_eligible_group" )
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
            gained = self.engine.explore_top_pixels(
                min( self.agsls_config.candidate_probe_evaluations, remaining ),
                mask=union_mask,
                score_field=self._probe_score_field(),
            )
            explored_in_stage += int( gained )
            ranked = self._rank_basins( persistence )
            eligible = self._eligible_basins( ranked )
            if not eligible:
                self._mark_latest_snapshot( decision="deferred_groups_lost", basins=ranked )
                self._record_decision_reason( "deferred_groups_lost" )
                return False
            decision_ready, decision_reason = self._should_choose_leader( eligible, explored_in_stage )
            if gained == 0:
                decision_ready = True
                decision_reason = "no_unexplored_cells"

        selected = eligible[ 0 ]
        new_bounds = self._padded_bounds( selected )
        state = self.engine.state
        if state is None:
            raise RuntimeError( "engine state missing" )
        old_bounds = state.bounds.copy()
        if np.allclose( new_bounds, old_bounds ):
            self._mark_latest_snapshot( decision="deferred_no_shrink", basins=eligible )
            self._record_decision_reason( "deferred_no_shrink" )
            return False
        self.engine.remap_to_bounds( new_bounds )
        self._record_zoom( selected, old_bounds, new_bounds, steps, decision_reason )
        self._record_decision_reason( decision_reason, accepted=True )
        return True

    def run( self, zoom_cycles: int | None = None, evaluations: int | None = None ) -> SearchRun:
        """Run AGSLS until the zoom or evaluation limit is reached."""

        limit = self.agsls_config.max_zoom_cycles if zoom_cycles is None else int( zoom_cycles )
        eval_limit = self.agsls_config.max_evaluations if evaluations is None else int( evaluations )
        self.active_max_evaluations = eval_limit
        self.engine.active_max_evaluations = eval_limit
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
                widths = state.bounds[ :, 1 ] - state.bounds[ :, 0 ]
                min_widths = self.agsls_config.min_side_fraction * ( self.original_bounds[ :, 1 ] - self.original_bounds[ :, 0 ] )
                if np.all( widths <= min_widths ):
                    break
                self.step()
                decision_rounds += 1
                if eval_limit is None and decision_rounds >= max_rounds:
                    break
        finally:
            self.active_max_evaluations = None
            self.engine.active_max_evaluations = None
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
                "schedule_summary": None if self.runtime_policy is None else self.runtime_policy.summary(),
                "decision_reason_counts": dict( sorted( self._decision_reason_counts.items() ) ),
                "zoom_acceptance_count": int( self._accepted_zoom_count ),
            },
        )

    def zoom_history( self ) -> list[ ZoomEvent ]:
        """Return recorded zoom events."""

        return list( self.zoom_events )

    def snapshot( self ):
        """Return the current SmoothLife snapshot."""

        return self.engine.snapshot()
