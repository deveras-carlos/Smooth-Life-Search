"""Literal 2D SmoothLife simulator with lazy objective evaluation."""

from __future__ import annotations

from typing import Callable

import numpy as np

from ..core import (
    KERNEL_PARAMETER_FIELDS,
    SMOOTHLIFE_PER_STEP_FIELDS,
    ZOOM_BOUNDARY_FIELDS,
    Basin,
    RuntimeSignals,
    SchedulePolicy,
    SearchRun,
    SmoothLifeSnapshot,
    normalize_bounds_2d,
)
from .basins import detect_basins
from .config import SmoothLifeConfig
from .dynamics import (
    exploitation_score_field,
    exploration_score_field,
    initial_field,
    laplacian,
    refresh_dynamics_fields,
    vitality,
)
from .evaluation import (
    blank_objective_cache,
    best_evaluated_flat_index,
    evaluation_points,
    normalized_objective_field,
    subpixel_best_offset,
)
from .kernels import build_disk_kernel, build_ring_kernel
from .presets import apply_preset
from .remap import remap_field_to_bounds
from .state import SmoothLifeState

Objective = Callable[[ np.ndarray ], float]


class SmoothLifeSearch:
    """Literal SmoothLife-inspired 2D simulator coupled to an optimization objective."""

    def __init__(
        self,
        objective: Objective,
        bounds: np.ndarray | list[ tuple[ float, float ] ],
        config: SmoothLifeConfig | None = None,
        runtime_policy: SchedulePolicy | None = None,
    ) -> None:
        self.objective = objective
        self.original_bounds = self._normalize_bounds( bounds )
        self.config = apply_preset( config or SmoothLifeConfig() )
        self.runtime_policy = runtime_policy
        self.rng = np.random.default_rng()
        self.inner_kernel = build_disk_kernel( self.config.inner_radius, self.config.anti_alias_radius )
        self.outer_kernel = build_ring_kernel( self.config.inner_radius, self.config.outer_radius, self.config.anti_alias_radius )
        self.state: SmoothLifeState | None = None
        self.snapshots: list[ SmoothLifeSnapshot ] = [ ]
        self.current_zoom_index = 0
        self.active_max_evaluations: int | None = None
        self.max_zoom_cycles = 1
        self._observed_best_value: float | None = None
        self._observed_local_best_value: float | None = None
        self._last_global_improvement = 0.0
        self._last_stage_improvement = 0.0
        self.reset()

    @staticmethod
    def _normalize_bounds( bounds: np.ndarray | list[ tuple[ float, float ] ] ) -> np.ndarray:
        return normalize_bounds_2d( bounds, owner="SmoothLifeSearch" )

    def _worst_value( self ) -> float:
        return -np.inf if self.config.maximize else np.inf

    def _is_better( self, candidate_value: float, incumbent_value: float ) -> bool:
        if self.config.maximize:
            return candidate_value > incumbent_value + 1e-12
        return candidate_value < incumbent_value - 1e-12

    def _is_better_or_equal( self, candidate_value: float, incumbent_value: float ) -> bool:
        if self.config.maximize:
            return candidate_value >= incumbent_value - 1e-12
        return candidate_value <= incumbent_value + 1e-12

    def _initial_field( self ) -> np.ndarray:
        return initial_field( self.config, self.rng )

    def _blank_objective_cache( self ) -> tuple[ np.ndarray, np.ndarray, np.ndarray ]:
        return blank_objective_cache( self.config.grid_shape )

    def _default_point( self, bounds: np.ndarray ) -> np.ndarray:
        return np.mean( bounds, axis=1 )

    def _rebuild_kernels( self ) -> None:
        self.inner_kernel = build_disk_kernel( self.config.inner_radius, self.config.anti_alias_radius )
        self.outer_kernel = build_ring_kernel( self.config.inner_radius, self.config.outer_radius, self.config.anti_alias_radius )

    def set_zoom_index( self, zoom_index: int, max_zoom_cycles: int | None = None ) -> None:
        self.current_zoom_index = max( int( zoom_index ), 0 )
        if max_zoom_cycles is not None:
            self.max_zoom_cycles = max( int( max_zoom_cycles ), 1 )

    def reset( self, seed: int | None = None, bounds: np.ndarray | None = None ) -> None:
        """Reset the field and the lazy objective cache."""

        if seed is not None:
            self.rng = np.random.default_rng( seed )
        if self.runtime_policy is not None:
            self.runtime_policy.reset_tracking()
        current_bounds = self.original_bounds.copy() if bounds is None else self._normalize_bounds( bounds )
        field = self._initial_field()
        objective_values, objective_field, evaluated_mask = self._blank_objective_cache()
        zeros = np.zeros_like( field )
        default_point = self._default_point( current_bounds )
        worst_value = self._worst_value()
        self.state = SmoothLifeState(
            field=field,
            objective_values=objective_values,
            objective_field=objective_field,
            evaluated_mask=evaluated_mask,
            inner_fill=zeros.copy(),
            outer_fill=zeros.copy(),
            transition_field=zeros.copy(),
            support_ema=zeros.copy(),
            bounds=current_bounds,
            best_point=default_point.copy(),
            best_value=worst_value,
            local_best_point=default_point.copy(),
            local_best_value=worst_value,
            box_best_point=default_point.copy(),
            box_best_value=worst_value,
            step_index=0,
            evaluations=0,
        )
        self._refresh_dynamics_fields()
        self._bootstrap_exploration()
        self._refresh_dynamics_fields()
        self._reset_support_ema()
        self.snapshots = [ self.snapshot() ]
        self._reset_improvement_trackers()

    def _require_state( self ) -> SmoothLifeState:
        if self.state is None:
            raise RuntimeError( "simulator has not been initialized" )
        return self.state

    def _improvement_amount( self, previous: float | None, current: float ) -> float:
        if previous is None or not np.isfinite( previous ) or not np.isfinite( current ):
            return 0.0
        if self.config.maximize:
            return max( 0.0, float( current ) - float( previous ) )
        return max( 0.0, float( previous ) - float( current ) )

    def _reset_improvement_trackers( self ) -> None:
        state = self._require_state()
        self._observed_best_value = float( state.best_value )
        self._observed_local_best_value = float( state.local_best_value )
        self._last_global_improvement = 0.0
        self._last_stage_improvement = 0.0

    def _update_improvement_trackers( self ) -> None:
        state = self._require_state()
        current_best = float( state.best_value )
        current_local = float( state.local_best_value )
        self._last_global_improvement = self._improvement_amount( self._observed_best_value, current_best )
        self._last_stage_improvement = self._improvement_amount( self._observed_local_best_value, current_local )
        self._observed_best_value = current_best
        self._observed_local_best_value = current_local

    def _pixel_center( self, row: int, col: int, bounds: np.ndarray ) -> np.ndarray:
        height, width = self.config.grid_shape
        x = bounds[ 0, 0 ] + ( ( float( col ) + 0.5 ) / width ) * ( bounds[ 0, 1 ] - bounds[ 0, 0 ] )
        y = bounds[ 1, 0 ] + ( ( float( row ) + 0.5 ) / height ) * ( bounds[ 1, 1 ] - bounds[ 1, 0 ] )
        return np.asarray( [ x, y ], dtype=float )

    def _flat_index_to_point( self, flat_index: int, bounds: np.ndarray ) -> np.ndarray:
        row, col = np.unravel_index( int( flat_index ), self.config.grid_shape )
        return self._pixel_center( int( row ), int( col ), bounds )

    def grid_world_points( self, bounds: np.ndarray | None = None ) -> np.ndarray:
        """Return the flattened pixel-center coordinates for the active grid."""

        resolved_bounds = self.current_bounds() if bounds is None else self._normalize_bounds( bounds )
        height, width = self.config.grid_shape
        local_x = ( np.arange( width, dtype=float ) + 0.5 ) / width
        local_y = ( np.arange( height, dtype=float ) + 0.5 ) / height
        grid_x, grid_y = np.meshgrid( local_x, local_y, indexing="xy" )
        x = resolved_bounds[ 0, 0 ] + grid_x * ( resolved_bounds[ 0, 1 ] - resolved_bounds[ 0, 0 ] )
        y = resolved_bounds[ 1, 0 ] + grid_y * ( resolved_bounds[ 1, 1 ] - resolved_bounds[ 1, 0 ] )
        return np.column_stack( ( x.ravel(), y.ravel() ) )

    @staticmethod
    def vitality( field: np.ndarray ) -> np.ndarray:
        return vitality( field )

    def alive_mask( self, threshold: float ) -> np.ndarray:
        state = self._require_state()
        return np.abs( state.field ) <= float( threshold )

    def support_field( self ) -> np.ndarray:
        state = self._require_state()
        return self.vitality( state.field ) * state.objective_field

    def _reset_support_ema( self ) -> None:
        state = self._require_state()
        state.support_ema = self.support_field().copy()

    def _update_support_ema( self ) -> None:
        state = self._require_state()
        support = self.support_field()
        alpha = float( self.config.support_ema_alpha )
        if alpha <= 0.0:
            state.support_ema = support.copy()
            return
        state.support_ema = ( 1.0 - alpha ) * state.support_ema + alpha * support

    def smoothed_support_field( self ) -> np.ndarray:
        state = self._require_state()
        if self.config.support_ema_alpha <= 0.0:
            return self.support_field()
        return state.support_ema.copy()

    def current_bounds( self ) -> np.ndarray:
        return self._require_state().bounds.copy()

    def best_point( self ) -> np.ndarray:
        return self._require_state().best_point.copy()

    def best_value( self ) -> float:
        return float( self._require_state().best_value )

    def runtime_signals( self ) -> RuntimeSignals:
        state = self._require_state()
        widths = state.bounds[ :, 1 ] - state.bounds[ :, 0 ]
        remaining_budget = None
        if self.active_max_evaluations is not None:
            remaining_budget = max( int( self.active_max_evaluations ) - int( state.evaluations ), 0 )
        return RuntimeSignals(
            step_index=int( state.step_index ),
            zoom_index=int( self.current_zoom_index ),
            evaluations=int( state.evaluations ),
            remaining_budget=remaining_budget,
            total_budget=self.active_max_evaluations,
            explored_fraction=float( np.mean( state.evaluated_mask ) ),
            current_box_widths=( float( widths[ 0 ] ), float( widths[ 1 ] ) ),
            best_value=float( state.best_value ),
            local_best_value=float( state.local_best_value ),
            global_improvement=float( self._last_global_improvement ),
            stage_improvement=float( self._last_stage_improvement ),
            max_zoom_cycles=int( self.max_zoom_cycles ),
        )

    def apply_runtime_overrides( self, overrides: dict[ str, float | int ], *, rebuild_kernels: bool ) -> dict[ str, float | int ]:
        changed: dict[ str, float | int ] = { }
        if not overrides:
            return changed
        for field_name, value in overrides.items():
            if not hasattr( self.config, field_name ):
                continue
            current = getattr( self.config, field_name )
            if current == value:
                continue
            setattr( self.config, field_name, value )
            changed[ field_name ] = value
        if not changed:
            return changed
        self.config.__post_init__()
        if rebuild_kernels and any( field_name in KERNEL_PARAMETER_FIELDS for field_name in changed ):
            self._rebuild_kernels()
        if self.state is not None:
            if "objective_gamma" in changed:
                self._refresh_objective_field()
            self._refresh_dynamics_fields()
            if "support_ema_alpha" in changed:
                self._reset_support_ema()
        return changed

    def apply_runtime_policy( self, allowed_fields: set[ str ] | frozenset[ str ], *, rebuild_kernels: bool ) -> dict[ str, float | int ]:
        if self.runtime_policy is None:
            return { }
        overrides = self.runtime_policy.resolve( self.runtime_signals(), allowed_fields )
        return self.apply_runtime_overrides( overrides, rebuild_kernels=rebuild_kernels )

    def _refresh_dynamics_fields( self ) -> None:
        state = self._require_state()
        inner_fill, outer_fill, transition = refresh_dynamics_fields(
            state.field,
            state.objective_field,
            self.config,
            self.inner_kernel,
            self.outer_kernel,
        )
        state.inner_fill = inner_fill
        state.outer_fill = outer_fill
        state.transition_field = transition

    def _refresh_objective_field( self ) -> None:
        state = self._require_state()
        state.objective_field = normalized_objective_field(
            state.objective_values,
            state.evaluated_mask,
            maximize=self.config.maximize,
            gamma=self.config.objective_gamma,
        )

    def _update_best_records( self ) -> None:
        state = self._require_state()
        best_flat = best_evaluated_flat_index(
            state.objective_values,
            state.evaluated_mask,
            maximize=self.config.maximize,
        )
        if best_flat is None:
            return
        flat_values = state.objective_values.ravel()
        candidate_value = float( flat_values[ best_flat ] )
        candidate_point, has_subpixel_offset = self._subpixel_best_candidate( best_flat, state )
        if has_subpixel_offset and self.config.subpixel_confirm:
            confirmed = self._confirm_subpixel_candidate( candidate_point )
            if confirmed is not None and self._is_better( confirmed, candidate_value ):
                candidate_value = float( confirmed )
            else:
                candidate_point = self._flat_index_to_point( best_flat, state.bounds )
        state.box_best_point = candidate_point.copy()
        state.box_best_value = candidate_value
        if not np.isfinite( state.best_value ) or self._is_better( candidate_value, state.best_value ):
            state.best_point = candidate_point.copy()
            state.best_value = candidate_value
        if not np.isfinite( state.local_best_value ) or self._is_better( candidate_value, state.local_best_value ):
            state.local_best_point = candidate_point.copy()
            state.local_best_value = candidate_value

    def _subpixel_best_point( self, best_flat: int, state: SmoothLifeState ) -> np.ndarray:
        return self._subpixel_best_candidate( best_flat, state )[ 0 ]

    def _subpixel_best_candidate( self, best_flat: int, state: SmoothLifeState ) -> tuple[ np.ndarray, bool ]:
        row, col = np.unravel_index( int( best_flat ), self.config.grid_shape )
        if not self.config.subpixel_best_point:
            return self._pixel_center( int( row ), int( col ), state.bounds ), False
        drow, dcol = subpixel_best_offset(
            state.objective_values,
            state.evaluated_mask,
            int( row ),
            int( col ),
            maximize=self.config.maximize,
        )
        if drow == 0.0 and dcol == 0.0:
            return self._pixel_center( int( row ), int( col ), state.bounds ), False
        height, width = self.config.grid_shape
        x = state.bounds[ 0, 0 ] + ( ( float( col ) + 0.5 + dcol ) / width ) * ( state.bounds[ 0, 1 ] - state.bounds[ 0, 0 ] )
        y = state.bounds[ 1, 0 ] + ( ( float( row ) + 0.5 + drow ) / height ) * ( state.bounds[ 1, 1 ] - state.bounds[ 1, 0 ] )
        return np.asarray( [ x, y ], dtype=float ), True

    def _confirmation_budget_available( self ) -> bool:
        state = self._require_state()
        if self.active_max_evaluations is None:
            return True
        return int( state.evaluations ) < int( self.active_max_evaluations )

    def _confirm_subpixel_candidate( self, candidate_point: np.ndarray ) -> float | None:
        if not self._confirmation_budget_available():
            return None
        state = self._require_state()
        value = float( self.objective( np.asarray( candidate_point, dtype=float ) ) )
        state.evaluations += 1
        return value

    def _evaluation_points( self, rows: np.ndarray, cols: np.ndarray, bounds: np.ndarray ) -> np.ndarray:
        return evaluation_points( rows, cols, bounds, self.config.grid_shape )

    def evaluate_pixels( self, rows: np.ndarray, cols: np.ndarray ) -> int:
        """Evaluate a batch of pixels that have not been explored yet."""

        state = self._require_state()
        if rows.size == 0 or cols.size == 0:
            return 0
        rows = np.asarray( rows, dtype=int )
        cols = np.asarray( cols, dtype=int )
        unexplored = ~state.evaluated_mask[ rows, cols ]
        if not np.any( unexplored ):
            return 0
        rows = rows[ unexplored ]
        cols = cols[ unexplored ]
        points = self._evaluation_points( rows, cols, state.bounds )
        values = np.asarray( [ float( self.objective( point ) ) for point in points ], dtype=float )
        state.objective_values[ rows, cols ] = values
        state.evaluated_mask[ rows, cols ] = True
        state.evaluations += int( values.size )
        self._refresh_objective_field()
        self._update_best_records()
        self._refresh_dynamics_fields()
        self._update_support_ema()
        self._update_improvement_trackers()
        return int( values.size )

    def _top_unexplored_indices(
        self,
        score_field: np.ndarray,
        limit: int,
        mask: np.ndarray | None = None,
    ) -> tuple[ np.ndarray, np.ndarray ]:
        state = self._require_state()
        if limit <= 0:
            return np.asarray( [ ], dtype=int ), np.asarray( [ ], dtype=int )
        candidate_mask = ~state.evaluated_mask
        if mask is not None:
            candidate_mask &= np.asarray( mask, dtype=bool )
        flat_indices = np.flatnonzero( candidate_mask.ravel() )
        if flat_indices.size == 0:
            return np.asarray( [ ], dtype=int ), np.asarray( [ ], dtype=int )
        scores = np.asarray( score_field, dtype=float ).ravel()[ flat_indices ]
        scores = np.where( np.isfinite( scores ), scores, -np.inf )
        resolved_limit = min( int( limit ), int( flat_indices.size ) )
        if resolved_limit <= 0:
            return np.asarray( [ ], dtype=int ), np.asarray( [ ], dtype=int )
        if resolved_limit == flat_indices.size:
            top_order = np.argsort( scores )[ ::-1 ]
        else:
            partial = np.argpartition( scores, -resolved_limit )[ -resolved_limit: ]
            top_order = partial[ np.argsort( scores[ partial ] )[ ::-1 ] ]
        chosen = flat_indices[ top_order ]
        rows, cols = np.unravel_index( chosen, self.config.grid_shape )
        return np.asarray( rows, dtype=int ), np.asarray( cols, dtype=int )

    def exploration_score_field( self ) -> np.ndarray:
        state = self._require_state()
        return exploration_score_field( state.field, state.transition_field )

    def exploitation_score_field( self ) -> np.ndarray:
        state = self._require_state()
        return exploitation_score_field(
            state.field,
            state.transition_field,
            state.objective_field,
            state.evaluated_mask,
        )

    def explore_top_pixels( self, limit: int, mask: np.ndarray | None = None, score_field: np.ndarray | None = None ) -> int:
        """Evaluate the highest-scoring unexplored pixels, optionally within a mask."""

        scores = self.exploration_score_field() if score_field is None else np.asarray( score_field, dtype=float )
        rows, cols = self._top_unexplored_indices( scores, limit, mask=mask )
        return self.evaluate_pixels( rows, cols )

    def _bootstrap_exploration( self ) -> int:
        return self.explore_top_pixels( self.config.evaluations_per_step, score_field=self.vitality( self._require_state().field ) )

    def _laplacian( self, field: np.ndarray ) -> np.ndarray:
        return laplacian( field )

    def _capture_snapshot( self, force: bool = False, selected_basin_bbox: np.ndarray | None = None ) -> None:
        if not self.config.store_all_snapshots and not force:
            return
        state = self._require_state()
        if not force and state.step_index % self.config.snapshot_interval != 0:
            return
        snapshot = self.snapshot()
        snapshot.selected_basin_bbox = None if selected_basin_bbox is None else np.asarray( selected_basin_bbox, dtype=float )
        self.snapshots.append( snapshot )

    def step( self, n: int = 1 ) -> None:
        """Advance the SmoothLife field and lazily explore new pixels."""

        state = self._require_state()
        for _ in range( n ):
            self.apply_runtime_policy( SMOOTHLIFE_PER_STEP_FIELDS, rebuild_kernels=False )
            laplacian = self._laplacian( state.field )
            target_state = state.transition_field
            if self.config.time_mode == "continuous":
                updated = state.field + self.config.dt * ( target_state - state.field ) + self.config.diffusion * laplacian
            else:
                updated = ( 1.0 - self.config.dt ) * state.field + self.config.dt * target_state + self.config.diffusion * laplacian
            state.field = np.clip( updated, self.config.field_floor, self.config.field_ceiling )
            state.step_index += 1
            self._refresh_dynamics_fields()
            self.explore_top_pixels( self.config.evaluations_per_step )
            self._capture_snapshot()

    def basin_candidates(
        self,
        threshold_quantile: float = 0.88,
        min_cells: int = 16,
        alive_core_threshold: float = 0.30,
        cluster_eps_pixels: float = 2.5,
        cluster_min_samples: int = 6,
        basin_envelope_quantile_offset: float = 0.08,
        basin_envelope_growth_pixels: int = 1,
        support_field: np.ndarray | None = None,
    ) -> list[ Basin ]:
        """Extract promising dense groups from the current support field."""

        state = self._require_state()
        return detect_basins(
            support_field=self.support_field() if support_field is None else np.asarray( support_field, dtype=float ),
            objective_field=state.objective_field,
            bounds=state.bounds,
            alive_mask=self.alive_mask( alive_core_threshold ),
            evaluated_mask=state.evaluated_mask,
            objective_values=state.objective_values,
            maximize=self.config.maximize,
            threshold_quantile=threshold_quantile,
            min_cells=min_cells,
            cluster_eps_pixels=cluster_eps_pixels,
            cluster_min_samples=cluster_min_samples,
            basin_envelope_quantile_offset=basin_envelope_quantile_offset,
            basin_envelope_growth_pixels=basin_envelope_growth_pixels,
            global_best_point=state.best_point,
            global_best_value=state.best_value,
        )

    def snapshot( self ) -> SmoothLifeSnapshot:
        """Create an immutable copy of the current state."""

        state = self._require_state()
        explored_fraction = float( np.mean( state.evaluated_mask ) )
        return SmoothLifeSnapshot(
            step_index=state.step_index,
            bounds=state.bounds.copy(),
            field=state.field.copy(),
            inner_fill=state.inner_fill.copy(),
            outer_fill=state.outer_fill.copy(),
            objective_field=state.objective_field.copy(),
            transition_field=state.transition_field.copy(),
            evaluated_mask=state.evaluated_mask.copy(),
            best_point=state.best_point.copy(),
            best_value=float( state.best_value ),
            local_best_point=state.local_best_point.copy(),
            local_best_value=float( state.local_best_value ),
            box_best_point=state.box_best_point.copy(),
            box_best_value=float( state.box_best_value ),
            metadata={
                "evaluations": int( state.evaluations ),
                "explored_fraction": explored_fraction,
            },
        )

    def remap_to_bounds( self, new_bounds: np.ndarray ) -> None:
        """Zoom the full grid into a new box and clear the lazy objective cache."""

        state = self._require_state()
        new_bounds = self._normalize_bounds( new_bounds )
        state.field = np.clip( remap_field_to_bounds( state.field, state.bounds, new_bounds ), self.config.field_floor, self.config.field_ceiling )
        state.bounds = new_bounds
        objective_values, objective_field, evaluated_mask = self._blank_objective_cache()
        state.objective_values = objective_values
        state.objective_field = objective_field
        state.evaluated_mask = evaluated_mask
        default_point = self._default_point( new_bounds )
        state.local_best_point = default_point.copy()
        state.local_best_value = self._worst_value()
        state.box_best_point = default_point.copy()
        state.box_best_value = self._worst_value()
        self._refresh_dynamics_fields()
        self._bootstrap_exploration()
        self._refresh_dynamics_fields()
        self._reset_support_ema()
        self._reset_improvement_trackers()
        self._capture_snapshot( force=True )

    def run( self, steps: int | None = None, evaluations: int | None = None ) -> SearchRun:
        """Run the simulator for a fixed number of steps or until a budget is reached."""

        if steps is None and evaluations is None:
            steps = self.config.snapshot_interval
        state = self._require_state()
        target_evaluations = None if evaluations is None else int( evaluations )
        self.active_max_evaluations = target_evaluations
        self.apply_runtime_policy( ZOOM_BOUNDARY_FIELDS, rebuild_kernels=True )
        remaining_steps = 0 if steps is None else int( steps )
        try:
            while True:
                if remaining_steps <= 0 and steps is not None:
                    break
                if target_evaluations is not None and state.evaluations >= target_evaluations:
                    break
                self.step( 1 )
                if steps is not None:
                    remaining_steps -= 1
        finally:
            self.active_max_evaluations = None
        schedule_summary = None if self.runtime_policy is None else self.runtime_policy.summary()
        return SearchRun(
            best_point=state.best_point.copy(),
            best_value=float( state.best_value ),
            evaluations=int( state.evaluations ),
            bounds=state.bounds.copy(),
            snapshots=list( self.snapshots ),
            zoom_events=[ ],
            metadata={
                "mode": self.config.run_mode,
                "steps": int( state.step_index ),
                "schedule_summary": schedule_summary,
            },
        )
