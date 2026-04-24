from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
import sys
from unittest.mock import MagicMock, patch

import numpy as np

import main
from smooth_life_search import (
    AGSLSConfig,
    AdaptiveGridSmoothLifeSearch,
    Basin,
    SmoothLifeConfig,
    SmoothLifeSearch,
    ackley,
    render_run_frames,
    save_run_animation,
    sphere,
)
from smooth_life_search.agsls.scoring import score_basins
from smooth_life_search.smoothlife.basins import detect_basins
from smooth_life_search.smoothlife.kernels import build_disk_kernel, build_ring_kernel
from smooth_life_search.smoothlife.transition import smoothlife_transition


class CountingObjective:
    def __init__( self ) -> None:
        self.count = 0

    def __call__( self, point: np.ndarray ) -> float:
        self.count += 1
        return sphere( point )


def _bbox_world(
    shape: tuple[ int, int ],
    bounds: np.ndarray,
    row0: int,
    col0: int,
    row1: int,
    col1: int,
) -> np.ndarray:
    height, width = shape
    return np.array(
        [
            [
                bounds[ 0, 0 ] + ( col0 / width ) * ( bounds[ 0, 1 ] - bounds[ 0, 0 ] ),
                bounds[ 0, 0 ] + ( col1 / width ) * ( bounds[ 0, 1 ] - bounds[ 0, 0 ] ),
            ],
            [
                bounds[ 1, 0 ] + ( row0 / height ) * ( bounds[ 1, 1 ] - bounds[ 1, 0 ] ),
                bounds[ 1, 0 ] + ( row1 / height ) * ( bounds[ 1, 1 ] - bounds[ 1, 0 ] ),
            ],
        ],
        dtype=float,
    )


def _make_basin(
    shape: tuple[ int, int ],
    bounds: np.ndarray,
    row0: int,
    col0: int,
    row1: int,
    col1: int,
    *,
    score: float,
    alive_density: float,
    best_value: float,
) -> Basin:
    mask = np.zeros( shape, dtype=bool )
    mask[ row0:row1, col0:col1 ] = True
    bbox_world = _bbox_world( shape, bounds, row0, col0, row1, col1 )
    centroid_grid = np.asarray( [ 0.5 * ( col0 + col1 - 1 ), 0.5 * ( row0 + row1 - 1 ) ], dtype=float )
    centroid_world = np.asarray(
        [
            0.5 * ( bbox_world[ 0, 0 ] + bbox_world[ 0, 1 ] ),
            0.5 * ( bbox_world[ 1, 0 ] + bbox_world[ 1, 1 ] ),
        ],
        dtype=float,
    )
    return Basin(
        mask=mask,
        centroid_grid=centroid_grid,
        centroid_world=centroid_world,
        bbox_grid=( row0, col0, row1 - 1, col1 - 1 ),
        bbox_world=bbox_world,
        support_mass=float( np.count_nonzero( mask ) ),
        objective_score=0.9,
        stability_score=0.8,
        alive_density=alive_density,
        basin_best_point=centroid_world.copy(),
        basin_best_value=best_value,
        combined_score=score,
    )


class TestSmoothLifeKernels( unittest.TestCase ):
    def test_disk_and_ring_kernels_are_normalized( self ) -> None:
        disk = build_disk_kernel( radius=10.0, anti_alias_radius=1.0 )
        ring = build_ring_kernel( inner_radius=10.0, outer_radius=12.0, anti_alias_radius=1.0 )
        self.assertAlmostEqual( float( disk.sum() ), 1.0, places=6 )
        self.assertAlmostEqual( float( ring.sum() ), 1.0, places=6 )


class TestSmoothLifeSearch( unittest.TestCase ):
    def test_pixel_centers_cover_expected_world_coordinates( self ) -> None:
        config = SmoothLifeConfig( grid_shape=( 200, 200 ), evaluations_per_step=1, preset="search" )
        search = SmoothLifeSearch( sphere, bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ], config=config )
        points = search.grid_world_points( np.asarray( [ [ -10.0, 10.0 ], [ -10.0, 10.0 ] ], dtype=float ) )
        self.assertTrue( np.allclose( points[ 0 ], np.asarray( [ -9.95, -9.95 ] ) ) )
        self.assertTrue( np.allclose( points[ 199 ], np.asarray( [ 9.95, -9.95 ] ) ) )
        self.assertTrue( np.allclose( points[ -1 ], np.asarray( [ 9.95, 9.95 ] ) ) )

    def test_reset_is_lazy_and_counts_real_objective_calls( self ) -> None:
        objective = CountingObjective()
        config = SmoothLifeConfig( grid_shape=( 200, 200 ), evaluations_per_step=7, preset="search" )
        search = SmoothLifeSearch( objective, bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ], config=config )
        objective.count = 0
        search.reset( seed=7 )
        self.assertEqual( objective.count, 7 )
        self.assertEqual( search.state.evaluations, 7 )
        self.assertEqual( int( search.state.evaluated_mask.sum() ), 7 )

    def test_remap_bootstraps_without_full_grid_evaluation( self ) -> None:
        objective = CountingObjective()
        config = SmoothLifeConfig( grid_shape=( 64, 64 ), evaluations_per_step=5, preset="search" )
        search = SmoothLifeSearch( objective, bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ], config=config )
        objective.count = 0
        search.reset( seed=5 )
        before = objective.count
        search.remap_to_bounds( np.asarray( [ [ -5.0, 0.0 ], [ -4.0, 1.0 ] ], dtype=float ) )
        self.assertEqual( objective.count - before, 5 )
        self.assertEqual( int( search.state.evaluated_mask.sum() ), 5 )

    def test_each_step_evaluates_only_new_pixels( self ) -> None:
        objective = CountingObjective()
        config = SmoothLifeConfig( grid_shape=( 40, 40 ), evaluations_per_step=4, snapshot_interval=1, preset="search" )
        search = SmoothLifeSearch( objective, bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ], config=config )
        objective.count = 0
        search.reset( seed=3 )
        self.assertEqual( objective.count, 4 )
        search.step( 3 )
        self.assertEqual( search.state.evaluations, objective.count )
        self.assertEqual( int( search.state.evaluated_mask.sum() ), objective.count )
        self.assertEqual( objective.count, 16 )

    def test_field_stays_signed_inside_bounds( self ) -> None:
        config = SmoothLifeConfig( grid_shape=( 48, 48 ), evaluations_per_step=5, snapshot_interval=1, preset="search" )
        search = SmoothLifeSearch( sphere, bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ], config=config )
        search.reset( seed=9 )
        run = search.run( steps=6 )
        for snapshot in run.snapshots:
            self.assertLessEqual( float( np.max( snapshot.field ) ), 1.0 + 1e-12 )
            self.assertGreaterEqual( float( np.min( snapshot.field ) ), -1.0 - 1e-12 )

    def test_objective_guidance_pulls_better_cells_closer_to_zero( self ) -> None:
        config = SmoothLifeConfig( objective_coupling=0.6, preset="search" )
        field = np.ones( ( 1, 2 ), dtype=float )
        inner_fill = np.full( ( 1, 2 ), 0.3, dtype=float )
        outer_fill = np.full( ( 1, 2 ), 0.3, dtype=float )
        objective_field = np.asarray( [ [ 1.0, 0.0 ] ], dtype=float )
        target_state, _ = smoothlife_transition( field, inner_fill, outer_fill, config, objective_field )
        self.assertLess( abs( float( target_state[ 0, 0 ] ) ), abs( float( target_state[ 0, 1 ] ) ) )

    def test_objective_is_part_of_transition_function( self ) -> None:
        base = SmoothLifeConfig( grid_shape=( 40, 40 ), evaluations_per_step=4, run_mode="search", objective_coupling=0.0, snapshot_interval=1, preset="search" )
        guided = SmoothLifeConfig( grid_shape=( 40, 40 ), evaluations_per_step=4, run_mode="search", objective_coupling=0.6, snapshot_interval=1, preset="search" )
        plain = SmoothLifeSearch( ackley, bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ], config=base )
        biased = SmoothLifeSearch( ackley, bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ], config=guided )
        plain.reset( seed=3 )
        biased.reset( seed=3 )
        plain.step( 1 )
        biased.step( 1 )
        self.assertFalse( np.allclose( plain.state.transition_field, biased.state.transition_field ) )


class TestDenseGroups( unittest.TestCase ):
    def test_clustering_splits_distant_groups_and_merges_close_ones( self ) -> None:
        bounds = np.asarray( [ [ -10.0, 10.0 ], [ -10.0, 10.0 ] ], dtype=float )
        support = np.zeros( ( 20, 20 ), dtype=float )
        support[ 2:5, 2:5 ] = 1.0
        support[ 12:15, 12:15 ] = 1.0
        objective = np.ones_like( support )
        alive = support > 0.0
        evaluated = alive.copy()
        values = np.full_like( support, np.nan )
        values[ evaluated ] = 1.0
        far = detect_basins(
            support_field=support,
            objective_field=objective,
            bounds=bounds,
            alive_mask=alive,
            evaluated_mask=evaluated,
            objective_values=values,
            maximize=False,
            threshold_quantile=0.8,
            min_cells=4,
            cluster_eps_pixels=2.5,
            cluster_min_samples=3,
        )
        self.assertEqual( len( far ), 2 )

        merged_support = np.zeros( ( 20, 20 ), dtype=float )
        merged_support[ 2:5, 2:5 ] = 1.0
        merged_support[ 5:8, 5:8 ] = 1.0
        merged_alive = merged_support > 0.0
        merged_values = np.full_like( merged_support, np.nan )
        merged_values[ merged_alive ] = 1.0
        merged = detect_basins(
            support_field=merged_support,
            objective_field=objective,
            bounds=bounds,
            alive_mask=merged_alive,
            evaluated_mask=merged_alive,
            objective_values=merged_values,
            maximize=False,
            threshold_quantile=0.8,
            min_cells=4,
            cluster_eps_pixels=2.5,
            cluster_min_samples=3,
        )
        self.assertEqual( len( merged ), 1 )

    def test_core_envelope_bounds_preserve_nearby_incumbent( self ) -> None:
        bounds = np.asarray( [ [ 0.0, 16.0 ], [ 0.0, 16.0 ] ], dtype=float )
        support = np.zeros( ( 16, 16 ), dtype=float )
        support[ 4:12, 4:12 ] = 0.6
        support[ 5:11, 5:11 ] = 1.0
        objective = np.full_like( support, 0.5 )
        alive = support > 0.0
        evaluated = np.zeros_like( support, dtype=bool )
        evaluated[ 8, 4 ] = True
        evaluated[ 8, 8 ] = True
        objective[ 8, 4 ] = 1.0
        objective[ 8, 8 ] = 0.7
        values = np.full_like( support, np.inf )
        values[ 8, 4 ] = 0.0
        values[ 8, 8 ] = 1.0
        incumbent = np.asarray( [ 4.5, 8.5 ], dtype=float )

        basins = detect_basins(
            support_field=support,
            objective_field=objective,
            bounds=bounds,
            alive_mask=alive,
            evaluated_mask=evaluated,
            objective_values=values,
            maximize=False,
            threshold_quantile=0.88,
            min_cells=8,
            cluster_eps_pixels=2.5,
            cluster_min_samples=3,
            basin_envelope_quantile_offset=0.08,
            basin_envelope_growth_pixels=0,
            global_best_point=incumbent,
            global_best_value=0.0,
        )

        self.assertEqual( len( basins ), 1 )
        basin = basins[ 0 ]
        self.assertIsNotNone( basin.core_bbox_world )
        core_contains_incumbent = bool( np.all( ( incumbent >= basin.core_bbox_world[ :, 0 ] ) & ( incumbent <= basin.core_bbox_world[ :, 1 ] ) ) )
        envelope_contains_incumbent = bool( np.all( ( incumbent >= basin.bbox_world[ :, 0 ] ) & ( incumbent <= basin.bbox_world[ :, 1 ] ) ) )
        self.assertFalse( core_contains_incumbent )
        self.assertTrue( envelope_contains_incumbent )
        self.assertTrue( basin.incumbent_in_envelope )
        self.assertEqual( basin.evaluated_count, 2 )
        self.assertAlmostEqual( basin.best_objective_score, 1.0 )

    def test_alive_density_changes_group_ranking( self ) -> None:
        bounds = np.asarray( [ [ -10.0, 10.0 ], [ -10.0, 10.0 ] ], dtype=float )
        high_density = _make_basin( ( 16, 16 ), bounds, 2, 2, 6, 6, score=0.0, alive_density=0.9, best_value=1.0 )
        low_density = _make_basin( ( 16, 16 ), bounds, 8, 8, 12, 12, score=0.0, alive_density=0.2, best_value=1.0 )
        high_density.support_mass = 10.0
        low_density.support_mass = 10.0
        high_density.objective_score = 0.8
        low_density.objective_score = 0.8
        high_density.stability_score = 0.8
        low_density.stability_score = 0.8
        scored = score_basins(
            [ low_density, high_density ],
            AGSLSConfig( alive_density_weight=1.5, mass_weight=0.0, objective_weight=1.0, stability_weight=1.0, area_penalty=0.0 ),
        )
        self.assertIs( scored[ 0 ], high_density )


class TestAGSLS( unittest.TestCase ):
    def test_global_and_local_best_are_monotonic_across_snapshots( self ) -> None:
        smoothlife = SmoothLifeConfig( grid_shape=( 48, 48 ), evaluations_per_step=4, objective_coupling=0.35, snapshot_interval=1, preset="search" )
        agsls = AGSLSConfig(
            max_zoom_cycles=3,
            initial_steps_per_zoom=10,
            min_steps_per_zoom=4,
            zoom_decay=0.6,
            min_basin_cells=8,
            max_evaluations=400,
        )
        search = AdaptiveGridSmoothLifeSearch(
            ackley,
            bounds=[ ( -5.0, 5.0 ), ( -5.0, 5.0 ) ],
            smoothlife_config=smoothlife,
            agsls_config=agsls,
        )
        search.reset( seed=19 )
        run = search.run()
        global_trace = np.asarray( [ snapshot.best_value for snapshot in run.snapshots ], dtype=float )
        self.assertTrue( np.all( np.diff( global_trace ) <= 1e-12 ) )
        self.assertAlmostEqual( float( global_trace[ -1 ] ), float( run.best_value ), places=12 )

    @staticmethod
    def _late_stage_signals(
        *,
        zoom_index: int = 0,
        max_zoom_cycles: int = 5,
        global_improvement: float = 1.0,
        stage_improvement: float = 1.0,
    ):
        from smooth_life_search.core import RuntimeSignals

        return RuntimeSignals(
            step_index=0,
            zoom_index=zoom_index,
            evaluations=0,
            remaining_budget=100,
            total_budget=100,
            explored_fraction=0.0,
            current_box_widths=(1.0, 1.0),
            best_value=1.0,
            local_best_value=1.0,
            global_improvement=global_improvement,
            stage_improvement=stage_improvement,
            max_zoom_cycles=max_zoom_cycles,
        )

    def test_late_stage_reactive_default_returns_base( self ) -> None:
        from smooth_life_search import FieldSchedule

        schedule = FieldSchedule(
            field_name="evaluations_per_step",
            mode="late_stage_reactive",
            base_value=16,
            high_value=8,
            zoom_fraction_threshold=0.6,
            plateau_threshold=1e-4,
        )
        signals = self._late_stage_signals( zoom_index=1, max_zoom_cycles=5, global_improvement=1.0 )
        self.assertEqual( schedule.evaluate( signals ), 16 )

    def test_late_stage_reactive_fires_on_zoom_fraction( self ) -> None:
        from smooth_life_search import FieldSchedule

        schedule = FieldSchedule(
            field_name="evaluations_per_step",
            mode="late_stage_reactive",
            base_value=16,
            high_value=8,
            zoom_fraction_threshold=0.6,
            plateau_threshold=1e-4,
        )
        signals = self._late_stage_signals( zoom_index=3, max_zoom_cycles=5, global_improvement=1.0 )
        self.assertEqual( schedule.evaluate( signals ), 8 )

    def test_late_stage_reactive_fires_on_plateau( self ) -> None:
        from smooth_life_search import FieldSchedule

        schedule = FieldSchedule(
            field_name="evaluations_per_step",
            mode="late_stage_reactive",
            base_value=16,
            high_value=8,
            zoom_fraction_threshold=0.6,
            plateau_threshold=1e-4,
        )
        signals = self._late_stage_signals(
            zoom_index=0, max_zoom_cycles=5, global_improvement=1e-7, stage_improvement=1e-7
        )
        self.assertEqual( schedule.evaluate( signals ), 8 )

    def test_smoothlife_per_step_fields_reresolve_each_decision( self ) -> None:
        from smooth_life_search import FieldSchedule, SchedulePolicy

        policy = SchedulePolicy(
            schedules=( FieldSchedule(
                field_name="evaluations_per_step",
                mode="late_stage_reactive",
                base_value=4,
                high_value=4,
                zoom_fraction_threshold=0.6,
                plateau_threshold=1e-4,
            ), ),
            family="smoothlife_cadence",
            schedule_kind="late_stage_reactive",
        )
        smoothlife = SmoothLifeConfig( grid_shape=( 32, 32 ), evaluations_per_step=4, preset="search", snapshot_interval=1 )
        agsls = AGSLSConfig( max_zoom_cycles=2, max_evaluations=200 )
        search = AdaptiveGridSmoothLifeSearch(
            sphere,
            bounds=[ ( -5.0, 5.0 ), ( -5.0, 5.0 ) ],
            smoothlife_config=smoothlife,
            agsls_config=agsls,
            runtime_policy=policy,
        )
        search.reset( seed=3 )
        run = search.run()
        summary = run.metadata.get( "schedule_summary" )
        self.assertIsNotNone( summary )
        field_summaries = summary[ "field_summaries" ]
        evals_summary = next( entry for entry in field_summaries if entry[ "field_name" ] == "evaluations_per_step" )
        self.assertGreaterEqual( int( evals_summary[ "resolution_count" ] ), 1 )

    def test_zoom_limit_scales_with_budget( self ) -> None:
        agsls = AGSLSConfig( max_zoom_cycles=5, zoom_cycles_budget_baseline=800 )
        search = AdaptiveGridSmoothLifeSearch( sphere, bounds=[ ( -1.0, 1.0 ), ( -1.0, 1.0 ) ], agsls_config=agsls )
        self.assertEqual( search._effective_zoom_limit( 1600, 5 ), 6 )
        self.assertEqual( search._effective_zoom_limit( 3200, 5 ), 7 )
        self.assertEqual( search._effective_zoom_limit( 6400, 5 ), 8 )
        self.assertEqual( search._effective_zoom_limit( 12800, 5 ), 9 )

    def test_zoom_limit_unchanged_at_small_budget( self ) -> None:
        agsls = AGSLSConfig( max_zoom_cycles=5, zoom_cycles_budget_baseline=800 )
        search = AdaptiveGridSmoothLifeSearch( sphere, bounds=[ ( -1.0, 1.0 ), ( -1.0, 1.0 ) ], agsls_config=agsls )
        self.assertEqual( search._effective_zoom_limit( 400, 5 ), 5 )
        self.assertEqual( search._effective_zoom_limit( 800, 5 ), 5 )
        self.assertEqual( search._effective_zoom_limit( None, 5 ), 5 )

    def test_zoom_limit_scaling_disabled_when_baseline_none( self ) -> None:
        agsls = AGSLSConfig( max_zoom_cycles=5, zoom_cycles_budget_baseline=None )
        search = AdaptiveGridSmoothLifeSearch( sphere, bounds=[ ( -1.0, 1.0 ), ( -1.0, 1.0 ) ], agsls_config=agsls )
        self.assertEqual( search._effective_zoom_limit( 6400, 5 ), 5 )
        self.assertEqual( search._effective_zoom_limit( 12800, 5 ), 5 )

    def test_late_stage_trigger_fires_on_plateau_and_weak_shrink( self ) -> None:
        smoothlife = SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=6, snapshot_interval=1, preset="search" )
        agsls = AGSLSConfig( max_zoom_cycles=3, late_stage_max_rounds=2, late_stage_eval_batch=3, max_evaluations=120 )
        search = AdaptiveGridSmoothLifeSearch( sphere, bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ], smoothlife_config=smoothlife, agsls_config=agsls )
        search.reset( seed=5 )
        state = search.engine.state
        self.assertIsNotNone( state )
        basin = _make_basin(
            search.engine.config.grid_shape,
            state.bounds,
            4,
            4,
            12,
            12,
            score=0.85,
            alive_density=0.7,
            best_value=float( state.local_best_value ),
        )
        basin.incumbent_in_envelope = True
        weak_bounds = state.bounds.copy()
        weak_bounds[ 0, 0 ] += 0.25
        weak_bounds[ 0, 1 ] -= 0.25
        weak_bounds[ 1, 0 ] += 0.25
        weak_bounds[ 1, 1 ] -= 0.25
        late_state = {
            "late_stage_mode": False,
            "plateau": False,
            "small_box": False,
            "late": False,
            "intensification_rounds": 0,
        }
        should_intensify, projected_shrink_ratio, exit_reason = search._should_intensify(
            box_id=0,
            basin=basin,
            old_bounds=state.bounds.copy(),
            new_bounds=weak_bounds,
            late_stage_state=late_state,
        )
        self.assertTrue( should_intensify )
        self.assertEqual( exit_reason, "weak_shrink" )
        self.assertGreaterEqual( projected_shrink_ratio, agsls.late_stage_min_shrink_ratio )

        search._box_round_counts[ 0 ] = 1
        search.engine._last_global_improvement = 0.0
        search.engine._last_stage_improvement = 0.0
        plateau_state = search._late_stage_state( box_id=0, active_limit=3, bounds=state.bounds.copy() )
        self.assertTrue( bool( plateau_state[ "plateau" ] ) )
        should_intensify, _, exit_reason = search._should_intensify(
            box_id=0,
            basin=basin,
            old_bounds=state.bounds.copy(),
            new_bounds=weak_bounds,
            late_stage_state=plateau_state,
        )
        self.assertTrue( should_intensify )
        self.assertIn( exit_reason, ( "weak_shrink", "late_box" ) )

    def test_late_stage_cadence_lowers_eval_batch_only_when_qualifying( self ) -> None:
        search = AdaptiveGridSmoothLifeSearch(
            sphere,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=16, snapshot_interval=1, preset="search" ),
            agsls_config=AGSLSConfig( late_stage_eval_batch=8 ),
        )
        self.assertEqual(
            search._late_stage_step_batch(
                16,
                late_stage_state={
                    "late_stage_mode": False,
                    "plateau": False,
                    "small_box": False,
                    "late": False,
                    "intensification_rounds": 0,
                },
            ),
            16,
        )
        self.assertEqual(
            search._late_stage_step_batch(
                16,
                late_stage_state={
                    "late_stage_mode": True,
                    "plateau": True,
                    "small_box": False,
                    "late": False,
                    "intensification_rounds": 0,
                },
            ),
            8,
        )

    def test_direct_world_point_evaluation_updates_best_without_touching_grid_cache( self ) -> None:
        search = AdaptiveGridSmoothLifeSearch(
            sphere,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=4, snapshot_interval=1, preset="search" ),
            agsls_config=AGSLSConfig(),
        )
        search.reset( seed=12 )
        state = search.engine.state
        self.assertIsNotNone( state )
        baseline_evaluations = int( state.evaluations )
        baseline_grid_evaluations = int( state.evaluated_mask.sum() )
        probe = np.asarray( [ [ 0.0, 0.0 ] ], dtype=float )
        first = search._evaluate_arbitrary_points( probe )
        second = search._evaluate_arbitrary_points( probe )
        self.assertTrue( np.allclose( first, second ) )
        self.assertEqual( int( state.evaluations ), baseline_evaluations + 2 )
        self.assertEqual( int( state.evaluated_mask.sum() ), baseline_grid_evaluations )
        self.assertTrue( np.allclose( state.best_point, probe[ 0 ] ) )
        self.assertAlmostEqual( float( state.best_value ), 0.0, places=12 )

    def test_translation_config_validates_fraction_knobs( self ) -> None:
        config = AGSLSConfig(
            late_stage_translation_enabled=True,
            late_stage_translation_step_fraction=0.25,
            late_stage_translation_min_offset_fraction=0.10,
        )
        self.assertTrue( bool( config.late_stage_translation_enabled ) )
        invalid_cases = (
            {"late_stage_translation_step_fraction": 0.0},
            {"late_stage_translation_step_fraction": 1.5},
            {"late_stage_translation_min_offset_fraction": 0.0},
            {"late_stage_translation_min_offset_fraction": 1.5},
        )
        for kwargs in invalid_cases:
            with self.subTest( kwargs=kwargs ):
                with self.assertRaises( ValueError ):
                    AGSLSConfig( **kwargs )

    def test_translation_target_uses_weighted_elites_deterministically( self ) -> None:
        search = AdaptiveGridSmoothLifeSearch(
            sphere,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=4, snapshot_interval=1, preset="search" ),
            agsls_config=AGSLSConfig( late_stage_elite_k=3 ),
        )
        search.reset( seed=21 )
        state = search.engine.state
        self.assertIsNotNone( state )
        state.evaluated_mask.fill( False )
        state.objective_values.fill( np.nan )
        basin = _make_basin(
            search.engine.config.grid_shape,
            state.bounds,
            6,
            6,
            12,
            12,
            score=0.9,
            alive_density=0.7,
            best_value=0.0,
        )
        elite_cells = (
            ( 8, 8, 0.2 ),
            ( 7, 7, 0.1 ),
            ( 6, 6, 0.1 ),
        )
        for row, col, value in elite_cells:
            state.evaluated_mask[ row, col ] = True
            state.objective_values[ row, col ] = value
        target, kind = search._late_stage_translation_target( basin )
        expected_points = np.asarray(
            [
                search.engine._pixel_center( 6, 6, state.bounds ),
                search.engine._pixel_center( 7, 7, state.bounds ),
                search.engine._pixel_center( 8, 8, state.bounds ),
            ],
            dtype=float,
        )
        expected = np.average( expected_points, axis=0, weights=np.asarray( [ 3.0, 2.0, 1.0 ], dtype=float ) )
        self.assertEqual( kind, "elite_weighted" )
        self.assertTrue( np.allclose( target, expected ) )

    def test_translation_target_falls_back_to_basin_best_then_centroid( self ) -> None:
        search = AdaptiveGridSmoothLifeSearch(
            sphere,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=4, snapshot_interval=1, preset="search" ),
            agsls_config=AGSLSConfig(),
        )
        search.reset( seed=22 )
        state = search.engine.state
        self.assertIsNotNone( state )
        state.evaluated_mask.fill( False )
        basin = _make_basin(
            search.engine.config.grid_shape,
            state.bounds,
            4,
            4,
            10,
            10,
            score=0.8,
            alive_density=0.7,
            best_value=0.0,
        )
        basin.basin_best_point = np.asarray( [ 1.5, -2.0 ], dtype=float )
        target, kind = search._late_stage_translation_target( basin )
        self.assertEqual( kind, "basin_best" )
        self.assertTrue( np.allclose( target, basin.basin_best_point ) )
        basin.basin_best_point = None
        target, kind = search._late_stage_translation_target( basin )
        self.assertEqual( kind, "centroid" )
        self.assertTrue( np.allclose( target, basin.centroid_world ) )

    def test_translation_returns_none_when_dense_group_offset_is_too_small( self ) -> None:
        search = AdaptiveGridSmoothLifeSearch(
            sphere,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=4, snapshot_interval=1, preset="search" ),
            agsls_config=AGSLSConfig(
                late_stage_translation_enabled=True,
                late_stage_translation_step_fraction=0.25,
                late_stage_translation_min_offset_fraction=0.50,
            ),
        )
        search.reset( seed=23 )
        state = search.engine.state
        self.assertIsNotNone( state )
        basin = _make_basin(
            search.engine.config.grid_shape,
            state.bounds,
            8,
            8,
            14,
            14,
            score=0.8,
            alive_density=0.7,
            best_value=float( state.local_best_value ),
        )
        basin.basin_best_point = np.asarray( [ 1.0, 0.0 ], dtype=float )
        standard_shrink = state.bounds.copy()
        standard_shrink[ 0, 0 ] += 0.5
        standard_shrink[ 0, 1 ] -= 0.5
        standard_shrink[ 1, 0 ] += 0.5
        standard_shrink[ 1, 1 ] -= 0.5
        self.assertIsNone( search._run_late_stage_translation( basin, state.bounds.copy(), standard_shrink, "weak_shrink" ) )

    def test_translation_clips_near_edges_and_preserves_incumbent( self ) -> None:
        search = AdaptiveGridSmoothLifeSearch(
            sphere,
            bounds=[ ( 0.0, 10.0 ), ( 0.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=4, snapshot_interval=1, preset="search" ),
            agsls_config=AGSLSConfig(
                late_stage_translation_enabled=True,
                late_stage_translation_step_fraction=0.35,
                late_stage_translation_min_offset_fraction=0.05,
            ),
        )
        search.reset( seed=24 )
        state = search.engine.state
        self.assertIsNotNone( state )
        incumbent = np.asarray( [ 0.75, 5.0 ], dtype=float )
        incumbent_value = sphere( incumbent )
        state.best_point = incumbent.copy()
        state.best_value = incumbent_value
        state.local_best_point = incumbent.copy()
        state.local_best_value = incumbent_value
        state.box_best_point = incumbent.copy()
        state.box_best_value = incumbent_value
        basin = _make_basin(
            search.engine.config.grid_shape,
            state.bounds,
            18,
            18,
            24,
            24,
            score=0.9,
            alive_density=0.8,
            best_value=10.0,
        )
        basin.basin_best_point = np.asarray( [ 9.75, 5.0 ], dtype=float )
        basin.better_than_incumbent = False
        standard_shrink = np.asarray( [ [ 7.5, 9.5 ], [ 4.0, 6.0 ] ], dtype=float )
        result = search._run_late_stage_translation( basin, state.bounds.copy(), standard_shrink, "weak_shrink" )
        self.assertIsNotNone( result )
        refined_bounds, summary = result
        self.assertTrue( bool( summary[ "translation_ran" ] ) )
        self.assertEqual( summary[ "translation_target_kind" ], "basin_best" )
        self.assertTrue( np.all( refined_bounds[ :, 0 ] >= state.bounds[ :, 0 ] ) )
        self.assertTrue( np.all( refined_bounds[ :, 1 ] <= state.bounds[ :, 1 ] ) )
        self.assertTrue( bool( np.all( incumbent >= refined_bounds[ :, 0 ] ) and np.all( incumbent <= refined_bounds[ :, 1 ] ) ) )
        applied = np.asarray( summary[ "translation_applied_vector" ], dtype=float )
        caps = search.agsls_config.late_stage_translation_step_fraction * ( state.bounds[ :, 1 ] - state.bounds[ :, 0 ] )
        self.assertTrue( np.all( np.abs( applied ) <= caps + 1e-12 ) )

    def test_translation_accepts_zoom_and_skips_intensification( self ) -> None:
        def objective( point: np.ndarray ) -> float:
            return -float( np.asarray( point, dtype=float )[ 0 ] )

        search = AdaptiveGridSmoothLifeSearch(
            objective,
            bounds=[ ( 0.0, 10.0 ), ( 0.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=8, snapshot_interval=1, preset="search" ),
            agsls_config=AGSLSConfig(
                max_zoom_cycles=1,
                max_evaluations=200,
                late_stage_translation_enabled=True,
                late_stage_translation_step_fraction=0.45,
                late_stage_translation_min_offset_fraction=0.08,
                late_stage_microgrid_enabled=False,
                zoom_padding=0.0,
            ),
        )
        search.reset( seed=25 )
        state = search.engine.state
        self.assertIsNotNone( state )
        incumbent = np.asarray( [ 1.0, 5.0 ], dtype=float )
        incumbent_value = objective( incumbent )
        state.best_point = incumbent.copy()
        state.best_value = incumbent_value
        state.local_best_point = incumbent.copy()
        state.local_best_value = incumbent_value
        state.box_best_point = incumbent.copy()
        state.box_best_value = incumbent_value
        search.engine._reset_improvement_trackers()
        basin = _make_basin(
            search.engine.config.grid_shape,
            state.bounds,
            16,
            16,
            24,
            24,
            score=0.95,
            alive_density=0.8,
            best_value=objective( np.asarray( [ 9.0, 5.0 ], dtype=float ) ),
        )
        basin.basin_best_point = np.asarray( [ 9.0, 5.0 ], dtype=float )
        basin.incumbent_in_envelope = True
        basin.better_than_incumbent = True
        real_remap = search.engine.remap_to_bounds
        search.engine.remap_to_bounds = MagicMock( wraps=real_remap )
        search.engine.explore_top_pixels = MagicMock( return_value=0 )
        weak_bounds = np.asarray( [ [ 0.25, 9.75 ], [ 0.25, 9.75 ] ], dtype=float )
        late_stage_state = {
            "late_stage_mode": False,
            "plateau": False,
            "small_box": False,
            "late": False,
            "intensification_rounds": 0,
        }
        with patch.object( search, "_late_stage_state", return_value=late_stage_state ):
            with patch.object( search, "_persistence_map", return_value=np.zeros( search.engine.config.grid_shape ) ):
                with patch.object( search, "_rank_basins", return_value=[ basin ] ):
                    with patch.object( search, "_probe_basins", return_value=None ):
                        with patch.object( search, "_padded_bounds", return_value=weak_bounds ):
                            accepted = search.step()
        self.assertTrue( accepted )
        search.engine.remap_to_bounds.assert_called_once()
        self.assertEqual( search._decision_reason_counts.get( "late_stage_intensify", 0 ), 0 )
        self.assertEqual( search._decision_trace[ -1 ][ "decision_reason" ], "late_stage_translate_zoom" )
        diagnostics = search.zoom_events[ 0 ].diagnostics
        self.assertTrue( bool( diagnostics.get( "translation_ran" ) ) )
        self.assertEqual( diagnostics.get( "decision_reason" ), "late_stage_translate_zoom" )
        self.assertEqual( diagnostics.get( "late_stage_exit_reason" ), "translated_zoom" )
        self.assertFalse( bool( diagnostics.get( "microgrid_ran" ) ) )

    def test_microgrid_still_runs_when_translation_does_not_fire( self ) -> None:
        target = np.asarray( [ 2.25, -0.5 ], dtype=float )

        def controlled_objective( point: np.ndarray ) -> float:
            return float( np.sum( ( np.asarray( point, dtype=float ) - target ) ** 2 ) )

        search = AdaptiveGridSmoothLifeSearch(
            controlled_objective,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=4, snapshot_interval=1, preset="search" ),
            agsls_config=AGSLSConfig(
                max_zoom_cycles=1,
                max_evaluations=200,
                late_stage_microgrid_enabled=True,
                late_stage_microgrid_resolution=5,
                late_stage_microgrid_centers=2,
                late_stage_microgrid_side_fraction=0.20,
                late_stage_translation_enabled=True,
                late_stage_translation_step_fraction=0.25,
                late_stage_translation_min_offset_fraction=0.10,
            ),
        )
        search.reset( seed=26 )
        state = search.engine.state
        self.assertIsNotNone( state )
        incumbent = np.asarray( [ 1.0, -0.5 ], dtype=float )
        incumbent_value = controlled_objective( incumbent )
        state.best_point = incumbent.copy()
        state.best_value = incumbent_value
        state.local_best_point = incumbent.copy()
        state.local_best_value = incumbent_value
        state.box_best_point = incumbent.copy()
        state.box_best_value = incumbent_value
        search.engine._reset_improvement_trackers()
        basin = _make_basin(
            search.engine.config.grid_shape,
            state.bounds,
            14,
            14,
            20,
            20,
            score=0.95,
            alive_density=0.8,
            best_value=controlled_objective( np.asarray( [ 6.0, 6.0 ], dtype=float ) ),
        )
        basin.basin_best_point = np.asarray( [ 6.0, 6.0 ], dtype=float )
        basin.basin_best_value = controlled_objective( basin.basin_best_point )
        basin.better_than_incumbent = False
        search.engine.explore_top_pixels = MagicMock( return_value=0 )
        search.engine.remap_to_bounds = MagicMock()
        late_stage_state = {
            "late_stage_mode": True,
            "plateau": True,
            "small_box": False,
            "late": False,
            "intensification_rounds": 0,
        }
        with patch.object( search, "_late_stage_state", return_value=late_stage_state ):
            with patch.object( search, "_persistence_map", return_value=np.zeros( search.engine.config.grid_shape ) ):
                with patch.object( search, "_rank_basins", return_value=[ basin ] ):
                    with patch.object( search, "_probe_basins", return_value=None ):
                        with patch.object( search, "_run_late_stage_translation", return_value=None ) as translation_mock:
                            accepted = search.step()
        self.assertTrue( accepted )
        translation_mock.assert_called_once()
        diagnostics = search.zoom_events[ 0 ].diagnostics
        self.assertFalse( bool( diagnostics.get( "translation_ran" ) ) )
        self.assertTrue( bool( diagnostics.get( "microgrid_ran" ) ) )

    def test_translation_enabled_beats_disabled_on_controlled_late_stage_problem( self ) -> None:
        def objective( point: np.ndarray ) -> float:
            return -float( np.asarray( point, dtype=float )[ 0 ] )

        def build_search( enabled: bool ) -> AdaptiveGridSmoothLifeSearch:
            search = AdaptiveGridSmoothLifeSearch(
                objective,
                bounds=[ ( 0.0, 10.0 ), ( 0.0, 10.0 ) ],
                smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=8, snapshot_interval=1, preset="search" ),
                agsls_config=AGSLSConfig(
                    max_zoom_cycles=1,
                    max_evaluations=200,
                    late_stage_translation_enabled=enabled,
                    late_stage_translation_step_fraction=0.45,
                    late_stage_translation_min_offset_fraction=0.08,
                    late_stage_microgrid_enabled=False,
                    zoom_padding=0.0,
                ),
            )
            search.reset( seed=27 )
            state = search.engine.state
            self.assertIsNotNone( state )
            incumbent = np.asarray( [ 0.0, 5.0 ], dtype=float )
            incumbent_value = objective( incumbent )
            state.best_point = incumbent.copy()
            state.best_value = incumbent_value
            state.local_best_point = incumbent.copy()
            state.local_best_value = incumbent_value
            state.box_best_point = incumbent.copy()
            state.box_best_value = incumbent_value
            search.engine._reset_improvement_trackers()
            return search

        late_stage_state = {
            "late_stage_mode": False,
            "plateau": False,
            "small_box": False,
            "late": False,
            "intensification_rounds": 0,
        }
        enabled_search = build_search( True )
        disabled_search = build_search( False )
        enabled_state = enabled_search.engine.state
        disabled_state = disabled_search.engine.state
        self.assertIsNotNone( enabled_state )
        self.assertIsNotNone( disabled_state )
        basin_enabled = _make_basin(
            enabled_search.engine.config.grid_shape,
            enabled_state.bounds,
            16,
            16,
            24,
            24,
            score=0.95,
            alive_density=0.8,
            best_value=objective( np.asarray( [ 9.0, 5.0 ], dtype=float ) ),
        )
        basin_enabled.basin_best_point = np.asarray( [ 9.0, 5.0 ], dtype=float )
        basin_enabled.incumbent_in_envelope = True
        basin_enabled.better_than_incumbent = True
        basin_disabled = _make_basin(
            disabled_search.engine.config.grid_shape,
            disabled_state.bounds,
            16,
            16,
            24,
            24,
            score=0.95,
            alive_density=0.8,
            best_value=objective( np.asarray( [ 9.0, 5.0 ], dtype=float ) ),
        )
        basin_disabled.basin_best_point = np.asarray( [ 9.0, 5.0 ], dtype=float )
        basin_disabled.incumbent_in_envelope = True
        basin_disabled.better_than_incumbent = True
        disabled_search.engine.explore_top_pixels = MagicMock( return_value=0 )
        weak_bounds = np.asarray( [ [ 0.25, 9.75 ], [ 0.25, 9.75 ] ], dtype=float )
        with patch.object( enabled_search, "_late_stage_state", return_value=late_stage_state ):
            with patch.object( enabled_search, "_persistence_map", return_value=np.zeros( enabled_search.engine.config.grid_shape ) ):
                with patch.object( enabled_search, "_rank_basins", return_value=[ basin_enabled ] ):
                    with patch.object( enabled_search, "_probe_basins", return_value=None ):
                        with patch.object( enabled_search, "_padded_bounds", return_value=weak_bounds ):
                            enabled_search.step()
        with patch.object( disabled_search, "_late_stage_state", return_value=late_stage_state ):
            with patch.object( disabled_search, "_persistence_map", return_value=np.zeros( disabled_search.engine.config.grid_shape ) ):
                with patch.object( disabled_search, "_rank_basins", return_value=[ basin_disabled ] ):
                    with patch.object( disabled_search, "_probe_basins", return_value=None ):
                        with patch.object( disabled_search, "_padded_bounds", return_value=weak_bounds ):
                            disabled_search.step()
        self.assertLess( float( enabled_state.best_value ), float( disabled_state.best_value ) )
        self.assertTrue( bool( enabled_search._decision_trace[ -1 ][ "translation_ran" ] ) )
        self.assertFalse( bool( disabled_search._decision_trace[ -1 ][ "translation_ran" ] ) )

    def test_microgrid_candidates_dedupe_incumbent_and_clip_near_edges( self ) -> None:
        agsls = AGSLSConfig(
            late_stage_microgrid_enabled=True,
            late_stage_microgrid_centers=3,
            late_stage_microgrid_side_fraction=0.30,
        )
        search = AdaptiveGridSmoothLifeSearch(
            sphere,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=4, snapshot_interval=1, preset="search" ),
            agsls_config=agsls,
        )
        search.reset( seed=4 )
        state = search.engine.state
        self.assertIsNotNone( state )
        shared_point = np.asarray( [ 9.5, 9.5 ], dtype=float )
        state.best_point = shared_point.copy()
        state.best_value = 0.0
        state.local_best_point = shared_point.copy()
        state.local_best_value = 0.0
        state.box_best_point = shared_point.copy()
        state.box_best_value = 0.0
        state.evaluated_mask[ 23, 23 ] = True
        state.objective_values[ 23, 23 ] = 0.5
        state.evaluated_mask[ 22, 23 ] = True
        state.objective_values[ 22, 23 ] = 0.75
        basin = _make_basin(
            search.engine.config.grid_shape,
            state.bounds,
            20,
            20,
            24,
            24,
            score=0.9,
            alive_density=0.7,
            best_value=0.0,
        )
        basin.basin_best_point = shared_point.copy()
        candidates = search._late_stage_microgrid_candidates( basin, state.bounds.copy() )
        self.assertEqual( [ str( candidate[ "kind" ] ) for candidate in candidates ][ 0 ], "basin_best" )
        self.assertNotIn( "incumbent", [ str( candidate[ "kind" ] ) for candidate in candidates ] )
        self.assertLessEqual( len( candidates ), agsls.late_stage_microgrid_centers )
        sample_bounds = search._late_stage_microgrid_sample_bounds( shared_point, state.bounds.copy() )
        self.assertTrue( np.all( sample_bounds[ :, 0 ] >= state.bounds[ :, 0 ] ) )
        self.assertTrue( np.all( sample_bounds[ :, 1 ] <= state.bounds[ :, 1 ] ) )
        min_widths = search._minimum_zoom_widths( state.bounds.copy() )
        self.assertTrue( np.all( ( sample_bounds[ :, 1 ] - sample_bounds[ :, 0 ] ) >= min_widths - 1e-12 ) )

    def test_disabled_or_budget_limited_microgrid_returns_none( self ) -> None:
        disabled = AdaptiveGridSmoothLifeSearch(
            sphere,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=4, snapshot_interval=1, preset="search" ),
            agsls_config=AGSLSConfig( late_stage_microgrid_enabled=False ),
        )
        disabled.reset( seed=8 )
        disabled_state = disabled.engine.state
        self.assertIsNotNone( disabled_state )
        basin = _make_basin(
            disabled.engine.config.grid_shape,
            disabled_state.bounds,
            4,
            4,
            10,
            10,
            score=0.8,
            alive_density=0.7,
            best_value=float( disabled_state.local_best_value ),
        )
        self.assertIsNone( disabled._run_late_stage_microgrid( basin, disabled_state.bounds.copy() ) )

        enabled = AdaptiveGridSmoothLifeSearch(
            sphere,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=4, snapshot_interval=1, preset="search" ),
            agsls_config=AGSLSConfig(
                late_stage_microgrid_enabled=True,
                late_stage_microgrid_resolution=5,
                late_stage_microgrid_centers=3,
            ),
        )
        enabled.reset( seed=8 )
        enabled_state = enabled.engine.state
        self.assertIsNotNone( enabled_state )
        enabled.active_max_evaluations = int( enabled_state.evaluations ) + 10
        try:
            self.assertIsNone( enabled._run_late_stage_microgrid( basin, enabled_state.bounds.copy() ) )
        finally:
            enabled.active_max_evaluations = None

    def test_microgrid_refines_late_stage_zoom_around_incumbent( self ) -> None:
        target = np.asarray( [ 2.25, -0.5 ], dtype=float )

        def controlled_objective( point: np.ndarray ) -> float:
            return float( np.sum( ( np.asarray( point, dtype=float ) - target ) ** 2 ) )

        agsls = AGSLSConfig(
            max_zoom_cycles=1,
            max_evaluations=200,
            late_stage_microgrid_enabled=True,
            late_stage_microgrid_resolution=5,
            late_stage_microgrid_centers=2,
            late_stage_microgrid_side_fraction=0.20,
        )
        search = AdaptiveGridSmoothLifeSearch(
            controlled_objective,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=4, snapshot_interval=1, preset="search" ),
            agsls_config=agsls,
        )
        search.reset( seed=9 )
        state = search.engine.state
        self.assertIsNotNone( state )
        incumbent = np.asarray( [ 1.0, -0.5 ], dtype=float )
        incumbent_value = controlled_objective( incumbent )
        state.best_point = incumbent.copy()
        state.best_value = incumbent_value
        state.local_best_point = incumbent.copy()
        state.local_best_value = incumbent_value
        state.box_best_point = incumbent.copy()
        state.box_best_value = incumbent_value
        search.engine._reset_improvement_trackers()
        basin = _make_basin(
            search.engine.config.grid_shape,
            state.bounds,
            14,
            14,
            20,
            20,
            score=0.95,
            alive_density=0.8,
            best_value=controlled_objective( np.asarray( [ 6.0, 6.0 ], dtype=float ) ),
        )
        basin.basin_best_point = np.asarray( [ 6.0, 6.0 ], dtype=float )
        basin.basin_best_value = controlled_objective( basin.basin_best_point )
        basin.better_than_incumbent = False
        search.engine.explore_top_pixels = MagicMock( return_value=0 )
        search.engine.remap_to_bounds = MagicMock()
        late_stage_state = {
            "late_stage_mode": True,
            "plateau": True,
            "small_box": False,
            "late": False,
            "intensification_rounds": 0,
        }
        with patch.object( search, "_late_stage_state", return_value=late_stage_state ):
            with patch.object( search, "_persistence_map", return_value=np.zeros( search.engine.config.grid_shape ) ):
                with patch.object( search, "_rank_basins", return_value=[ basin ] ):
                    with patch.object( search, "_probe_basins", return_value=None ):
                        accepted = search.step()
        self.assertTrue( accepted )
        self.assertEqual( len( search.zoom_events ), 1 )
        self.assertAlmostEqual( float( state.best_value ), 0.0, places=12 )
        diagnostics = search.zoom_events[ 0 ].diagnostics
        self.assertTrue( bool( diagnostics.get( "microgrid_ran" ) ) )
        self.assertEqual( diagnostics.get( "microgrid_winning_center_kind" ), "incumbent" )
        self.assertTrue( np.allclose( np.asarray( diagnostics.get( "microgrid_winning_point" ), dtype=float ), target ) )
        refined_bounds = np.asarray( diagnostics.get( "microgrid_refined_bounds" ), dtype=float )
        self.assertTrue( np.all( refined_bounds[ :, 0 ] <= target ) )
        self.assertTrue( np.all( refined_bounds[ :, 1 ] >= target ) )

    def test_enabled_microgrid_beats_disabled_on_controlled_late_stage_problem( self ) -> None:
        target = np.asarray( [ 2.25, -0.5 ], dtype=float )

        def controlled_objective( point: np.ndarray ) -> float:
            return float( np.sum( ( np.asarray( point, dtype=float ) - target ) ** 2 ) )

        def build_search( enabled: bool ) -> AdaptiveGridSmoothLifeSearch:
            search = AdaptiveGridSmoothLifeSearch(
                controlled_objective,
                bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
                smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=4, snapshot_interval=1, preset="search" ),
                agsls_config=AGSLSConfig(
                    max_zoom_cycles=1,
                    max_evaluations=200,
                    late_stage_microgrid_enabled=enabled,
                    late_stage_microgrid_resolution=5,
                    late_stage_microgrid_centers=2,
                    late_stage_microgrid_side_fraction=0.20,
                ),
            )
            search.reset( seed=10 )
            state = search.engine.state
            self.assertIsNotNone( state )
            incumbent = np.asarray( [ 1.0, -0.5 ], dtype=float )
            incumbent_value = controlled_objective( incumbent )
            state.best_point = incumbent.copy()
            state.best_value = incumbent_value
            state.local_best_point = incumbent.copy()
            state.local_best_value = incumbent_value
            state.box_best_point = incumbent.copy()
            state.box_best_value = incumbent_value
            search.engine._reset_improvement_trackers()
            return search

        late_stage_state = {
            "late_stage_mode": True,
            "plateau": True,
            "small_box": False,
            "late": False,
            "intensification_rounds": 0,
        }
        enabled_search = build_search( True )
        disabled_search = build_search( False )
        enabled_state = enabled_search.engine.state
        disabled_state = disabled_search.engine.state
        self.assertIsNotNone( enabled_state )
        self.assertIsNotNone( disabled_state )
        basin_enabled = _make_basin(
            enabled_search.engine.config.grid_shape,
            enabled_state.bounds,
            14,
            14,
            20,
            20,
            score=0.95,
            alive_density=0.8,
            best_value=controlled_objective( np.asarray( [ 6.0, 6.0 ], dtype=float ) ),
        )
        basin_enabled.basin_best_point = np.asarray( [ 6.0, 6.0 ], dtype=float )
        basin_enabled.basin_best_value = controlled_objective( basin_enabled.basin_best_point )
        basin_disabled = _make_basin(
            disabled_search.engine.config.grid_shape,
            disabled_state.bounds,
            14,
            14,
            20,
            20,
            score=0.95,
            alive_density=0.8,
            best_value=controlled_objective( np.asarray( [ 6.0, 6.0 ], dtype=float ) ),
        )
        basin_disabled.basin_best_point = np.asarray( [ 6.0, 6.0 ], dtype=float )
        basin_disabled.basin_best_value = controlled_objective( basin_disabled.basin_best_point )
        enabled_search.engine.explore_top_pixels = MagicMock( return_value=0 )
        disabled_search.engine.explore_top_pixels = MagicMock( return_value=0 )
        enabled_search.engine.remap_to_bounds = MagicMock()
        disabled_search.engine.remap_to_bounds = MagicMock()
        with patch.object( enabled_search, "_late_stage_state", return_value=late_stage_state ):
            with patch.object( enabled_search, "_persistence_map", return_value=np.zeros( enabled_search.engine.config.grid_shape ) ):
                with patch.object( enabled_search, "_rank_basins", return_value=[ basin_enabled ] ):
                    with patch.object( enabled_search, "_probe_basins", return_value=None ):
                        enabled_search.step()
        with patch.object( disabled_search, "_late_stage_state", return_value=late_stage_state ):
            with patch.object( disabled_search, "_persistence_map", return_value=np.zeros( disabled_search.engine.config.grid_shape ) ):
                with patch.object( disabled_search, "_rank_basins", return_value=[ basin_disabled ] ):
                    with patch.object( disabled_search, "_probe_basins", return_value=None ):
                        disabled_search.step()
        self.assertLess( float( enabled_state.best_value ), float( disabled_state.best_value ) )
        self.assertTrue( bool( enabled_search._decision_trace[ -1 ][ "microgrid_ran" ] ) )
        self.assertFalse( bool( disabled_search._decision_trace[ -1 ][ "microgrid_ran" ] ) )

    def test_zoom_cycles_shrink_bounds_and_become_more_frequent( self ) -> None:
        smoothlife = SmoothLifeConfig( grid_shape=( 48, 48 ), evaluations_per_step=4, objective_coupling=0.35, snapshot_interval=1, preset="search" )
        agsls = AGSLSConfig(
            max_zoom_cycles=3,
            initial_steps_per_zoom=12,
            min_steps_per_zoom=4,
            zoom_decay=0.5,
            min_basin_cells=8,
            max_evaluations=500,
        )
        search = AdaptiveGridSmoothLifeSearch( sphere, bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ], smoothlife_config=smoothlife, agsls_config=agsls )
        search.reset( seed=11 )
        run = search.run()
        self.assertGreaterEqual( len( run.zoom_events ), 1 )
        widths = [ event.new_bounds[ :, 1 ] - event.new_bounds[ :, 0 ] for event in run.zoom_events ]
        for previous, current in zip( widths, widths[ 1: ] ):
            self.assertTrue( np.all( current <= previous + 1e-12 ) )
        step_counts = [ event.steps_per_zoom for event in run.zoom_events ]
        for previous, current in zip( step_counts, step_counts[ 1: ] ):
            self.assertLessEqual( current, previous )

    def test_density_gate_can_defer_zoom( self ) -> None:
        smoothlife = SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=3, snapshot_interval=1, preset="search" )
        agsls = AGSLSConfig( max_zoom_cycles=1, min_alive_density=0.95, candidate_probe_evaluations=0, max_evaluations=100 )
        search = AdaptiveGridSmoothLifeSearch( sphere, bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ], smoothlife_config=smoothlife, agsls_config=agsls )
        search.reset( seed=1 )
        state = search.engine.state
        self.assertIsNotNone( state )
        basin = _make_basin(
            search.engine.config.grid_shape,
            state.bounds,
            2,
            2,
            6,
            6,
            score=0.9,
            alive_density=0.2,
            best_value=float( state.local_best_value ),
        )
        with patch.object( search, "_persistence_map", return_value=np.zeros( search.engine.config.grid_shape ) ):
            with patch.object( search, "_rank_basins", return_value=[ basin ] ):
                with patch.object( search, "_probe_basins", return_value=None ):
                    accepted = search.step()
        self.assertFalse( accepted )
        self.assertEqual( len( search.zoom_events ), 0 )
        self.assertEqual( search.engine.snapshots[ -1 ].metadata.get( "zoom_decision" ), "deferred_no_eligible_group" )

    def test_multi_group_stage_explores_until_cap_then_zooms( self ) -> None:
        smoothlife = SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=3, snapshot_interval=1, preset="search" )
        agsls = AGSLSConfig(
            max_zoom_cycles=1,
            candidate_probe_evaluations=2,
            undecided_stage_max_evaluations=4,
            dominance_margin=0.20,
            similarity_margin=0.05,
            max_evaluations=100,
        )
        search = AdaptiveGridSmoothLifeSearch( sphere, bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ], smoothlife_config=smoothlife, agsls_config=agsls )
        search.reset( seed=2 )
        state = search.engine.state
        self.assertIsNotNone( state )
        basin_a = _make_basin(
            search.engine.config.grid_shape,
            state.bounds,
            2,
            2,
            7,
            7,
            score=0.70,
            alive_density=0.7,
            best_value=float( state.local_best_value ),
        )
        basin_b = _make_basin(
            search.engine.config.grid_shape,
            state.bounds,
            10,
            10,
            15,
            15,
            score=0.60,
            alive_density=0.68,
            best_value=float( state.local_best_value ),
        )
        search.engine.explore_top_pixels = MagicMock( side_effect=[ 2, 2 ] )
        search.engine.remap_to_bounds = MagicMock()
        with patch.object( search, "_persistence_map", return_value=np.zeros( search.engine.config.grid_shape ) ):
            with patch.object( search, "_rank_basins", return_value=[ basin_a, basin_b ] ):
                with patch.object( search, "_probe_basins", return_value=None ):
                    accepted = search.step()
        self.assertTrue( accepted )
        self.assertGreaterEqual( search.engine.explore_top_pixels.call_count, 2 )
        self.assertEqual( len( search.zoom_events ), 1 )
        self.assertEqual( search.engine.snapshots[ -1 ].metadata.get( "zoom_decision" ), "accepted" )
        self.assertEqual( search.engine.snapshots[ -1 ].metadata.get( "zoom_reason" ), "exploration_cap" )

    def test_similar_groups_probe_once_before_zooming( self ) -> None:
        smoothlife = SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=3, snapshot_interval=1, preset="search" )
        agsls = AGSLSConfig(
            max_zoom_cycles=1,
            candidate_probe_evaluations=2,
            undecided_stage_max_evaluations=6,
            dominance_margin=0.20,
            similarity_margin=0.05,
            max_evaluations=100,
        )
        search = AdaptiveGridSmoothLifeSearch( sphere, bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ], smoothlife_config=smoothlife, agsls_config=agsls )
        search.reset( seed=3 )
        state = search.engine.state
        self.assertIsNotNone( state )
        basin_a = _make_basin(
            search.engine.config.grid_shape,
            state.bounds,
            2,
            2,
            7,
            7,
            score=0.70,
            alive_density=0.7,
            best_value=float( state.local_best_value ),
        )
        basin_b = _make_basin(
            search.engine.config.grid_shape,
            state.bounds,
            10,
            10,
            15,
            15,
            score=0.68,
            alive_density=0.68,
            best_value=float( state.local_best_value ),
        )
        search.engine.explore_top_pixels = MagicMock( return_value=2 )
        search.engine.remap_to_bounds = MagicMock()
        with patch.object( search, "_persistence_map", return_value=np.zeros( search.engine.config.grid_shape ) ):
            with patch.object( search, "_rank_basins", return_value=[ basin_a, basin_b ] ):
                with patch.object( search, "_probe_basins", return_value=None ):
                    accepted = search.step()
        self.assertTrue( accepted )
        self.assertEqual( search.engine.explore_top_pixels.call_count, 1 )
        self.assertEqual( search.engine.explore_top_pixels.call_args.args[ 0 ], 2 )
        self.assertEqual( search.engine.snapshots[ -1 ].metadata.get( "zoom_reason" ), "similar_groups" )

    def test_deferred_no_shrink_routes_into_late_stage_intensification( self ) -> None:
        smoothlife = SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=6, snapshot_interval=1, preset="search" )
        agsls = AGSLSConfig( max_zoom_cycles=1, late_stage_max_rounds=1, late_stage_eval_batch=2, max_evaluations=100 )
        search = AdaptiveGridSmoothLifeSearch( sphere, bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ], smoothlife_config=smoothlife, agsls_config=agsls )
        search.reset( seed=4 )
        state = search.engine.state
        self.assertIsNotNone( state )
        basin = _make_basin(
            search.engine.config.grid_shape,
            state.bounds,
            2,
            2,
            8,
            8,
            score=0.9,
            alive_density=0.7,
            best_value=float( state.local_best_value ),
        )
        basin.incumbent_in_envelope = True
        search.engine.explore_top_pixels = MagicMock( return_value=2 )
        search.engine.remap_to_bounds = MagicMock()
        with patch.object( search, "_persistence_map", return_value=np.zeros( search.engine.config.grid_shape ) ):
            with patch.object( search, "_rank_basins", return_value=[ basin ] ):
                with patch.object( search, "_probe_basins", return_value=None ):
                    with patch.object( search, "_padded_bounds", return_value=state.bounds.copy() ):
                        accepted = search.step()
        self.assertFalse( accepted )
        search.engine.remap_to_bounds.assert_not_called()
        self.assertEqual( search.engine.snapshots[ -1 ].metadata.get( "zoom_decision" ), "late_stage_intensify" )
        self.assertEqual( search.engine.snapshots[ -1 ].metadata.get( "zoom_reason" ), "late_stage_intensify" )
        self.assertEqual( search._decision_trace[ -1 ][ "decision_reason" ], "late_stage_intensify" )
        self.assertTrue( bool( search._decision_trace[ -1 ][ "late_stage_mode" ] ) )
        self.assertIn( "projected_shrink_ratio", search._decision_trace[ -1 ] )
        self.assertIn( "focus_mask_coverage", search._decision_trace[ -1 ] )

    def test_late_stage_resume_zoom_records_diagnostics( self ) -> None:
        smoothlife = SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=6, snapshot_interval=1, preset="search" )
        agsls = AGSLSConfig( max_zoom_cycles=1, late_stage_max_rounds=1, late_stage_eval_batch=2, max_evaluations=100 )
        search = AdaptiveGridSmoothLifeSearch( sphere, bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ], smoothlife_config=smoothlife, agsls_config=agsls )
        search.reset( seed=6 )
        state = search.engine.state
        self.assertIsNotNone( state )
        basin = _make_basin(
            search.engine.config.grid_shape,
            state.bounds,
            4,
            4,
            12,
            12,
            score=0.95,
            alive_density=0.75,
            best_value=float( state.local_best_value ),
        )
        basin.better_than_incumbent = True
        search._late_stage_round_counts[ 0 ] = 1
        smaller_bounds = state.bounds.copy()
        smaller_bounds[ 0, 0 ] += 2.0
        smaller_bounds[ 0, 1 ] -= 2.0
        smaller_bounds[ 1, 0 ] += 2.0
        smaller_bounds[ 1, 1 ] -= 2.0
        search.engine.remap_to_bounds = MagicMock()
        with patch.object( search, "_persistence_map", return_value=np.zeros( search.engine.config.grid_shape ) ):
            with patch.object( search, "_rank_basins", return_value=[ basin ] ):
                with patch.object( search, "_probe_basins", return_value=None ):
                    with patch.object( search, "_padded_bounds", return_value=smaller_bounds ):
                        accepted = search.step()
        self.assertTrue( accepted )
        self.assertEqual( len( search.zoom_events ), 1 )
        self.assertEqual( search.zoom_events[ 0 ].diagnostics.get( "decision_reason" ), "late_stage_resume_zoom" )
        self.assertEqual( search.zoom_events[ 0 ].diagnostics.get( "late_stage_exit_reason" ), "resume_zoom" )
        self.assertEqual( search._decision_trace[ -1 ][ "decision_reason" ], "late_stage_resume_zoom" )
        self.assertIn( "projected_shrink_ratio", search.zoom_events[ 0 ].diagnostics )
        self.assertIn( "focus_mask_coverage", search.zoom_events[ 0 ].diagnostics )

    def test_known_optimum_stays_inside_zoom_bounds_for_observed_failed_seeds( self ) -> None:
        smoothlife = SmoothLifeConfig( grid_shape=( 64, 64 ), evaluations_per_step=8, objective_coupling=0.35, snapshot_interval=9999, preset="search" )
        agsls = AGSLSConfig( max_zoom_cycles=5, max_evaluations=1600 )
        cases = (
            ( "sphere", sphere, [ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ], ( 0, 1, 8, 13, 15 ) ),
            ( "ackley", ackley, [ ( -5.0, 5.0 ), ( -5.0, 5.0 ) ], ( 0, 3, 6, 11, 15, 16 ) ),
        )
        optimum = np.asarray( [ 0.0, 0.0 ], dtype=float )
        for objective_name, objective, bounds, seeds in cases:
            for seed in seeds:
                with self.subTest( objective=objective_name, seed=seed ):
                    search = AdaptiveGridSmoothLifeSearch( objective, bounds=bounds, smoothlife_config=smoothlife, agsls_config=agsls )
                    search.reset( seed=seed )
                    run = search.run( evaluations=1600 )
                    self.assertGreater( len( run.zoom_events ), 0 )
                    for event in run.zoom_events:
                        contains_optimum = bool( np.all( ( optimum >= event.new_bounds[ :, 0 ] ) & ( optimum <= event.new_bounds[ :, 1 ] ) ) )
                        self.assertTrue( contains_optimum )


class TestLateStagePatternSearch( unittest.TestCase ):
    def test_config_validates_knobs( self ) -> None:
        config = AGSLSConfig(
            late_stage_exploiter="pattern_search",
            late_stage_pattern_search_initial_step_fraction=0.20,
            late_stage_pattern_search_min_step_fraction=0.01,
            late_stage_pattern_search_shrink=0.6,
            late_stage_pattern_search_max_iterations=10,
            late_stage_pattern_search_max_evaluations=15,
            late_stage_pattern_search_reuse_tolerance_cells=0.4,
        )
        self.assertEqual( config.late_stage_exploiter, "pattern_search" )
        bad_kwargs = (
            {"late_stage_exploiter": "bogus"},
            {"late_stage_pattern_search_initial_step_fraction": 0.0},
            {"late_stage_pattern_search_initial_step_fraction": 1.5},
            {"late_stage_pattern_search_min_step_fraction": 0.0},
            {"late_stage_pattern_search_min_step_fraction": 0.5, "late_stage_pattern_search_initial_step_fraction": 0.1},
            {"late_stage_pattern_search_shrink": 0.0},
            {"late_stage_pattern_search_shrink": 1.0},
            {"late_stage_pattern_search_max_iterations": 0},
            {"late_stage_pattern_search_max_evaluations": 0},
            {"late_stage_pattern_search_reuse_tolerance_cells": -0.1},
            {"late_stage_pattern_search_reuse_tolerance_cells": 1.5},
        )
        for kwargs in bad_kwargs:
            with self.subTest( kwargs=kwargs ):
                with self.assertRaises( ValueError ):
                    AGSLSConfig( **kwargs )

    def test_reuse_helper_returns_cached_value_within_tolerance( self ) -> None:
        search = AdaptiveGridSmoothLifeSearch(
            sphere,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=4, preset="search" ),
            agsls_config=AGSLSConfig(),
        )
        search.reset( seed=3 )
        state = search.engine.state
        self.assertIsNotNone( state )
        row, col = 12, 12
        state.evaluated_mask[ row, col ] = True
        state.objective_values[ row, col ] = 7.5
        pixel_center = search.engine._pixel_center( row, col, state.bounds )
        cached, reused = search._reuse_evaluated_value( pixel_center, state.bounds )
        self.assertTrue( reused )
        self.assertEqual( cached, 7.5 )
        _, no_reuse = search._reuse_evaluated_value( pixel_center + np.asarray( [ 50.0, 0.0 ] ), state.bounds )
        self.assertFalse( no_reuse )
        row2, col2 = 5, 5
        self.assertFalse( bool( state.evaluated_mask[ row2, col2 ] ) )
        pixel2 = search.engine._pixel_center( row2, col2, state.bounds )
        _, miss = search._reuse_evaluated_value( pixel2, state.bounds )
        self.assertFalse( miss )

    def test_pattern_search_refines_around_incumbent( self ) -> None:
        target = np.asarray( [ 2.25, -0.5 ], dtype=float )

        def controlled_objective( point: np.ndarray ) -> float:
            return float( np.sum( ( np.asarray( point, dtype=float ) - target ) ** 2 ) )

        search = AdaptiveGridSmoothLifeSearch(
            controlled_objective,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=4, snapshot_interval=1, preset="search" ),
            agsls_config=AGSLSConfig(
                max_zoom_cycles=1,
                max_evaluations=500,
                late_stage_exploiter="pattern_search",
                late_stage_pattern_search_initial_step_fraction=0.3,
                late_stage_pattern_search_min_step_fraction=0.001,
                late_stage_pattern_search_shrink=0.5,
                late_stage_pattern_search_max_iterations=40,
                late_stage_pattern_search_max_evaluations=60,
            ),
        )
        search.reset( seed=9 )
        state = search.engine.state
        self.assertIsNotNone( state )
        incumbent = np.asarray( [ 1.0, -0.5 ], dtype=float )
        incumbent_value = controlled_objective( incumbent )
        state.best_point = incumbent.copy()
        state.best_value = incumbent_value
        state.local_best_point = incumbent.copy()
        state.local_best_value = incumbent_value
        state.box_best_point = incumbent.copy()
        state.box_best_value = incumbent_value
        search.engine._reset_improvement_trackers()
        basin = _make_basin(
            search.engine.config.grid_shape,
            state.bounds,
            14, 14, 20, 20,
            score=0.95, alive_density=0.8,
            best_value=controlled_objective( np.asarray( [ 6.0, 6.0 ], dtype=float ) ),
        )
        basin.basin_best_point = np.asarray( [ 6.0, 6.0 ], dtype=float )
        basin.basin_best_value = controlled_objective( basin.basin_best_point )
        basin.better_than_incumbent = False
        search.engine.explore_top_pixels = MagicMock( return_value=0 )
        search.engine.remap_to_bounds = MagicMock()
        late_stage_state = {
            "late_stage_mode": True,
            "plateau": True,
            "small_box": False,
            "late": False,
            "intensification_rounds": 0,
        }
        with patch.object( search, "_late_stage_state", return_value=late_stage_state ):
            with patch.object( search, "_persistence_map", return_value=np.zeros( search.engine.config.grid_shape ) ):
                with patch.object( search, "_rank_basins", return_value=[ basin ] ):
                    with patch.object( search, "_probe_basins", return_value=None ):
                        accepted = search.step()
        self.assertTrue( accepted )
        self.assertEqual( len( search.zoom_events ), 1 )
        self.assertLess( float( state.best_value ), 0.05 )
        diagnostics = search.zoom_events[ 0 ].diagnostics
        self.assertTrue( bool( diagnostics.get( "pattern_search_ran" ) ) )
        self.assertFalse( bool( diagnostics.get( "microgrid_ran" ) ) )
        self.assertEqual( diagnostics.get( "pattern_search_seed_kind" ), "incumbent" )
        refined_bounds = np.asarray( diagnostics.get( "pattern_search_refined_bounds" ), dtype=float )
        self.assertTrue( np.all( refined_bounds[ :, 0 ] <= target ) )
        self.assertTrue( np.all( refined_bounds[ :, 1 ] >= target ) )

    def test_pattern_search_reuses_nearby_evaluated_pixels( self ) -> None:
        counting = CountingObjective()
        search = AdaptiveGridSmoothLifeSearch(
            counting,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=4, snapshot_interval=1, preset="search" ),
            agsls_config=AGSLSConfig(
                max_zoom_cycles=1,
                max_evaluations=500,
                late_stage_exploiter="pattern_search",
                late_stage_pattern_search_initial_step_fraction=0.05,
                late_stage_pattern_search_min_step_fraction=0.02,
                late_stage_pattern_search_shrink=0.5,
                late_stage_pattern_search_max_iterations=4,
                late_stage_pattern_search_max_evaluations=20,
                late_stage_pattern_search_reuse_tolerance_cells=0.5,
            ),
        )
        search.reset( seed=11 )
        state = search.engine.state
        self.assertIsNotNone( state )
        seed_row, seed_col = 12, 12
        seed_point = search.engine._pixel_center( seed_row, seed_col, state.bounds )
        state.best_point = seed_point.copy()
        state.best_value = sphere( seed_point )
        state.local_best_point = seed_point.copy()
        state.local_best_value = state.best_value
        state.box_best_point = seed_point.copy()
        state.box_best_value = state.best_value
        pre_populated = 0
        for d_row, d_col in ( ( 0, 1 ), ( 0, -1 ), ( 1, 0 ), ( -1, 0 ) ):
            row = seed_row + d_row
            col = seed_col + d_col
            state.evaluated_mask[ row, col ] = True
            state.objective_values[ row, col ] = sphere( search.engine._pixel_center( row, col, state.bounds ) )
            pre_populated += 1
        search.engine._reset_improvement_trackers()
        basin = _make_basin(
            search.engine.config.grid_shape,
            state.bounds,
            seed_row - 4, seed_col - 4, seed_row + 4, seed_col + 4,
            score=0.95, alive_density=0.8,
            best_value=float( state.best_value ),
        )
        basin.basin_best_point = seed_point.copy()
        basin.basin_best_value = float( state.best_value )
        basin.better_than_incumbent = False
        calls_before = counting.count
        result = search._run_late_stage_pattern_search( basin, state.bounds.copy() )
        calls_after = counting.count
        self.assertIsNotNone( result )
        _, summary = result
        self.assertTrue( bool( summary[ "pattern_search_ran" ] ) )
        fresh = int( summary[ "pattern_search_evaluations_spent" ] )
        reused = int( summary[ "pattern_search_evaluations_reused" ] )
        self.assertGreaterEqual( reused, 1 )
        self.assertEqual( calls_after - calls_before, fresh )

    def test_pattern_search_respects_fresh_budget( self ) -> None:
        def objective( point: np.ndarray ) -> float:
            return float( np.sum( np.asarray( point, dtype=float ) ** 2 ) )

        search = AdaptiveGridSmoothLifeSearch(
            objective,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=4, snapshot_interval=1, preset="search" ),
            agsls_config=AGSLSConfig(
                max_zoom_cycles=1,
                late_stage_exploiter="pattern_search",
                late_stage_pattern_search_initial_step_fraction=0.3,
                late_stage_pattern_search_min_step_fraction=0.001,
                late_stage_pattern_search_shrink=0.5,
                late_stage_pattern_search_max_iterations=20,
                late_stage_pattern_search_max_evaluations=5,
            ),
        )
        search.reset( seed=13 )
        state = search.engine.state
        self.assertIsNotNone( state )
        seed = np.asarray( [ 1.0, 1.0 ], dtype=float )
        state.best_point = seed.copy()
        state.best_value = float( objective( seed ) )
        state.local_best_point = seed.copy()
        state.local_best_value = state.best_value
        state.box_best_point = seed.copy()
        state.box_best_value = state.best_value
        search.engine._reset_improvement_trackers()
        basin = _make_basin(
            search.engine.config.grid_shape, state.bounds,
            10, 10, 18, 18,
            score=0.9, alive_density=0.7,
            best_value=float( state.best_value ),
        )
        basin.basin_best_point = seed.copy()
        result = search._run_late_stage_pattern_search( basin, state.bounds.copy() )
        self.assertIsNotNone( result )
        _, summary = result
        self.assertLessEqual( int( summary[ "pattern_search_evaluations_spent" ] ), 5 )

    def test_pattern_search_terminates_on_min_step( self ) -> None:
        def objective( point: np.ndarray ) -> float:
            return float( np.sum( np.asarray( point, dtype=float ) ** 2 ) )

        search = AdaptiveGridSmoothLifeSearch(
            objective,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=4, snapshot_interval=1, preset="search" ),
            agsls_config=AGSLSConfig(
                max_zoom_cycles=1,
                max_evaluations=400,
                late_stage_exploiter="pattern_search",
                late_stage_pattern_search_initial_step_fraction=0.1,
                late_stage_pattern_search_min_step_fraction=0.05,
                late_stage_pattern_search_shrink=0.5,
                late_stage_pattern_search_max_iterations=30,
                late_stage_pattern_search_max_evaluations=40,
            ),
        )
        search.reset( seed=17 )
        state = search.engine.state
        self.assertIsNotNone( state )
        seed = np.asarray( [ 0.0, 0.0 ], dtype=float )
        state.best_point = seed.copy()
        state.best_value = float( objective( seed ) )
        state.local_best_point = seed.copy()
        state.local_best_value = state.best_value
        state.box_best_point = seed.copy()
        state.box_best_value = state.best_value
        search.engine._reset_improvement_trackers()
        basin = _make_basin(
            search.engine.config.grid_shape, state.bounds,
            10, 10, 18, 18,
            score=0.9, alive_density=0.7,
            best_value=float( state.best_value ),
        )
        basin.basin_best_point = seed.copy()
        result = search._run_late_stage_pattern_search( basin, state.bounds.copy() )
        self.assertIsNotNone( result )
        _, summary = result
        final_step = np.asarray( summary[ "pattern_search_final_step" ], dtype=float )
        widths = state.bounds[ :, 1 ] - state.bounds[ :, 0 ]
        self.assertTrue( np.all( final_step <= 0.05 * widths + 1e-12 ) )

    def test_dispatcher_routes_correctly( self ) -> None:
        def objective( point: np.ndarray ) -> float:
            return float( np.sum( np.asarray( point, dtype=float ) ** 2 ) )

        def build_search( exploiter: str ) -> AdaptiveGridSmoothLifeSearch:
            search = AdaptiveGridSmoothLifeSearch(
                objective,
                bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
                smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=4, snapshot_interval=1, preset="search" ),
                agsls_config=AGSLSConfig(
                    max_zoom_cycles=1,
                    max_evaluations=400,
                    late_stage_exploiter=exploiter,
                    late_stage_microgrid_enabled=True,
                    late_stage_microgrid_resolution=3,
                    late_stage_microgrid_centers=2,
                ),
            )
            search.reset( seed=19 )
            state = search.engine.state
            assert state is not None
            seed = np.asarray( [ 1.0, 1.0 ], dtype=float )
            state.best_point = seed.copy()
            state.best_value = float( objective( seed ) )
            state.local_best_point = seed.copy()
            state.local_best_value = state.best_value
            state.box_best_point = seed.copy()
            state.box_best_value = state.best_value
            search.engine._reset_improvement_trackers()
            return search

        for exploiter, expected_microgrid, expected_pattern in (
            ( "microgrid", True, False ),
            ( "pattern_search", False, True ),
            ( "none", False, False ),
        ):
            with self.subTest( exploiter=exploiter ):
                search = build_search( exploiter )
                state = search.engine.state
                self.assertIsNotNone( state )
                basin = _make_basin(
                    search.engine.config.grid_shape, state.bounds,
                    10, 10, 18, 18,
                    score=0.9, alive_density=0.7,
                    best_value=float( state.best_value ),
                )
                basin.basin_best_point = np.asarray( state.best_point, dtype=float ).copy()
                basin.better_than_incumbent = False
                kind, result = search._run_late_stage_exploiter( basin, state.bounds.copy() )
                if exploiter == "none":
                    self.assertEqual( kind, "none" )
                    self.assertIsNone( result )
                else:
                    self.assertEqual( kind, exploiter )
                    self.assertIsNotNone( result )
                    _, summary = result
                    self.assertEqual( bool( summary.get( "microgrid_ran", False ) ), expected_microgrid )
                    self.assertEqual( bool( summary.get( "pattern_search_ran", False ) ), expected_pattern )

    def test_tuning_registers_late_stage_exploiter_family( self ) -> None:
        from smooth_life_search.benchmark.tuning import FAMILY_REGISTRY, _late_stage_exploiter_levels
        from smooth_life_search.benchmark.exploitation import DEFAULT_EXPLOITATION_FAMILIES

        self.assertIn( "late_stage_exploiter", FAMILY_REGISTRY )
        self.assertIn( "late_stage_exploiter", DEFAULT_EXPLOITATION_FAMILIES )
        levels = _late_stage_exploiter_levels( SmoothLifeConfig(), AGSLSConfig() )
        kinds = [ level.get( "late_stage_exploiter" ) for level in levels ]
        self.assertIn( "none", kinds )
        self.assertIn( "microgrid", kinds )
        self.assertIn( "pattern_search", kinds )


class TestLateStagePeriodicLocalSearch( unittest.TestCase ):
    def test_config_validates_knobs( self ) -> None:
        config = AGSLSConfig(
            late_stage_periodic_local_search_enabled=True,
            late_stage_periodic_local_search_interval_steps=4,
            late_stage_periodic_local_search_initial_step_fraction=0.20,
            late_stage_periodic_local_search_min_step_fraction=0.02,
            late_stage_periodic_local_search_shrink=0.6,
            late_stage_periodic_local_search_max_iterations=6,
            late_stage_periodic_local_search_max_evaluations=12,
        )
        self.assertTrue( bool( config.late_stage_periodic_local_search_enabled ) )
        bad_kwargs = (
            {"late_stage_periodic_local_search_interval_steps": 0},
            {"late_stage_periodic_local_search_initial_step_fraction": 0.0},
            {"late_stage_periodic_local_search_initial_step_fraction": 1.5},
            {"late_stage_periodic_local_search_min_step_fraction": 0.0},
            {
                "late_stage_periodic_local_search_min_step_fraction": 0.5,
                "late_stage_periodic_local_search_initial_step_fraction": 0.1,
            },
            {"late_stage_periodic_local_search_shrink": 0.0},
            {"late_stage_periodic_local_search_shrink": 1.0},
            {"late_stage_periodic_local_search_max_iterations": 0},
            {"late_stage_periodic_local_search_max_evaluations": 0},
        )
        for kwargs in bad_kwargs:
            with self.subTest( kwargs=kwargs ):
                with self.assertRaises( ValueError ):
                    AGSLSConfig( **kwargs )

    def test_budget_split_evenly_distributes_remainder( self ) -> None:
        search = AdaptiveGridSmoothLifeSearch( sphere, bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ] )
        self.assertEqual( search._periodic_local_search_budget_split( 3, 5 ), ( 2, 2, 1 ) )
        self.assertEqual( search._periodic_local_search_budget_split( 4, 2 ), ( 1, 1, 0, 0 ) )
        self.assertEqual( search._periodic_local_search_budget_split( 0, 5 ), () )

    def test_elite_seed_selection_uses_whole_box_ordered_by_value_then_row_col( self ) -> None:
        search = AdaptiveGridSmoothLifeSearch(
            sphere,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=1, preset="search" ),
            agsls_config=AGSLSConfig( late_stage_elite_k=3 ),
        )
        search.reset( seed=3 )
        state = search.engine.state
        self.assertIsNotNone( state )
        state.evaluated_mask.fill( False )
        state.objective_values.fill( np.nan )
        ranked = (
            ( 4, 7, 1.0 ),
            ( 5, 5, 1.0 ),
            ( 5, 6, 1.0 ),
            ( 12, 12, 2.0 ),
        )
        for row, col, value in ranked:
            state.evaluated_mask[ row, col ] = True
            state.objective_values[ row, col ] = value
        seeds = search._late_stage_periodic_elite_seeds( state.bounds.copy() )
        expected = [
            search.engine._pixel_center( 4, 7, state.bounds ),
            search.engine._pixel_center( 5, 5, state.bounds ),
            search.engine._pixel_center( 5, 6, state.bounds ),
        ]
        self.assertEqual( len( seeds ), 3 )
        for actual, target in zip( seeds, expected ):
            self.assertTrue( np.allclose( actual, target ) )

    def test_periodic_local_search_improves_best_records_without_zooming( self ) -> None:
        seed_target = np.asarray( [ 1.0, 0.0 ], dtype=float )

        def objective( point: np.ndarray ) -> float:
            return float( np.sum( ( np.asarray( point, dtype=float ) - seed_target ) ** 2 ) )

        search = AdaptiveGridSmoothLifeSearch(
            objective,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=1, snapshot_interval=1, preset="search" ),
            agsls_config=AGSLSConfig(
                max_zoom_cycles=1,
                max_evaluations=100,
                late_stage_periodic_local_search_enabled=True,
                late_stage_periodic_local_search_interval_steps=1,
                late_stage_periodic_local_search_initial_step_fraction=0.05,
                late_stage_periodic_local_search_min_step_fraction=0.01,
                late_stage_periodic_local_search_shrink=0.5,
                late_stage_periodic_local_search_max_iterations=8,
                late_stage_periodic_local_search_max_evaluations=8,
            ),
        )
        search.reset( seed=5 )
        state = search.engine.state
        self.assertIsNotNone( state )
        row, col = 12, 12
        seed = search.engine._pixel_center( row, col, state.bounds )
        initial_value = float( objective( seed ) )
        state.evaluated_mask.fill( False )
        state.objective_values.fill( np.nan )
        state.evaluated_mask[ row, col ] = True
        state.objective_values[ row, col ] = initial_value
        state.best_point = seed.copy()
        state.best_value = initial_value
        state.local_best_point = seed.copy()
        state.local_best_value = initial_value
        state.box_best_point = seed.copy()
        state.box_best_value = initial_value
        search.engine._reset_improvement_trackers()
        bounds_before = state.bounds.copy()
        summary = search._run_periodic_downhill_local_search( state.bounds.copy(), [ seed.copy() ] )
        self.assertIsNotNone( summary )
        self.assertTrue( bool( summary[ "periodic_local_search_last_improved" ] ) )
        self.assertLess( float( state.best_value ), initial_value )
        self.assertEqual( len( search.zoom_events ), 0 )
        self.assertTrue( np.allclose( state.bounds, bounds_before ) )

    def test_hook_does_not_run_when_disabled_or_outside_late_stage( self ) -> None:
        smoothlife = SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=1, snapshot_interval=1, preset="search" )
        disabled = AdaptiveGridSmoothLifeSearch(
            sphere,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=smoothlife,
            agsls_config=AGSLSConfig( late_stage_periodic_local_search_enabled=False ),
        )
        disabled.reset( seed=7 )
        late_state = {
            "late_stage_mode": True,
            "plateau": False,
            "small_box": False,
            "late": True,
            "intensification_rounds": 0,
        }
        with patch.object( disabled, "_run_periodic_downhill_local_search", return_value={} ) as run_mock:
            disabled._persistence_map( 2, box_id=0, late_stage_state=late_state )
        self.assertEqual( run_mock.call_count, 0 )
        self.assertEqual( disabled._late_stage_step_counts, {} )

        enabled = AdaptiveGridSmoothLifeSearch(
            sphere,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=smoothlife,
            agsls_config=AGSLSConfig(
                late_stage_periodic_local_search_enabled=True,
                late_stage_periodic_local_search_interval_steps=1,
            ),
        )
        enabled.reset( seed=9 )
        not_late_state = {
            "late_stage_mode": False,
            "plateau": False,
            "small_box": False,
            "late": False,
            "intensification_rounds": 0,
        }
        with patch.object( enabled, "_run_periodic_downhill_local_search", return_value={} ) as run_mock:
            enabled._persistence_map( 2, box_id=0, late_stage_state=not_late_state )
        self.assertEqual( run_mock.call_count, 0 )
        self.assertEqual( enabled._late_stage_step_counts, {} )

    def test_periodic_cadence_starts_after_k_steps_and_resets_per_box( self ) -> None:
        search = AdaptiveGridSmoothLifeSearch(
            sphere,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=1, snapshot_interval=1, preset="search" ),
            agsls_config=AGSLSConfig(
                late_stage_periodic_local_search_enabled=True,
                late_stage_periodic_local_search_interval_steps=2,
            ),
        )
        search.reset( seed=11 )
        late_state = {
            "late_stage_mode": True,
            "plateau": True,
            "small_box": False,
            "late": False,
            "intensification_rounds": 0,
        }
        invocation = {
            "periodic_local_search_last_step_index": 0,
            "periodic_local_search_last_seed_count": 1,
            "periodic_local_search_last_best_point": [ 0.0, 0.0 ],
            "periodic_local_search_last_best_value": 0.0,
            "periodic_local_search_last_improved": False,
            "periodic_local_search_last_evaluations_spent": 0,
        }
        with patch.object( search, "_late_stage_periodic_elite_seeds", return_value=[ np.asarray( [ 0.0, 0.0 ], dtype=float ) ] ):
            with patch.object( search, "_run_periodic_downhill_local_search", return_value=invocation ) as run_mock:
                search._persistence_map( 1, box_id=0, late_stage_state=late_state )
                self.assertEqual( run_mock.call_count, 0 )
                search._persistence_map( 1, box_id=0, late_stage_state=late_state )
                self.assertEqual( run_mock.call_count, 1 )
                search._persistence_map( 2, box_id=0, late_stage_state=late_state )
                self.assertEqual( run_mock.call_count, 2 )
                search._persistence_map( 1, box_id=1, late_stage_state=late_state )
                self.assertEqual( run_mock.call_count, 2 )
                search._persistence_map( 1, box_id=1, late_stage_state=late_state )
                self.assertEqual( run_mock.call_count, 3 )
        self.assertEqual( search._late_stage_step_counts[ 0 ], 4 )
        self.assertEqual( search._late_stage_step_counts[ 1 ], 2 )
        self.assertEqual( search._periodic_local_search_summaries[ 0 ][ "periodic_local_search_runs" ], 2 )
        self.assertEqual( search._periodic_local_search_summaries[ 1 ][ "periodic_local_search_runs" ], 1 )

    def test_periodic_local_search_respects_empty_seeds_and_budget_limits( self ) -> None:
        search = AdaptiveGridSmoothLifeSearch(
            sphere,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig( grid_shape=( 24, 24 ), evaluations_per_step=1, preset="search" ),
            agsls_config=AGSLSConfig(
                max_evaluations=10,
                late_stage_periodic_local_search_enabled=True,
                late_stage_periodic_local_search_max_evaluations=4,
            ),
        )
        search.reset( seed=13 )
        state = search.engine.state
        self.assertIsNotNone( state )
        self.assertIsNone( search._run_periodic_downhill_local_search( state.bounds.copy(), [] ) )
        row, col = 12, 12
        seed = search.engine._pixel_center( row, col, state.bounds )
        state.evaluated_mask.fill( False )
        state.objective_values.fill( np.nan )
        state.evaluated_mask[ row, col ] = True
        state.objective_values[ row, col ] = sphere( seed )
        search.active_max_evaluations = int( state.evaluations )
        search.engine.active_max_evaluations = int( state.evaluations )
        try:
            self.assertIsNone( search._run_periodic_downhill_local_search( state.bounds.copy(), [ seed ] ) )
        finally:
            search.active_max_evaluations = None
            search.engine.active_max_evaluations = None

    def test_periodic_diagnostics_flow_into_snapshot_trace_and_zoom( self ) -> None:
        search = AdaptiveGridSmoothLifeSearch(
            sphere,
            bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ],
            smoothlife_config=SmoothLifeConfig(
                grid_shape=( 24, 24 ),
                evaluations_per_step=1,
                snapshot_interval=1,
                preset="search",
            ),
            agsls_config=AGSLSConfig(
                max_zoom_cycles=1,
                initial_steps_per_zoom=1,
                min_steps_per_zoom=1,
                late_stage_periodic_local_search_enabled=True,
                late_stage_periodic_local_search_interval_steps=1,
            ),
        )
        search.reset( seed=17 )
        state = search.engine.state
        self.assertIsNotNone( state )
        basin = _make_basin(
            search.engine.config.grid_shape,
            state.bounds,
            4,
            4,
            12,
            12,
            score=0.95,
            alive_density=0.75,
            best_value=float( state.local_best_value ),
        )
        basin.better_than_incumbent = True
        smaller_bounds = state.bounds.copy()
        smaller_bounds[ 0, 0 ] += 2.0
        smaller_bounds[ 0, 1 ] -= 2.0
        smaller_bounds[ 1, 0 ] += 2.0
        smaller_bounds[ 1, 1 ] -= 2.0
        late_state = {
            "late_stage_mode": True,
            "plateau": True,
            "small_box": False,
            "late": False,
            "intensification_rounds": 0,
        }
        summary = {
            "periodic_local_search_last_step_index": 1,
            "periodic_local_search_last_seed_count": 1,
            "periodic_local_search_last_best_point": [ 0.0, 0.0 ],
            "periodic_local_search_last_best_value": 0.0,
            "periodic_local_search_last_improved": True,
            "periodic_local_search_last_evaluations_spent": 2,
        }
        with patch.object( search, "_late_stage_state", return_value=late_state ):
            with patch.object( search, "_late_stage_periodic_elite_seeds", return_value=[ state.best_point.copy() ] ):
                with patch.object( search, "_run_periodic_downhill_local_search", return_value=summary ):
                    with patch.object( search, "_rank_basins", return_value=[ basin ] ):
                        with patch.object( search, "_probe_basins", return_value=None ):
                            with patch.object( search, "_padded_bounds", return_value=smaller_bounds ):
                                accepted = search.step()
        self.assertTrue( accepted )
        snapshot_metadata = search.engine.snapshots[ -1 ].metadata
        self.assertTrue( bool( snapshot_metadata.get( "periodic_local_search_ran" ) ) )
        self.assertEqual( snapshot_metadata.get( "periodic_local_search_runs" ), 1 )
        self.assertEqual( search._decision_trace[ -1 ].get( "periodic_local_search_runs" ), 1 )
        self.assertEqual( search.zoom_events[ 0 ].diagnostics.get( "periodic_local_search_runs" ), 1 )
        self.assertEqual( search.zoom_events[ 0 ].diagnostics.get( "periodic_local_search_total_evaluations_spent" ), 2 )


class TestVisualization( unittest.TestCase ):
    def test_can_export_gif_animation( self ) -> None:
        config = SmoothLifeConfig( grid_shape=( 32, 32 ), evaluations_per_step=4, snapshot_interval=1, preset="search" )
        search = SmoothLifeSearch( sphere, bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ], config=config )
        search.reset( seed=13 )
        run = search.run( steps=4 )
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path( tmpdir ) / "run.gif"
            save_run_animation( run, output, scale=2, duration_ms=40 )
            self.assertTrue( output.exists() )
            self.assertGreater( output.stat().st_size, 0 )

    def test_can_render_frames_for_gui_viewer( self ) -> None:
        config = SmoothLifeConfig( grid_shape=( 32, 32 ), evaluations_per_step=4, snapshot_interval=1, preset="search" )
        search = SmoothLifeSearch( sphere, bounds=[ ( -10.0, 10.0 ), ( -10.0, 10.0 ) ], config=config )
        search.reset( seed=17 )
        run = search.run( steps=3 )
        frames = render_run_frames( run, scale=2 )
        self.assertEqual( len( frames ), len( run.snapshots ) )
        self.assertGreater( frames[ 0 ].size[ 0 ], 64 )
        self.assertGreater( frames[ 0 ].size[ 1 ], 64 )


class TestCli( unittest.TestCase ):
    def test_module_entrypoint_executes_main( self ) -> None:
        result = subprocess.run(
            [ sys.executable, "-m", "smooth_life_search.input.cli", "--help" ],
            cwd=Path( __file__ ).resolve().parents[ 2 ],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual( result.returncode, 0 )
        self.assertIn( "usage:", result.stdout )
        self.assertIn( "simulate", result.stdout )

    def test_single_command_can_trigger_gui_viewer( self ) -> None:
        stdout = StringIO()
        with patch( "smooth_life_search.input.cli.open_run_viewer" ) as open_viewer:
            with redirect_stdout( stdout ):
                exit_code = main.main(
                    [
                        "single",
                        "--objective",
                        "ackley",
                        "--dimension",
                        "2",
                        "--seed",
                        "3",
                        "--budget",
                        "200",
                        "--grid-height",
                        "24",
                        "--grid-width",
                        "24",
                        "--zoom-cycles",
                        "1",
                        "--show",
                    ]
                )
        self.assertEqual( exit_code, 0 )
        open_viewer.assert_called_once()

    def test_single_command_runs( self ) -> None:
        stdout = StringIO()
        with redirect_stdout( stdout ):
            exit_code = main.main(
                [
                    "single",
                    "--objective",
                    "ackley",
                    "--dimension",
                    "2",
                    "--seed",
                    "3",
                    "--budget",
                    "200",
                    "--grid-height",
                    "32",
                    "--grid-width",
                    "32",
                    "--zoom-cycles",
                    "2",
                ]
            )
        self.assertEqual( exit_code, 0 )
        output = stdout.getvalue()
        self.assertIn( "objective: ackley", output )
        self.assertIn( "best value:", output )

    def test_simulate_command_runs( self ) -> None:
        stdout = StringIO()
        with redirect_stdout( stdout ):
            exit_code = main.main(
                [
                    "simulate",
                    "--objective",
                    "sphere",
                    "--dimension",
                    "2",
                    "--seed",
                    "2",
                    "--steps",
                    "6",
                    "--grid-height",
                    "32",
                    "--grid-width",
                    "32",
                    "--preset",
                    "paper_glider",
                ]
            )
        self.assertEqual( exit_code, 0 )
        output = stdout.getvalue()
        self.assertIn( "objective: sphere", output )
        self.assertIn( "steps:", output )

    def test_benchmark_command_runs( self ) -> None:
        stdout = StringIO()
        with redirect_stdout( stdout ):
            exit_code = main.main(
                [
                    "benchmark",
                    "--objective",
                    "sphere",
                    "--dimension",
                    "2",
                    "--seed-start",
                    "0",
                    "--trials",
                    "2",
                    "--budget",
                    "200",
                    "--grid-height",
                    "24",
                    "--grid-width",
                    "24",
                    "--zoom-cycles",
                    "1",
                    "--success-threshold",
                    "2.0",
                ]
            )
        self.assertEqual( exit_code, 0 )
        output = stdout.getvalue()
        self.assertIn( "objective: sphere", output )
        self.assertIn( "median best value:", output )


if __name__ == "__main__":
    unittest.main()
