from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
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
        local_trace = np.asarray( [ snapshot.local_best_value for snapshot in run.snapshots ], dtype=float )
        self.assertTrue( np.all( np.diff( global_trace ) <= 1e-12 ) )
        self.assertTrue( np.all( np.diff( local_trace ) <= 1e-12 ) )
        self.assertAlmostEqual( float( global_trace[ -1 ] ), float( run.best_value ), places=12 )

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
    def test_single_command_can_trigger_gui_viewer( self ) -> None:
        stdout = StringIO()
        with patch( "smooth_life_search.cli.open_run_viewer" ) as open_viewer:
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
