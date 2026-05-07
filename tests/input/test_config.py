from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from smooth_life_search import PointCloudSearchConfig
from smooth_life_search.input import load_config_file, merge_config_overrides
from smooth_life_search.input.cli import _build_configs, build_parser, run_point_cloud_command


class TestConfigLoading(unittest.TestCase):
    def test_loads_json_and_toml_config_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "run.json"
            toml_path = Path(tmpdir) / "run.toml"
            json_path.write_text('{"command": "run", "budget": 100}', encoding="utf-8")
            toml_path.write_text('command = "run"\nbudget = 200\n', encoding="utf-8")

            self.assertEqual(load_config_file(json_path), {"command": "run", "budget": 100})
            self.assertEqual(load_config_file(toml_path), {"command": "run", "budget": 200})

    def test_rejects_unknown_extension_and_non_object_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "run.yaml"
            yaml_path.write_text("budget: 100", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, ".json or .toml"):
                load_config_file(yaml_path)

            json_path = Path(tmpdir) / "list.json"
            json_path.write_text("[1, 2, 3]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "root must be an object"):
                load_config_file(json_path)

    def test_cli_overrides_skip_none_values(self) -> None:
        merged = merge_config_overrides({"budget": 100, "seed": 1}, {"budget": None, "seed": 7})
        self.assertEqual(merged, {"budget": 100, "seed": 7})

    def test_cli_point_cloud_defaults_match_config_defaults(self) -> None:
        args = build_parser().parse_args(["point-cloud", "--objective", "sphere", "--budget", "100"])
        defaults = PointCloudSearchConfig()

        self.assertEqual(args.batch_size, defaults.batch_size)
        self.assertEqual(args.initial_design_size, defaults.initial_design_size)
        self.assertEqual(args.portfolio_size, defaults.portfolio_size)
        self.assertEqual(args.density_grid_height, defaults.density_grid_shape[0])
        self.assertEqual(args.density_grid_width, defaults.density_grid_shape[1])
        self.assertEqual(args.trust_regions_enabled, defaults.trust_regions_enabled)
        self.assertEqual(args.surrogate_enabled, defaults.surrogate_enabled)
        self.assertEqual(args.local_refinement_enabled, defaults.local_refinement_enabled)
        self.assertEqual(args.local_refinement_max_evaluations, defaults.local_refinement_max_evaluations)
        self.assertEqual(args.early_stop_enabled, defaults.early_stop_enabled)
        self.assertEqual(args.early_stop_value, defaults.early_stop_value)
        self.assertIsNone(args.target_value)
        self.assertEqual(args.region_stall_patience, defaults.region_stall_patience)
        self.assertEqual(args.region_cooldown_batches, defaults.region_cooldown_batches)
        self.assertEqual(args.global_exploration_floor, defaults.global_exploration_floor)
        self.assertEqual(args.region_stencil_fraction, defaults.region_stencil_fraction)
        self.assertEqual(args.anisotropic_regions_enabled, defaults.anisotropic_regions_enabled)
        self.assertEqual(args.region_anisotropy_max, defaults.region_anisotropy_max)
        self.assertEqual(args.region_geometry_min_samples, defaults.region_geometry_min_samples)
        self.assertEqual(args.local_refinement_method, defaults.local_refinement_method)
        self.assertEqual(args.local_refinement_damping, defaults.local_refinement_damping)
        self.assertEqual(args.active_subspace_size, defaults.active_subspace_size)
        self.assertEqual(
            args.surrogate_full_quadratic_max_dimension,
            defaults.surrogate_full_quadratic_max_dimension,
        )
        self.assertEqual(args.projection_axes, defaults.projection_axes)
        self.assertEqual(args.projection_ensemble_enabled, defaults.projection_ensemble_enabled)
        self.assertEqual(args.projection_ensemble_size, defaults.projection_ensemble_size)
        self.assertEqual(args.projection_ensemble_refresh_batches, defaults.projection_ensemble_refresh_batches)
        self.assertEqual(args.coherent_probes_enabled, defaults.coherent_probes_enabled)
        self.assertEqual(args.surrogate_ranking_enabled, defaults.surrogate_ranking_enabled)
        self.assertEqual(args.candidate_pool_multiplier, defaults.candidate_pool_multiplier)
        self.assertEqual(args.surrogate_ranking_neighbor_count, defaults.surrogate_ranking_neighbor_count)
        self.assertEqual(args.surrogate_reliability_enabled, defaults.surrogate_reliability_enabled)
        self.assertEqual(args.surrogate_rank_weight_min, defaults.surrogate_rank_weight_min)
        self.assertEqual(args.surrogate_rank_weight_max, defaults.surrogate_rank_weight_max)
        self.assertEqual(args.probe_recenter_enabled, defaults.probe_recenter_enabled)
        self.assertEqual(args.probe_recenter_max_restarts, defaults.probe_recenter_max_restarts)
        self.assertEqual(args.basin_polishing_enabled, defaults.basin_polishing_enabled)
        self.assertEqual(args.basin_polishing_min_dimension, defaults.basin_polishing_min_dimension)
        self.assertEqual(args.basin_polishing_activation_ratio, defaults.basin_polishing_activation_ratio)
        self.assertEqual(args.successful_direction_memory_size, defaults.successful_direction_memory_size)
        self.assertEqual(args.direction_refinement_enabled, defaults.direction_refinement_enabled)
        self.assertEqual(args.direction_refinement_max_evaluations, defaults.direction_refinement_max_evaluations)
        self.assertEqual(args.direction_line_search_mode, defaults.direction_line_search_mode)
        self.assertEqual(args.direction_line_search_max_steps, defaults.direction_line_search_max_steps)
        self.assertEqual(args.direction_line_search_min_step_fraction, defaults.direction_line_search_min_step_fraction)
        self.assertEqual(args.linkage_blocks_enabled, defaults.linkage_blocks_enabled)
        self.assertEqual(args.linkage_update_interval_batches, defaults.linkage_update_interval_batches)
        self.assertEqual(args.linkage_neighbor_count, defaults.linkage_neighbor_count)
        self.assertEqual(args.cross_block_lbfgs_enabled, defaults.cross_block_lbfgs_enabled)
        self.assertEqual(args.cross_block_lbfgs_memory_size, defaults.cross_block_lbfgs_memory_size)
        self.assertEqual(args.cooperative_refinement_enabled, defaults.cooperative_refinement_enabled)
        self.assertEqual(args.cooperative_min_dimension, defaults.cooperative_min_dimension)
        self.assertEqual(args.cooperative_group_size, defaults.cooperative_group_size)
        self.assertEqual(args.cooperative_groups_per_batch, defaults.cooperative_groups_per_batch)
        self.assertEqual(args.cooperative_frontier_enabled, defaults.cooperative_frontier_enabled)
        self.assertEqual(args.cooperative_frontier_fraction, defaults.cooperative_frontier_fraction)
        self.assertEqual(args.axis_coverage_pressure, defaults.axis_coverage_pressure)
        self.assertEqual(args.active_set_max_fraction, defaults.active_set_max_fraction)
        self.assertEqual(args.active_set_expand_interval_batches, defaults.active_set_expand_interval_batches)

    def test_cli_can_disable_point_cloud_features(self) -> None:
        args = build_parser().parse_args(
            [
                "point-cloud",
                "--objective",
                "sphere",
                "--budget",
                "100",
                "--no-trust-regions",
                "--no-anisotropic-regions",
                "--no-surrogate",
                "--no-projection-ensemble",
                "--no-local-refinement",
                "--no-coherent-probes",
                "--no-surrogate-ranking",
                "--no-surrogate-reliability",
                "--no-probe-recenter",
                "--no-basin-polishing",
                "--no-direction-refinement",
                "--no-linkage-blocks",
                "--no-cross-block-lbfgs",
                "--no-cooperative-refinement",
                "--no-cooperative-frontier",
            ]
        )

        self.assertFalse(args.trust_regions_enabled)
        self.assertFalse(args.anisotropic_regions_enabled)
        self.assertFalse(args.surrogate_enabled)
        self.assertFalse(args.projection_ensemble_enabled)
        self.assertFalse(args.local_refinement_enabled)
        self.assertFalse(args.coherent_probes_enabled)
        self.assertFalse(args.surrogate_ranking_enabled)
        self.assertFalse(args.surrogate_reliability_enabled)
        self.assertFalse(args.probe_recenter_enabled)
        self.assertFalse(args.basin_polishing_enabled)
        self.assertFalse(args.direction_refinement_enabled)
        self.assertFalse(args.linkage_blocks_enabled)
        self.assertFalse(args.cross_block_lbfgs_enabled)
        self.assertFalse(args.cooperative_refinement_enabled)
        self.assertFalse(args.cooperative_frontier_enabled)

    def test_removed_evolutionary_engine_flags_are_rejected(self) -> None:
        parser = build_parser()
        removed_flags = ["--no-shade", "--no-cma-region", "--no-restart-strategy", "--source-credit-temperature"]
        for flag in removed_flags:
            with self.subTest(flag=flag), redirect_stderr(StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(["point-cloud", "--objective", "sphere", "--budget", "100", flag])

    def test_cli_can_configure_point_cloud_controls(self) -> None:
        args = build_parser().parse_args(
            [
                "point-cloud",
                "--objective",
                "sphere",
                "--budget",
                "100",
                "--batch-size",
                "11",
                "--initial-design-size",
                "13",
                "--portfolio-size",
                "2",
                "--region-initial-radius-fraction",
                "0.2",
                "--early-stop-enabled",
                "--early-stop-value",
                "0.001",
                "--target-value",
                "0.0001",
                "--region-stall-patience",
                "5",
                "--region-cooldown-batches",
                "6",
                "--global-exploration-floor",
                "0.2",
                "--region-stencil-fraction",
                "0.5",
                "--region-anisotropy-max",
                "7",
                "--region-geometry-min-samples",
                "9",
                "--active-subspace-size",
                "5",
                "--surrogate-full-quadratic-max-dimension",
                "4",
                "--projection-axes",
                "1",
                "3",
                "--projection-ensemble-size",
                "3",
                "--projection-ensemble-refresh-batches",
                "5",
                "--local-refinement-max-evaluations",
                "21",
                "--local-refinement-method",
                "levenberg-marquardt",
                "--local-refinement-damping",
                "1e-4",
                "--candidate-pool-multiplier",
                "4",
                "--surrogate-ranking-neighbor-count",
                "24",
                "--surrogate-rank-weight-min",
                "0.25",
                "--surrogate-rank-weight-max",
                "0.75",
                "--probe-recenter-max-restarts",
                "3",
                "--basin-polishing-min-dimension",
                "20",
                "--basin-polishing-activation-ratio",
                "0.15",
                "--successful-direction-memory-size",
                "12",
                "--direction-refinement-max-evaluations",
                "64",
                "--direction-line-search-mode",
                "opportunistic",
                "--direction-line-search-max-steps",
                "5",
                "--direction-line-search-min-step-fraction",
                "1e-5",
                "--linkage-update-interval-batches",
                "4",
                "--linkage-neighbor-count",
                "5",
                "--cross-block-lbfgs-memory-size",
                "10",
                "--cooperative-min-dimension",
                "200",
                "--cooperative-group-size",
                "12",
                "--cooperative-groups-per-batch",
                "6",
                "--cooperative-frontier-fraction",
                "0.5",
                "--axis-coverage-pressure",
                "0.3",
                "--active-set-max-fraction",
                "0.4",
                "--active-set-expand-interval-batches",
                "7",
            ]
        )

        self.assertEqual(args.batch_size, 11)
        self.assertEqual(args.initial_design_size, 13)
        self.assertEqual(args.portfolio_size, 2)
        self.assertAlmostEqual(args.region_initial_radius_fraction, 0.2)
        self.assertTrue(args.early_stop_enabled)
        self.assertAlmostEqual(args.early_stop_value, 0.001)
        self.assertAlmostEqual(args.target_value, 0.0001)
        self.assertEqual(args.region_stall_patience, 5)
        self.assertEqual(args.region_cooldown_batches, 6)
        self.assertAlmostEqual(args.global_exploration_floor, 0.2)
        self.assertAlmostEqual(args.region_stencil_fraction, 0.5)
        self.assertAlmostEqual(args.region_anisotropy_max, 7.0)
        self.assertEqual(args.region_geometry_min_samples, 9)
        self.assertEqual(args.active_subspace_size, 5)
        self.assertEqual(args.surrogate_full_quadratic_max_dimension, 4)
        self.assertEqual(args.projection_axes, [1, 3])
        self.assertEqual(args.projection_ensemble_size, 3)
        self.assertEqual(args.projection_ensemble_refresh_batches, 5)
        self.assertEqual(args.local_refinement_max_evaluations, 21)
        self.assertEqual(args.local_refinement_method, "levenberg-marquardt")
        self.assertAlmostEqual(args.local_refinement_damping, 1e-4)
        self.assertEqual(args.candidate_pool_multiplier, 4)
        self.assertEqual(args.surrogate_ranking_neighbor_count, 24)
        self.assertAlmostEqual(args.surrogate_rank_weight_min, 0.25)
        self.assertAlmostEqual(args.surrogate_rank_weight_max, 0.75)
        self.assertEqual(args.probe_recenter_max_restarts, 3)
        self.assertEqual(args.basin_polishing_min_dimension, 20)
        self.assertAlmostEqual(args.basin_polishing_activation_ratio, 0.15)
        self.assertEqual(args.successful_direction_memory_size, 12)
        self.assertEqual(args.direction_refinement_max_evaluations, 64)
        self.assertEqual(args.direction_line_search_mode, "opportunistic")
        self.assertEqual(args.direction_line_search_max_steps, 5)
        self.assertAlmostEqual(args.direction_line_search_min_step_fraction, 1e-5)
        self.assertEqual(args.linkage_update_interval_batches, 4)
        self.assertEqual(args.linkage_neighbor_count, 5)
        self.assertEqual(args.cross_block_lbfgs_memory_size, 10)
        self.assertEqual(args.cooperative_min_dimension, 200)
        self.assertEqual(args.cooperative_group_size, 12)
        self.assertEqual(args.cooperative_groups_per_batch, 6)
        self.assertAlmostEqual(args.cooperative_frontier_fraction, 0.5)
        self.assertAlmostEqual(args.axis_coverage_pressure, 0.3)
        self.assertAlmostEqual(args.active_set_max_fraction, 0.4)
        self.assertEqual(args.active_set_expand_interval_batches, 7)

    def test_cli_builds_lean_high_dimensional_point_cloud_config(self) -> None:
        args = build_parser().parse_args(
            [
                "point-cloud",
                "--objective",
                "sphere",
                "--dimension",
                "30",
                "--budget",
                "100",
                "--candidate-pool-multiplier",
                "5",
                "--surrogate-ranking-neighbor-count",
                "20",
                "--projection-ensemble-size",
                "3",
                "--surrogate-rank-weight-min",
                "0.2",
                "--surrogate-rank-weight-max",
                "0.7",
                "--probe-recenter-max-restarts",
                "6",
                "--basin-polishing-activation-ratio",
                "0.2",
                "--successful-direction-memory-size",
                "18",
                "--direction-refinement-max-evaluations",
                "48",
                "--direction-line-search-max-steps",
                "6",
                "--linkage-neighbor-count",
                "4",
                "--cross-block-lbfgs-memory-size",
                "11",
                "--cooperative-min-dimension",
                "120",
                "--cooperative-group-size",
                "10",
                "--cooperative-groups-per-batch",
                "5",
                "--cooperative-frontier-fraction",
                "0.45",
                "--axis-coverage-pressure",
                "0.25",
                "--active-set-max-fraction",
                "0.3",
                "--active-set-expand-interval-batches",
                "6",
            ]
        )

        _bounds, _smoothlife, point_cloud = _build_configs(args)

        self.assertTrue(point_cloud.coherent_probes_enabled)
        self.assertTrue(point_cloud.surrogate_ranking_enabled)
        self.assertEqual(point_cloud.candidate_pool_multiplier, 5)
        self.assertEqual(point_cloud.surrogate_ranking_neighbor_count, 20)
        self.assertTrue(point_cloud.projection_ensemble_enabled)
        self.assertEqual(point_cloud.projection_ensemble_size, 3)
        self.assertTrue(point_cloud.surrogate_reliability_enabled)
        self.assertAlmostEqual(point_cloud.surrogate_rank_weight_min, 0.2)
        self.assertAlmostEqual(point_cloud.surrogate_rank_weight_max, 0.7)
        self.assertTrue(point_cloud.probe_recenter_enabled)
        self.assertEqual(point_cloud.probe_recenter_max_restarts, 6)
        self.assertTrue(point_cloud.basin_polishing_enabled)
        self.assertAlmostEqual(point_cloud.basin_polishing_activation_ratio, 0.2)
        self.assertEqual(point_cloud.successful_direction_memory_size, 18)
        self.assertTrue(point_cloud.direction_refinement_enabled)
        self.assertEqual(point_cloud.direction_refinement_max_evaluations, 48)
        self.assertEqual(point_cloud.direction_line_search_mode, "bracketed")
        self.assertEqual(point_cloud.direction_line_search_max_steps, 6)
        self.assertTrue(point_cloud.linkage_blocks_enabled)
        self.assertEqual(point_cloud.linkage_neighbor_count, 4)
        self.assertTrue(point_cloud.cross_block_lbfgs_enabled)
        self.assertEqual(point_cloud.cross_block_lbfgs_memory_size, 11)
        self.assertTrue(point_cloud.cooperative_refinement_enabled)
        self.assertEqual(point_cloud.cooperative_min_dimension, 120)
        self.assertEqual(point_cloud.cooperative_group_size, 10)
        self.assertEqual(point_cloud.cooperative_groups_per_batch, 5)
        self.assertTrue(point_cloud.cooperative_frontier_enabled)
        self.assertAlmostEqual(point_cloud.cooperative_frontier_fraction, 0.45)
        self.assertAlmostEqual(point_cloud.axis_coverage_pressure, 0.25)
        self.assertAlmostEqual(point_cloud.active_set_max_fraction, 0.3)
        self.assertEqual(point_cloud.active_set_expand_interval_batches, 6)

    def test_target_value_enables_early_stop_in_point_cloud_config(self) -> None:
        args = build_parser().parse_args(
            [
                "point-cloud",
                "--objective",
                "sphere",
                "--budget",
                "100",
                "--target-value",
                "1e-9",
            ]
        )

        _bounds, _smoothlife, point_cloud = _build_configs(args)

        self.assertTrue(point_cloud.early_stop_enabled)
        self.assertAlmostEqual(point_cloud.early_stop_value, 1e-9)

    def test_point_cloud_accepts_nd_dimension_and_projection_axes(self) -> None:
        args = build_parser().parse_args(
            [
                "point-cloud",
                "--objective",
                "sphere",
                "--dimension",
                "5",
                "--budget",
                "100",
                "--projection-axes",
                "1",
                "4",
            ]
        )

        bounds, _smoothlife, point_cloud = _build_configs(args)

        self.assertEqual(len(bounds), 5)
        self.assertEqual(point_cloud.projection_axes, (1, 4))

    def test_simulate_and_himmelblau_reject_nd_dimension(self) -> None:
        simulate_args = build_parser().parse_args(
            ["simulate", "--objective", "sphere", "--dimension", "5", "--steps", "1"]
        )
        with self.assertRaisesRegex(ValueError, "simulate"):
            _build_configs(simulate_args)

        himmelblau_args = build_parser().parse_args(
            ["point-cloud", "--objective", "himmelblau", "--dimension", "5", "--budget", "100"]
        )
        with self.assertRaisesRegex(ValueError, "himmelblau"):
            _build_configs(himmelblau_args)

    def test_point_cloud_nd_rejects_visualization_outputs(self) -> None:
        args = build_parser().parse_args(
            [
                "point-cloud",
                "--objective",
                "sphere",
                "--dimension",
                "5",
                "--budget",
                "100",
                "--gif",
                "nd.gif",
            ]
        )

        with self.assertRaisesRegex(ValueError, "visualization"):
            run_point_cloud_command(args)

    def test_agsls_command_is_removed(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["agsls", "--objective", "sphere", "--budget", "100"])


if __name__ == "__main__":
    unittest.main()
