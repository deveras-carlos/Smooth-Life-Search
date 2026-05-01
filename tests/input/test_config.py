from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from smooth_life_search import PointCloudSearchConfig
from smooth_life_search.input import load_config_file, merge_config_overrides
from smooth_life_search.input.cli import build_parser


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
        self.assertEqual(args.region_stall_patience, defaults.region_stall_patience)
        self.assertEqual(args.region_cooldown_batches, defaults.region_cooldown_batches)
        self.assertEqual(args.global_exploration_floor, defaults.global_exploration_floor)
        self.assertEqual(args.region_stencil_fraction, defaults.region_stencil_fraction)

    def test_cli_can_disable_point_cloud_features(self) -> None:
        args = build_parser().parse_args(
            [
                "point-cloud",
                "--objective",
                "sphere",
                "--budget",
                "100",
                "--no-trust-regions",
                "--no-surrogate",
                "--no-local-refinement",
            ]
        )

        self.assertFalse(args.trust_regions_enabled)
        self.assertFalse(args.surrogate_enabled)
        self.assertFalse(args.local_refinement_enabled)

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
                "--region-stall-patience",
                "5",
                "--region-cooldown-batches",
                "6",
                "--global-exploration-floor",
                "0.2",
                "--region-stencil-fraction",
                "0.5",
                "--local-refinement-max-evaluations",
                "21",
            ]
        )

        self.assertEqual(args.batch_size, 11)
        self.assertEqual(args.initial_design_size, 13)
        self.assertEqual(args.portfolio_size, 2)
        self.assertAlmostEqual(args.region_initial_radius_fraction, 0.2)
        self.assertTrue(args.early_stop_enabled)
        self.assertAlmostEqual(args.early_stop_value, 0.001)
        self.assertEqual(args.region_stall_patience, 5)
        self.assertEqual(args.region_cooldown_batches, 6)
        self.assertAlmostEqual(args.global_exploration_floor, 0.2)
        self.assertAlmostEqual(args.region_stencil_fraction, 0.5)
        self.assertEqual(args.local_refinement_max_evaluations, 21)

    def test_agsls_command_is_removed(self) -> None:
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(["agsls", "--objective", "sphere", "--budget", "100"])


if __name__ == "__main__":
    unittest.main()
