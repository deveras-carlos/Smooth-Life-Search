from __future__ import annotations

import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

from smooth_life_search import MatrixSmoothLifeConfig
from smooth_life_search.input import load_config_file, merge_config_overrides
from smooth_life_search.input.cli import _build_configs, build_parser, run_matrix_command


class TestConfigLoading(unittest.TestCase):
    def test_loads_json_and_toml_config_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "run.json"
            toml_path = Path(tmpdir) / "run.toml"
            json_path.write_text('{"command": "matrix", "budget": 100}', encoding="utf-8")
            toml_path.write_text('command = "matrix"\nbudget = 200\n', encoding="utf-8")

            self.assertEqual(load_config_file(json_path), {"command": "matrix", "budget": 100})
            self.assertEqual(load_config_file(toml_path), {"command": "matrix", "budget": 200})

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

    def test_cli_matrix_defaults_match_config_defaults(self) -> None:
        args = build_parser().parse_args(["matrix", "--objective", "sphere", "--budget", "100"])
        defaults = MatrixSmoothLifeConfig()

        self.assertEqual(args.max_steps, defaults.max_steps)
        self.assertIsNone(args.matrix_height)
        self.assertIsNone(args.matrix_width)
        self.assertEqual(args.decoder, defaults.decoder)
        self.assertEqual(args.decoder_gain, defaults.decoder_gain)
        self.assertEqual(args.projection_seed_offset, defaults.projection_seed_offset)
        self.assertEqual(args.cell_alive_threshold, defaults.cell_alive_threshold)
        self.assertEqual(args.local_decoder_block_size, defaults.local_decoder_block_size)
        self.assertEqual(args.local_decoder_overlap, defaults.local_decoder_overlap)
        self.assertEqual(args.local_credit_enabled, defaults.local_credit_enabled)
        self.assertEqual(args.local_credit_strength, defaults.local_credit_strength)
        self.assertEqual(args.local_credit_learning_rate, defaults.local_credit_learning_rate)
        self.assertEqual(args.local_credit_decay, defaults.local_credit_decay)
        self.assertEqual(args.local_credit_clip, defaults.local_credit_clip)
        self.assertEqual(args.patch_probe_enabled, defaults.patch_probe_enabled)
        self.assertEqual(args.patch_probe_interval_evaluations, defaults.patch_probe_interval_evaluations)
        self.assertEqual(args.patch_probe_count, defaults.patch_probe_count)
        self.assertEqual(args.patch_probe_step, defaults.patch_probe_step)
        self.assertEqual(args.steps_per_evaluation, defaults.steps_per_evaluation)
        self.assertEqual(args.elite_pull_strength, defaults.elite_pull_strength)
        self.assertEqual(args.failure_damping, defaults.failure_damping)
        self.assertEqual(args.reward_decay, defaults.reward_decay)
        self.assertEqual(args.reward_boost, defaults.reward_boost)
        self.assertEqual(args.mutation_noise, defaults.mutation_noise)
        self.assertEqual(args.mutation_decay, defaults.mutation_decay)
        self.assertEqual(args.advantage_strength, defaults.advantage_strength)
        self.assertEqual(args.advantage_decay, defaults.advantage_decay)
        self.assertEqual(args.direction_strength, defaults.direction_strength)
        self.assertEqual(args.direction_decay, defaults.direction_decay)
        self.assertEqual(args.temperature_init, defaults.temperature_init)
        self.assertEqual(args.temperature_decay, defaults.temperature_decay)
        self.assertEqual(args.temperature_reheat, defaults.temperature_reheat)
        self.assertEqual(args.stagnation_reheat_evaluations, defaults.stagnation_reheat_evaluations)
        self.assertEqual(args.matrix_line_search_enabled, defaults.matrix_line_search_enabled)
        self.assertEqual(args.matrix_line_search_alphas, defaults.matrix_line_search_alphas)
        self.assertEqual(args.early_stop_enabled, defaults.early_stop_enabled)
        self.assertEqual(args.early_stop_value, defaults.early_stop_value)
        self.assertIsNone(args.target_value)
        self.assertEqual(args.matrix_snapshot_interval, defaults.snapshot_interval)
        self.assertEqual(args.store_all_snapshots, defaults.store_all_snapshots)

    def test_cli_can_configure_matrix_controls(self) -> None:
        args = build_parser().parse_args(
            [
                "matrix",
                "--objective",
                "sphere",
                "--budget",
                "100",
                "--max-steps",
                "12",
                "--matrix-height",
                "20",
                "--matrix-width",
                "24",
                "--decoder",
                "random_projection",
                "--decoder-gain",
                "1.5",
                "--projection-seed-offset",
                "17",
                "--cell-alive-threshold",
                "0.2",
                "--local-decoder-block-size",
                "6",
                "--local-decoder-overlap",
                "1",
                "--no-local-credit",
                "--local-credit-strength",
                "0.45",
                "--local-credit-learning-rate",
                "0.25",
                "--local-credit-decay",
                "0.85",
                "--local-credit-clip",
                "3.5",
                "--no-patch-probes",
                "--patch-probe-interval-evaluations",
                "13",
                "--patch-probe-count",
                "2",
                "--patch-probe-step",
                "0.11",
                "--steps-per-evaluation",
                "3",
                "--elite-pull-strength",
                "0.2",
                "--failure-damping",
                "0.1",
                "--reward-decay",
                "0.9",
                "--reward-boost",
                "0.5",
                "--mutation-noise",
                "0.02",
                "--mutation-decay",
                "0.99",
                "--advantage-strength",
                "0.3",
                "--advantage-decay",
                "0.8",
                "--direction-strength",
                "0.4",
                "--direction-decay",
                "0.7",
                "--temperature-init",
                "1.2",
                "--temperature-decay",
                "0.98",
                "--temperature-reheat",
                "0.15",
                "--stagnation-reheat-evaluations",
                "9",
                "--no-matrix-line-search",
                "--matrix-line-search-alphas",
                "0.25",
                "0.75",
                "1.25",
                "--target-value",
                "1e-5",
                "--matrix-snapshot-interval",
                "5",
                "--no-store-all-snapshots",
            ]
        )

        _bounds, _smoothlife, matrix = _build_configs(args)
        self.assertEqual(matrix.max_evaluations, 100)
        self.assertEqual(matrix.max_steps, 12)
        self.assertEqual(matrix.matrix_shape, (20, 24))
        self.assertEqual(matrix.decoder, "random_projection")
        self.assertAlmostEqual(matrix.decoder_gain, 1.5)
        self.assertEqual(matrix.projection_seed_offset, 17)
        self.assertAlmostEqual(matrix.cell_alive_threshold, 0.2)
        self.assertEqual(matrix.local_decoder_block_size, 6)
        self.assertEqual(matrix.local_decoder_overlap, 1)
        self.assertFalse(matrix.local_credit_enabled)
        self.assertAlmostEqual(matrix.local_credit_strength, 0.45)
        self.assertAlmostEqual(matrix.local_credit_learning_rate, 0.25)
        self.assertAlmostEqual(matrix.local_credit_decay, 0.85)
        self.assertAlmostEqual(matrix.local_credit_clip, 3.5)
        self.assertFalse(matrix.patch_probe_enabled)
        self.assertEqual(matrix.patch_probe_interval_evaluations, 13)
        self.assertEqual(matrix.patch_probe_count, 2)
        self.assertAlmostEqual(matrix.patch_probe_step, 0.11)
        self.assertEqual(matrix.steps_per_evaluation, 3)
        self.assertAlmostEqual(matrix.elite_pull_strength, 0.2)
        self.assertAlmostEqual(matrix.failure_damping, 0.1)
        self.assertAlmostEqual(matrix.reward_decay, 0.9)
        self.assertAlmostEqual(matrix.reward_boost, 0.5)
        self.assertAlmostEqual(matrix.mutation_noise, 0.02)
        self.assertAlmostEqual(matrix.mutation_decay, 0.99)
        self.assertAlmostEqual(matrix.advantage_strength, 0.3)
        self.assertAlmostEqual(matrix.advantage_decay, 0.8)
        self.assertAlmostEqual(matrix.direction_strength, 0.4)
        self.assertAlmostEqual(matrix.direction_decay, 0.7)
        self.assertAlmostEqual(matrix.temperature_init, 1.2)
        self.assertAlmostEqual(matrix.temperature_decay, 0.98)
        self.assertAlmostEqual(matrix.temperature_reheat, 0.15)
        self.assertEqual(matrix.stagnation_reheat_evaluations, 9)
        self.assertFalse(matrix.matrix_line_search_enabled)
        self.assertEqual(matrix.matrix_line_search_alphas, (0.25, 0.75, 1.25))
        self.assertTrue(matrix.early_stop_enabled)
        self.assertAlmostEqual(matrix.early_stop_value or 0.0, 1e-5)
        self.assertEqual(matrix.snapshot_interval, 5)
        self.assertFalse(matrix.store_all_snapshots)

    def test_matrix_accepts_nd_and_point_cloud_command_is_removed(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["matrix", "--objective", "sphere", "--dimension", "5", "--budget", "12"])
        bounds, _smoothlife, _matrix = _build_configs(args)
        self.assertEqual(len(bounds), 5)

        with redirect_stderr(StringIO()):
            with self.assertRaises(SystemExit):
                parser.parse_args(["point-cloud", "--objective", "sphere", "--budget", "12"])

    def test_simulate_and_himmelblau_reject_invalid_dimensions(self) -> None:
        simulate_args = build_parser().parse_args(["simulate", "--objective", "sphere", "--dimension", "5"])
        with self.assertRaisesRegex(ValueError, "simulate currently supports only --dimension 2"):
            _build_configs(simulate_args)

        matrix_args = build_parser().parse_args(
            ["matrix", "--objective", "himmelblau", "--dimension", "5", "--budget", "20"]
        )
        with self.assertRaisesRegex(ValueError, "himmelblau requires --dimension 2"):
            _build_configs(matrix_args)

    def test_matrix_command_runs_small_nd_case(self) -> None:
        args = build_parser().parse_args(
            [
                "matrix",
                "--objective",
                "sphere",
                "--dimension",
                "5",
                "--budget",
                "8",
                "--seed",
                "3",
                "--no-store-all-snapshots",
            ]
        )
        payload = run_matrix_command(args)

        self.assertEqual(payload["mode"], "matrix")
        self.assertEqual(payload["dimension"], 5)
        self.assertLessEqual(payload["evaluations"], 8)
        self.assertEqual(payload["matrix_shape"], [16, 16])


if __name__ == "__main__":
    unittest.main()
