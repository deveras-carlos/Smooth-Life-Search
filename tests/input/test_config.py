from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from smooth_life_search.input import load_config_file, merge_config_overrides
from smooth_life_search.input.cli import build_parser
from smooth_life_search import AGSLSConfig


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

    def test_cli_agsls_defaults_match_config_defaults(self) -> None:
        args = build_parser().parse_args(["agsls", "--objective", "sphere", "--budget", "100"])
        defaults = AGSLSConfig()

        self.assertEqual(args.zoom_cycles, defaults.max_zoom_cycles)
        self.assertEqual(args.commit_fraction, defaults.commit_fraction)
        self.assertEqual(args.commit_steps, defaults.commit_steps_per_zoom)
        self.assertEqual(args.exploitation_steps, defaults.exploitation_steps_per_zoom)
        self.assertEqual(args.exploitation_shrink_fraction, defaults.exploitation_shrink_fraction)
        self.assertEqual(args.commit_guidance_top_k, defaults.commit_guidance_top_k)
        self.assertEqual(args.exploitation_guidance_sigma, defaults.exploitation_guidance_sigma)
        self.assertEqual(args.commit_surrogate_enabled, defaults.commit_surrogate_enabled)
        self.assertEqual(args.commit_surrogate_min_samples, defaults.commit_surrogate_min_samples)
        self.assertEqual(args.commit_surrogate_max_samples, defaults.commit_surrogate_max_samples)
        self.assertEqual(args.trust_region_enabled, defaults.trust_region_enabled)
        self.assertEqual(args.commit_trust_region_evaluations, defaults.commit_trust_region_evaluations)
        self.assertEqual(args.exploitation_trust_region_evaluations, defaults.exploitation_trust_region_evaluations)
        self.assertEqual(args.trust_region_candidate_pool_size, defaults.trust_region_candidate_pool_size)
        self.assertEqual(args.trust_region_initial_radius_fraction, defaults.trust_region_initial_radius_fraction)
        self.assertEqual(args.exploitation_valley_tracking_enabled, defaults.exploitation_valley_tracking_enabled)
        self.assertEqual(args.exploitation_valley_probe_evaluations, defaults.exploitation_valley_probe_evaluations)
        self.assertEqual(args.exploitation_valley_step_fraction, defaults.exploitation_valley_step_fraction)

    def test_cli_can_disable_commit_surrogate(self) -> None:
        args = build_parser().parse_args(
            ["agsls", "--objective", "sphere", "--budget", "100", "--no-commit-surrogate"]
        )

        self.assertFalse(args.commit_surrogate_enabled)

    def test_cli_can_configure_trust_region(self) -> None:
        args = build_parser().parse_args(
            [
                "agsls",
                "--objective",
                "sphere",
                "--budget",
                "100",
                "--no-trust-region",
                "--trust-region-commit-evals",
                "3",
                "--trust-region-exploitation-evals",
                "5",
                "--trust-region-candidates",
                "24",
                "--trust-region-initial-radius-fraction",
                "0.1",
            ]
        )

        self.assertFalse(args.trust_region_enabled)
        self.assertEqual(args.commit_trust_region_evaluations, 3)
        self.assertEqual(args.exploitation_trust_region_evaluations, 5)
        self.assertEqual(args.trust_region_candidate_pool_size, 24)
        self.assertAlmostEqual(args.trust_region_initial_radius_fraction, 0.1)

    def test_cli_can_configure_exploitation_valley_tracking(self) -> None:
        args = build_parser().parse_args(
            [
                "agsls",
                "--objective",
                "sphere",
                "--budget",
                "100",
                "--no-exploitation-valley-tracking",
                "--exploitation-valley-probes",
                "3",
                "--exploitation-valley-step-fraction",
                "0.05",
            ]
        )

        self.assertFalse(args.exploitation_valley_tracking_enabled)
        self.assertEqual(args.exploitation_valley_probe_evaluations, 3)
        self.assertAlmostEqual(args.exploitation_valley_step_fraction, 0.05)

if __name__ == "__main__":
    unittest.main()
