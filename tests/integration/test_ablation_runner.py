from __future__ import annotations

import unittest
from io import StringIO

import numpy as np

from tools.point_cloud_ablation import (
    AblationCase,
    AblationVariant,
    all_cases,
    format_markdown,
    parse_int_list,
    run_matrix,
    select_variants,
)


def _shifted_sphere(point: np.ndarray) -> float:
    target = np.linspace(-1.0, 1.0, point.size)
    shifted = np.asarray(point, dtype=float) - target
    return float(np.dot(shifted, shifted))


class TestPointCloudAblationRunner(unittest.TestCase):
    def test_runner_is_deterministic_and_reports_required_fields(self) -> None:
        case = AblationCase(
            name="smoke_shifted_sphere_12d_seed5",
            objective=_shifted_sphere,
            dimension=12,
            seed=5,
            budget=96,
            bounds=tuple([(-5.0, 5.0)] * 12),
            group="smoke",
        )
        variants = [
            AblationVariant("default", {}),
            AblationVariant("no_coherent_probes", {"coherent_probes_enabled": False}),
        ]

        first = run_matrix([case], variants)
        second = run_matrix([case], variants)

        self.assertEqual(len(first), 2)
        self.assertEqual(
            [(record["variant"], record["best_value"], record["best_source"]) for record in first],
            [(record["variant"], record["best_value"], record["best_source"]) for record in second],
        )
        for record in first:
            self.assertIn("source_counts", record)
            self.assertIn("source_improvements", record)
            self.assertIn("source_improvement_rate", record)
            self.assertIn("delta_vs_default", record)
            self.assertIn("ratio_vs_default", record)
            self.assertLessEqual(record["evaluations"], record["budget"])
            self.assertFalse(record["removed_source_hit"])

        self.assertEqual(first[0]["delta_vs_default"], 0.0)
        self.assertEqual(first[0]["ratio_vs_default"], 1.0)
        self.assertIsInstance(format_markdown(first), str)

    def test_variant_selection_rejects_unknown_names(self) -> None:
        self.assertEqual(
            [variant.name for variant in select_variants("default,no_coherent_probes")],
            ["default", "no_coherent_probes"],
        )
        with self.assertRaisesRegex(ValueError, "unknown ablation variants"):
            select_variants("default,missing")

    def test_filters_and_progress_output_are_stable(self) -> None:
        cases = all_cases("primary", dimensions=(30,), seeds=(7,), budget=64)
        stream = StringIO()

        records = run_matrix(cases, select_variants("default"), progress=True, progress_stream=stream)

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["dimension"], 30)
        self.assertEqual(records[0]["seed"], 7)
        self.assertEqual(records[0]["budget"], 64)
        self.assertIn("[1/1] rosenbrock_30d_seed7", stream.getvalue())
        self.assertEqual(parse_int_list("30, 50"), (30, 50))
        self.assertIsNone(parse_int_list(None))


if __name__ == "__main__":
    unittest.main()
