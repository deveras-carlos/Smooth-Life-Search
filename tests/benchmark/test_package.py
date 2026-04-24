from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import smooth_life_search.benchmark as benchmark
from smooth_life_search.benchmark.artifacts import append_ndjson, load_ndjson, reset_artifacts, write_csv


class TestBenchmarkPackage(unittest.TestCase):
    def test_public_package_exports_core_benchmark_api(self) -> None:
        self.assertIn("sphere", benchmark.OBJECTIVES)
        self.assertIs(benchmark.OBJECTIVES["sphere"], benchmark.sphere)
        self.assertTrue(hasattr(benchmark, "StudySpec"))
        self.assertTrue(hasattr(benchmark, "ExploitationStudySpec"))
        self.assertTrue(callable(benchmark.run_seeded_trials))

    def test_artifact_helpers_round_trip_records_and_reset_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            ndjson_path = output_dir / "trials.ndjson"
            append_ndjson(ndjson_path, {"trial": 1, "value": 2.0})
            self.assertEqual(load_ndjson(ndjson_path), [{"trial": 1, "value": 2.0}])

            csv_path = output_dir / "summary.csv"
            write_csv(csv_path, [{"a": 1, "b": 2}], ["a", "b"])
            self.assertIn("a,b", csv_path.read_text(encoding="utf-8"))

            reset_artifacts(output_dir, ["trials.ndjson", "summary.csv"], resume=False)
            self.assertFalse(ndjson_path.exists())
            self.assertFalse(csv_path.exists())


if __name__ == "__main__":
    unittest.main()
