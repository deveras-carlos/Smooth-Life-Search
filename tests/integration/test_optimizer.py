from __future__ import annotations

import unittest

from smooth_life_search import MatrixSmoothLifeConfig, MatrixSmoothLifeSearch, SmoothLifeConfig, sphere


class TestMatrixIntegration(unittest.TestCase):
    def test_matrix_run_records_archive_and_metadata(self) -> None:
        config = MatrixSmoothLifeConfig(
            max_evaluations=40,
            matrix_shape=(16, 16),
            snapshot_interval=4,
            store_all_snapshots=False,
        )
        search = MatrixSmoothLifeSearch(
            sphere,
            [(-5.0, 5.0), (-5.0, 5.0), (-5.0, 5.0)],
            SmoothLifeConfig(store_all_snapshots=False, preset="search"),
            config,
        )
        search.reset(seed=3)
        run = search.run()

        self.assertEqual(run.evaluations, config.max_evaluations)
        self.assertEqual(run.metadata["mode"], "matrix")
        self.assertEqual(run.metadata["archive_size"], run.evaluations)
        self.assertEqual(run.metadata["matrix_shape"], [16, 16])
        self.assertIn("source_counts", run.metadata)
        self.assertEqual(run.zoom_events, [])

    def test_python_api_requires_matrix_budget(self) -> None:
        search = MatrixSmoothLifeSearch(
            sphere,
            [(-5.0, 5.0), (-5.0, 5.0)],
            SmoothLifeConfig(store_all_snapshots=False),
            MatrixSmoothLifeConfig(max_evaluations=None, store_all_snapshots=False),
        )
        search.reset(seed=0)
        with self.assertRaisesRegex(ValueError, "max_evaluations"):
            search.run()


if __name__ == "__main__":
    unittest.main()
