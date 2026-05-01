from __future__ import annotations

import unittest

import numpy as np

from smooth_life_search.agsls.config import AGSLSConfig
from smooth_life_search.agsls.search import AdaptiveGridSmoothLifeSearch
from smooth_life_search.agsls.trust_region import (
    ArchiveQuadraticSurrogate,
    SampleArchive,
    fit_archive_quadratic,
    nearest_archive_distance,
)
from smooth_life_search.smoothlife.config import SmoothLifeConfig


class TestTrustRegionHelpers(unittest.TestCase):
    def test_sample_archive_persists_grid_samples_across_remaps(self) -> None:
        archive = SampleArchive.empty()
        values_a = np.asarray([[1.0, np.nan], [2.0, 3.0]], dtype=float)
        mask_a = np.asarray([[True, False], [True, True]], dtype=bool)
        bounds_a = np.asarray([[0.0, 2.0], [0.0, 2.0]], dtype=float)

        values_b = np.asarray([[4.0, 5.0], [np.nan, 6.0]], dtype=float)
        mask_b = np.asarray([[True, True], [False, True]], dtype=bool)
        bounds_b = np.asarray([[10.0, 12.0], [20.0, 22.0]], dtype=float)

        self.assertEqual(archive.add_grid(values_a, mask_a, bounds_a, (2, 2)), 3)
        self.assertEqual(archive.add_grid(values_b, mask_b, bounds_b, (2, 2)), 3)
        points, values = archive.arrays()

        self.assertEqual(points.shape, (6, 2))
        self.assertEqual(values.tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.assertTrue(np.any(np.all(np.isclose(points, [0.5, 0.5]), axis=1)))
        self.assertTrue(np.any(np.all(np.isclose(points, [10.5, 20.5]), axis=1)))

    def test_archive_quadratic_recovers_synthetic_minimizer(self) -> None:
        xs = np.linspace(-1.0, 1.0, 9)
        ys = np.linspace(-1.0, 1.0, 9)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        values = (points[:, 0] - 0.20) ** 2 + 2.0 * (points[:, 1] + 0.10) ** 2

        surrogate = fit_archive_quadratic(
            archive_points=points,
            archive_values=values,
            region_bounds=np.asarray([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
            center=np.asarray([0.0, 0.0], dtype=float),
            maximize=False,
            min_samples=12,
            max_samples=96,
            regularization=1e-10,
            max_condition=1e8,
        )

        self.assertTrue(surrogate.accepted, msg=surrogate.reason)
        self.assertIsNotNone(surrogate.point)
        self.assertTrue(np.allclose(surrogate.point, [0.20, -0.10], atol=1e-4))

    def test_acquisition_duplicate_filter_and_prediction_ranking(self) -> None:
        search = AdaptiveGridSmoothLifeSearch(
            lambda point: float(np.sum(point * point)),
            np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=float),
            SmoothLifeConfig(grid_shape=(32, 32), store_all_snapshots=False),
            AGSLSConfig(max_evaluations=200),
        )
        search.reset(seed=3)
        state = search.engine.state
        assert state is not None
        state.best_value = 0.5
        search.sample_archive = SampleArchive.empty()
        search.sample_archive.add_point(np.asarray([0.0, 0.0], dtype=float), 0.0)

        candidates = np.asarray([[0.0, 0.0], [0.8, 0.1], [0.2, 0.2]], dtype=float)
        surrogate = ArchiveQuadraticSurrogate(
            accepted=True,
            reason="accepted",
            sample_count=12,
            region_bounds=np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=float),
            coefficients=np.asarray([0.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
            target_min=0.0,
            target_span=1.0,
        )
        ranked = search._rank_trust_region_candidates(
            phase="exploitation",
            candidates=candidates,
            surrogate=surrogate,
            current_bounds=state.bounds,
        )

        self.assertEqual(int(ranked[0]), 1)
        distances = nearest_archive_distance(candidates, search.sample_archive.arrays()[0], state.bounds)
        self.assertEqual(float(distances[0]), 0.0)

    def test_direct_sample_updates_best_without_mutating_grid_cache(self) -> None:
        search = AdaptiveGridSmoothLifeSearch(
            lambda point: float(np.sum(point * point)),
            np.asarray([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
            SmoothLifeConfig(grid_shape=(32, 32), store_all_snapshots=False),
            AGSLSConfig(max_evaluations=200),
        )
        search.reset(seed=5)
        state = search.engine.state
        assert state is not None
        mask_before = state.evaluated_mask.copy()
        state.best_point = np.asarray([0.5, 0.5], dtype=float)
        state.best_value = 0.5

        value, improved = search._evaluate_direct_sample(np.asarray([0.0, 0.0], dtype=float))

        self.assertTrue(improved)
        self.assertEqual(value, 0.0)
        self.assertEqual(float(state.best_value), 0.0)
        self.assertTrue(np.array_equal(state.evaluated_mask, mask_before))
        self.assertGreater(len(search.sample_archive), 0)

    def test_trust_region_radius_expands_on_success_and_shrinks_on_failure(self) -> None:
        config = AGSLSConfig(
            max_evaluations=400,
            exploitation_trust_region_evaluations=4,
            trust_region_initial_radius_fraction=0.25,
            trust_region_expand_factor=1.4,
            trust_region_shrink_factor=0.5,
        )
        search = AdaptiveGridSmoothLifeSearch(
            lambda point: float((point[0] - 0.25) ** 2 + point[1] ** 2),
            np.asarray([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
            SmoothLifeConfig(grid_shape=(32, 32), store_all_snapshots=False),
            config,
        )
        search.reset(seed=7)
        state = search.engine.state
        assert state is not None
        state.best_point = np.asarray([0.0, 0.0], dtype=float)
        state.best_value = 0.25**2

        success = search._run_trust_region_batch("exploitation", None, state.bounds.copy())
        self.assertTrue(bool(success["trust_region_success"]))
        self.assertGreater(float(success["trust_region_radius_after"]), float(success["trust_region_radius_before"]))

        search.objective = lambda point: float(np.sum(point * point))
        state.best_point = np.asarray([0.0, 0.0], dtype=float)
        state.best_value = 0.0
        before = float(search._trust_region_state("exploitation").radius_fraction or 0.25)
        failure = search._run_trust_region_batch("exploitation", None, state.bounds.copy())
        self.assertFalse(bool(failure["trust_region_success"]))
        self.assertLess(float(failure["trust_region_radius_after"]), before)

    def test_run_metadata_emits_commit_or_exploitation_trust_events(self) -> None:
        search = AdaptiveGridSmoothLifeSearch(
            lambda point: float(np.sum(point * point)),
            np.asarray([[-2.0, 2.0], [-2.0, 2.0]], dtype=float),
            SmoothLifeConfig(grid_shape=(32, 32), store_all_snapshots=False, evaluations_per_step=8),
            AGSLSConfig(
                max_evaluations=600,
                max_zoom_cycles=2,
                exploration_fraction=0.05,
                commit_fraction=0.60,
                commit_steps_per_zoom=4,
                exploitation_steps_per_zoom=4,
                commit_trust_region_evaluations=2,
                exploitation_trust_region_evaluations=2,
            ),
        )
        search.reset(seed=11)
        run = search.run()
        events = run.metadata.get("trust_region_events", [])

        self.assertTrue(events)
        self.assertTrue(all(event["phase"] in ("commit", "exploitation") for event in events))
        self.assertEqual(run.metadata.get("sample_archive_size"), len(search.sample_archive))


if __name__ == "__main__":
    unittest.main()
