from __future__ import annotations

import unittest

import numpy as np

from smooth_life_search import MatrixSmoothLifeConfig, MatrixSmoothLifeSearch, SmoothLifeConfig, sphere


def _shifted_sphere(point: np.ndarray) -> float:
    shifted = np.asarray(point, dtype=float)
    return float(np.dot(shifted, shifted))


class TestMatrixSmoothLifeSearch(unittest.TestCase):
    def test_default_matrix_shape_is_near_square_and_large_enough(self) -> None:
        search = MatrixSmoothLifeSearch(
            sphere,
            [(-5.0, 5.0)] * 100,
            SmoothLifeConfig(store_all_snapshots=False),
            MatrixSmoothLifeConfig(max_evaluations=8, store_all_snapshots=False),
        )

        height, width = search.matrix_shape
        self.assertGreaterEqual(height, 16)
        self.assertGreaterEqual(width, 16)
        self.assertGreaterEqual(height * width, max(256, 4 * search.dimension))
        self.assertLessEqual(abs(height - width), 2)

    def test_rejects_explicit_shape_with_too_few_cells(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least dimension cells"):
            MatrixSmoothLifeSearch(
                sphere,
                [(-5.0, 5.0)] * 300,
                SmoothLifeConfig(store_all_snapshots=False),
                MatrixSmoothLifeConfig(max_evaluations=8, matrix_shape=(16, 16), store_all_snapshots=False),
            )

    def test_random_projection_decoder_is_deterministic_by_seed(self) -> None:
        config = MatrixSmoothLifeConfig(max_evaluations=8, store_all_snapshots=False, decoder="random_projection")
        first = MatrixSmoothLifeSearch(sphere, [(-5.0, 5.0)] * 5, SmoothLifeConfig(), config)
        second = MatrixSmoothLifeSearch(sphere, [(-5.0, 5.0)] * 5, SmoothLifeConfig(), config)
        first.reset(seed=11)
        second.reset(seed=11)
        field = np.linspace(-1.0, 1.0, first.matrix_shape[0] * first.matrix_shape[1]).reshape(first.matrix_shape)

        np.testing.assert_allclose(first.projection, second.projection)
        np.testing.assert_allclose(first.decode_field(field), second.decode_field(field))

    def test_local_sparse_decoder_is_deterministic_by_seed(self) -> None:
        config = MatrixSmoothLifeConfig(max_evaluations=8, store_all_snapshots=False)
        first = MatrixSmoothLifeSearch(sphere, [(-5.0, 5.0)] * 13, SmoothLifeConfig(), config)
        second = MatrixSmoothLifeSearch(sphere, [(-5.0, 5.0)] * 13, SmoothLifeConfig(), config)
        first.reset(seed=11)
        second.reset(seed=11)
        field = np.linspace(-1.0, 1.0, first.matrix_shape[0] * first.matrix_shape[1]).reshape(first.matrix_shape)

        np.testing.assert_array_equal(first.local_decoder_axes, second.local_decoder_axes)
        np.testing.assert_allclose(first.local_decoder_weights, second.local_decoder_weights)
        np.testing.assert_allclose(first.decode_field(field), second.decode_field(field))

    def test_neighborhood_features_use_alive_neighbors_and_edges(self) -> None:
        search = MatrixSmoothLifeSearch(
            sphere,
            [(-5.0, 5.0)] * 4,
            SmoothLifeConfig(store_all_snapshots=False),
            MatrixSmoothLifeConfig(max_evaluations=8, store_all_snapshots=False, cell_alive_threshold=0.5),
        )
        field = np.zeros(search.matrix_shape, dtype=float)
        field[0, 0] = 0.6
        field[0, 1] = 0.1
        field[2, 2] = -0.7

        features = search._neighborhood_features(field)

        self.assertIn(0.6, features[1, 1])
        self.assertIn(-0.7, features[1, 1])
        self.assertNotIn(0.1, features[1, 1])
        self.assertLessEqual(int(np.count_nonzero(features[0, 0])), 3)

    def test_local_sparse_decoder_covers_every_coordinate(self) -> None:
        search = MatrixSmoothLifeSearch(
            sphere,
            [(-5.0, 5.0)] * 31,
            SmoothLifeConfig(store_all_snapshots=False),
            MatrixSmoothLifeConfig(max_evaluations=8, store_all_snapshots=False),
        )

        self.assertEqual(search.config.decoder, "local_sparse")
        self.assertEqual(search.local_decoder_axis_counts.shape, (31,))
        self.assertTrue(np.all(search.local_decoder_axis_counts > 0.0))

    def test_decode_respects_bounds_and_zero_field_hits_center(self) -> None:
        bounds = [(-7.0, 13.0)] * 6
        search = MatrixSmoothLifeSearch(
            sphere,
            bounds,
            SmoothLifeConfig(store_all_snapshots=False),
            MatrixSmoothLifeConfig(max_evaluations=8, store_all_snapshots=False),
        )
        search.reset(seed=2)
        center = search.decode_field(np.zeros(search.matrix_shape, dtype=float))
        decoded = search.decode_field(np.ones(search.matrix_shape, dtype=float))

        np.testing.assert_allclose(center, np.asarray([3.0] * 6, dtype=float))
        self.assertTrue(np.all(decoded >= -7.0))
        self.assertTrue(np.all(decoded <= 13.0))

    def test_budget_is_exact_and_archive_deduplicates_fields(self) -> None:
        search = MatrixSmoothLifeSearch(
            sphere,
            [(-5.0, 5.0)] * 4,
            SmoothLifeConfig(store_all_snapshots=False),
            MatrixSmoothLifeConfig(max_evaluations=7, store_all_snapshots=False),
        )
        search.reset(seed=4)
        run = search.run()

        self.assertEqual(run.evaluations, 7)
        self.assertEqual(len(search.archive), 7)
        self.assertEqual(len({sample.field_key for sample in search.archive}), 7)

    def test_shifted_sphere_improves_over_neutral_center(self) -> None:
        bounds = [(-7.0, 13.0)] * 5
        center = np.asarray([3.0] * 5, dtype=float)
        center_value = _shifted_sphere(center)
        search = MatrixSmoothLifeSearch(
            _shifted_sphere,
            bounds,
            SmoothLifeConfig(store_all_snapshots=False),
            MatrixSmoothLifeConfig(max_evaluations=200, store_all_snapshots=False),
        )
        search.reset(seed=7)
        run = search.run()

        self.assertLess(run.best_value, center_value)
        self.assertGreater(run.metadata["reward_max"], 0.0)
        self.assertIn(
            run.metadata["best_source"],
            {"matrix_pulse", "matrix_random", "matrix", "matrix_line_search", "patch_probe"},
        )

    def test_constant_objective_non_improvement_stays_finite(self) -> None:
        def objective(_point: np.ndarray) -> float:
            return 1.0

        search = MatrixSmoothLifeSearch(
            objective,
            [(-5.0, 5.0)] * 3,
            SmoothLifeConfig(store_all_snapshots=False),
            MatrixSmoothLifeConfig(max_evaluations=20, store_all_snapshots=False),
        )
        search.reset(seed=5)
        run = search.run()

        self.assertTrue(np.all(np.isfinite(search.field)))
        self.assertTrue(np.all(np.isfinite(search.reward_field)))
        self.assertEqual(run.best_value, 1.0)
        self.assertEqual(run.metadata["stop_reason"], "budget_exhausted")

    def test_successful_decoded_direction_maps_back_to_matrix_space(self) -> None:
        search = MatrixSmoothLifeSearch(
            sphere,
            [(-5.0, 5.0)] * 5,
            SmoothLifeConfig(store_all_snapshots=False),
            MatrixSmoothLifeConfig(max_evaluations=8, store_all_snapshots=False),
        )
        search.reset(seed=9)
        decoded_delta = np.linspace(-0.1, 0.1, search.dimension)

        first = search._matrix_direction_from_decoded_delta(decoded_delta)
        second = search._matrix_direction_from_decoded_delta(decoded_delta)

        np.testing.assert_allclose(first, second)
        self.assertEqual(first.shape, search.matrix_shape)
        self.assertLessEqual(float(np.max(np.abs(first))), 1.0)

    def test_improvement_updates_advantage_and_direction_fields(self) -> None:
        search = MatrixSmoothLifeSearch(
            _shifted_sphere,
            [(-7.0, 13.0)] * 5,
            SmoothLifeConfig(store_all_snapshots=False),
            MatrixSmoothLifeConfig(max_evaluations=16, store_all_snapshots=False),
        )
        search.reset(seed=2)
        search._active_max_evaluations = 16
        self.assertTrue(search._evaluate_current_field("neutral"))
        search.field = search._field_for_decoder_raw(np.full(search.dimension, -0.25, dtype=float))
        search._refresh_dynamics()

        self.assertTrue(search._evaluate_current_field("test_improvement"))

        self.assertTrue(search._last_improved)
        self.assertGreater(float(np.linalg.norm(search.advantage_field)), 0.0)
        self.assertGreater(float(np.linalg.norm(search.direction_field)), 0.0)
        self.assertGreater(float(np.linalg.norm(search.local_credit_field)), 0.0)
        self.assertTrue(np.all(np.isfinite(search.temperature_field)))

    def test_worsening_update_assigns_negative_local_credit(self) -> None:
        search = MatrixSmoothLifeSearch(
            sphere,
            [(-5.0, 5.0)] * 4,
            SmoothLifeConfig(store_all_snapshots=False),
            MatrixSmoothLifeConfig(max_evaluations=8, store_all_snapshots=False),
        )
        previous = np.zeros(search.matrix_shape, dtype=float)
        current = previous.copy()
        current[4:7, 4:7] = 0.5

        search._update_local_credit_from_evaluation(previous, 1.0, current, 2.0)

        self.assertLess(float(np.min(search.neighborhood_credit)), 0.0)
        self.assertTrue(np.all(np.isfinite(search.local_credit_field)))

    def test_repeated_non_improvement_reheats_temperature(self) -> None:
        search = MatrixSmoothLifeSearch(
            lambda _point: 1.0,
            [(-5.0, 5.0)] * 4,
            SmoothLifeConfig(store_all_snapshots=False),
            MatrixSmoothLifeConfig(
                max_evaluations=8,
                store_all_snapshots=False,
                stagnation_reheat_evaluations=1,
                temperature_reheat=0.2,
            ),
        )
        search.reset(seed=3)
        before = float(np.mean(search.temperature_field))

        search._record_non_improvement_feedback()

        self.assertGreater(float(np.mean(search.temperature_field)), before)
        self.assertTrue(np.all(np.isfinite(search.temperature_field)))

    def test_matrix_line_search_respects_budget_and_can_improve(self) -> None:
        search = MatrixSmoothLifeSearch(
            _shifted_sphere,
            [(-7.0, 13.0)] * 5,
            SmoothLifeConfig(store_all_snapshots=False),
            MatrixSmoothLifeConfig(
                max_evaluations=8,
                store_all_snapshots=False,
                matrix_line_search_alphas=(0.5, 1.0, 1.5, 2.0),
            ),
        )
        search.reset(seed=2)
        search._active_max_evaluations = 8
        self.assertTrue(search._evaluate_current_field("neutral"))
        search.field = search._field_for_decoder_raw(np.full(search.dimension, -0.25, dtype=float))
        search._refresh_dynamics()
        self.assertTrue(search._evaluate_current_field("test_improvement"))
        before = float(search.best_value)

        search._run_matrix_line_search()

        self.assertLessEqual(len(search.archive), 8)
        self.assertEqual(len({sample.field_key for sample in search.archive}), len(search.archive))
        self.assertLessEqual(search.best_value, before)
        self.assertTrue(np.all(search.best_point >= -7.0))
        self.assertTrue(np.all(search.best_point <= 13.0))

    def test_disabled_matrix_line_search_emits_no_line_search_samples(self) -> None:
        search = MatrixSmoothLifeSearch(
            _shifted_sphere,
            [(-7.0, 13.0)] * 5,
            SmoothLifeConfig(store_all_snapshots=False),
            MatrixSmoothLifeConfig(max_evaluations=80, store_all_snapshots=False, matrix_line_search_enabled=False),
        )
        search.reset(seed=7)
        run = search.run()

        self.assertEqual(run.metadata["line_search_improvements"], 0)
        self.assertNotIn("matrix_line_search", run.metadata["source_counts"])

    def test_patch_probe_candidate_changes_only_selected_neighbors(self) -> None:
        search = MatrixSmoothLifeSearch(
            sphere,
            [(-5.0, 5.0)] * 4,
            SmoothLifeConfig(store_all_snapshots=False),
            MatrixSmoothLifeConfig(max_evaluations=8, store_all_snapshots=False),
        )
        search.reset(seed=4)
        row, col = 5, 5
        candidate = search._patch_probe_candidate(row, col)
        changed = set(map(tuple, np.argwhere(np.abs(candidate - search.field) > 1e-12)))
        expected = {(row + dy, col + dx) for dy, dx in search._neighbor_offsets()}

        self.assertEqual(changed, expected)
        self.assertNotIn((row, col), changed)

    def test_patch_probes_respect_budget_and_deduplicate_fields(self) -> None:
        search = MatrixSmoothLifeSearch(
            _shifted_sphere,
            [(-7.0, 13.0)] * 5,
            SmoothLifeConfig(store_all_snapshots=False),
            MatrixSmoothLifeConfig(
                max_evaluations=12,
                store_all_snapshots=False,
                matrix_line_search_enabled=False,
                patch_probe_interval_evaluations=2,
                patch_probe_count=2,
            ),
        )
        search.reset(seed=7)
        run = search.run()

        self.assertLessEqual(run.evaluations, 12)
        self.assertEqual(len({sample.field_key for sample in search.archive}), len(search.archive))
        self.assertGreaterEqual(run.metadata["patch_probe_count"], 0)
        self.assertTrue(np.all(np.isfinite(search.local_credit_field)))

    def test_disabled_patch_probes_emit_no_patch_samples(self) -> None:
        search = MatrixSmoothLifeSearch(
            _shifted_sphere,
            [(-7.0, 13.0)] * 5,
            SmoothLifeConfig(store_all_snapshots=False),
            MatrixSmoothLifeConfig(
                max_evaluations=40,
                store_all_snapshots=False,
                patch_probe_enabled=False,
            ),
        )
        search.reset(seed=7)
        run = search.run()

        self.assertEqual(run.metadata["patch_probe_count"], 0)
        self.assertNotIn("patch_probe", run.metadata["source_counts"])

    def test_snapshots_are_finite_and_nonblank(self) -> None:
        search = MatrixSmoothLifeSearch(
            sphere,
            [(-5.0, 5.0)] * 4,
            SmoothLifeConfig(store_all_snapshots=True),
            MatrixSmoothLifeConfig(max_evaluations=16, snapshot_interval=2, store_all_snapshots=True),
        )
        search.reset(seed=6)
        run = search.run()

        self.assertTrue(run.snapshots)
        snapshot = run.snapshots[-1]
        self.assertTrue(np.all(np.isfinite(snapshot.field)))
        self.assertTrue(np.all(np.isfinite(snapshot.objective_field)))
        self.assertGreater(float(np.max(snapshot.field) - np.min(snapshot.field)), 0.0)


if __name__ == "__main__":
    unittest.main()
