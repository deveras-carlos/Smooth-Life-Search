from __future__ import annotations

import unittest

import numpy as np

from smooth_life_search.point_cloud import fit_quadratic_surrogate


class TestPointCloudSurrogate(unittest.TestCase):
    def test_quadratic_surrogate_recovers_minimizer(self) -> None:
        xs = np.linspace(-1.0, 1.0, 11)
        ys = np.linspace(-1.0, 1.0, 11)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        values = (points[:, 0] - 0.2) ** 2 + 2.0 * (points[:, 1] + 0.3) ** 2

        result = fit_quadratic_surrogate(
            points,
            values,
            region_bounds=np.asarray([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
            center=np.asarray([0.0, 0.0], dtype=float),
            maximize=False,
            min_samples=12,
            max_samples=96,
            regularization=1e-10,
            max_condition=1e10,
        )

        self.assertTrue(result.accepted, msg=result.reason)
        self.assertTrue(np.allclose(result.point, [0.2, -0.3], atol=1e-4))

    def test_quadratic_surrogate_rejects_nonconvex_fit(self) -> None:
        xs = np.linspace(-1.0, 1.0, 9)
        ys = np.linspace(-1.0, 1.0, 9)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="xy")
        points = np.column_stack((grid_x.ravel(), grid_y.ravel()))
        values = (points[:, 0] - 0.2) ** 2 - (points[:, 1] + 0.3) ** 2

        result = fit_quadratic_surrogate(
            points,
            values,
            region_bounds=np.asarray([[-1.0, 1.0], [-1.0, 1.0]], dtype=float),
            center=np.asarray([0.0, 0.0], dtype=float),
            maximize=False,
            min_samples=12,
            max_samples=96,
            regularization=1e-10,
            max_condition=1e10,
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, "nonconvex_surrogate")


if __name__ == "__main__":
    unittest.main()
