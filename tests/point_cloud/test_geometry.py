from __future__ import annotations

import unittest

import numpy as np

from smooth_life_search.point_cloud.geometry import fit_region_geometry


class TestPointCloudRegionGeometry(unittest.TestCase):
    def test_identity_fallback_with_too_few_samples(self) -> None:
        geometry = fit_region_geometry(
            np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=float),
            np.asarray([1.0, 0.0], dtype=float),
            bounds=np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=float),
            center=np.asarray([0.5, 0.5], dtype=float),
            radius_fraction=0.25,
            maximize=False,
            min_samples=8,
            anisotropy_max=25.0,
        )

        self.assertEqual(geometry.reason, "insufficient_samples")
        self.assertTrue(np.allclose(geometry.basis, np.eye(2)))
        self.assertEqual(geometry.anisotropy, 1.0)

    def test_covariance_basis_aligns_with_elongated_cloud(self) -> None:
        direction = np.asarray([1.0, 0.45], dtype=float)
        direction = direction / np.linalg.norm(direction)
        normal = np.asarray([-direction[1], direction[0]], dtype=float)
        offsets = np.linspace(-0.35, 0.35, 48)
        minor = 0.01 * np.sin(np.linspace(0.0, 3.0 * np.pi, offsets.size))
        points = np.asarray([0.5, 0.5]) + offsets[:, None] * direction + minor[:, None] * normal
        values = offsets * offsets + 0.1 * minor * minor

        geometry = fit_region_geometry(
            points,
            values,
            bounds=np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=float),
            center=np.asarray([0.5, 0.5], dtype=float),
            radius_fraction=0.35,
            maximize=False,
            min_samples=8,
            anisotropy_max=25.0,
        )

        self.assertEqual(geometry.reason, "archive_covariance")
        self.assertGreater(abs(float(np.dot(geometry.basis[:, 0], direction))), 0.95)
        self.assertGreater(geometry.anisotropy, 2.0)

    def test_anisotropy_is_capped_and_finite(self) -> None:
        points = np.column_stack((np.linspace(0.1, 0.9, 64), np.full(64, 0.5)))
        points[:, 1] += np.linspace(-1e-6, 1e-6, 64)
        values = (points[:, 0] - 0.5) ** 2

        geometry = fit_region_geometry(
            points,
            values,
            bounds=np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=float),
            center=np.asarray([0.5, 0.5], dtype=float),
            radius_fraction=0.50,
            maximize=False,
            min_samples=8,
            anisotropy_max=3.0,
        )

        self.assertLessEqual(geometry.anisotropy, 3.0)
        self.assertTrue(np.all(np.isfinite(geometry.eigenvalues)))
        self.assertTrue(np.all(np.isfinite(geometry.axis_scales)))

    def test_nd_geometry_uses_capped_active_subspace(self) -> None:
        rng = np.random.default_rng(3)
        points = rng.normal(0.0, 0.05, size=(80, 5))
        points[:, 0] += np.linspace(-0.4, 0.4, points.shape[0])
        points += 0.5
        values = np.sum(np.square(points - 0.5), axis=1)

        geometry = fit_region_geometry(
            points,
            values,
            bounds=np.asarray([[0.0, 1.0]] * 5, dtype=float),
            center=np.full(5, 0.5, dtype=float),
            radius_fraction=0.50,
            maximize=False,
            min_samples=8,
            anisotropy_max=5.0,
            active_subspace_size=3,
        )

        self.assertEqual(geometry.reason, "archive_covariance")
        self.assertEqual(geometry.basis.shape, (5, 3))
        self.assertEqual(geometry.axis_scales.shape, (3,))
        self.assertLessEqual(geometry.anisotropy, 5.0)
        self.assertTrue(np.all(np.isfinite(geometry.basis)))


if __name__ == "__main__":
    unittest.main()
