from __future__ import annotations

import unittest

import numpy as np

from smooth_life_search.point_cloud import PointCloudArchive


class TestPointCloudArchive(unittest.TestCase):
    def test_archive_deduplicates_exact_points(self) -> None:
        archive = PointCloudArchive.empty()
        point = np.asarray([0.25, -0.5], dtype=float)

        first = archive.add(point, 1.0, source="test", batch_index=0)
        second = archive.add(point.copy(), 2.0, source="test", batch_index=0)
        points, values = archive.arrays()

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(len(archive), 1)
        self.assertEqual(points.shape, (1, 2))
        self.assertEqual(values.tolist(), [1.0])

    def test_archive_deduplicates_exact_points_in_5d(self) -> None:
        archive = PointCloudArchive.empty()
        point = np.asarray([0.25, -0.5, 1.0, 2.0, -3.0], dtype=float)

        first = archive.add(point, 1.0, source="test", batch_index=0)
        second = archive.add(point.copy(), 2.0, source="test", batch_index=0)
        points, values = archive.arrays()

        self.assertIsNotNone(first)
        self.assertIsNone(second)
        self.assertEqual(points.shape, (1, 5))
        self.assertEqual(values.tolist(), [1.0])

    def test_nearest_distance_is_normalized_by_bounds(self) -> None:
        archive = PointCloudArchive.empty()
        archive.add(np.asarray([0.0, 0.0], dtype=float), 0.0, source="test", batch_index=0)

        distances = archive.nearest_distance(
            np.asarray([[5.0, 0.0], [0.0, 10.0]], dtype=float),
            np.asarray([[0.0, 10.0], [0.0, 20.0]], dtype=float),
        )

        self.assertTrue(np.allclose(distances, [0.5, 0.5]))


if __name__ == "__main__":
    unittest.main()
