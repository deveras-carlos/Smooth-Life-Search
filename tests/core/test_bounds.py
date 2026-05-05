from __future__ import annotations

import unittest

import numpy as np

from smooth_life_search.core import normalize_bounds_nd


class TestBoundsND(unittest.TestCase):
    def test_normalize_bounds_nd_accepts_matching_dimension(self) -> None:
        bounds = normalize_bounds_nd([(-1.0, 1.0)] * 5, owner="test", dimension=5)

        self.assertEqual(bounds.shape, (5, 2))
        self.assertTrue(np.all(bounds[:, 0] == -1.0))
        self.assertTrue(np.all(bounds[:, 1] == 1.0))

    def test_normalize_bounds_nd_rejects_invalid_shapes_and_dimension_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2 dimensions"):
            normalize_bounds_nd([(0.0, 1.0)], owner="test")
        with self.assertRaisesRegex(ValueError, "expected 5D"):
            normalize_bounds_nd([(-1.0, 1.0)] * 4, owner="test", dimension=5)
        with self.assertRaisesRegex(ValueError, "lower < upper"):
            normalize_bounds_nd([(1.0, 1.0), (0.0, 1.0)], owner="test")


if __name__ == "__main__":
    unittest.main()
