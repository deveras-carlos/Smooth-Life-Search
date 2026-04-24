from __future__ import annotations

import unittest

import numpy as np

from smooth_life_search.input import ObjectiveSpec, resolve_objective


class TestObjectiveSpec(unittest.TestCase):
    def test_resolves_builtin_by_name_string_and_mapping(self) -> None:
        point = np.asarray([3.0, 4.0], dtype=float)
        self.assertEqual(resolve_objective("sphere")(point), 25.0)
        self.assertEqual(resolve_objective({"kind": "builtin", "name": "sphere"})(point), 25.0)
        self.assertEqual(resolve_objective(ObjectiveSpec.builtin("sphere"))(point), 25.0)

    def test_unknown_builtin_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown builtin objective"):
            resolve_objective("missing")

    def test_invalid_mapping_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty name"):
            resolve_objective({"kind": "builtin"})
        with self.assertRaisesRegex(ValueError, "unsupported objective kind"):
            resolve_objective({"kind": "csv_surface", "name": "surface"})


if __name__ == "__main__":
    unittest.main()
