from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

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
            resolve_objective({"kind": "bogus", "name": "surface"})

    def test_resolves_python_callable_import_path(self) -> None:
        objective = resolve_objective(
            {
                "kind": "import_path",
                "import_path": "smooth_life_search.benchmark.functions:sphere",
            }
        )
        self.assertEqual(objective(np.asarray([1.0, 2.0], dtype=float)), 5.0)

    def test_invalid_import_path_raises_clear_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "non-empty import_path"):
            resolve_objective({"kind": "import_path"})
        with self.assertRaisesRegex(ValueError, "module:function"):
            resolve_objective({"kind": "import_path", "import_path": "smooth_life_search.benchmark.functions.sphere"})

    def test_resolves_csv_surface_objective_with_bilinear_interpolation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "surface.csv"
            path.write_text(
                "x,y,value\n"
                "0,0,0\n"
                "1,0,10\n"
                "0,1,20\n"
                "1,1,30\n",
                encoding="utf-8",
            )
            objective = resolve_objective({"kind": "csv_surface", "csv_path": str(path)})
            self.assertEqual(objective(np.asarray([0.0, 0.0], dtype=float)), 0.0)
            self.assertEqual(objective(np.asarray([1.0, 1.0], dtype=float)), 30.0)
            self.assertEqual(objective(np.asarray([0.5, 0.5], dtype=float)), 15.0)

    def test_csv_surface_rejects_malformed_grids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            duplicate = Path(tmpdir) / "duplicate.csv"
            duplicate.write_text("x,y,value\n0,0,1\n0,0,2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                resolve_objective({"kind": "csv_surface", "csv_path": str(duplicate)})

            missing = Path(tmpdir) / "missing.csv"
            missing.write_text("x,y,value\n0,0,1\n1,0,2\n0,1,3\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "complete rectangular grid"):
                resolve_objective({"kind": "csv_surface", "csv_path": str(missing)})


if __name__ == "__main__":
    unittest.main()
