"""Objective specifications loaded from user-facing input."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import importlib
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from ..benchmark import OBJECTIVES
from ..core import Objective

ObjectiveKind = Literal["builtin", "import_path", "csv_surface"]


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    """Declarative description of an objective function."""

    kind: ObjectiveKind
    name: str | None = None
    import_path: str | None = None
    csv_path: str | None = None

    @classmethod
    def builtin(cls, name: str) -> "ObjectiveSpec":
        """Build a spec for one of the bundled benchmark objectives."""

        return cls(kind="builtin", name=name)

    @classmethod
    def python_callable(cls, import_path: str) -> "ObjectiveSpec":
        """Build a spec for a Python callable identified by ``module:function``."""

        return cls(kind="import_path", import_path=import_path)

    @classmethod
    def csv_surface(cls, csv_path: str | Path) -> "ObjectiveSpec":
        """Build a spec for a sampled 2D objective surface."""

        return cls(kind="csv_surface", csv_path=str(csv_path))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ObjectiveSpec":
        """Build a spec from a config-file dictionary."""

        kind = str(payload.get("kind", "builtin"))
        if kind == "builtin":
            name = payload.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("builtin objective specs require a non-empty name")
            return cls.builtin(name)
        if kind == "import_path":
            import_path = payload.get("import_path")
            if not isinstance(import_path, str) or not import_path:
                raise ValueError("import_path objective specs require a non-empty import_path")
            return cls.python_callable(import_path)
        if kind == "csv_surface":
            csv_path = payload.get("csv_path", payload.get("path"))
            if not isinstance(csv_path, str) or not csv_path:
                raise ValueError("csv_surface objective specs require a non-empty csv_path")
            return cls.csv_surface(csv_path)
        raise ValueError(f"unsupported objective kind: {kind}")


def _load_import_path(import_path: str) -> Objective:
    """Load a callable from ``module:function`` syntax."""

    module_name, separator, attribute_name = import_path.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("import_path objective specs must use 'module:function' syntax")
    module = importlib.import_module(module_name)
    candidate = getattr(module, attribute_name)
    if not callable(candidate):
        raise ValueError(f"imported objective is not callable: {import_path}")
    return candidate


@dataclass(frozen=True, slots=True)
class CsvSurfaceObjective:
    """2D sampled surface with bilinear interpolation."""

    x_values: np.ndarray
    y_values: np.ndarray
    values: np.ndarray

    def __call__(self, point: np.ndarray) -> float:
        arr = np.asarray(point, dtype=float)
        if arr.shape != (2,):
            raise ValueError("CSV surface objectives expect a 2D point")
        x = float(arr[0])
        y = float(arr[1])
        if x < float(self.x_values[0]) or x > float(self.x_values[-1]):
            raise ValueError("point x-coordinate is outside the CSV surface domain")
        if y < float(self.y_values[0]) or y > float(self.y_values[-1]):
            raise ValueError("point y-coordinate is outside the CSV surface domain")
        x0_idx = self._lower_interval_index(self.x_values, x)
        y0_idx = self._lower_interval_index(self.y_values, y)
        x0 = float(self.x_values[x0_idx])
        x1 = float(self.x_values[x0_idx + 1])
        y0 = float(self.y_values[y0_idx])
        y1 = float(self.y_values[y0_idx + 1])
        tx = (x - x0) / (x1 - x0)
        ty = (y - y0) / (y1 - y0)
        v00 = float(self.values[y0_idx, x0_idx])
        v10 = float(self.values[y0_idx, x0_idx + 1])
        v01 = float(self.values[y0_idx + 1, x0_idx])
        v11 = float(self.values[y0_idx + 1, x0_idx + 1])
        lower = (1.0 - tx) * v00 + tx * v10
        upper = (1.0 - tx) * v01 + tx * v11
        return float((1.0 - ty) * lower + ty * upper)

    @staticmethod
    def _lower_interval_index(axis: np.ndarray, value: float) -> int:
        if value >= float(axis[-1]):
            return int(axis.size - 2)
        return max(0, min(int(np.searchsorted(axis, value, side="right") - 1), int(axis.size - 2)))


def _load_csv_surface(csv_path: str) -> CsvSurfaceObjective:
    path = Path(csv_path)
    points: dict[tuple[float, float], float] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"x", "y", "value"}
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            raise ValueError("CSV surface files must include x, y, and value columns")
        for row in reader:
            try:
                key = (float(row["x"]), float(row["y"]))
                value = float(row["value"])
            except (TypeError, ValueError) as exc:
                raise ValueError("CSV surface rows must contain numeric x, y, and value fields") from exc
            if key in points:
                raise ValueError(f"duplicate CSV surface sample at x={key[0]}, y={key[1]}")
            points[key] = value
    if not points:
        raise ValueError("CSV surface files must contain at least one sample")
    x_values = np.asarray(sorted({key[0] for key in points}), dtype=float)
    y_values = np.asarray(sorted({key[1] for key in points}), dtype=float)
    if x_values.size < 2 or y_values.size < 2:
        raise ValueError("CSV surface objectives require at least two x and y coordinates")
    expected_count = int(x_values.size * y_values.size)
    if len(points) != expected_count:
        raise ValueError("CSV surface samples must form a complete rectangular grid")
    values = np.empty((y_values.size, x_values.size), dtype=float)
    for y_index, y in enumerate(y_values):
        for x_index, x in enumerate(x_values):
            try:
                values[y_index, x_index] = points[(float(x), float(y))]
            except KeyError as exc:
                raise ValueError("CSV surface samples must form a complete rectangular grid") from exc
    return CsvSurfaceObjective(x_values=x_values, y_values=y_values, values=values)


def resolve_objective(spec: ObjectiveSpec | Mapping[str, Any] | str) -> Objective:
    """Resolve an objective spec into a callable objective."""

    if isinstance(spec, str):
        resolved = ObjectiveSpec.builtin(spec)
    elif isinstance(spec, ObjectiveSpec):
        resolved = spec
    else:
        resolved = ObjectiveSpec.from_mapping(spec)

    if resolved.kind == "builtin":
        try:
            return OBJECTIVES[str(resolved.name)]
        except KeyError as exc:
            raise ValueError(f"unknown builtin objective: {resolved.name}") from exc
    if resolved.kind == "import_path":
        return _load_import_path(str(resolved.import_path))
    if resolved.kind == "csv_surface":
        return _load_csv_surface(str(resolved.csv_path))
    raise ValueError(f"unsupported objective kind: {resolved.kind}")
