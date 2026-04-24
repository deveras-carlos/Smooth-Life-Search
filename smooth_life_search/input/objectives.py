"""Objective specifications loaded from user-facing input."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any, Literal, Mapping

from ..benchmark import OBJECTIVES
from ..core import Objective

ObjectiveKind = Literal["builtin", "import_path"]


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    """Declarative description of an objective function."""

    kind: ObjectiveKind
    name: str | None = None
    import_path: str | None = None

    @classmethod
    def builtin(cls, name: str) -> "ObjectiveSpec":
        """Build a spec for one of the bundled benchmark objectives."""

        return cls(kind="builtin", name=name)

    @classmethod
    def python_callable(cls, import_path: str) -> "ObjectiveSpec":
        """Build a spec for a Python callable identified by ``module:function``."""

        return cls(kind="import_path", import_path=import_path)

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
    raise ValueError(f"unsupported objective kind: {resolved.kind}")
