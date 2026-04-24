"""Objective specifications loaded from user-facing input."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from ..benchmark import OBJECTIVES
from ..core import Objective

ObjectiveKind = Literal["builtin"]


@dataclass(frozen=True, slots=True)
class ObjectiveSpec:
    """Declarative description of an objective function."""

    kind: ObjectiveKind
    name: str

    @classmethod
    def builtin(cls, name: str) -> "ObjectiveSpec":
        """Build a spec for one of the bundled benchmark objectives."""

        return cls(kind="builtin", name=name)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "ObjectiveSpec":
        """Build a spec from a config-file dictionary."""

        kind = str(payload.get("kind", "builtin"))
        if kind != "builtin":
            raise ValueError(f"unsupported objective kind: {kind}")
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("builtin objective specs require a non-empty name")
        return cls.builtin(name)


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
            return OBJECTIVES[resolved.name]
        except KeyError as exc:
            raise ValueError(f"unknown builtin objective: {resolved.name}") from exc
    raise ValueError(f"unsupported objective kind: {resolved.kind}")
