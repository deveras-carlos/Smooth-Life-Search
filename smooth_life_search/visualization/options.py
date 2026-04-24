"""Render options for images, animations, and viewers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RenderOptions:
    """Common visualization settings."""

    scale: int = 2
    duration_ms: int = 90

    def __post_init__(self) -> None:
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.duration_ms <= 0:
            raise ValueError("duration_ms must be positive")
