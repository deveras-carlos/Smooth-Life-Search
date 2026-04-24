"""Shared benchmark objective registry and default bounds."""

from __future__ import annotations

from typing import Callable

import numpy as np

from .benchmark import ackley, griewank, himmelblau, rastrigin, rosenbrock, sphere

ObjectiveFn = Callable[[np.ndarray], float]

OBJECTIVES: dict[str, ObjectiveFn] = {
    "ackley": ackley,
    "griewank": griewank,
    "himmelblau": himmelblau,
    "rastrigin": rastrigin,
    "rosenbrock": rosenbrock,
    "sphere": sphere,
}

DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "ackley": (-10.0, 10.0),
    "griewank": (-10.0, 10.0),
    "himmelblau": (-6.0, 6.0),
    "rastrigin": (-5.12, 5.12),
    "rosenbrock": (-10.0, 10.0),
    "sphere": (-10.0, 10.0),
}

__all__ = ["DEFAULT_BOUNDS", "OBJECTIVES", "ObjectiveFn"]
