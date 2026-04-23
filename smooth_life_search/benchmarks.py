"""Common continuous optimization benchmarks."""

from __future__ import annotations

import math

import numpy as np


def _as_array(x: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError("benchmark functions expect a 1D input vector")
    return arr


def sphere(x: np.ndarray | list[float] | tuple[float, ...]) -> float:
    arr = _as_array(x)
    return float(np.dot(arr, arr))


def rosenbrock(x: np.ndarray | list[float] | tuple[float, ...]) -> float:
    arr = _as_array(x)
    if arr.size < 2:
        raise ValueError("rosenbrock requires at least 2 dimensions")
    return float(np.sum(100.0 * np.square(arr[1:] - np.square(arr[:-1])) + np.square(1.0 - arr[:-1])))


def ackley(x: np.ndarray | list[float] | tuple[float, ...]) -> float:
    arr = _as_array(x)
    n = arr.size
    term_1 = -20.0 * math.exp(-0.2 * math.sqrt(np.sum(np.square(arr)) / n))
    term_2 = -math.exp(np.sum(np.cos(2.0 * math.pi * arr)) / n)
    return float(term_1 + term_2 + 20.0 + math.e)


def rastrigin(x: np.ndarray | list[float] | tuple[float, ...]) -> float:
    arr = _as_array(x)
    n = arr.size
    return float(10.0 * n + np.sum(np.square(arr) - 10.0 * np.cos(2.0 * math.pi * arr)))


def griewank(x: np.ndarray | list[float] | tuple[float, ...]) -> float:
    arr = _as_array(x)
    i = np.arange(1, arr.size + 1, dtype=float)
    return float(np.sum(np.square(arr)) / 4000.0 - np.prod(np.cos(arr / np.sqrt(i))) + 1.0)


def himmelblau(x: np.ndarray | list[float] | tuple[float, ...]) -> float:
    arr = _as_array(x)
    if arr.size != 2:
        raise ValueError("himmelblau is defined for exactly 2 dimensions")
    x0, x1 = arr
    return float((x0 * x0 + x1 - 11.0) ** 2 + (x0 + x1 * x1 - 7.0) ** 2)
