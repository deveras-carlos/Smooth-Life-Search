"""Signed SmoothLife transition with bounded objective support."""

from __future__ import annotations

import numpy as np

from .config import SmoothLifeConfig


def sigmoid( x: np.ndarray, center: float | np.ndarray, width: float ) -> np.ndarray:
    """Smooth step used by SmoothLife transition rules."""

    scale = max( width, 1e-12 )
    return 1.0 / ( 1.0 + np.exp( -( x - center ) * 4.0 / scale ) )


def mix_intervals( x0: float, x1: float, inner_fill: np.ndarray, width: float ) -> np.ndarray:
    """Interpolate between birth and death thresholds based on inner fill."""

    mixer = sigmoid( inner_fill, 0.5, width )
    return x0 * ( 1.0 - mixer ) + x1 * mixer


def smoothlife_target_vitality(
    inner_fill: np.ndarray,
    outer_fill: np.ndarray,
    config: SmoothLifeConfig,
    objective_field: np.ndarray | None = None,
) -> np.ndarray:
    """Compute the canonical SmoothLife vitality target in [ 0, 1 ]."""

    low = mix_intervals( config.birth_low, config.death_low, inner_fill, config.alpha_m )
    high = mix_intervals( config.birth_high, config.death_high, inner_fill, config.alpha_m )
    base = sigmoid( outer_fill, low, config.alpha_n ) * ( 1.0 - sigmoid( outer_fill, high, config.alpha_n ) )
    return np.clip( base, 0.0, 1.0 )


def _resolve_polarity( field: np.ndarray ) -> np.ndarray:
    """Preserve the sign of existing structures and break exact-zero ties locally."""

    polarity = np.sign( field )
    unresolved = polarity == 0.0
    if not np.any( unresolved ):
        return polarity
    neighbor_sum = (
        np.roll( field, 1, axis=0 )
        + np.roll( field, -1, axis=0 )
        + np.roll( field, 1, axis=1 )
        + np.roll( field, -1, axis=1 )
    )
    neighbor_sign = np.sign( neighbor_sum )
    polarity = np.where( unresolved, neighbor_sign, polarity )
    polarity[ polarity == 0.0 ] = 1.0
    return polarity


def smoothlife_transition(
    field: np.ndarray,
    inner_fill: np.ndarray,
    outer_fill: np.ndarray,
    config: SmoothLifeConfig,
    objective_field: np.ndarray | None = None,
) -> tuple[ np.ndarray, np.ndarray ]:
    """Return the signed target state and the target vitality field."""

    target_vitality = smoothlife_target_vitality( inner_fill, outer_fill, config, objective_field )
    if objective_field is None or config.run_mode == "simulation" or config.objective_coupling <= 0.0:
        life_support = target_vitality
    else:
        objective = np.clip(np.nan_to_num(objective_field, nan=0.5, posinf=1.0, neginf=0.0), 0.0, 1.0)
        support_bias = ( 1.0 - config.objective_coupling ) + config.objective_coupling * objective
        life_support = np.clip( target_vitality * support_bias, 0.0, 1.0 )
    target_magnitude = 1.0 - life_support
    polarity = _resolve_polarity( field )
    target_state = polarity * target_magnitude
    return np.clip( target_state, -1.0, 1.0 ), target_vitality
