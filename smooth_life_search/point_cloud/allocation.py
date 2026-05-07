"""Deterministic candidate allocation for point-cloud search stages."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StageAllocationState:
    """Minimal state needed to choose lean point-cloud proposal weights."""

    dimension: int
    evolutionary_active: bool
    basin_polishing_active: bool
    deep_basin_polishing_active: bool
    cooperative_active: bool
    cooperative_heavy: bool
    trust_regions_enabled: bool
    coherent_probes_enabled: bool


def lean_stage_weights(state: StageAllocationState) -> dict[str, float]:
    """Return source weights for the current optimizer stage.

    The policy is intentionally deterministic and small:
    exploration searches broadly, basin polishing focuses near the incumbent,
    and large-D propagation reserves budget for coherent and cooperative
    group moves.
    """

    if not state.evolutionary_active:
        return {}

    if state.basin_polishing_active and state.cooperative_active and state.dimension >= 100:
        if state.deep_basin_polishing_active:
            return {
                "global": 0.02,
                "smoothlife_density": 0.04,
                "region": 0.14 if state.trust_regions_enabled else 0.0,
                "coherent": 0.18 if state.coherent_probes_enabled else 0.0,
                "exploit": 0.30,
                "cooperative": 0.32 if state.cooperative_heavy else 0.18,
            }
        return {
            "global": 0.02,
            "smoothlife_density": 0.04,
            "region": 0.12 if state.trust_regions_enabled else 0.0,
            "coherent": 0.30 if state.coherent_probes_enabled else 0.0,
            "exploit": 0.18,
            "cooperative": 0.34 if state.cooperative_heavy else 0.16,
        }

    if state.basin_polishing_active:
        return {
            "global": 0.04,
            "smoothlife_density": 0.08,
            "region": 0.24 if state.trust_regions_enabled else 0.0,
            "coherent": 0.18 if state.coherent_probes_enabled else 0.0,
            "exploit": 0.46,
            "cooperative": 0.0,
        }

    return {
        "global": 0.22,
        "smoothlife_density": 0.30,
        "region": 0.26 if state.trust_regions_enabled else 0.0,
        "coherent": 0.22 if state.coherent_probes_enabled else 0.0,
        "exploit": 0.0,
        "cooperative": 0.0,
    }
