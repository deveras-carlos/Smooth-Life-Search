"""Research-grade parallel tuning study harness for SmoothLifeSearch and AGSLS."""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, replace
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import statistics
import tempfile
import time
from typing import Any, Callable, Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .core import (
    AGSLS_PER_DECISION_FIELDS,
    FieldSchedule,
    SMOOTHLIFE_PER_STEP_FIELDS,
    SchedulePolicy,
    ZOOM_BOUNDARY_FIELDS,
)
from .agsls.config import AGSLSConfig
from .agsls.controller import AdaptiveGridSmoothLifeSearch
from .benchmarking import _success_mask
from .objectives import DEFAULT_BOUNDS, OBJECTIVES
from .smoothlife.config import SmoothLifeConfig
from .smoothlife.simulator import SmoothLifeSearch


DEFAULT_TUNING_OBJECTIVES = ("sphere", "ackley", "rastrigin", "griewank", "rosenbrock", "himmelblau")
DEFAULT_SUCCESS_THRESHOLDS = {
    "sphere": 0.1,
    "ackley": 1.0,
    "rastrigin": 2.5,
    "griewank": 0.25,
    "rosenbrock": 5.0,
    "himmelblau": 1.0,
}
FAMILY_SCOPES = frozenset({"static_only", "sls_per_step", "agsls_per_step", "agsls_decision", "agsls_zoom_boundary"})
PREFLIGHT_ARTIFACT_NAMES = (
    "per_objective_leaderboard",
    "overall_rank",
    "parameter_effects",
    "adaptive_required",
    "adaptive_paired_effects",
    "finalists",
    "family_manifest",
    "report",
)

SMOOTHLIFE_FIELD_NAMES = frozenset(SmoothLifeConfig.__dataclass_fields__)
AGSLS_FIELD_NAMES = frozenset(AGSLSConfig.__dataclass_fields__)
ADAPTIVE_FIELD_NAMES = SMOOTHLIFE_PER_STEP_FIELDS | AGSLS_PER_DECISION_FIELDS | ZOOM_BOUNDARY_FIELDS

FIELD_SCOPE_SCHEDULE_KINDS: dict[str, tuple[str, ...]] = {
    "static_only": (),
    "sls_per_step": ("zoom_linear", "budget_sigmoid", "plateau_reactive", "late_stage_reactive"),
    "agsls_per_step": ("zoom_linear", "budget_sigmoid", "plateau_reactive", "late_stage_reactive"),
    "agsls_decision": ("zoom_linear", "budget_sigmoid", "plateau_reactive", "decision_gap_reactive", "late_stage_reactive"),
    "agsls_zoom_boundary": ("zoom_linear", "budget_sigmoid", "plateau_reactive", "late_stage_reactive"),
}


@dataclass(slots=True)
class CandidateSpec:
    """One candidate configuration to benchmark."""

    stage_name: str
    variant: str
    config_id: str
    config_label: str
    family: str
    family_scope: str
    schedule_kind: str
    smoothlife_config: dict[str, Any]
    agsls_config: dict[str, Any]
    runtime_policy: dict[str, Any] | None = None
    smoothlife_overrides: dict[str, Any] = field(default_factory=dict)
    agsls_overrides: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class StudySpec:
    """Decision-complete study specification."""

    output_dir: Path
    objectives: tuple[str, ...] = DEFAULT_TUNING_OBJECTIVES
    baseline_budgets: tuple[int, ...] = (400, 800, 1600)
    static_budgets: tuple[int, ...] = (400, 800)
    adaptive_screen_budgets: tuple[int, ...] = (400, 800, 1600)
    adaptive_confirmation_budgets: tuple[int, ...] = (800, 1600, 3200)
    interaction_budgets: tuple[int, ...] = (800, 1600)
    final_confirmation_budgets: tuple[int, ...] = (3200, 6400)
    variants: tuple[str, ...] = ("sls", "agsls")
    seed_start: int = 0
    stage1_seeds: int = 24
    stage2_seeds: int = 16
    stage3_seeds: int = 20
    stage4_seeds: int = 32
    stage5_seeds: int = 12
    stage6_seeds: int = 64
    stage1_grid_shape: tuple[int, int] = (64, 64)
    intermediate_grid_shape: tuple[int, int] = (64, 64)
    final_grid_shape: tuple[int, int] = (128, 128)
    interaction_top_k: int = 6
    finalist_limit: int = 8
    workers: int | None = None
    resume: bool = True
    keep_snapshots_in_finalists: bool = True
    parameter_families: tuple[str, ...] | None = None
    success_thresholds: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_SUCCESS_THRESHOLDS))
    profile: str = "maximal"
    strict_adaptive_rule: bool = True
    preflight: bool = True

    def resolved_workers(self) -> int:
        if self.workers is not None:
            return max(int(self.workers), 1)
        return max(1, (mp.cpu_count() or 2) - 1)


@dataclass(slots=True)
class StudySummary:
    """High-level study outcome."""

    output_dir: Path
    total_trials: int
    completed_trials: int
    skipped_trials: int
    trials_path: Path
    per_objective_leaderboard_path: Path
    overall_rank_path: Path
    parameter_effects_path: Path
    adaptive_required_path: Path
    adaptive_paired_effects_path: Path
    runtime_efficiency_path: Path
    family_manifest_path: Path
    finalists_path: Path
    report_path: Path


@dataclass(slots=True)
class ParameterFamilyDefinition:
    """Family of related knobs used for ablations and schedules."""

    name: str
    applicable_variants: tuple[str, ...]
    family_scope: str
    adaptive_variants: tuple[str, ...]
    schedule_kinds: tuple[str, ...]
    build_levels: Callable[[SmoothLifeConfig, AGSLSConfig], tuple[dict[str, Any], ...]]

    def __post_init__(self) -> None:
        if self.family_scope not in FAMILY_SCOPES:
            raise ValueError(f"unknown family scope: {self.family_scope}")


def _base_smoothlife_config(grid_shape: tuple[int, int], *, store_all_snapshots: bool) -> SmoothLifeConfig:
    return SmoothLifeConfig(
        grid_shape=grid_shape,
        preset="search",
        run_mode="search",
        store_all_snapshots=store_all_snapshots,
    )


def _base_agsls_config() -> AGSLSConfig:
    return AGSLSConfig(max_evaluations=None)


def _split_overrides(overrides: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    smoothlife: dict[str, Any] = {}
    agsls: dict[str, Any] = {}
    for field_name, value in overrides.items():
        if field_name in SMOOTHLIFE_FIELD_NAMES:
            smoothlife[field_name] = value
        elif field_name in AGSLS_FIELD_NAMES:
            agsls[field_name] = value
    return smoothlife, agsls


def _slugify(*parts: str) -> str:
    joined = "__".join(part.strip().lower().replace(" ", "_").replace("/", "_") for part in parts if part)
    return joined.replace(".", "_")


def _transition_window_levels(smoothlife: SmoothLifeConfig, _agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    deltas = (-0.03, -0.015, 0.0, 0.015, 0.03)
    return tuple(
        {
            "birth_low": smoothlife.birth_low + delta,
            "birth_high": smoothlife.birth_high + delta,
            "death_low": smoothlife.death_low + delta,
            "death_high": smoothlife.death_high + delta,
        }
        for delta in deltas
    )


def _transition_alpha_levels(smoothlife: SmoothLifeConfig, _agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    scales = (0.6, 0.8, 1.0, 1.2, 1.4)
    return tuple(
        {
            "alpha_n": smoothlife.alpha_n * scale,
            "alpha_m": smoothlife.alpha_m * scale,
        }
        for scale in scales
    )


def _kernel_geometry_levels(smoothlife: SmoothLifeConfig, _agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    scales = (0.7, 0.85, 1.0, 1.15, 1.3)
    return tuple(
        {
            "inner_radius": smoothlife.inner_radius * scale,
            "outer_radius": smoothlife.outer_radius * scale,
            "anti_alias_radius": max(0.5, smoothlife.anti_alias_radius * scale),
        }
        for scale in scales
    )


def _time_integration_levels(smoothlife: SmoothLifeConfig, _agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    values = (0.10, 0.175, smoothlife.dt, 0.325, 0.40)
    return tuple({"dt": value} for value in values)


def _diffusion_levels(_smoothlife: SmoothLifeConfig, _agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    return tuple({"diffusion": value} for value in (0.0, 0.05, 0.10, 0.20, 0.30))


def _objective_guidance_levels(_smoothlife: SmoothLifeConfig, _agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    return tuple({"objective_coupling": value} for value in (0.0, 0.15, 0.30, 0.45, 0.60))


def _evaluation_batch_levels(_smoothlife: SmoothLifeConfig, _agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    return tuple({"evaluations_per_step": value} for value in (4, 6, 8, 12, 16))


def _zoom_schedule_levels(_smoothlife: SmoothLifeConfig, _agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    return (
        {"initial_steps_per_zoom": 16, "min_steps_per_zoom": 4, "zoom_decay": 0.50},
        {"initial_steps_per_zoom": 24, "min_steps_per_zoom": 6, "zoom_decay": 0.625},
        {"initial_steps_per_zoom": 32, "min_steps_per_zoom": 8, "zoom_decay": 0.75},
        {"initial_steps_per_zoom": 40, "min_steps_per_zoom": 10, "zoom_decay": 0.875},
        {"initial_steps_per_zoom": 48, "min_steps_per_zoom": 12, "zoom_decay": 1.00},
    )


def _max_zoom_cycles_levels(_smoothlife: SmoothLifeConfig, agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    defaults = (2, 3, agsls.max_zoom_cycles, 7, 9)
    return tuple({"max_zoom_cycles": value} for value in defaults)


def _basin_quantile_levels(_smoothlife: SmoothLifeConfig, _agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    return tuple({"basin_quantile": value} for value in (0.75, 0.82, 0.88, 0.92, 0.96))


def _alive_core_threshold_levels(_smoothlife: SmoothLifeConfig, agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    values = (0.15, 0.225, agsls.alive_core_threshold, 0.375, 0.45)
    return tuple({"alive_core_threshold": value} for value in values)


def _alive_density_levels(_smoothlife: SmoothLifeConfig, _agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    return tuple({"min_alive_density": value} for value in (0.05, 0.125, 0.20, 0.275, 0.35))


def _min_basin_cells_levels(_smoothlife: SmoothLifeConfig, agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    values = (8, 16, agsls.min_basin_cells, 32, 48)
    return tuple({"min_basin_cells": value} for value in values)


def _zoom_padding_levels(_smoothlife: SmoothLifeConfig, _agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    return tuple({"zoom_padding": value} for value in (0.05, 0.10, 0.20, 0.30, 0.40))


def _basin_scoring_levels(_smoothlife: SmoothLifeConfig, agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    return (
        {
            "mass_weight": 1.50,
            "alive_density_weight": 1.00,
            "objective_weight": 1.00,
            "stability_weight": 0.75,
            "area_penalty": 0.10,
        },
        {
            "mass_weight": 1.25,
            "alive_density_weight": 0.85,
            "objective_weight": 1.25,
            "stability_weight": 0.75,
            "area_penalty": 0.15,
        },
        {
            "mass_weight": agsls.mass_weight,
            "alive_density_weight": agsls.alive_density_weight,
            "objective_weight": agsls.objective_weight,
            "stability_weight": agsls.stability_weight,
            "area_penalty": agsls.area_penalty,
        },
        {
            "mass_weight": 0.75,
            "alive_density_weight": 0.75,
            "objective_weight": 1.75,
            "stability_weight": 1.00,
            "area_penalty": 0.20,
        },
        {
            "mass_weight": 0.50,
            "alive_density_weight": 0.60,
            "objective_weight": 2.00,
            "stability_weight": 1.25,
            "area_penalty": 0.25,
        },
    )


def _decision_margin_levels(_smoothlife: SmoothLifeConfig, _agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    return (
        {"dominance_margin": 0.10, "similarity_margin": 0.01},
        {"dominance_margin": 0.15, "similarity_margin": 0.03},
        {"dominance_margin": 0.20, "similarity_margin": 0.05},
        {"dominance_margin": 0.30, "similarity_margin": 0.08},
        {"dominance_margin": 0.40, "similarity_margin": 0.12},
    )


def _probe_budget_levels(_smoothlife: SmoothLifeConfig, _agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    return (
        {"candidate_probe_evaluations": 0, "undecided_stage_max_evaluations": 0},
        {"candidate_probe_evaluations": 4, "undecided_stage_max_evaluations": 16},
        {"candidate_probe_evaluations": 16, "undecided_stage_max_evaluations": 64},
        {"candidate_probe_evaluations": 32, "undecided_stage_max_evaluations": 128},
        {"candidate_probe_evaluations": 48, "undecided_stage_max_evaluations": 192},
    )


def _cluster_shape_levels(_smoothlife: SmoothLifeConfig, _agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    return (
        {"cluster_eps_pixels": 1.5, "cluster_min_samples": 3},
        {"cluster_eps_pixels": 2.0, "cluster_min_samples": 4},
        {"cluster_eps_pixels": 2.5, "cluster_min_samples": 6},
        {"cluster_eps_pixels": 3.0, "cluster_min_samples": 8},
        {"cluster_eps_pixels": 3.5, "cluster_min_samples": 10},
    )


def _smoothlife_cadence_levels(_smoothlife: SmoothLifeConfig, _agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    return tuple({"evaluations_per_step": value} for value in (24, 20, 16, 12, 8))


def _agsls_cadence_levels(_smoothlife: SmoothLifeConfig, _agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    return tuple({"initial_steps_per_zoom": value} for value in (48, 40, 32, 24, 16))


def _late_stage_microgrid_levels(_smoothlife: SmoothLifeConfig, agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    return (
        {
            "late_stage_microgrid_enabled": False,
            "late_stage_microgrid_centers": agsls.late_stage_microgrid_centers,
            "late_stage_microgrid_side_fraction": agsls.late_stage_microgrid_side_fraction,
        },
        {
            "late_stage_microgrid_enabled": True,
            "late_stage_microgrid_centers": 2,
            "late_stage_microgrid_side_fraction": 0.20,
        },
        {
            "late_stage_microgrid_enabled": True,
            "late_stage_microgrid_centers": 3,
            "late_stage_microgrid_side_fraction": 0.25,
        },
        {
            "late_stage_microgrid_enabled": True,
            "late_stage_microgrid_centers": 4,
            "late_stage_microgrid_side_fraction": 0.35,
        },
    )


def _late_stage_exploiter_levels(_smoothlife: SmoothLifeConfig, agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    return (
        {"late_stage_exploiter": "none"},
        {
            "late_stage_exploiter": "microgrid",
            "late_stage_microgrid_enabled": True,
            "late_stage_microgrid_resolution": agsls.late_stage_microgrid_resolution,
            "late_stage_microgrid_centers": agsls.late_stage_microgrid_centers,
            "late_stage_microgrid_side_fraction": agsls.late_stage_microgrid_side_fraction,
        },
        {
            "late_stage_exploiter": "pattern_search",
            "late_stage_pattern_search_initial_step_fraction": agsls.late_stage_pattern_search_initial_step_fraction,
            "late_stage_pattern_search_min_step_fraction": agsls.late_stage_pattern_search_min_step_fraction,
            "late_stage_pattern_search_shrink": agsls.late_stage_pattern_search_shrink,
            "late_stage_pattern_search_max_iterations": agsls.late_stage_pattern_search_max_iterations,
            "late_stage_pattern_search_max_evaluations": agsls.late_stage_pattern_search_max_evaluations,
        },
        {
            "late_stage_exploiter": "pattern_search",
            "late_stage_pattern_search_initial_step_fraction": 0.08,
            "late_stage_pattern_search_min_step_fraction": 0.002,
            "late_stage_pattern_search_shrink": 0.5,
            "late_stage_pattern_search_max_iterations": 18,
            "late_stage_pattern_search_max_evaluations": 30,
        },
    )


def _late_stage_translation_levels(_smoothlife: SmoothLifeConfig, _agsls: AGSLSConfig) -> tuple[dict[str, Any], ...]:
    return (
        {
            "late_stage_translation_enabled": False,
            "late_stage_translation_step_fraction": 0.25,
            "late_stage_translation_min_offset_fraction": 0.10,
        },
        {
            "late_stage_translation_enabled": True,
            "late_stage_translation_step_fraction": 0.15,
            "late_stage_translation_min_offset_fraction": 0.20,
        },
        {
            "late_stage_translation_enabled": True,
            "late_stage_translation_step_fraction": 0.25,
            "late_stage_translation_min_offset_fraction": 0.12,
        },
        {
            "late_stage_translation_enabled": True,
            "late_stage_translation_step_fraction": 0.35,
            "late_stage_translation_min_offset_fraction": 0.08,
        },
    )


FAMILY_DEFINITIONS: tuple[ParameterFamilyDefinition, ...] = (
    ParameterFamilyDefinition(
        "transition_window",
        ("sls", "agsls"),
        "agsls_zoom_boundary",
        ("agsls",),
        ("zoom_linear", "budget_sigmoid", "plateau_reactive"),
        _transition_window_levels,
    ),
    ParameterFamilyDefinition(
        "transition_alpha",
        ("sls", "agsls"),
        "agsls_zoom_boundary",
        ("agsls",),
        ("zoom_linear", "budget_sigmoid", "plateau_reactive"),
        _transition_alpha_levels,
    ),
    ParameterFamilyDefinition(
        "kernel_geometry",
        ("sls", "agsls"),
        "agsls_zoom_boundary",
        ("agsls",),
        ("zoom_linear", "budget_sigmoid", "plateau_reactive"),
        _kernel_geometry_levels,
    ),
    ParameterFamilyDefinition(
        "time_integration",
        ("sls", "agsls"),
        "sls_per_step",
        ("sls", "agsls"),
        ("zoom_linear", "budget_sigmoid", "plateau_reactive"),
        _time_integration_levels,
    ),
    ParameterFamilyDefinition(
        "diffusion",
        ("sls", "agsls"),
        "sls_per_step",
        ("sls", "agsls"),
        ("zoom_linear", "budget_sigmoid", "plateau_reactive"),
        _diffusion_levels,
    ),
    ParameterFamilyDefinition(
        "objective_guidance",
        ("sls", "agsls"),
        "sls_per_step",
        ("sls", "agsls"),
        ("zoom_linear", "budget_sigmoid", "plateau_reactive"),
        _objective_guidance_levels,
    ),
    ParameterFamilyDefinition(
        "evaluation_batch",
        ("sls", "agsls"),
        "sls_per_step",
        ("sls", "agsls"),
        ("zoom_linear", "budget_sigmoid", "plateau_reactive"),
        _evaluation_batch_levels,
    ),
    ParameterFamilyDefinition(
        "zoom_schedule",
        ("agsls",),
        "agsls_decision",
        ("agsls",),
        ("zoom_linear", "budget_sigmoid", "plateau_reactive", "decision_gap_reactive"),
        _zoom_schedule_levels,
    ),
    ParameterFamilyDefinition(
        "max_zoom_cycles",
        ("agsls",),
        "static_only",
        (),
        (),
        _max_zoom_cycles_levels,
    ),
    ParameterFamilyDefinition(
        "basin_quantile",
        ("agsls",),
        "agsls_decision",
        ("agsls",),
        ("zoom_linear", "budget_sigmoid", "plateau_reactive", "decision_gap_reactive"),
        _basin_quantile_levels,
    ),
    ParameterFamilyDefinition(
        "alive_core_threshold",
        ("agsls",),
        "agsls_decision",
        ("agsls",),
        ("zoom_linear", "budget_sigmoid", "plateau_reactive", "decision_gap_reactive"),
        _alive_core_threshold_levels,
    ),
    ParameterFamilyDefinition(
        "alive_density_gate",
        ("agsls",),
        "agsls_decision",
        ("agsls",),
        ("zoom_linear", "budget_sigmoid", "plateau_reactive", "decision_gap_reactive"),
        _alive_density_levels,
    ),
    ParameterFamilyDefinition(
        "min_basin_cells",
        ("agsls",),
        "agsls_decision",
        ("agsls",),
        ("zoom_linear", "budget_sigmoid", "plateau_reactive", "decision_gap_reactive"),
        _min_basin_cells_levels,
    ),
    ParameterFamilyDefinition(
        "zoom_padding",
        ("agsls",),
        "agsls_decision",
        ("agsls",),
        ("zoom_linear", "budget_sigmoid", "plateau_reactive", "decision_gap_reactive"),
        _zoom_padding_levels,
    ),
    ParameterFamilyDefinition(
        "basin_scoring",
        ("agsls",),
        "agsls_decision",
        ("agsls",),
        ("zoom_linear", "budget_sigmoid", "plateau_reactive", "decision_gap_reactive"),
        _basin_scoring_levels,
    ),
    ParameterFamilyDefinition(
        "decision_margins",
        ("agsls",),
        "agsls_decision",
        ("agsls",),
        ("zoom_linear", "budget_sigmoid", "plateau_reactive", "decision_gap_reactive"),
        _decision_margin_levels,
    ),
    ParameterFamilyDefinition(
        "probe_budget",
        ("agsls",),
        "agsls_decision",
        ("agsls",),
        ("zoom_linear", "budget_sigmoid", "plateau_reactive", "decision_gap_reactive"),
        _probe_budget_levels,
    ),
    ParameterFamilyDefinition(
        "cluster_shape",
        ("agsls",),
        "agsls_zoom_boundary",
        ("agsls",),
        ("zoom_linear", "budget_sigmoid", "plateau_reactive"),
        _cluster_shape_levels,
    ),
    ParameterFamilyDefinition(
        "smoothlife_cadence",
        ("agsls",),
        "sls_per_step",
        ("agsls",),
        ("zoom_linear", "budget_sigmoid", "plateau_reactive", "late_stage_reactive"),
        _smoothlife_cadence_levels,
    ),
    ParameterFamilyDefinition(
        "agsls_cadence",
        ("agsls",),
        "agsls_decision",
        ("agsls",),
        ("zoom_linear", "budget_sigmoid", "plateau_reactive", "late_stage_reactive"),
        _agsls_cadence_levels,
    ),
    ParameterFamilyDefinition(
        "late_stage_microgrid",
        ("agsls",),
        "static_only",
        (),
        (),
        _late_stage_microgrid_levels,
    ),
    ParameterFamilyDefinition(
        "late_stage_translation",
        ("agsls",),
        "static_only",
        (),
        (),
        _late_stage_translation_levels,
    ),
    ParameterFamilyDefinition(
        "late_stage_exploiter",
        ("agsls",),
        "static_only",
        (),
        (),
        _late_stage_exploiter_levels,
    ),
)

FAMILY_REGISTRY = {definition.name: definition for definition in FAMILY_DEFINITIONS}


def _filtered_families(spec: StudySpec) -> tuple[ParameterFamilyDefinition, ...]:
    if spec.parameter_families is None:
        return FAMILY_DEFINITIONS
    allowed = set(spec.parameter_families)
    return tuple(definition for definition in FAMILY_DEFINITIONS if definition.name in allowed)


def _family_scope(definition_name: str) -> str:
    if definition_name == "baseline":
        return "static_only"
    return FAMILY_REGISTRY[definition_name].family_scope


def _record_overrides(record: dict[str, Any]) -> dict[str, Any]:
    overrides = dict(record.get("smoothlife_overrides", {}))
    overrides.update(record.get("agsls_overrides", {}))
    return overrides


def _allowed_schedule_kinds(definition: ParameterFamilyDefinition, variant: str) -> tuple[str, ...]:
    if variant not in definition.adaptive_variants:
        return ()
    if definition.family_scope == "sls_per_step" and variant == "sls":
        return tuple(kind for kind in definition.schedule_kinds if kind in {"budget_sigmoid", "plateau_reactive"})
    return definition.schedule_kinds


def _apply_candidate_overrides(
    smoothlife: SmoothLifeConfig,
    agsls: AGSLSConfig,
    overrides: dict[str, Any],
) -> tuple[SmoothLifeConfig, AGSLSConfig]:
    smoothlife_overrides, agsls_overrides = _split_overrides(overrides)
    next_smoothlife = replace(smoothlife, **smoothlife_overrides) if smoothlife_overrides else replace(smoothlife)
    next_agsls = replace(agsls, **agsls_overrides) if agsls_overrides else replace(agsls)
    return next_smoothlife, next_agsls


def _candidate_from_configs(
    *,
    stage_name: str,
    variant: str,
    config_suffix: str,
    config_label: str,
    family: str,
    family_scope: str,
    schedule_kind: str,
    smoothlife: SmoothLifeConfig,
    agsls: AGSLSConfig,
    runtime_policy: SchedulePolicy | None = None,
    overrides: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
) -> CandidateSpec:
    smoothlife_overrides, agsls_overrides = _split_overrides(overrides or {})
    merged_metadata = dict(metadata or {})
    merged_metadata.setdefault("family_scope", family_scope)
    return CandidateSpec(
        stage_name=stage_name,
        variant=variant,
        config_id=_slugify(stage_name, variant, config_suffix),
        config_label=config_label,
        family=family,
        family_scope=family_scope,
        schedule_kind=schedule_kind,
        smoothlife_config=asdict(smoothlife),
        agsls_config=asdict(agsls),
        runtime_policy=None if runtime_policy is None else runtime_policy.to_payload(),
        smoothlife_overrides=smoothlife_overrides,
        agsls_overrides=agsls_overrides,
        metadata=merged_metadata,
    )


def _schedule_threshold(schedule_kind: str) -> float:
    if schedule_kind == "decision_gap_reactive":
        return 0.05
    return 1e-4


def _stage1_candidates(spec: StudySpec) -> list[CandidateSpec]:
    smoothlife = _base_smoothlife_config(spec.stage1_grid_shape, store_all_snapshots=False)
    agsls = _base_agsls_config()
    candidates: list[CandidateSpec] = []
    if "sls" in spec.variants:
        candidates.append(
            _candidate_from_configs(
                stage_name="stage1_baseline",
                variant="sls",
                config_suffix="baseline",
                config_label="baseline",
                family="baseline",
                family_scope="static_only",
                schedule_kind="constant",
                smoothlife=smoothlife,
                agsls=agsls,
            )
        )
    if "agsls" in spec.variants:
        candidates.append(
            _candidate_from_configs(
                stage_name="stage1_baseline",
                variant="agsls",
                config_suffix="baseline",
                config_label="baseline",
                family="baseline",
                family_scope="static_only",
                schedule_kind="constant",
                smoothlife=smoothlife,
                agsls=agsls,
            )
        )
    return candidates


def _stage2_candidates(spec: StudySpec) -> list[CandidateSpec]:
    smoothlife = _base_smoothlife_config(spec.intermediate_grid_shape, store_all_snapshots=False)
    agsls = _base_agsls_config()
    candidates: list[CandidateSpec] = []
    for definition in _filtered_families(spec):
        levels = definition.build_levels(smoothlife, agsls)
        for variant in spec.variants:
            if variant not in definition.applicable_variants:
                continue
            for index, overrides in enumerate(levels):
                smoothlife_cfg, agsls_cfg = _apply_candidate_overrides(smoothlife, agsls, overrides)
                candidates.append(
                    _candidate_from_configs(
                        stage_name="stage2_static_ablation",
                        variant=variant,
                        config_suffix=f"{definition.name}_level_{index}",
                        config_label=f"{definition.name}:L{index}",
                        family=definition.name,
                        family_scope=definition.family_scope,
                        schedule_kind="constant",
                        smoothlife=smoothlife_cfg,
                        agsls=agsls_cfg,
                        overrides=overrides,
                        metadata={"level_index": index},
                    )
                )
    return candidates


def _config_records(records: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for record in records:
        mapping.setdefault(str(record["config_id"]), record)
    return mapping


def _bucket_stats(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for record in records:
        key = (str(record["config_id"]), str(record["objective"]), int(record["budget"]))
        grouped.setdefault(key, []).append(record)
    stats: list[dict[str, Any]] = []
    for (_config_id, objective, budget), group in grouped.items():
        best_values = [float(item["best_value"]) for item in group]
        wall_times = [float(item["wall_time_s"]) for item in group]
        seeds = len(group)
        maximize = bool(group[0].get("maximize", False))
        q1, q3 = np.quantile(best_values, [0.25, 0.75])
        stats.append(
            {
                "config_id": str(group[0]["config_id"]),
                "config_label": str(group[0]["config_label"]),
                "stage_name": str(group[0]["stage_name"]),
                "variant": str(group[0]["variant"]),
                "family": str(group[0]["family"]),
                "family_scope": str(group[0]["family_scope"]),
                "schedule_kind": str(group[0]["schedule_kind"]),
                "objective": objective,
                "budget": budget,
                "maximize": maximize,
                "median_best_value": float(np.median(best_values)),
                "iqr_low": float(q1),
                "iqr_high": float(q3),
                "iqr_width": float(q3 - q1),
                "success_rate": float(np.mean([float(item["success"]) for item in group])),
                "mean_wall_time_s": float(np.mean(wall_times)),
                "trials": seeds,
            }
        )
    return stats


def _ranking_sort_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    median = float(row["median_best_value"])
    median_key = -median if bool(row.get("maximize", False)) else median
    return (
        median_key,
        -float(row["success_rate"]),
        float(row["iqr_width"]),
        str(row["config_id"]),
    )


def _rank_bucket_rows(rows: list[dict[str, Any]], rank_field: str) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=_ranking_sort_key)
    for index, row in enumerate(ranked, start=1):
        row[rank_field] = index
    return ranked


def _annotate_bucket_ranks(stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_variant_bucket: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    by_pooled_bucket: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in stats:
        by_variant_bucket.setdefault((str(row["variant"]), str(row["objective"]), int(row["budget"])), []).append(row)
        by_pooled_bucket.setdefault((str(row["objective"]), int(row["budget"])), []).append(row)
    for rows in by_variant_bucket.values():
        _rank_bucket_rows(rows, "variant_rank")
    for rows in by_pooled_bucket.values():
        _rank_bucket_rows(rows, "pooled_rank")
    return stats


def _overall_rank_rows(stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_scope: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in stats:
        by_scope.setdefault((str(row["variant"]), str(row["config_id"])), []).append(row)
        by_scope.setdefault(("pooled", str(row["config_id"])), []).append(row)
    for (scope, config_id), group in by_scope.items():
        rank_field = "pooled_rank" if scope == "pooled" else "variant_rank"
        mean_rank = statistics.fmean(float(item[rank_field]) for item in group)
        win_buckets = sum(1 for item in group if int(item[rank_field]) == 1)
        rows.append(
            {
                "scope": scope,
                "variant": str(group[0]["variant"]),
                "config_id": config_id,
                "config_label": str(group[0]["config_label"]),
                "stage_name": str(group[0]["stage_name"]),
                "family": str(group[0]["family"]),
                "family_scope": str(group[0]["family_scope"]),
                "schedule_kind": str(group[0]["schedule_kind"]),
                "mean_rank": float(mean_rank),
                "win_buckets": int(win_buckets),
                "bucket_count": len(group),
            }
        )
    rows.sort(key=lambda row: (str(row["scope"]), float(row["mean_rank"]), -int(row["win_buckets"]), str(row["config_id"])))
    return rows


def _best_rows_by_family(
    overall_rows: Iterable[dict[str, Any]],
    *,
    variant: str,
    families: Iterable[str] | None = None,
    schedule_kinds: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    best: dict[str, dict[str, Any]] = {}
    allowed_families = None if families is None else set(families)
    for row in overall_rows:
        if str(row["scope"]) != variant or str(row["variant"]) != variant:
            continue
        family = str(row["family"])
        if allowed_families is not None and family not in allowed_families:
            continue
        if schedule_kinds is not None and str(row["schedule_kind"]) not in schedule_kinds:
            continue
        incumbent = best.get(family)
        if incumbent is None or (float(row["mean_rank"]), str(row["config_id"])) < (float(incumbent["mean_rank"]), str(incumbent["config_id"])):
            best[family] = row
    return best


def _stage_records(records: Iterable[dict[str, Any]], stage_names: set[str]) -> list[dict[str, Any]]:
    return [record for record in records if str(record["stage_name"]) in stage_names]


def _overall_rows_for_records(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stats = _annotate_bucket_ranks(_bucket_stats(records))
    overall = _overall_rank_rows(stats)
    return stats, overall


def _best_constant_records(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    stage2_records = _stage_records(records, {"stage2_static_ablation"})
    if not stage2_records:
        return {}
    _stats, overall = _overall_rows_for_records(stage2_records)
    source_map = _config_records(stage2_records)
    best_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for variant in ("sls", "agsls"):
        for family, row in _best_rows_by_family(overall, variant=variant).items():
            best_rows[(variant, family)] = source_map[str(row["config_id"])]
    return best_rows


def _best_adaptive_records(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    stage3_records = _stage_records(records, {"stage3_adaptive_screen"})
    if not stage3_records:
        return {}
    _stats, overall = _overall_rows_for_records(stage3_records)
    source_map = _config_records(stage3_records)
    best_rows: dict[tuple[str, str], dict[str, Any]] = {}
    for variant in ("sls", "agsls"):
        rows = _best_rows_by_family(overall, variant=variant)
        for family, row in rows.items():
            best_rows[(variant, family)] = source_map[str(row["config_id"])]
    return best_rows


def _representative_stage4_records(records: list[dict[str, Any]]) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], dict[str, Any]]]:
    stage4_records = _stage_records(records, {"stage4_adaptive_confirmation"})
    if not stage4_records:
        return {}, {}
    _stats, overall = _overall_rows_for_records(stage4_records)
    source_map = _config_records(stage4_records)
    representative_rows: dict[tuple[str, str], dict[str, Any]] = {}
    representative_records: dict[tuple[str, str], dict[str, Any]] = {}
    for variant in ("sls", "agsls"):
        rows = _best_rows_by_family(overall, variant=variant)
        for family, row in rows.items():
            key = (variant, family)
            representative_rows[key] = row
            representative_records[key] = source_map[str(row["config_id"])]
    return representative_rows, representative_records


def _stage3_candidates(spec: StudySpec, records: list[dict[str, Any]]) -> list[CandidateSpec]:
    constant_records = _best_constant_records(records)
    candidates: list[CandidateSpec] = []
    for definition in _filtered_families(spec):
        smoothlife = _base_smoothlife_config(spec.intermediate_grid_shape, store_all_snapshots=False)
        agsls = _base_agsls_config()
        levels = definition.build_levels(smoothlife, agsls)
        low_values = levels[0]
        high_values = levels[-1]
        for variant in spec.variants:
            schedule_kinds = _allowed_schedule_kinds(definition, variant)
            if not schedule_kinds:
                continue
            source = constant_records.get((variant, definition.name))
            if source is None:
                continue
            best_smoothlife = SmoothLifeConfig(**source["smoothlife_config"])
            best_agsls = AGSLSConfig(**source["agsls_config"])
            base_values = _record_overrides(source)
            if not base_values:
                base_values = dict(levels[2])
            for schedule_kind in schedule_kinds:
                schedules: list[FieldSchedule] = []
                for field_name, base_value in base_values.items():
                    if field_name not in ADAPTIVE_FIELD_NAMES:
                        continue
                    schedules.append(
                        FieldSchedule(
                            field_name=field_name,
                            mode=schedule_kind,
                            base_value=base_value,
                            low_value=low_values.get(field_name, base_value),
                            high_value=high_values.get(field_name, base_value),
                            plateau_threshold=_schedule_threshold(schedule_kind),
                        )
                    )
                if not schedules:
                    continue
                policy = SchedulePolicy(tuple(schedules), family=definition.name, schedule_kind=schedule_kind)
                candidates.append(
                    _candidate_from_configs(
                        stage_name="stage3_adaptive_screen",
                        variant=variant,
                        config_suffix=f"{definition.name}_{schedule_kind}",
                        config_label=f"{definition.name}:{schedule_kind}",
                        family=definition.name,
                        family_scope=definition.family_scope,
                        schedule_kind=schedule_kind,
                        smoothlife=best_smoothlife,
                        agsls=best_agsls,
                        runtime_policy=policy,
                        overrides=base_values,
                        metadata={"source_config_id": str(source["config_id"])},
                    )
                )
    return candidates


def _stage4_candidates(spec: StudySpec, records: list[dict[str, Any]]) -> list[CandidateSpec]:
    constant_records = _best_constant_records(records)
    adaptive_records = _best_adaptive_records(records)
    candidates: list[CandidateSpec] = []
    for definition in _filtered_families(spec):
        for variant in spec.variants:
            if variant not in definition.applicable_variants:
                continue
            constant_source = constant_records.get((variant, definition.name))
            if constant_source is not None:
                candidates.append(
                    _candidate_from_configs(
                        stage_name="stage4_adaptive_confirmation",
                        variant=variant,
                        config_suffix=f"{definition.name}_constant",
                        config_label=f"{definition.name}:constant",
                        family=definition.name,
                        family_scope=definition.family_scope,
                        schedule_kind="constant",
                        smoothlife=SmoothLifeConfig(**constant_source["smoothlife_config"]),
                        agsls=AGSLSConfig(**constant_source["agsls_config"]),
                        overrides=_record_overrides(constant_source),
                        metadata={"source_config_id": str(constant_source["config_id"])},
                    )
                )
            adaptive_source = adaptive_records.get((variant, definition.name))
            if adaptive_source is None:
                continue
            candidates.append(
                _candidate_from_configs(
                    stage_name="stage4_adaptive_confirmation",
                    variant=variant,
                    config_suffix=f"{definition.name}_{adaptive_source['schedule_kind']}",
                    config_label=f"{definition.name}:{adaptive_source['schedule_kind']}",
                    family=definition.name,
                    family_scope=definition.family_scope,
                    schedule_kind=str(adaptive_source["schedule_kind"]),
                    smoothlife=SmoothLifeConfig(**adaptive_source["smoothlife_config"]),
                    agsls=AGSLSConfig(**adaptive_source["agsls_config"]),
                    runtime_policy=SchedulePolicy.from_payload(adaptive_source.get("runtime_policy")),
                    overrides=_record_overrides(adaptive_source),
                    metadata={"source_config_id": str(adaptive_source["config_id"])},
                )
            )
    return candidates


def _family_component(record: dict[str, Any], label: str, grid_shape: tuple[int, int]) -> tuple[dict[str, Any], dict[str, Any], SchedulePolicy | None]:
    family_name = str(record["family"])
    definition = FAMILY_REGISTRY[family_name]
    base_smoothlife = _base_smoothlife_config(grid_shape, store_all_snapshots=False)
    base_agsls = _base_agsls_config()
    levels = definition.build_levels(base_smoothlife, base_agsls)
    level_lookup = {"low": levels[0], "default": _record_overrides(record) or dict(levels[2]), "high": levels[-1]}
    chosen_overrides = dict(level_lookup[label])
    smoothlife_overrides, agsls_overrides = _split_overrides(chosen_overrides)
    runtime_policy = SchedulePolicy.from_payload(record.get("runtime_policy"))
    if runtime_policy is None:
        return smoothlife_overrides, agsls_overrides, None
    rebuilt_schedules: list[FieldSchedule] = []
    for schedule in runtime_policy.schedules:
        base_value = chosen_overrides.get(schedule.field_name, schedule.base_value)
        rebuilt_schedules.append(
            FieldSchedule(
                field_name=schedule.field_name,
                mode=schedule.mode,
                base_value=base_value,
                low_value=levels[0].get(schedule.field_name, schedule.low_value if schedule.low_value is not None else base_value),
                high_value=levels[-1].get(schedule.field_name, schedule.high_value if schedule.high_value is not None else base_value),
                plateau_threshold=schedule.plateau_threshold,
            )
        )
    return smoothlife_overrides, agsls_overrides, SchedulePolicy(tuple(rebuilt_schedules), family=family_name, schedule_kind=str(record["schedule_kind"]))


def _stage5_candidates(spec: StudySpec, records: list[dict[str, Any]]) -> list[CandidateSpec]:
    representative_rows, representative_records = _representative_stage4_records(records)
    candidates: list[CandidateSpec] = []
    for variant in spec.variants:
        variant_rows = [row for key, row in representative_rows.items() if key[0] == variant]
        variant_rows.sort(key=lambda row: (float(row["mean_rank"]), str(row["family"])))
        top_rows = variant_rows[: max(spec.interaction_top_k, 0)]
        family_pairs = itertools.combinations([str(row["family"]) for row in top_rows], 2)
        for family_a, family_b in family_pairs:
            record_a = representative_records[(variant, family_a)]
            record_b = representative_records[(variant, family_b)]
            for label_a, label_b in itertools.product(("low", "default", "high"), repeat=2):
                smoothlife = _base_smoothlife_config(spec.intermediate_grid_shape, store_all_snapshots=False)
                agsls = _base_agsls_config()
                smoothlife_overrides_a, agsls_overrides_a, policy_a = _family_component(record_a, label_a, spec.intermediate_grid_shape)
                smoothlife_overrides_b, agsls_overrides_b, policy_b = _family_component(record_b, label_b, spec.intermediate_grid_shape)
                merged_overrides = dict(smoothlife_overrides_a)
                merged_overrides.update(smoothlife_overrides_b)
                merged_overrides.update(agsls_overrides_a)
                merged_overrides.update(agsls_overrides_b)
                smoothlife_cfg, agsls_cfg = _apply_candidate_overrides(smoothlife, agsls, merged_overrides)
                merged_policy = None
                merged_schedules: list[FieldSchedule] = []
                for policy in (policy_a, policy_b):
                    if policy is None:
                        continue
                    merged_schedules.extend(policy.schedules)
                if merged_schedules:
                    merged_policy = SchedulePolicy(tuple(merged_schedules), family=f"{family_a}+{family_b}", schedule_kind="interaction")
                candidates.append(
                    _candidate_from_configs(
                        stage_name="stage5_interaction",
                        variant=variant,
                        config_suffix=f"{family_a}_{label_a}__{family_b}_{label_b}",
                        config_label=f"{family_a}:{label_a}+{family_b}:{label_b}",
                        family=f"{family_a}+{family_b}",
                        family_scope="static_only",
                        schedule_kind="interaction",
                        smoothlife=smoothlife_cfg,
                        agsls=agsls_cfg,
                        runtime_policy=merged_policy,
                        overrides=merged_overrides,
                        metadata={
                            "family_a": family_a,
                            "family_b": family_b,
                            "label_a": label_a,
                            "label_b": label_b,
                            "representative_kind_a": str(record_a["schedule_kind"]),
                            "representative_kind_b": str(record_b["schedule_kind"]),
                        },
                    )
                )
    return candidates


def _select_finalists(records: list[dict[str, Any]], limit: int) -> dict[str, list[dict[str, Any]]]:
    prefinal_records = _stage_records(
        records,
        {
            "stage1_baseline",
            "stage2_static_ablation",
            "stage3_adaptive_screen",
            "stage4_adaptive_confirmation",
            "stage5_interaction",
        },
    )
    stats = _annotate_bucket_ranks(_bucket_stats(prefinal_records))
    overall = _overall_rank_rows(stats)
    finalists: dict[str, list[dict[str, Any]]] = {"sls": [], "agsls": []}
    config_map = _config_records(prefinal_records)
    for variant in finalists:
        rows = [row for row in overall if str(row["scope"]) == variant and str(row["variant"]) == variant]
        rows = sorted(rows, key=lambda row: (float(row["mean_rank"]), str(row["config_id"])))
        for row in rows[: max(limit, 0)]:
            source = config_map[str(row["config_id"])]
            finalists[variant].append(
                {
                    "source_config_id": str(row["config_id"]),
                    "source_stage_name": str(row["stage_name"]),
                    "config_label": str(row["config_label"]),
                    "family": str(row["family"]),
                    "family_scope": str(row["family_scope"]),
                    "schedule_kind": str(row["schedule_kind"]),
                    "smoothlife_config": source["smoothlife_config"],
                    "agsls_config": source["agsls_config"],
                    "runtime_policy": source.get("runtime_policy"),
                    "smoothlife_overrides": source.get("smoothlife_overrides", {}),
                    "agsls_overrides": source.get("agsls_overrides", {}),
                }
            )
    return finalists


def _stage6_candidates(spec: StudySpec, finalists: dict[str, list[dict[str, Any]]]) -> list[CandidateSpec]:
    candidates: list[CandidateSpec] = []
    for variant, finalist_rows in finalists.items():
        for finalist in finalist_rows:
            smoothlife = SmoothLifeConfig(**finalist["smoothlife_config"])
            smoothlife = replace(
                smoothlife,
                grid_shape=spec.final_grid_shape,
                store_all_snapshots=spec.keep_snapshots_in_finalists,
            )
            agsls = AGSLSConfig(**finalist["agsls_config"])
            runtime_policy = SchedulePolicy.from_payload(finalist.get("runtime_policy"))
            candidates.append(
                _candidate_from_configs(
                    stage_name="stage6_final_confirmation",
                    variant=variant,
                    config_suffix=f"{finalist['source_config_id']}_final",
                    config_label=f"final:{finalist['config_label']}",
                    family=str(finalist["family"]),
                    family_scope=str(finalist["family_scope"]),
                    schedule_kind=str(finalist["schedule_kind"]),
                    smoothlife=smoothlife,
                    agsls=agsls,
                    runtime_policy=runtime_policy,
                    overrides={**finalist.get("smoothlife_overrides", {}), **finalist.get("agsls_overrides", {})},
                    metadata={"source_config_id": finalist["source_config_id"]},
                )
            )
    return candidates


def _seeds(seed_start: int, count: int) -> list[int]:
    return list(range(int(seed_start), int(seed_start) + int(count)))


def _trial_payloads(
    spec: StudySpec,
    *,
    candidates: Iterable[CandidateSpec],
    stage_name: str,
    budgets: Iterable[int],
    seed_count: int,
) -> list[dict[str, Any]]:
    seeds = _seeds(spec.seed_start, seed_count)
    payloads: list[dict[str, Any]] = []
    for candidate in candidates:
        for objective_name in spec.objectives:
            bound = DEFAULT_BOUNDS[objective_name]
            bounds = [(bound[0], bound[1]), (bound[0], bound[1])]
            for budget in budgets:
                for seed in seeds:
                    trial_key = _slugify(stage_name, candidate.variant, candidate.config_id, objective_name, str(budget), str(seed))
                    payloads.append(
                        {
                            "trial_key": trial_key,
                            "stage_name": stage_name,
                            "variant": candidate.variant,
                            "objective": objective_name,
                            "bounds": bounds,
                            "budget": int(budget),
                            "seed": int(seed),
                            "config_id": candidate.config_id,
                            "config_label": candidate.config_label,
                            "family": candidate.family,
                            "family_scope": candidate.family_scope,
                            "schedule_kind": candidate.schedule_kind,
                            "smoothlife_config": candidate.smoothlife_config,
                            "agsls_config": candidate.agsls_config,
                            "runtime_policy": candidate.runtime_policy,
                            "smoothlife_overrides": candidate.smoothlife_overrides,
                            "agsls_overrides": candidate.agsls_overrides,
                            "metadata": candidate.metadata,
                            "success_threshold": float(spec.success_thresholds[objective_name]),
                        }
                    )
    return payloads


def _execute_trial(payload: dict[str, Any]) -> dict[str, Any]:
    objective_name = str(payload["objective"])
    objective = OBJECTIVES[objective_name]
    bounds = payload["bounds"]
    smoothlife = SmoothLifeConfig(**payload["smoothlife_config"])
    agsls = AGSLSConfig(**payload["agsls_config"])
    runtime_policy = SchedulePolicy.from_payload(payload.get("runtime_policy"))
    budget = int(payload["budget"])
    seed = int(payload["seed"])
    start = time.perf_counter()
    if str(payload["variant"]) == "sls":
        search = SmoothLifeSearch(objective, bounds, smoothlife, runtime_policy=runtime_policy)
        search.reset(seed=seed)
        run = search.run(evaluations=budget)
    else:
        search = AdaptiveGridSmoothLifeSearch(objective, bounds, smoothlife, agsls, runtime_policy=runtime_policy)
        search.reset(seed=seed)
        run = search.run(evaluations=budget)
    wall_time = time.perf_counter() - start
    best_value = float(run.best_value)
    success = bool(_success_mask(np.asarray([best_value], dtype=float), float(payload["success_threshold"]), maximize=smoothlife.maximize)[0])
    schedule_summary = run.metadata.get("schedule_summary")
    decision_reason_counts = run.metadata.get("decision_reason_counts", {})
    zoom_acceptance_count = int(run.metadata.get("zoom_acceptance_count", len(run.zoom_events)))
    return {
        "trial_key": str(payload["trial_key"]),
        "stage_name": str(payload["stage_name"]),
        "variant": str(payload["variant"]),
        "objective": objective_name,
        "budget": budget,
        "seed": seed,
        "config_id": str(payload["config_id"]),
        "config_label": str(payload["config_label"]),
        "family": str(payload["family"]),
        "family_scope": str(payload["family_scope"]),
        "schedule_kind": str(payload["schedule_kind"]),
        "best_value": best_value,
        "best_point": np.asarray(run.best_point, dtype=float).tolist(),
        "evaluations": int(run.evaluations),
        "zoom_events": len(run.zoom_events),
        "wall_time_s": float(wall_time),
        "success": success,
        "maximize": bool(smoothlife.maximize),
        "smoothlife_config": payload["smoothlife_config"],
        "agsls_config": payload["agsls_config"],
        "runtime_policy": payload.get("runtime_policy"),
        "smoothlife_overrides": payload.get("smoothlife_overrides", {}),
        "agsls_overrides": payload.get("agsls_overrides", {}),
        "schedule_summary": schedule_summary,
        "scheduled_fields": [] if schedule_summary is None else list(schedule_summary.get("scheduled_fields", [])),
        "decision_reason_counts": decision_reason_counts,
        "zoom_acceptance_count": zoom_acceptance_count,
        "metadata": payload.get("metadata", {}),
    }


def _load_trial_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _append_trial_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def _run_trials(
    spec: StudySpec,
    trials_path: Path,
    payloads: list[dict[str, Any]],
    *,
    completed_keys: set[str],
) -> tuple[int, int]:
    pending = [payload for payload in payloads if str(payload["trial_key"]) not in completed_keys]
    skipped = len(payloads) - len(pending)
    if not pending:
        return 0, skipped
    if spec.resolved_workers() == 1:
        completed = 0
        for payload in pending:
            record = _execute_trial(payload)
            _append_trial_record(trials_path, record)
            completed_keys.add(str(record["trial_key"]))
            completed += 1
        return completed, skipped
    ctx = mp.get_context("spawn")
    completed = 0
    with ProcessPoolExecutor(max_workers=spec.resolved_workers(), mp_context=ctx) as executor:
        futures = {executor.submit(_execute_trial, payload): payload for payload in pending}
        for future in as_completed(futures):
            record = future.result()
            _append_trial_record(trials_path, record)
            completed_keys.add(str(record["trial_key"]))
            completed += 1
    return completed, skipped


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _parameter_effect_rows(stats: list[dict[str, Any]], overall_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bucket_lookup: dict[str, list[dict[str, Any]]] = {}
    for row in stats:
        bucket_lookup.setdefault(str(row["config_id"]), []).append(row)
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in overall_rows:
        if str(row["scope"]) == "pooled":
            continue
        grouped.setdefault((str(row["variant"]), str(row["family"]), str(row["schedule_kind"])), []).append(row)
    effects: list[dict[str, Any]] = []
    for (variant, family, schedule_kind), rows in grouped.items():
        best = min(rows, key=lambda row: (float(row["mean_rank"]), str(row["config_id"])))
        buckets = bucket_lookup.get(str(best["config_id"]), [])
        effects.append(
            {
                "variant": variant,
                "family": family,
                "family_scope": str(best["family_scope"]),
                "schedule_kind": schedule_kind,
                "best_config_id": str(best["config_id"]),
                "best_config_label": str(best["config_label"]),
                "stage_name": str(best["stage_name"]),
                "mean_rank": float(best["mean_rank"]),
                "win_buckets": int(best["win_buckets"]),
                "bucket_count": int(best["bucket_count"]),
                "evaluated_buckets": len(buckets),
            }
        )
    effects.sort(key=lambda row: (row["variant"], row["family"], float(row["mean_rank"])))
    return effects


def _runtime_efficiency_rows(stats: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(
        stats,
        key=lambda row: (
            str(row["variant"]),
            str(row["objective"]),
            int(row["budget"]),
            float(row["mean_wall_time_s"]),
            str(row["config_id"]),
        ),
    )
    return [
        {
            "stage_name": str(row["stage_name"]),
            "variant": str(row["variant"]),
            "objective": str(row["objective"]),
            "budget": int(row["budget"]),
            "config_id": str(row["config_id"]),
            "config_label": str(row["config_label"]),
            "family": str(row["family"]),
            "family_scope": str(row["family_scope"]),
            "schedule_kind": str(row["schedule_kind"]),
            "mean_wall_time_s": float(row["mean_wall_time_s"]),
            "trials": int(row["trials"]),
        }
        for row in rows
    ]


def _paired_improvement(constant_value: float, adaptive_value: float, *, maximize: bool) -> float:
    if maximize:
        return float(adaptive_value) - float(constant_value)
    return float(constant_value) - float(adaptive_value)


def _stable_seed(*parts: object) -> int:
    seed = 0
    for part in parts:
        for char in str(part):
            seed = (seed * 131 + ord(char)) % (2**32 - 1)
    return seed or 1


def _adaptive_paired_trials(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage4_records = _stage_records(records, {"stage4_adaptive_confirmation"})
    if not stage4_records:
        return []
    by_family: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in stage4_records:
        if str(record["family"]) == "baseline":
            continue
        by_family.setdefault((str(record["variant"]), str(record["family"])), []).append(record)
    paired: list[dict[str, Any]] = []
    for (variant, family), family_records in by_family.items():
        constant_rows = {
            (str(row["objective"]), int(row["budget"]), int(row["seed"])): row
            for row in family_records
            if str(row["schedule_kind"]) == "constant"
        }
        adaptive_rows = [
            row for row in family_records if str(row["schedule_kind"]) != "constant"
        ]
        if not constant_rows or not adaptive_rows:
            continue
        adaptive_config_id = str(adaptive_rows[0]["config_id"])
        adaptive_schedule_kind = str(adaptive_rows[0]["schedule_kind"])
        family_scope = str(adaptive_rows[0]["family_scope"])
        maximize = bool(adaptive_rows[0].get("maximize", False))
        for adaptive_row in adaptive_rows:
            key = (str(adaptive_row["objective"]), int(adaptive_row["budget"]), int(adaptive_row["seed"]))
            constant_row = constant_rows.get(key)
            if constant_row is None:
                continue
            improvement = _paired_improvement(float(constant_row["best_value"]), float(adaptive_row["best_value"]), maximize=maximize)
            paired.append(
                {
                    "variant": variant,
                    "family": family,
                    "family_scope": family_scope,
                    "objective": str(adaptive_row["objective"]),
                    "budget": int(adaptive_row["budget"]),
                    "seed": int(adaptive_row["seed"]),
                    "maximize": maximize,
                    "best_constant_config_id": str(constant_row["config_id"]),
                    "best_adaptive_config_id": adaptive_config_id,
                    "best_adaptive_schedule_kind": adaptive_schedule_kind,
                    "constant_best_value": float(constant_row["best_value"]),
                    "adaptive_best_value": float(adaptive_row["best_value"]),
                    "paired_improvement": float(improvement),
                    "adaptive_win": bool(improvement > 0.0),
                }
            )
    paired.sort(key=lambda row: (row["variant"], row["family"], row["objective"], int(row["budget"]), int(row["seed"])))
    return paired


def _adaptive_paired_effect_rows(paired_trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = {}
    for row in paired_trials:
        grouped.setdefault((str(row["variant"]), str(row["family"]), str(row["objective"]), int(row["budget"])), []).append(row)
    effect_rows: list[dict[str, Any]] = []
    for (variant, family, objective, budget), rows in grouped.items():
        deltas = np.asarray([float(row["paired_improvement"]) for row in rows], dtype=float)
        effect_rows.append(
            {
                "variant": variant,
                "family": family,
                "family_scope": str(rows[0]["family_scope"]),
                "objective": objective,
                "budget": budget,
                "best_constant_config_id": str(rows[0]["best_constant_config_id"]),
                "best_adaptive_config_id": str(rows[0]["best_adaptive_config_id"]),
                "best_adaptive_schedule_kind": str(rows[0]["best_adaptive_schedule_kind"]),
                "pair_count": len(rows),
                "paired_win_rate": float(np.mean([float(row["adaptive_win"]) for row in rows])),
                "median_paired_improvement": float(np.median(deltas)),
                "mean_paired_improvement": float(np.mean(deltas)),
            }
        )
    effect_rows.sort(key=lambda row: (row["variant"], row["family"], row["objective"], int(row["budget"])))
    return effect_rows


def _bootstrap_ci(values: np.ndarray, *, resamples: int = 1000, seed: int = 0) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(resamples, values.size))
    sample_medians = np.median(values[indices], axis=1)
    low, high = np.quantile(sample_medians, [0.025, 0.975])
    return float(low), float(high)


def _adaptive_required_rows(spec: StudySpec, paired_trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    highest_budget = max(spec.adaptive_confirmation_budgets) if spec.adaptive_confirmation_budgets else 0
    by_family: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in paired_trials:
        by_family.setdefault((str(row["variant"]), str(row["family"])), []).append(row)
    rows: list[dict[str, Any]] = []
    for definition in _filtered_families(spec):
        for variant in definition.applicable_variants:
            if variant not in spec.variants:
                continue
            runtime_capable = bool(_allowed_schedule_kinds(definition, variant))
            family_trials = by_family.get((variant, definition.name), [])
            high_budget_trials = [row for row in family_trials if int(row["budget"]) == highest_budget]
            deltas = np.asarray([float(row["paired_improvement"]) for row in high_budget_trials], dtype=float)
            objective_medians: dict[str, float] = {}
            for objective_name in spec.objectives:
                objective_rows = [row for row in high_budget_trials if str(row["objective"]) == objective_name]
                if not objective_rows:
                    continue
                objective_medians[objective_name] = float(np.median([float(row["paired_improvement"]) for row in objective_rows]))
            high_budget_objective_wins = sum(1 for value in objective_medians.values() if value >= 0.0)
            paired_win_rate = float(np.mean([float(row["adaptive_win"]) for row in high_budget_trials])) if high_budget_trials else 0.0
            median_paired_improvement = float(np.median(deltas)) if deltas.size else 0.0
            ci_low, ci_high = _bootstrap_ci(deltas, seed=_stable_seed(variant, definition.name, highest_budget)) if deltas.size else (0.0, 0.0)
            best_constant_config_id = "" if not high_budget_trials else str(high_budget_trials[0]["best_constant_config_id"])
            best_adaptive_config_id = "" if not high_budget_trials else str(high_budget_trials[0]["best_adaptive_config_id"])
            classification = "not_runtime_adaptable"
            if runtime_capable:
                classification = "constant_sufficient"
                required_objective_wins = min(4, len(spec.objectives))
                if spec.strict_adaptive_rule:
                    if deltas.size and ci_low > 0.0 and paired_win_rate >= 0.60 and high_budget_objective_wins >= required_objective_wins:
                        classification = "adaptive_required"
                    elif deltas.size and median_paired_improvement > 0.0:
                        classification = "adaptive_helpful"
                else:
                    if deltas.size and median_paired_improvement > 0.0 and paired_win_rate >= 0.60:
                        classification = "adaptive_required"
                    elif deltas.size and median_paired_improvement > 0.0:
                        classification = "adaptive_helpful"
            rows.append(
                {
                    "variant": variant,
                    "family": definition.name,
                    "family_scope": definition.family_scope,
                    "best_constant_config_id": best_constant_config_id,
                    "best_adaptive_config_id": best_adaptive_config_id,
                    "paired_win_rate": paired_win_rate,
                    "median_paired_improvement": median_paired_improvement,
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "high_budget_objective_wins": int(high_budget_objective_wins),
                    "compared_pairs": int(len(high_budget_trials)),
                    "classification": classification,
                }
            )
    rows.sort(key=lambda row: (row["variant"], row["family"]))
    return rows


def _field_manifest_entries() -> dict[str, dict[str, dict[str, Any]]]:
    smoothlife_scope_map = {field_name: "static_only" for field_name in SMOOTHLIFE_FIELD_NAMES}
    for field_name in SMOOTHLIFE_PER_STEP_FIELDS:
        smoothlife_scope_map[field_name] = "sls_per_step"
    for field_name in ZOOM_BOUNDARY_FIELDS:
        smoothlife_scope_map[field_name] = "agsls_zoom_boundary"
    agsls_scope_map = {field_name: "static_only" for field_name in AGSLS_FIELD_NAMES}
    for field_name in AGSLS_PER_DECISION_FIELDS:
        agsls_scope_map[field_name] = "agsls_decision"

    def _entry(owner: str, field_name: str, scope: str) -> dict[str, Any]:
        adaptive_variants: tuple[str, ...]
        if scope == "sls_per_step":
            adaptive_variants = ("sls", "agsls")
        elif scope in {"agsls_per_step", "agsls_decision", "agsls_zoom_boundary"}:
            adaptive_variants = ("agsls",)
        else:
            adaptive_variants = ()
        return {
            "owner": owner,
            "scope": scope,
            "adaptive_variants": list(adaptive_variants),
            "compatible_schedule_kinds": list(FIELD_SCOPE_SCHEDULE_KINDS[scope]),
        }

    return {
        "smoothlife": {field_name: _entry("smoothlife", field_name, smoothlife_scope_map[field_name]) for field_name in sorted(SMOOTHLIFE_FIELD_NAMES)},
        "agsls": {field_name: _entry("agsls", field_name, agsls_scope_map[field_name]) for field_name in sorted(AGSLS_FIELD_NAMES)},
    }


def _family_manifest(spec: StudySpec) -> dict[str, Any]:
    families: dict[str, Any] = {}
    for definition in _filtered_families(spec):
        families[definition.name] = {
            "family_scope": definition.family_scope,
            "applicable_variants": list(definition.applicable_variants),
            "adaptive_variants": list(definition.adaptive_variants),
            "schedule_kinds": list(definition.schedule_kinds),
        }
    manifest = _field_manifest_entries()
    manifest["families"] = families
    manifest["profile"] = spec.profile
    return manifest


def _report_markdown(records: list[dict[str, Any]], adaptive_rows: list[dict[str, Any]]) -> str:
    final_records = _stage_records(records, {"stage6_final_confirmation"})
    source_records = final_records if final_records else records
    stats = _annotate_bucket_ranks(_bucket_stats(source_records))
    overall = _overall_rank_rows(stats)
    winners: dict[str, dict[str, Any] | None] = {"sls": None, "agsls": None, "pooled": None}
    for scope in winners:
        scoped_rows = [row for row in overall if str(row["scope"]) == scope]
        winners[scope] = None if not scoped_rows else min(scoped_rows, key=lambda row: (float(row["mean_rank"]), str(row["config_id"])))
    required_rows = [row for row in adaptive_rows if str(row["classification"]) == "adaptive_required"]
    lines = ["# Tune Benchmark Report", "", "## Winners"]
    for scope, title in (("sls", "Best SLS"), ("agsls", "Best AGSLS"), ("pooled", "Best Pooled")):
        row = winners[scope]
        if row is None:
            lines.append(f"- {title}: none")
            continue
        lines.append(
            f"- {title}: `{row['config_label']}` (`{row['config_id']}`), family `{row['family']}`, "
            f"schedule `{row['schedule_kind']}`, mean rank {float(row['mean_rank']):.3f}"
        )
    lines.extend(["", "## Adaptive Required Families"])
    if not required_rows:
        lines.append("- None")
    else:
        for row in required_rows:
            lines.append(
                f"- `{row['variant']}` / `{row['family']}`: median improvement {float(row['median_paired_improvement']):.6f}, "
                f"95% CI [{float(row['ci_low']):.6f}, {float(row['ci_high']):.6f}], win rate {float(row['paired_win_rate']):.3f}"
            )
    return "\n".join(lines) + "\n"


def _write_artifacts(spec: StudySpec, records: list[dict[str, Any]], finalists: dict[str, list[dict[str, Any]]]) -> dict[str, Path]:
    stats = _annotate_bucket_ranks(_bucket_stats(records))
    overall_rows = _overall_rank_rows(stats)
    parameter_rows = _parameter_effect_rows(stats, overall_rows)
    paired_trials = _adaptive_paired_trials(records)
    adaptive_effect_rows = _adaptive_paired_effect_rows(paired_trials)
    adaptive_rows = _adaptive_required_rows(spec, paired_trials)
    runtime_rows = _runtime_efficiency_rows(stats)
    manifest = _family_manifest(spec)
    report = _report_markdown(records, adaptive_rows)
    output_dir = spec.output_dir
    per_objective_path = output_dir / "per_objective_leaderboard.csv"
    overall_path = output_dir / "overall_rank.csv"
    parameter_path = output_dir / "parameter_effects.csv"
    adaptive_path = output_dir / "adaptive_required.csv"
    adaptive_effects_path = output_dir / "adaptive_paired_effects.csv"
    runtime_path = output_dir / "runtime_efficiency.csv"
    family_manifest_path = output_dir / "family_manifest.json"
    finalists_path = output_dir / "finalists.json"
    report_path = output_dir / "report.md"
    _write_csv(
        per_objective_path,
        sorted(stats, key=lambda row: (row["variant"], row["objective"], int(row["budget"]), int(row["variant_rank"]), row["config_id"])),
        [
            "stage_name",
            "variant",
            "objective",
            "budget",
            "config_id",
            "config_label",
            "family",
            "family_scope",
            "schedule_kind",
            "median_best_value",
            "iqr_low",
            "iqr_high",
            "iqr_width",
            "success_rate",
            "trials",
            "variant_rank",
            "pooled_rank",
        ],
    )
    _write_csv(
        overall_path,
        overall_rows,
        [
            "scope",
            "variant",
            "config_id",
            "config_label",
            "stage_name",
            "family",
            "family_scope",
            "schedule_kind",
            "mean_rank",
            "win_buckets",
            "bucket_count",
        ],
    )
    _write_csv(
        parameter_path,
        parameter_rows,
        [
            "variant",
            "family",
            "family_scope",
            "schedule_kind",
            "best_config_id",
            "best_config_label",
            "stage_name",
            "mean_rank",
            "win_buckets",
            "bucket_count",
            "evaluated_buckets",
        ],
    )
    _write_csv(
        adaptive_path,
        adaptive_rows,
        [
            "variant",
            "family",
            "family_scope",
            "best_constant_config_id",
            "best_adaptive_config_id",
            "paired_win_rate",
            "median_paired_improvement",
            "ci_low",
            "ci_high",
            "high_budget_objective_wins",
            "compared_pairs",
            "classification",
        ],
    )
    _write_csv(
        adaptive_effects_path,
        adaptive_effect_rows,
        [
            "variant",
            "family",
            "family_scope",
            "objective",
            "budget",
            "best_constant_config_id",
            "best_adaptive_config_id",
            "best_adaptive_schedule_kind",
            "pair_count",
            "paired_win_rate",
            "median_paired_improvement",
            "mean_paired_improvement",
        ],
    )
    _write_csv(
        runtime_path,
        runtime_rows,
        [
            "stage_name",
            "variant",
            "objective",
            "budget",
            "config_id",
            "config_label",
            "family",
            "family_scope",
            "schedule_kind",
            "mean_wall_time_s",
            "trials",
        ],
    )
    with family_manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    with finalists_path.open("w", encoding="utf-8") as handle:
        json.dump(finalists, handle, indent=2, sort_keys=True)
    report_path.write_text(report, encoding="utf-8")
    return {
        "per_objective_leaderboard": per_objective_path,
        "overall_rank": overall_path,
        "parameter_effects": parameter_path,
        "adaptive_required": adaptive_path,
        "adaptive_paired_effects": adaptive_effects_path,
        "runtime_efficiency": runtime_path,
        "family_manifest": family_manifest_path,
        "finalists": finalists_path,
        "report": report_path,
    }


def _run_stage(
    spec: StudySpec,
    trials_path: Path,
    *,
    stage_name: str,
    candidates: list[CandidateSpec],
    budgets: tuple[int, ...],
    seed_count: int,
    completed_keys: set[str],
) -> tuple[int, int]:
    payloads = _trial_payloads(spec, candidates=candidates, stage_name=stage_name, budgets=budgets, seed_count=seed_count)
    return _run_trials(spec, trials_path, payloads, completed_keys=completed_keys)


def _artifact_paths(summary: StudySummary) -> dict[str, Path]:
    return {
        "per_objective_leaderboard": summary.per_objective_leaderboard_path,
        "overall_rank": summary.overall_rank_path,
        "parameter_effects": summary.parameter_effects_path,
        "adaptive_required": summary.adaptive_required_path,
        "adaptive_paired_effects": summary.adaptive_paired_effects_path,
        "runtime_efficiency": summary.runtime_efficiency_path,
        "family_manifest": summary.family_manifest_path,
        "finalists": summary.finalists_path,
        "report": summary.report_path,
    }


def _prepare_output_dir(spec: StudySpec) -> Path:
    output_dir = Path(spec.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if spec.resume:
        return output_dir
    for filename in (
        "trials.ndjson",
        "per_objective_leaderboard.csv",
        "overall_rank.csv",
        "parameter_effects.csv",
        "adaptive_required.csv",
        "adaptive_paired_effects.csv",
        "runtime_efficiency.csv",
        "family_manifest.json",
        "finalists.json",
        "report.md",
    ):
        path = output_dir / filename
        if path.exists():
            path.unlink()
    return output_dir


def _run_preflight(spec: StudySpec) -> None:
    if not spec.preflight or spec.resolved_workers() <= 1:
        return
    with tempfile.TemporaryDirectory() as tempdir:
        base_dir = Path(tempdir)
        common = dict(
            objectives=("sphere", "ackley"),
            baseline_budgets=(40,),
            static_budgets=(40,),
            adaptive_screen_budgets=(40,),
            adaptive_confirmation_budgets=(60,),
            interaction_budgets=(60,),
            final_confirmation_budgets=(80,),
            stage1_seeds=1,
            stage2_seeds=1,
            stage3_seeds=1,
            stage4_seeds=1,
            stage5_seeds=1,
            stage6_seeds=1,
            interaction_top_k=2,
            finalist_limit=2,
            parameter_families=("diffusion", "objective_guidance", "basin_quantile"),
            resume=False,
            keep_snapshots_in_finalists=False,
            profile="smoke",
            strict_adaptive_rule=spec.strict_adaptive_rule,
            preflight=False,
        )
        serial_summary = _run_tuning_study_impl(StudySpec(output_dir=base_dir / "serial", workers=1, **common))
        parallel_summary = _run_tuning_study_impl(StudySpec(output_dir=base_dir / "parallel", workers=2, **common))
        serial_artifacts = _artifact_paths(serial_summary)
        parallel_artifacts = _artifact_paths(parallel_summary)
        for artifact_name in PREFLIGHT_ARTIFACT_NAMES:
            serial_text = serial_artifacts[artifact_name].read_text(encoding="utf-8")
            parallel_text = parallel_artifacts[artifact_name].read_text(encoding="utf-8")
            if serial_text != parallel_text:
                raise RuntimeError(f"preflight failed: serial and parallel artifacts differ for {artifact_name}")


def _run_tuning_study_impl(spec: StudySpec) -> StudySummary:
    output_dir = _prepare_output_dir(spec)
    trials_path = output_dir / "trials.ndjson"
    completed_records = _load_trial_records(trials_path)
    completed_keys = {str(record["trial_key"]) for record in completed_records}
    total_completed = 0
    total_skipped = 0

    stage1_candidates = _stage1_candidates(spec)
    completed, skipped = _run_stage(
        spec,
        trials_path,
        stage_name="stage1_baseline",
        candidates=stage1_candidates,
        budgets=tuple(spec.baseline_budgets),
        seed_count=spec.stage1_seeds,
        completed_keys=completed_keys,
    )
    total_completed += completed
    total_skipped += skipped
    completed_records = _load_trial_records(trials_path)

    stage2_candidates = _stage2_candidates(spec)
    completed, skipped = _run_stage(
        spec,
        trials_path,
        stage_name="stage2_static_ablation",
        candidates=stage2_candidates,
        budgets=tuple(spec.static_budgets),
        seed_count=spec.stage2_seeds,
        completed_keys=completed_keys,
    )
    total_completed += completed
    total_skipped += skipped
    completed_records = _load_trial_records(trials_path)

    stage3_candidates = _stage3_candidates(spec, completed_records)
    completed, skipped = _run_stage(
        spec,
        trials_path,
        stage_name="stage3_adaptive_screen",
        candidates=stage3_candidates,
        budgets=tuple(spec.adaptive_screen_budgets),
        seed_count=spec.stage3_seeds,
        completed_keys=completed_keys,
    )
    total_completed += completed
    total_skipped += skipped
    completed_records = _load_trial_records(trials_path)

    stage4_candidates = _stage4_candidates(spec, completed_records)
    completed, skipped = _run_stage(
        spec,
        trials_path,
        stage_name="stage4_adaptive_confirmation",
        candidates=stage4_candidates,
        budgets=tuple(spec.adaptive_confirmation_budgets),
        seed_count=spec.stage4_seeds,
        completed_keys=completed_keys,
    )
    total_completed += completed
    total_skipped += skipped
    completed_records = _load_trial_records(trials_path)

    stage5_candidates = _stage5_candidates(spec, completed_records)
    completed, skipped = _run_stage(
        spec,
        trials_path,
        stage_name="stage5_interaction",
        candidates=stage5_candidates,
        budgets=tuple(spec.interaction_budgets),
        seed_count=spec.stage5_seeds,
        completed_keys=completed_keys,
    )
    total_completed += completed
    total_skipped += skipped
    completed_records = _load_trial_records(trials_path)

    finalists = _select_finalists(completed_records, spec.finalist_limit)
    stage6_candidates = _stage6_candidates(spec, finalists)
    completed, skipped = _run_stage(
        spec,
        trials_path,
        stage_name="stage6_final_confirmation",
        candidates=stage6_candidates,
        budgets=tuple(spec.final_confirmation_budgets),
        seed_count=spec.stage6_seeds,
        completed_keys=completed_keys,
    )
    total_completed += completed
    total_skipped += skipped
    completed_records = _load_trial_records(trials_path)
    artifacts = _write_artifacts(spec, completed_records, finalists)
    total_trials = len(completed_records)
    return StudySummary(
        output_dir=output_dir,
        total_trials=total_trials,
        completed_trials=total_completed,
        skipped_trials=total_skipped,
        trials_path=trials_path,
        per_objective_leaderboard_path=artifacts["per_objective_leaderboard"],
        overall_rank_path=artifacts["overall_rank"],
        parameter_effects_path=artifacts["parameter_effects"],
        adaptive_required_path=artifacts["adaptive_required"],
        adaptive_paired_effects_path=artifacts["adaptive_paired_effects"],
        runtime_efficiency_path=artifacts["runtime_efficiency"],
        family_manifest_path=artifacts["family_manifest"],
        finalists_path=artifacts["finalists"],
        report_path=artifacts["report"],
    )


def run_tuning_study(spec: StudySpec) -> StudySummary:
    """Run the multi-stage parallel tuning study and materialize artifacts."""

    _run_preflight(spec)
    return _run_tuning_study_impl(spec)


__all__ = ["StudySpec", "StudySummary", "run_tuning_study"]
