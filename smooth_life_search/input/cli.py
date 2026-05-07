"""Packaged command-line entrypoint for SmoothLife simulation and point-cloud search."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .. import (
    PointCloudSearchConfig,
    PointCloudSmoothLifeSearch,
    SmoothLifeConfig,
    SmoothLifeSearch,
    open_run_viewer,
    run_seeded_trials,
    save_run_animation,
    summarize_results,
)
from ..benchmark import DEFAULT_BOUNDS, OBJECTIVES, ObjectiveFn
from .config import load_config_file


def _build_bounds(
    objective_name: str | None,
    dimension: int,
    lower: float | None,
    upper: float | None,
    *,
    command: str,
) -> list[tuple[float, float]]:
    if objective_name is None:
        raise ValueError("objective is required")
    if dimension < 2:
        raise ValueError("point-cloud requires --dimension >= 2")
    if command == "simulate" and dimension != 2:
        raise ValueError("simulate currently supports only --dimension 2")
    if objective_name == "himmelblau" and dimension != 2:
        raise ValueError("himmelblau requires --dimension 2")
    if (lower is None) != (upper is None):
        raise ValueError("pass both --lower and --upper together, or neither")
    if lower is None or upper is None:
        lower, upper = DEFAULT_BOUNDS[objective_name]
    if upper <= lower:
        raise ValueError("upper must be greater than lower")
    return [(lower, upper) for _ in range(int(dimension))]


def _build_configs(args: argparse.Namespace) -> tuple[list[tuple[float, float]], SmoothLifeConfig, PointCloudSearchConfig]:
    bounds = _build_bounds(args.objective, args.dimension, args.lower, args.upper, command=args.command)
    cloud_defaults = PointCloudSearchConfig()
    target_value = getattr(args, "target_value", None)
    early_stop_enabled = getattr(args, "early_stop_enabled", cloud_defaults.early_stop_enabled)
    early_stop_value = getattr(args, "early_stop_value", cloud_defaults.early_stop_value)
    if target_value is not None:
        early_stop_enabled = True
        early_stop_value = target_value
    smoothlife = SmoothLifeConfig(
        grid_shape=(args.grid_height, args.grid_width),
        dt=args.dt,
        diffusion=args.diffusion,
        objective_coupling=args.objective_coupling,
        objective_gamma=args.objective_gamma,
        best_improvement_tolerance=args.best_improvement_tolerance,
        run_mode="simulation" if args.command == "simulate" else "search",
        preset=args.preset,
        maximize=args.maximize,
    )
    point_cloud = PointCloudSearchConfig(
        max_evaluations=args.budget,
        batch_size=getattr(args, "batch_size", cloud_defaults.batch_size),
        initial_design_size=getattr(args, "initial_design_size", cloud_defaults.initial_design_size),
        max_batches=getattr(args, "max_batches", cloud_defaults.max_batches),
        density_grid_shape=(
            getattr(args, "density_grid_height", cloud_defaults.density_grid_shape[0]),
            getattr(args, "density_grid_width", cloud_defaults.density_grid_shape[1]),
        ),
        density_elite_fraction=getattr(args, "density_elite_fraction", cloud_defaults.density_elite_fraction),
        density_sigma_fraction=getattr(args, "density_sigma_fraction", cloud_defaults.density_sigma_fraction),
        density_smooth_steps=getattr(args, "density_smooth_steps", cloud_defaults.density_smooth_steps),
        portfolio_size=getattr(args, "portfolio_size", cloud_defaults.portfolio_size),
        elite_fraction=getattr(args, "elite_fraction", cloud_defaults.elite_fraction),
        trust_regions_enabled=getattr(args, "trust_regions_enabled", cloud_defaults.trust_regions_enabled),
        region_initial_radius_fraction=getattr(
            args,
            "region_initial_radius_fraction",
            cloud_defaults.region_initial_radius_fraction,
        ),
        region_min_radius_fraction=getattr(args, "region_min_radius_fraction", cloud_defaults.region_min_radius_fraction),
        region_max_radius_fraction=getattr(args, "region_max_radius_fraction", cloud_defaults.region_max_radius_fraction),
        region_expand_factor=getattr(args, "region_expand_factor", cloud_defaults.region_expand_factor),
        region_shrink_factor=getattr(args, "region_shrink_factor", cloud_defaults.region_shrink_factor),
        region_candidate_fraction=getattr(args, "region_candidate_fraction", cloud_defaults.region_candidate_fraction),
        density_candidate_fraction=getattr(args, "density_candidate_fraction", cloud_defaults.density_candidate_fraction),
        global_candidate_fraction=getattr(args, "global_candidate_fraction", cloud_defaults.global_candidate_fraction),
        exploit_candidate_fraction=getattr(args, "exploit_candidate_fraction", cloud_defaults.exploit_candidate_fraction),
        early_stop_enabled=early_stop_enabled,
        early_stop_value=early_stop_value,
        region_stall_patience=getattr(args, "region_stall_patience", cloud_defaults.region_stall_patience),
        region_cooldown_batches=getattr(args, "region_cooldown_batches", cloud_defaults.region_cooldown_batches),
        global_exploration_floor=getattr(args, "global_exploration_floor", cloud_defaults.global_exploration_floor),
        region_stencil_fraction=getattr(args, "region_stencil_fraction", cloud_defaults.region_stencil_fraction),
        anisotropic_regions_enabled=getattr(
            args,
            "anisotropic_regions_enabled",
            cloud_defaults.anisotropic_regions_enabled,
        ),
        region_anisotropy_max=getattr(args, "region_anisotropy_max", cloud_defaults.region_anisotropy_max),
        region_geometry_min_samples=getattr(
            args,
            "region_geometry_min_samples",
            cloud_defaults.region_geometry_min_samples,
        ),
        active_subspace_size=getattr(args, "active_subspace_size", cloud_defaults.active_subspace_size),
        surrogate_enabled=getattr(args, "surrogate_enabled", cloud_defaults.surrogate_enabled),
        surrogate_min_samples=getattr(args, "surrogate_min_samples", cloud_defaults.surrogate_min_samples),
        surrogate_max_samples=getattr(args, "surrogate_max_samples", cloud_defaults.surrogate_max_samples),
        surrogate_full_quadratic_max_dimension=getattr(
            args,
            "surrogate_full_quadratic_max_dimension",
            cloud_defaults.surrogate_full_quadratic_max_dimension,
        ),
        projection_ensemble_enabled=getattr(
            args,
            "projection_ensemble_enabled",
            cloud_defaults.projection_ensemble_enabled,
        ),
        projection_ensemble_size=getattr(
            args,
            "projection_ensemble_size",
            cloud_defaults.projection_ensemble_size,
        ),
        projection_ensemble_refresh_batches=getattr(
            args,
            "projection_ensemble_refresh_batches",
            cloud_defaults.projection_ensemble_refresh_batches,
        ),
        local_refinement_enabled=getattr(args, "local_refinement_enabled", cloud_defaults.local_refinement_enabled),
        local_refinement_start_evaluations=getattr(
            args,
            "local_refinement_start_evaluations",
            cloud_defaults.local_refinement_start_evaluations,
        ),
        local_refinement_max_evaluations=getattr(
            args,
            "local_refinement_max_evaluations",
            cloud_defaults.local_refinement_max_evaluations,
        ),
        local_refinement_step_fraction=getattr(
            args,
            "local_refinement_step_fraction",
            cloud_defaults.local_refinement_step_fraction,
        ),
        local_refinement_method=getattr(args, "local_refinement_method", cloud_defaults.local_refinement_method),
        local_refinement_damping=getattr(args, "local_refinement_damping", cloud_defaults.local_refinement_damping),
        high_dimensional_refinement_enabled=getattr(
            args,
            "high_dimensional_refinement_enabled",
            cloud_defaults.high_dimensional_refinement_enabled,
        ),
        high_dimensional_min_dimension=getattr(
            args,
            "high_dimensional_min_dimension",
            cloud_defaults.high_dimensional_min_dimension,
        ),
        block_refinement_overlap=getattr(args, "block_refinement_overlap", cloud_defaults.block_refinement_overlap),
        block_refinement_blocks_per_pass=getattr(
            args,
            "block_refinement_blocks_per_pass",
            cloud_defaults.block_refinement_blocks_per_pass,
        ),
        dimension_scaled_batches_enabled=getattr(
            args,
            "dimension_scaled_batches_enabled",
            cloud_defaults.dimension_scaled_batches_enabled,
        ),
        dimension_scaled_batch_max=getattr(
            args,
            "dimension_scaled_batch_max",
            cloud_defaults.dimension_scaled_batch_max,
        ),
        coherent_probes_enabled=getattr(
            args,
            "coherent_probes_enabled",
            cloud_defaults.coherent_probes_enabled,
        ),
        surrogate_ranking_enabled=getattr(
            args,
            "surrogate_ranking_enabled",
            cloud_defaults.surrogate_ranking_enabled,
        ),
        candidate_pool_multiplier=getattr(
            args,
            "candidate_pool_multiplier",
            cloud_defaults.candidate_pool_multiplier,
        ),
        surrogate_ranking_neighbor_count=getattr(
            args,
            "surrogate_ranking_neighbor_count",
            cloud_defaults.surrogate_ranking_neighbor_count,
        ),
        probe_recenter_enabled=getattr(
            args,
            "probe_recenter_enabled",
            cloud_defaults.probe_recenter_enabled,
        ),
        probe_recenter_max_restarts=getattr(
            args,
            "probe_recenter_max_restarts",
            cloud_defaults.probe_recenter_max_restarts,
        ),
        basin_polishing_enabled=getattr(
            args,
            "basin_polishing_enabled",
            cloud_defaults.basin_polishing_enabled,
        ),
        basin_polishing_min_dimension=getattr(
            args,
            "basin_polishing_min_dimension",
            cloud_defaults.basin_polishing_min_dimension,
        ),
        basin_polishing_activation_ratio=getattr(
            args,
            "basin_polishing_activation_ratio",
            cloud_defaults.basin_polishing_activation_ratio,
        ),
        successful_direction_memory_size=getattr(
            args,
            "successful_direction_memory_size",
            cloud_defaults.successful_direction_memory_size,
        ),
        direction_refinement_enabled=getattr(
            args,
            "direction_refinement_enabled",
            cloud_defaults.direction_refinement_enabled,
        ),
        direction_refinement_max_evaluations=getattr(
            args,
            "direction_refinement_max_evaluations",
            cloud_defaults.direction_refinement_max_evaluations,
        ),
        direction_line_search_mode=getattr(
            args,
            "direction_line_search_mode",
            cloud_defaults.direction_line_search_mode,
        ),
        direction_line_search_max_steps=getattr(
            args,
            "direction_line_search_max_steps",
            cloud_defaults.direction_line_search_max_steps,
        ),
        direction_line_search_min_step_fraction=getattr(
            args,
            "direction_line_search_min_step_fraction",
            cloud_defaults.direction_line_search_min_step_fraction,
        ),
        linkage_blocks_enabled=getattr(
            args,
            "linkage_blocks_enabled",
            cloud_defaults.linkage_blocks_enabled,
        ),
        linkage_update_interval_batches=getattr(
            args,
            "linkage_update_interval_batches",
            cloud_defaults.linkage_update_interval_batches,
        ),
        linkage_neighbor_count=getattr(
            args,
            "linkage_neighbor_count",
            cloud_defaults.linkage_neighbor_count,
        ),
        cross_block_lbfgs_enabled=getattr(
            args,
            "cross_block_lbfgs_enabled",
            cloud_defaults.cross_block_lbfgs_enabled,
        ),
        cross_block_lbfgs_memory_size=getattr(
            args,
            "cross_block_lbfgs_memory_size",
            cloud_defaults.cross_block_lbfgs_memory_size,
        ),
        cooperative_refinement_enabled=getattr(
            args,
            "cooperative_refinement_enabled",
            cloud_defaults.cooperative_refinement_enabled,
        ),
        cooperative_min_dimension=getattr(
            args,
            "cooperative_min_dimension",
            cloud_defaults.cooperative_min_dimension,
        ),
        cooperative_group_size=getattr(
            args,
            "cooperative_group_size",
            cloud_defaults.cooperative_group_size,
        ),
        cooperative_groups_per_batch=getattr(
            args,
            "cooperative_groups_per_batch",
            cloud_defaults.cooperative_groups_per_batch,
        ),
        cooperative_frontier_enabled=getattr(
            args,
            "cooperative_frontier_enabled",
            cloud_defaults.cooperative_frontier_enabled,
        ),
        cooperative_frontier_fraction=getattr(
            args,
            "cooperative_frontier_fraction",
            cloud_defaults.cooperative_frontier_fraction,
        ),
        axis_coverage_pressure=getattr(
            args,
            "axis_coverage_pressure",
            cloud_defaults.axis_coverage_pressure,
        ),
        active_set_max_fraction=getattr(
            args,
            "active_set_max_fraction",
            cloud_defaults.active_set_max_fraction,
        ),
        active_set_expand_interval_batches=getattr(
            args,
            "active_set_expand_interval_batches",
            cloud_defaults.active_set_expand_interval_batches,
        ),
        projection_axes=(
            None
            if getattr(args, "projection_axes", cloud_defaults.projection_axes) is None
            else tuple(int(axis) for axis in getattr(args, "projection_axes"))
        ),
        surrogate_reliability_enabled=getattr(
            args,
            "surrogate_reliability_enabled",
            cloud_defaults.surrogate_reliability_enabled,
        ),
        surrogate_rank_weight_min=getattr(
            args,
            "surrogate_rank_weight_min",
            cloud_defaults.surrogate_rank_weight_min,
        ),
        surrogate_rank_weight_max=getattr(
            args,
            "surrogate_rank_weight_max",
            cloud_defaults.surrogate_rank_weight_max,
        ),
        best_improvement_tolerance=args.best_improvement_tolerance,
    )
    return bounds, smoothlife, point_cloud


def _objective_from_args(args: argparse.Namespace) -> ObjectiveFn:
    if args.objective is None:
        raise ValueError("objective is required")
    return OBJECTIVES[args.objective]


def _snapshot_payload(snapshot: Any) -> dict[str, Any]:
    return {
        "step_index": snapshot.step_index,
        "bounds": snapshot.bounds.tolist(),
        "best_point": snapshot.best_point.tolist(),
        "best_value": snapshot.best_value,
        "metadata": dict(snapshot.metadata),
    }


def _maybe_write_animation(run: Any, gif_path: str | None) -> str | None:
    if not gif_path:
        return None
    output = save_run_animation(run, Path(gif_path))
    return str(output)


def _maybe_show_animation(run: Any, show: bool, title: str) -> None:
    if not show:
        return
    run.metadata.setdefault("title", title)
    open_run_viewer(run, title=title)


def run_simulation(args: argparse.Namespace) -> dict[str, Any]:
    objective = _objective_from_args(args)
    bounds, smoothlife, _point_cloud = _build_configs(args)
    engine = SmoothLifeSearch(objective, bounds, smoothlife)
    engine.reset(seed=args.seed)
    run = engine.run(steps=args.steps)
    run.metadata.update({"mode": "simulation", "objective": args.objective})
    gif_path = _maybe_write_animation(run, args.gif)
    _maybe_show_animation(run, args.show, title="SmoothLife Simulation")
    return {
        "mode": "simulate",
        "objective": args.objective,
        "seed": args.seed,
        "steps": args.steps,
        "best_value": run.best_value,
        "best_point": run.best_point.tolist(),
        "bounds": run.bounds.tolist(),
        "snapshots": [_snapshot_payload(snapshot) for snapshot in run.snapshots],
        "gif_path": gif_path,
    }


def run_point_cloud_command(args: argparse.Namespace) -> dict[str, Any]:
    objective = _objective_from_args(args)
    bounds, smoothlife, point_cloud = _build_configs(args)
    if args.dimension != 2 and (args.gif or args.show):
        raise ValueError("point-cloud GIF/show visualization currently supports only --dimension 2")
    target_value = point_cloud.early_stop_value if point_cloud.early_stop_enabled else None
    search = PointCloudSmoothLifeSearch(objective, bounds, smoothlife, point_cloud)
    search.reset(seed=args.seed)
    run = search.run(evaluations=args.budget)
    run.metadata.update({"mode": "point-cloud", "objective": args.objective})
    gif_path = _maybe_write_animation(run, args.gif)
    _maybe_show_animation(run, args.show, title="Point-Cloud SmoothLife")
    return {
        "mode": "point-cloud",
        "objective": args.objective,
        "dimension": args.dimension,
        "seed": args.seed,
        "budget": args.budget,
        "evaluations": run.evaluations,
        "best_value": run.best_value,
        "best_point": run.best_point.tolist(),
        "bounds": run.bounds.tolist(),
        "archive_size": run.metadata.get("archive_size", 0),
        "portfolio": run.metadata.get("portfolio", []),
        "batch_events": run.metadata.get("batch_events", []),
        "region_events": run.metadata.get("region_events", []),
        "trust_region_events": run.metadata.get("trust_region_events", []),
        "stop_reason": run.metadata.get("stop_reason", ""),
        "target_value": target_value,
        "local_refinement_stalled": run.metadata.get("local_refinement_stalled", False),
        "active_region_count": run.metadata.get("active_region_count", 0),
        "sleeping_region_count": run.metadata.get("sleeping_region_count", 0),
        "snapshots": [_snapshot_payload(snapshot) for snapshot in run.snapshots],
        "gif_path": gif_path,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    objective = _objective_from_args(args)
    bounds, smoothlife, point_cloud = _build_configs(args)
    seeds = list(range(args.seed_start, args.seed_start + args.trials))

    def search_factory(_seed: int) -> PointCloudSmoothLifeSearch:
        return PointCloudSmoothLifeSearch(
            objective,
            bounds,
            replace(smoothlife, maximize=args.maximize),
            replace(point_cloud, max_evaluations=args.budget),
        )

    results = run_seeded_trials(search_factory, seeds)
    summary = summarize_results(results, success_threshold=args.success_threshold, maximize=args.maximize)
    return {
        "mode": "benchmark",
        "objective": args.objective,
        "dimension": args.dimension,
        "seed_start": args.seed_start,
        "trials": args.trials,
        "budget": args.budget,
        "maximize": args.maximize,
        "success_threshold": args.success_threshold,
        "median_best_value": summary.median_best_value,
        "iqr_best_value": list(summary.iqr_best_value),
        "success_rate": summary.success_rate,
        "best_values": summary.best_values.tolist(),
        "seeds": seeds,
    }


def _print_point_cloud(payload: dict[str, Any], show_batches: bool) -> None:
    print(f"objective: {payload['objective']}")
    print(f"dimension: {payload['dimension']}")
    print(f"seed: {payload['seed']}")
    print(f"budget: {payload['budget']}")
    print(f"evaluations: {payload['evaluations']}")
    print(f"best value: {payload['best_value']:e}")
    print(f"best point: {payload['best_point']}")
    print(f"archive size: {payload['archive_size']}")
    print(f"portfolio size: {len(payload['portfolio'])}")
    print(f"active regions: {payload['active_region_count']}")
    print(f"sleeping regions: {payload['sleeping_region_count']}")
    print(f"stop reason: {payload['stop_reason']}")
    if payload["target_value"] is not None:
        print(f"target value: {payload['target_value']:e}")
    if show_batches:
        print("batch events:")
        for event in payload["batch_events"]:
            print(
                "  "
                f"batch={event['batch_index']} kind={event['kind']} "
                f"spent={event['evaluations_spent']} improved={event['improved']} "
                f"best={event['best_after']:e}"
            )
    if payload["gif_path"] is not None:
        print(f"gif: {payload['gif_path']}")


def _print_simulation(payload: dict[str, Any]) -> None:
    print(f"objective: {payload['objective']}")
    print(f"seed: {payload['seed']}")
    print(f"steps: {payload['steps']}")
    print(f"best value: {payload['best_value']:.8f}")
    print(f"best point: {payload['best_point']}")
    if payload["gif_path"] is not None:
        print(f"gif: {payload['gif_path']}")


def _print_benchmark(payload: dict[str, Any]) -> None:
    print(f"objective: {payload['objective']}")
    print(f"dimension: {payload['dimension']}")
    print(f"trials: {payload['trials']}")
    print(f"seed start: {payload['seed_start']}")
    print(f"budget: {payload['budget']}")
    print(f"success threshold: {payload['success_threshold']}")
    print(f"median best value: {payload['median_best_value']:.8f}")
    print(f"iqr best value: [{payload['iqr_best_value'][0]:.8f}, {payload['iqr_best_value'][1]:.8f}]")
    print(f"success rate: {payload['success_rate']:.4f}")


def build_parser() -> argparse.ArgumentParser:
    cloud_defaults = PointCloudSearchConfig()
    parser = argparse.ArgumentParser(
        description="Run SmoothLife simulation and point-cloud SmoothLife optimization jobs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Optional JSON/TOML file providing parser defaults.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--objective", choices=sorted(OBJECTIVES), default=None, help="Benchmark objective to optimize.")
    shared.add_argument("--dimension", type=int, default=2, help="Problem dimension. Point-cloud supports N-D; simulate/rendering are 2D.")
    shared.add_argument("--budget", type=int, default=None, help="Objective evaluation budget.")
    shared.add_argument("--lower", type=float, default=None, help="Lower bound for every coordinate.")
    shared.add_argument("--upper", type=float, default=None, help="Upper bound for every coordinate.")
    shared.add_argument("--maximize", action="store_true", help="Treat the objective as a maximization problem.")
    shared.add_argument("--seed", type=int, default=0, help="RNG seed for the run.")
    shared.add_argument("--grid-height", type=int, default=128, help="SmoothLife grid height.")
    shared.add_argument("--grid-width", type=int, default=128, help="SmoothLife grid width.")
    shared.add_argument("--dt", type=float, default=0.25, help="SmoothLife time-step.")
    shared.add_argument("--diffusion", type=float, default=0.10, help="Diffusion strength.")
    shared.add_argument("--objective-coupling", type=float, default=0.30, help="How strongly the objective affects the transition function.")
    shared.add_argument("--objective-gamma", type=float, default=1.0, help="Exponent applied to normalized objective values.")
    shared.add_argument("--best-improvement-tolerance", type=float, default=0.0, help="Minimum objective improvement required to update the best record.")
    shared.add_argument("--preset", type=str, default="search", help="SmoothLife preset name.")
    shared.add_argument("--gif", type=str, default=None, help="Optional GIF output path.")
    shared.add_argument("--show", action="store_true", help="Open the animation in a Tk GUI viewer after the run completes.")

    cloud_shared = argparse.ArgumentParser(add_help=False)
    cloud_shared.add_argument("--batch-size", type=int, default=cloud_defaults.batch_size, help="Candidate batch size.")
    cloud_shared.add_argument("--initial-design-size", type=int, default=cloud_defaults.initial_design_size, help="Initial archive design size.")
    cloud_shared.add_argument("--max-batches", type=int, default=cloud_defaults.max_batches, help="Optional maximum candidate batches.")
    cloud_shared.add_argument("--density-grid-height", type=int, default=cloud_defaults.density_grid_shape[0], help="Derived density grid height.")
    cloud_shared.add_argument("--density-grid-width", type=int, default=cloud_defaults.density_grid_shape[1], help="Derived density grid width.")
    cloud_shared.add_argument("--density-elite-fraction", type=float, default=cloud_defaults.density_elite_fraction, help="Archive fraction used to rasterize density.")
    cloud_shared.add_argument("--density-sigma-fraction", type=float, default=cloud_defaults.density_sigma_fraction, help="RBF sigma for density rasterization in normalized units.")
    cloud_shared.add_argument("--density-smooth-steps", type=int, default=cloud_defaults.density_smooth_steps, help="SmoothLife-style density smoothing steps.")
    cloud_shared.add_argument("--portfolio-size", type=int, default=cloud_defaults.portfolio_size, help="Maximum adaptive proposal regions.")
    cloud_shared.add_argument("--elite-fraction", type=float, default=cloud_defaults.elite_fraction, help="Archive fraction used to seed regions.")
    cloud_shared.add_argument("--no-trust-regions", dest="trust_regions_enabled", action="store_false", default=cloud_defaults.trust_regions_enabled, help="Disable adaptive region candidate batches.")
    cloud_shared.add_argument("--region-initial-radius-fraction", type=float, default=cloud_defaults.region_initial_radius_fraction, help="Initial region radius in normalized box units.")
    cloud_shared.add_argument("--region-min-radius-fraction", type=float, default=cloud_defaults.region_min_radius_fraction, help="Minimum region radius in normalized box units.")
    cloud_shared.add_argument("--region-max-radius-fraction", type=float, default=cloud_defaults.region_max_radius_fraction, help="Maximum region radius in normalized box units.")
    cloud_shared.add_argument("--region-expand-factor", type=float, default=cloud_defaults.region_expand_factor, help="Radius multiplier after region success.")
    cloud_shared.add_argument("--region-shrink-factor", type=float, default=cloud_defaults.region_shrink_factor, help="Radius multiplier after region failure.")
    cloud_shared.add_argument("--region-candidate-fraction", type=float, default=cloud_defaults.region_candidate_fraction, help="Batch fraction drawn from proposal regions.")
    cloud_shared.add_argument("--density-candidate-fraction", type=float, default=cloud_defaults.density_candidate_fraction, help="Batch fraction sampled from the density view.")
    cloud_shared.add_argument("--global-candidate-fraction", type=float, default=cloud_defaults.global_candidate_fraction, help="Batch fraction sampled globally.")
    cloud_shared.add_argument("--exploit-candidate-fraction", type=float, default=cloud_defaults.exploit_candidate_fraction, help="Batch fraction probing near the incumbent.")
    cloud_shared.add_argument("--early-stop-enabled", action="store_true", default=cloud_defaults.early_stop_enabled, help="Allow explicit value-threshold early stopping.")
    cloud_shared.add_argument("--early-stop-value", type=float, default=cloud_defaults.early_stop_value, help="Best objective value threshold for explicit early stopping.")
    cloud_shared.add_argument("--target-value", type=float, default=None, help="Shorthand target that enables explicit early stopping at this best value.")
    cloud_shared.add_argument("--region-stall-patience", type=int, default=cloud_defaults.region_stall_patience, help="Failed region batches before a region cools down.")
    cloud_shared.add_argument("--region-cooldown-batches", type=int, default=cloud_defaults.region_cooldown_batches, help="Batches a stalled region sleeps before receiving candidates again.")
    cloud_shared.add_argument("--global-exploration-floor", type=float, default=cloud_defaults.global_exploration_floor, help="Fraction of density candidates reserved for global low-discrepancy exploration.")
    cloud_shared.add_argument("--region-stencil-fraction", type=float, default=cloud_defaults.region_stencil_fraction, help="Fraction of region candidates used for coordinate/cross stencils.")
    cloud_shared.add_argument("--no-anisotropic-regions", dest="anisotropic_regions_enabled", action="store_false", default=cloud_defaults.anisotropic_regions_enabled, help="Disable archive-derived rotated region geometry.")
    cloud_shared.add_argument("--region-anisotropy-max", type=float, default=cloud_defaults.region_anisotropy_max, help="Maximum major/minor axis ratio for anisotropic proposal regions.")
    cloud_shared.add_argument("--region-geometry-min-samples", type=int, default=cloud_defaults.region_geometry_min_samples, help="Minimum nearby samples needed to derive anisotropic region geometry.")
    cloud_shared.add_argument("--active-subspace-size", type=int, default=cloud_defaults.active_subspace_size, help="Maximum local geometry subspace dimension for N-D regions.")
    cloud_shared.add_argument("--no-surrogate", dest="surrogate_enabled", action="store_false", default=cloud_defaults.surrogate_enabled, help="Disable local quadratic surrogate region candidates.")
    cloud_shared.add_argument("--surrogate-min-samples", type=int, default=cloud_defaults.surrogate_min_samples, help="Minimum samples for local quadratic fits.")
    cloud_shared.add_argument("--surrogate-max-samples", type=int, default=cloud_defaults.surrogate_max_samples, help="Maximum samples for local quadratic fits.")
    cloud_shared.add_argument("--surrogate-full-quadratic-max-dimension", type=int, default=cloud_defaults.surrogate_full_quadratic_max_dimension, help="Maximum dimension using full quadratic surrogate fits.")
    cloud_shared.add_argument("--projection-axes", type=int, nargs=2, metavar=("I", "J"), default=cloud_defaults.projection_axes, help="Optional coordinate axes used for 2D density projections.")
    cloud_shared.add_argument("--no-projection-ensemble", dest="projection_ensemble_enabled", action="store_false", default=cloud_defaults.projection_ensemble_enabled, help="Disable deterministic multi-projection SmoothLife density views.")
    cloud_shared.add_argument("--projection-ensemble-size", type=int, default=cloud_defaults.projection_ensemble_size, help="Maximum 2D projection frames blended for density sampling.")
    cloud_shared.add_argument("--projection-ensemble-refresh-batches", type=int, default=cloud_defaults.projection_ensemble_refresh_batches, help="Batches between projection-ensemble diagnostic refreshes.")
    cloud_shared.add_argument("--no-local-refinement", dest="local_refinement_enabled", action="store_false", default=cloud_defaults.local_refinement_enabled, help="Disable finite-difference local refinement probes.")
    cloud_shared.add_argument("--local-refinement-start-evaluations", type=int, default=cloud_defaults.local_refinement_start_evaluations, help="Archive size before local refinement can run.")
    cloud_shared.add_argument("--local-refinement-max-evaluations", type=int, default=cloud_defaults.local_refinement_max_evaluations, help="Maximum local-refinement evaluations per run.")
    cloud_shared.add_argument("--local-refinement-step-fraction", type=float, default=cloud_defaults.local_refinement_step_fraction, help="Maximum local-refinement step in normalized box units.")
    cloud_shared.add_argument("--local-refinement-method", choices=("bfgs", "levenberg-marquardt", "hybrid"), default=cloud_defaults.local_refinement_method, help="Finite-difference local refinement direction strategy.")
    cloud_shared.add_argument("--local-refinement-damping", type=float, default=cloud_defaults.local_refinement_damping, help="Initial damping used by Levenberg-Marquardt local refinement.")
    cloud_shared.add_argument("--no-high-dimensional-refinement", dest="high_dimensional_refinement_enabled", action="store_false", default=cloud_defaults.high_dimensional_refinement_enabled, help="Disable block-local refinement for high-dimensional runs.")
    cloud_shared.add_argument("--high-dimensional-min-dimension", type=int, default=cloud_defaults.high_dimensional_min_dimension, help="Dimension threshold where high-dimensional refinement activates.")
    cloud_shared.add_argument("--block-refinement-overlap", type=int, default=cloud_defaults.block_refinement_overlap, help="Coordinate overlap between consecutive high-dimensional refinement blocks.")
    cloud_shared.add_argument("--block-refinement-blocks-per-pass", type=int, default=cloud_defaults.block_refinement_blocks_per_pass, help="Number of high-dimensional refinement blocks tried per local pass.")
    cloud_shared.add_argument("--no-dimension-scaled-batches", dest="dimension_scaled_batches_enabled", action="store_false", default=cloud_defaults.dimension_scaled_batches_enabled, help="Disable dimension-scaled candidate batch sizing.")
    cloud_shared.add_argument("--dimension-scaled-batch-max", type=int, default=cloud_defaults.dimension_scaled_batch_max, help="Maximum effective batch size when dimension-scaled batches are active.")
    cloud_shared.add_argument("--no-coherent-probes", dest="coherent_probes_enabled", action="store_false", default=cloud_defaults.coherent_probes_enabled, help="Disable high-dimensional coherent coordinate probes.")
    cloud_shared.add_argument("--no-surrogate-ranking", dest="surrogate_ranking_enabled", action="store_false", default=cloud_defaults.surrogate_ranking_enabled, help="Disable surrogate-ranked preselection for high-dimensional proposal pools.")
    cloud_shared.add_argument("--candidate-pool-multiplier", type=int, default=cloud_defaults.candidate_pool_multiplier, help="Proposal pool multiplier used before surrogate-ranked preselection.")
    cloud_shared.add_argument("--surrogate-ranking-neighbor-count", type=int, default=cloud_defaults.surrogate_ranking_neighbor_count, help="Nearest archive samples used for surrogate proposal ranking.")
    cloud_shared.add_argument("--no-probe-recenter", dest="probe_recenter_enabled", action="store_false", default=cloud_defaults.probe_recenter_enabled, help="Disable recentering when finite-difference probes improve the incumbent.")
    cloud_shared.add_argument("--probe-recenter-max-restarts", type=int, default=cloud_defaults.probe_recenter_max_restarts, help="Maximum gradient recomputations after probe improvements.")
    cloud_shared.add_argument("--no-basin-polishing", dest="basin_polishing_enabled", action="store_false", default=cloud_defaults.basin_polishing_enabled, help="Disable high-dimensional basin-polishing allocation.")
    cloud_shared.add_argument("--basin-polishing-min-dimension", type=int, default=cloud_defaults.basin_polishing_min_dimension, help="Dimension threshold where basin-polishing mode may activate.")
    cloud_shared.add_argument("--basin-polishing-activation-ratio", type=float, default=cloud_defaults.basin_polishing_activation_ratio, help="Current/baseline target ratio needed to enter basin-polishing mode.")
    cloud_shared.add_argument("--successful-direction-memory-size", type=int, default=cloud_defaults.successful_direction_memory_size, help="Recent successful normalized directions kept for polishing.")
    cloud_shared.add_argument("--no-direction-refinement", dest="direction_refinement_enabled", action="store_false", default=cloud_defaults.direction_refinement_enabled, help="Disable derivative-free successful-direction line-search polishing.")
    cloud_shared.add_argument("--direction-refinement-max-evaluations", type=int, default=cloud_defaults.direction_refinement_max_evaluations, help="Maximum evaluations spent by one direction-refinement pass.")
    cloud_shared.add_argument("--direction-line-search-mode", choices=("opportunistic", "bracketed"), default=cloud_defaults.direction_line_search_mode, help="Derivative-free direction refinement line-search policy.")
    cloud_shared.add_argument("--direction-line-search-max-steps", type=int, default=cloud_defaults.direction_line_search_max_steps, help="Maximum trial step scales per direction line-search side.")
    cloud_shared.add_argument("--direction-line-search-min-step-fraction", type=float, default=cloud_defaults.direction_line_search_min_step_fraction, help="Smallest normalized step fraction used by direction line search.")
    cloud_shared.add_argument("--no-linkage-blocks", dest="linkage_blocks_enabled", action="store_false", default=cloud_defaults.linkage_blocks_enabled, help="Disable linkage-aware high-dimensional refinement blocks.")
    cloud_shared.add_argument("--linkage-update-interval-batches", type=int, default=cloud_defaults.linkage_update_interval_batches, help="Batches between archive-derived linkage score refreshes.")
    cloud_shared.add_argument("--linkage-neighbor-count", type=int, default=cloud_defaults.linkage_neighbor_count, help="Strong neighbors included in linkage-aware blocks.")
    cloud_shared.add_argument("--no-cross-block-lbfgs", dest="cross_block_lbfgs_enabled", action="store_false", default=cloud_defaults.cross_block_lbfgs_enabled, help="Disable limited-memory cross-block BFGS directions.")
    cloud_shared.add_argument("--cross-block-lbfgs-memory-size", type=int, default=cloud_defaults.cross_block_lbfgs_memory_size, help="Number of accepted cross-block secant pairs kept.")
    cloud_shared.add_argument("--no-cooperative-refinement", dest="cooperative_refinement_enabled", action="store_false", default=cloud_defaults.cooperative_refinement_enabled, help="Disable large-D active-set cooperative refinement.")
    cloud_shared.add_argument("--cooperative-min-dimension", type=int, default=cloud_defaults.cooperative_min_dimension, help="Dimension threshold where cooperative active-set refinement activates.")
    cloud_shared.add_argument("--cooperative-group-size", type=int, default=cloud_defaults.cooperative_group_size, help="Coordinate group size for cooperative refinement; defaults to active subspace size.")
    cloud_shared.add_argument("--cooperative-groups-per-batch", type=int, default=cloud_defaults.cooperative_groups_per_batch, help="Cooperative coordinate groups attempted per batch.")
    cloud_shared.add_argument("--no-cooperative-frontier", dest="cooperative_frontier_enabled", action="store_false", default=cloud_defaults.cooperative_frontier_enabled, help="Disable frontier-biased cooperative group scheduling.")
    cloud_shared.add_argument("--cooperative-frontier-fraction", type=float, default=cloud_defaults.cooperative_frontier_fraction, help="Fraction of cooperative groups reserved for frontier propagation.")
    cloud_shared.add_argument("--axis-coverage-pressure", type=float, default=cloud_defaults.axis_coverage_pressure, help="Weight assigned to under-covered axes in cooperative active-set scoring.")
    cloud_shared.add_argument("--active-set-max-fraction", type=float, default=cloud_defaults.active_set_max_fraction, help="Maximum fraction of axes kept in the large-D active set.")
    cloud_shared.add_argument("--active-set-expand-interval-batches", type=int, default=cloud_defaults.active_set_expand_interval_batches, help="Batches between active-set coverage expansion refreshes.")
    cloud_shared.add_argument("--no-surrogate-reliability", dest="surrogate_reliability_enabled", action="store_false", default=cloud_defaults.surrogate_reliability_enabled, help="Disable reliability gating for surrogate-ranked proposal preselection.")
    cloud_shared.add_argument("--surrogate-rank-weight-min", type=float, default=cloud_defaults.surrogate_rank_weight_min, help="Minimum prediction weight when surrogate ranking reliability is poor.")
    cloud_shared.add_argument("--surrogate-rank-weight-max", type=float, default=cloud_defaults.surrogate_rank_weight_max, help="Maximum prediction weight when surrogate ranking reliability is strong.")

    point_cloud = subparsers.add_parser("point-cloud", parents=[shared, cloud_shared], help="Run point-cloud SmoothLife optimization.")
    point_cloud.add_argument("--show-batches", action="store_true", help="Print per-batch details.")
    point_cloud.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    simulate = subparsers.add_parser("simulate", parents=[shared], help="Run the literal SmoothLife simulator.")
    simulate.add_argument("--steps", type=int, default=24, help="Number of SmoothLife steps to run.")
    simulate.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    benchmark = subparsers.add_parser("benchmark", parents=[shared, cloud_shared], help="Run repeated seeded point-cloud trials.")
    benchmark.add_argument("--seed-start", type=int, default=0, help="First seed in the benchmark sweep.")
    benchmark.add_argument("--trials", type=int, default=10, help="Number of seeded runs.")
    benchmark.add_argument("--success-threshold", type=float, default=0.1, help="Best-value threshold for counting a run as successful.")
    benchmark.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=None)
    config_args, _remaining = config_parser.parse_known_args(raw_argv)
    parser = build_parser()
    args = parser.parse_args(raw_argv)
    if config_args.config is not None:
        explicit_dests = {
            token[2:].replace("-", "_")
            for token in raw_argv
            if token.startswith("--") and token != "--config"
        }
        if "no_trust_regions" in explicit_dests:
            explicit_dests.add("trust_regions_enabled")
        if "no_surrogate" in explicit_dests:
            explicit_dests.add("surrogate_enabled")
        if "no_local_refinement" in explicit_dests:
            explicit_dests.add("local_refinement_enabled")
        if "no_anisotropic_regions" in explicit_dests:
            explicit_dests.add("anisotropic_regions_enabled")
        if "no_projection_ensemble" in explicit_dests:
            explicit_dests.add("projection_ensemble_enabled")
        if "no_high_dimensional_refinement" in explicit_dests:
            explicit_dests.add("high_dimensional_refinement_enabled")
        if "no_dimension_scaled_batches" in explicit_dests:
            explicit_dests.add("dimension_scaled_batches_enabled")
        if "no_coherent_probes" in explicit_dests:
            explicit_dests.add("coherent_probes_enabled")
        if "no_surrogate_ranking" in explicit_dests:
            explicit_dests.add("surrogate_ranking_enabled")
        if "no_probe_recenter" in explicit_dests:
            explicit_dests.add("probe_recenter_enabled")
        if "no_basin_polishing" in explicit_dests:
            explicit_dests.add("basin_polishing_enabled")
        if "no_direction_refinement" in explicit_dests:
            explicit_dests.add("direction_refinement_enabled")
        if "no_surrogate_reliability" in explicit_dests:
            explicit_dests.add("surrogate_reliability_enabled")
        if "no_linkage_blocks" in explicit_dests:
            explicit_dests.add("linkage_blocks_enabled")
        if "no_cross_block_lbfgs" in explicit_dests:
            explicit_dests.add("cross_block_lbfgs_enabled")
        if "no_cooperative_refinement" in explicit_dests:
            explicit_dests.add("cooperative_refinement_enabled")
        if "no_cooperative_frontier" in explicit_dests:
            explicit_dests.add("cooperative_frontier_enabled")
        if "early_stop_enabled" in explicit_dests:
            explicit_dests.add("early_stop_enabled")
        if "target_value" in explicit_dests:
            explicit_dests.update({"early_stop_enabled", "early_stop_value"})
        elif "early_stop_value" in explicit_dests:
            explicit_dests.add("target_value")
        for key, value in load_config_file(config_args.config).items():
            if key not in explicit_dests:
                setattr(args, key, value)
    try:
        if args.command == "point-cloud":
            payload = run_point_cloud_command(args)
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                _print_point_cloud(payload, show_batches=args.show_batches)
            return 0
        if args.command == "simulate":
            payload = run_simulation(args)
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                _print_simulation(payload)
            return 0
        if args.command == "benchmark":
            payload = run_benchmark(args)
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                _print_benchmark(payload)
            return 0
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 1


__all__ = ["build_parser", "main", "run_benchmark", "run_point_cloud_command", "run_simulation"]


if __name__ == "__main__":
    raise SystemExit(main())
