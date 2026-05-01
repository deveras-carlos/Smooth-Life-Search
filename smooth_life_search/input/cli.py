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


def _build_bounds(objective_name: str | None, dimension: int, lower: float | None, upper: float | None) -> list[tuple[float, float]]:
    if objective_name is None:
        raise ValueError("objective is required")
    if dimension != 2:
        raise ValueError("this implementation currently supports only 2D problems")
    if objective_name == "himmelblau" and dimension != 2:
        raise ValueError("himmelblau requires --dimension 2")
    if (lower is None) != (upper is None):
        raise ValueError("pass both --lower and --upper together, or neither")
    if lower is None or upper is None:
        lower, upper = DEFAULT_BOUNDS[objective_name]
    if upper <= lower:
        raise ValueError("upper must be greater than lower")
    return [(lower, upper), (lower, upper)]


def _build_configs(args: argparse.Namespace) -> tuple[list[tuple[float, float]], SmoothLifeConfig, PointCloudSearchConfig]:
    bounds = _build_bounds(args.objective, args.dimension, args.lower, args.upper)
    cloud_defaults = PointCloudSearchConfig()
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
        surrogate_enabled=getattr(args, "surrogate_enabled", cloud_defaults.surrogate_enabled),
        surrogate_min_samples=getattr(args, "surrogate_min_samples", cloud_defaults.surrogate_min_samples),
        surrogate_max_samples=getattr(args, "surrogate_max_samples", cloud_defaults.surrogate_max_samples),
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
    shared.add_argument("--dimension", type=int, default=2, help="Problem dimension. Only 2D is currently supported.")
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
    cloud_shared.add_argument("--no-surrogate", dest="surrogate_enabled", action="store_false", default=cloud_defaults.surrogate_enabled, help="Disable local quadratic surrogate region candidates.")
    cloud_shared.add_argument("--surrogate-min-samples", type=int, default=cloud_defaults.surrogate_min_samples, help="Minimum samples for local quadratic fits.")
    cloud_shared.add_argument("--surrogate-max-samples", type=int, default=cloud_defaults.surrogate_max_samples, help="Maximum samples for local quadratic fits.")
    cloud_shared.add_argument("--no-local-refinement", dest="local_refinement_enabled", action="store_false", default=cloud_defaults.local_refinement_enabled, help="Disable finite-difference local refinement probes.")
    cloud_shared.add_argument("--local-refinement-start-evaluations", type=int, default=cloud_defaults.local_refinement_start_evaluations, help="Archive size before local refinement can run.")
    cloud_shared.add_argument("--local-refinement-max-evaluations", type=int, default=cloud_defaults.local_refinement_max_evaluations, help="Maximum local-refinement evaluations per run.")
    cloud_shared.add_argument("--local-refinement-step-fraction", type=float, default=cloud_defaults.local_refinement_step_fraction, help="Maximum local-refinement step in normalized box units.")

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
