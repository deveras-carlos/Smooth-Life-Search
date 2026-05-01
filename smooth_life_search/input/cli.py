"""Packaged command-line entrypoint for SmoothLife Search and AGSLS."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .. import (
    AGSLSConfig,
    AdaptiveGridSmoothLifeSearch,
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


def _build_configs(args: argparse.Namespace) -> tuple[list[tuple[float, float]], SmoothLifeConfig, AGSLSConfig]:
    bounds = _build_bounds(args.objective, args.dimension, args.lower, args.upper)
    smooth_defaults = SmoothLifeConfig()
    agsls_defaults = AGSLSConfig()
    smoothlife = SmoothLifeConfig(
        grid_shape=(args.grid_height, args.grid_width),
        dt=args.dt,
        diffusion=args.diffusion,
        objective_coupling=args.objective_coupling,
        objective_gamma=args.objective_gamma,
        best_improvement_tolerance=args.best_improvement_tolerance,
        objective_guidance_mode=getattr(args, "objective_guidance_mode", smooth_defaults.objective_guidance_mode),
        objective_rbf_top_k=getattr(args, "objective_rbf_top_k", smooth_defaults.objective_rbf_top_k),
        objective_rbf_sigma=getattr(args, "objective_rbf_sigma", smooth_defaults.objective_rbf_sigma),
        objective_rbf_temperature=getattr(args, "objective_rbf_temperature", smooth_defaults.objective_rbf_temperature),
        objective_uncertainty_weight=getattr(args, "objective_uncertainty_weight", smooth_defaults.objective_uncertainty_weight),
        objective_drift_strength=getattr(args, "objective_drift_strength", smooth_defaults.objective_drift_strength),
        objective_drift_clip=getattr(args, "objective_drift_clip", smooth_defaults.objective_drift_clip),
        run_mode="simulation" if args.command == "simulate" else "search",
        preset=args.preset,
        maximize=args.maximize,
    )
    agsls = AGSLSConfig(
        max_evaluations=args.budget,
        max_zoom_cycles=getattr(args, "zoom_cycles", agsls_defaults.max_zoom_cycles),
        exploration_fraction=getattr(args, "exploration_fraction", agsls_defaults.exploration_fraction),
        commit_fraction=getattr(args, "commit_fraction", agsls_defaults.commit_fraction),
        exploration_steps_per_tick=getattr(args, "exploration_steps", agsls_defaults.exploration_steps_per_tick),
        commit_steps_per_zoom=getattr(args, "commit_steps", agsls_defaults.commit_steps_per_zoom),
        exploitation_steps_per_zoom=getattr(args, "exploitation_steps", agsls_defaults.exploitation_steps_per_zoom),
        basin_quantile=getattr(args, "basin_quantile", agsls_defaults.basin_quantile),
        min_basin_cells=getattr(args, "min_basin_cells", agsls_defaults.min_basin_cells),
        min_alive_density=getattr(args, "min_alive_density", agsls_defaults.min_alive_density),
        commit_min_shrink_fraction=getattr(args, "commit_min_shrink_fraction", agsls_defaults.commit_min_shrink_fraction),
        commit_min_explored_fraction=getattr(args, "commit_min_explored_fraction", agsls_defaults.commit_min_explored_fraction),
        exploitation_shrink_fraction=getattr(args, "exploitation_shrink_fraction", agsls_defaults.exploitation_shrink_fraction),
        commit_guidance_top_k=getattr(args, "commit_guidance_top_k", agsls_defaults.commit_guidance_top_k),
        commit_guidance_sigma=getattr(args, "commit_guidance_sigma", agsls_defaults.commit_guidance_sigma),
        commit_guidance_temperature=getattr(args, "commit_guidance_temperature", agsls_defaults.commit_guidance_temperature),
        commit_uncertainty_weight=getattr(args, "commit_uncertainty_weight", agsls_defaults.commit_uncertainty_weight),
        commit_drift_strength=getattr(args, "commit_drift_strength", agsls_defaults.commit_drift_strength),
        exploitation_guidance_top_k=getattr(args, "exploitation_guidance_top_k", agsls_defaults.exploitation_guidance_top_k),
        exploitation_guidance_sigma=getattr(args, "exploitation_guidance_sigma", agsls_defaults.exploitation_guidance_sigma),
        exploitation_guidance_temperature=getattr(args, "exploitation_guidance_temperature", agsls_defaults.exploitation_guidance_temperature),
        exploitation_uncertainty_weight=getattr(args, "exploitation_uncertainty_weight", agsls_defaults.exploitation_uncertainty_weight),
        exploitation_drift_strength=getattr(args, "exploitation_drift_strength", agsls_defaults.exploitation_drift_strength),
        commit_surrogate_enabled=getattr(args, "commit_surrogate_enabled", agsls_defaults.commit_surrogate_enabled),
        commit_surrogate_min_samples=getattr(args, "commit_surrogate_min_samples", agsls_defaults.commit_surrogate_min_samples),
        commit_surrogate_max_samples=getattr(args, "commit_surrogate_max_samples", agsls_defaults.commit_surrogate_max_samples),
        trust_region_enabled=getattr(args, "trust_region_enabled", agsls_defaults.trust_region_enabled),
        commit_trust_region_evaluations=getattr(
            args,
            "commit_trust_region_evaluations",
            agsls_defaults.commit_trust_region_evaluations,
        ),
        exploitation_trust_region_evaluations=getattr(
            args,
            "exploitation_trust_region_evaluations",
            agsls_defaults.exploitation_trust_region_evaluations,
        ),
        trust_region_candidate_pool_size=getattr(
            args,
            "trust_region_candidate_pool_size",
            agsls_defaults.trust_region_candidate_pool_size,
        ),
        trust_region_initial_radius_fraction=getattr(
            args,
            "trust_region_initial_radius_fraction",
            agsls_defaults.trust_region_initial_radius_fraction,
        ),
        exploitation_valley_tracking_enabled=getattr(
            args,
            "exploitation_valley_tracking_enabled",
            agsls_defaults.exploitation_valley_tracking_enabled,
        ),
        exploitation_valley_probe_evaluations=getattr(
            args,
            "exploitation_valley_probe_evaluations",
            agsls_defaults.exploitation_valley_probe_evaluations,
        ),
        exploitation_valley_step_fraction=getattr(
            args,
            "exploitation_valley_step_fraction",
            agsls_defaults.exploitation_valley_step_fraction,
        ),
    )
    return bounds, smoothlife, agsls


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


def _zoom_payload(run: Any) -> list[dict[str, Any]]:
    return [
        {
            "zoom_index": event.zoom_index,
            "steps_per_zoom": event.steps_per_zoom,
            "old_bounds": event.old_bounds.tolist(),
            "new_bounds": event.new_bounds.tolist(),
            "selected_basin_score": event.selected_basin_score,
            "selected_basin_bbox": event.selected_basin_bbox.tolist(),
            "evaluation_count": event.evaluation_count,
            "diagnostics": dict(event.diagnostics),
        }
        for event in run.zoom_events
    ]


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
    bounds, smoothlife, _ = _build_configs(args)
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


def run_agsls_command(args: argparse.Namespace) -> dict[str, Any]:
    objective = _objective_from_args(args)
    bounds, smoothlife, agsls = _build_configs(args)
    controller = AdaptiveGridSmoothLifeSearch(objective, bounds, smoothlife, agsls)
    controller.reset(seed=args.seed)
    run = controller.run(zoom_cycles=args.zoom_cycles, evaluations=args.budget)
    run.metadata.update({"mode": "agsls", "objective": args.objective})
    gif_path = _maybe_write_animation(run, args.gif)
    _maybe_show_animation(run, args.show, title="AGSLS")
    return {
        "mode": "agsls",
        "objective": args.objective,
        "dimension": args.dimension,
        "seed": args.seed,
        "budget": args.budget,
        "evaluations": run.evaluations,
        "best_value": run.best_value,
        "best_point": run.best_point.tolist(),
        "bounds": run.bounds.tolist(),
        "zoom_events": _zoom_payload(run),
        "phase_counts": run.metadata.get("phase_counts", {}),
        "decision_trace": run.metadata.get("decision_trace", []),
        "trust_region_events": run.metadata.get("trust_region_events", []),
        "snapshots": [_snapshot_payload(snapshot) for snapshot in run.snapshots],
        "gif_path": gif_path,
    }


def run_single(args: argparse.Namespace) -> dict[str, Any]:
    return run_agsls_command(args) | {"mode": "single"}


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    objective = _objective_from_args(args)
    bounds, smoothlife, agsls = _build_configs(args)
    seeds = list(range(args.seed_start, args.seed_start + args.trials))

    def search_factory(_seed: int) -> AdaptiveGridSmoothLifeSearch:
        return AdaptiveGridSmoothLifeSearch(
            objective,
            bounds,
            replace(smoothlife, maximize=args.maximize),
            replace(agsls, max_evaluations=args.budget),
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


def _print_single(payload: dict[str, Any], show_stages: bool) -> None:
    print(f"objective: {payload['objective']}")
    print(f"dimension: {payload['dimension']}")
    print(f"seed: {payload['seed']}")
    print(f"budget: {payload['budget']}")
    print(f"evaluations: {payload['evaluations']}")
    print(f"best value: {payload['best_value']:e}")
    print(f"best point: {payload['best_point']}")
    print(f"phase counts: {payload['phase_counts']}")
    if show_stages:
        print("zoom events:")
        for event in payload["zoom_events"]:
            diagnostics = event["diagnostics"]
            print(
                "  "
                f"zoom={event['zoom_index']} phase={diagnostics.get('phase')} "
                f"reason={diagnostics.get('zoom_reason')} bounds={event['new_bounds']}"
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
    agsls_defaults = AGSLSConfig()
    parser = argparse.ArgumentParser(
        description="Run SmoothLife simulation and three-phase AGSLS jobs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Optional JSON/TOML file providing parser defaults.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--objective", choices=sorted(OBJECTIVES), default=None, help="Benchmark objective to optimize.")
    shared.add_argument("--dimension", type=int, default=2, help="Problem dimension. Only 2D is currently supported.")
    shared.add_argument("--budget", type=int, default=None, help="Objective evaluation budget for AGSLS.")
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

    agsls_shared = argparse.ArgumentParser(add_help=False)
    agsls_shared.add_argument("--zoom-cycles", type=int, default=agsls_defaults.max_zoom_cycles, help="Maximum number of AGSLS zooms.")
    agsls_shared.add_argument("--exploration-fraction", type=float, default=agsls_defaults.exploration_fraction, help="Budget fraction spent with SmoothLife only.")
    agsls_shared.add_argument("--commit-fraction", type=float, default=agsls_defaults.commit_fraction, help="Budget fraction where commit ends and exploitation begins.")
    agsls_shared.add_argument("--exploration-steps", type=int, default=agsls_defaults.exploration_steps_per_tick, help="SmoothLife steps per exploration tick.")
    agsls_shared.add_argument("--commit-steps", type=int, default=agsls_defaults.commit_steps_per_zoom, help="SmoothLife steps before each commit zoom.")
    agsls_shared.add_argument("--exploitation-steps", type=int, default=agsls_defaults.exploitation_steps_per_zoom, help="SmoothLife steps before each exploitation zoom.")
    agsls_shared.add_argument("--basin-quantile", type=float, default=agsls_defaults.basin_quantile, help="Support-field quantile used to detect basins.")
    agsls_shared.add_argument("--min-basin-cells", type=int, default=agsls_defaults.min_basin_cells, help="Minimum cells for a basin to be zoom-eligible.")
    agsls_shared.add_argument("--min-alive-density", type=float, default=agsls_defaults.min_alive_density, help="Minimum alive density for a basin to be zoom-eligible.")
    agsls_shared.add_argument("--commit-min-shrink-fraction", type=float, default=agsls_defaults.commit_min_shrink_fraction, help="Commit zoom side floor as a fraction of current bounds.")
    agsls_shared.add_argument("--commit-min-explored-fraction", type=float, default=agsls_defaults.commit_min_explored_fraction, help="Minimum active-box explored fraction before commit can zoom.")
    agsls_shared.add_argument("--exploitation-shrink-fraction", type=float, default=agsls_defaults.exploitation_shrink_fraction, help="Aggressive exploitation side shrink factor.")
    agsls_shared.add_argument("--commit-guidance-top-k", type=int, default=agsls_defaults.commit_guidance_top_k, help="Top evaluated samples used for commit RBF guidance.")
    agsls_shared.add_argument("--commit-guidance-sigma", type=float, default=agsls_defaults.commit_guidance_sigma, help="Commit RBF guidance sigma in normalized box units.")
    agsls_shared.add_argument("--commit-guidance-temperature", type=float, default=agsls_defaults.commit_guidance_temperature, help="Commit RBF softmax temperature.")
    agsls_shared.add_argument("--commit-uncertainty-weight", type=float, default=agsls_defaults.commit_uncertainty_weight, help="Commit unevaluated-cell bonus near guided support.")
    agsls_shared.add_argument("--commit-drift-strength", type=float, default=agsls_defaults.commit_drift_strength, help="Commit objective-gradient drift strength.")
    agsls_shared.add_argument("--exploitation-guidance-top-k", type=int, default=agsls_defaults.exploitation_guidance_top_k, help="Top evaluated samples used for exploitation RBF guidance.")
    agsls_shared.add_argument("--exploitation-guidance-sigma", type=float, default=agsls_defaults.exploitation_guidance_sigma, help="Exploitation RBF guidance sigma in normalized box units.")
    agsls_shared.add_argument("--exploitation-guidance-temperature", type=float, default=agsls_defaults.exploitation_guidance_temperature, help="Exploitation RBF softmax temperature.")
    agsls_shared.add_argument("--exploitation-uncertainty-weight", type=float, default=agsls_defaults.exploitation_uncertainty_weight, help="Exploitation unevaluated-cell bonus near guided support.")
    agsls_shared.add_argument("--exploitation-drift-strength", type=float, default=agsls_defaults.exploitation_drift_strength, help="Exploitation objective-gradient drift strength.")
    agsls_shared.add_argument("--no-commit-surrogate", dest="commit_surrogate_enabled", action="store_false", default=agsls_defaults.commit_surrogate_enabled, help="Disable the commit-phase quadratic surrogate zoom helper.")
    agsls_shared.add_argument("--commit-surrogate-min-samples", type=int, default=agsls_defaults.commit_surrogate_min_samples, help="Minimum evaluated samples required for commit surrogate fitting.")
    agsls_shared.add_argument("--commit-surrogate-max-samples", type=int, default=agsls_defaults.commit_surrogate_max_samples, help="Maximum evaluated samples used for commit surrogate fitting.")
    agsls_shared.add_argument("--no-trust-region", dest="trust_region_enabled", action="store_false", default=agsls_defaults.trust_region_enabled, help="Disable trust-region acquisition batches in commit/exploitation.")
    agsls_shared.add_argument("--trust-region-commit-evals", dest="commit_trust_region_evaluations", type=int, default=agsls_defaults.commit_trust_region_evaluations, help="Maximum trust-region probes per commit decision.")
    agsls_shared.add_argument("--trust-region-exploitation-evals", dest="exploitation_trust_region_evaluations", type=int, default=agsls_defaults.exploitation_trust_region_evaluations, help="Maximum trust-region probes per exploitation decision.")
    agsls_shared.add_argument("--trust-region-candidates", dest="trust_region_candidate_pool_size", type=int, default=agsls_defaults.trust_region_candidate_pool_size, help="Candidate pool size for trust-region acquisition.")
    agsls_shared.add_argument("--trust-region-initial-radius-fraction", type=float, default=agsls_defaults.trust_region_initial_radius_fraction, help="Initial trust-region radius as a fraction of active bounds.")
    agsls_shared.add_argument("--no-exploitation-valley-tracking", dest="exploitation_valley_tracking_enabled", action="store_false", default=agsls_defaults.exploitation_valley_tracking_enabled, help="Disable exploitation valley/manifold probe tracking.")
    agsls_shared.add_argument("--exploitation-valley-probes", dest="exploitation_valley_probe_evaluations", type=int, default=agsls_defaults.exploitation_valley_probe_evaluations, help="Maximum objective probes used for exploitation valley tracking.")
    agsls_shared.add_argument("--exploitation-valley-step-fraction", type=float, default=agsls_defaults.exploitation_valley_step_fraction, help="Initial normalized step used by exploitation valley tracking.")

    single = subparsers.add_parser("single", parents=[shared, agsls_shared], help="Run one AGSLS optimization job.")
    single.add_argument("--show-stages", action="store_true", help="Print per-zoom details.")
    single.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    agsls = subparsers.add_parser("agsls", parents=[shared, agsls_shared], help="Run AGSLS explicitly.")
    agsls.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    simulate = subparsers.add_parser("simulate", parents=[shared], help="Run the literal SmoothLife simulator without adaptive zoom.")
    simulate.add_argument("--steps", type=int, default=24, help="Number of SmoothLife steps to run.")
    simulate.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    benchmark = subparsers.add_parser("benchmark", parents=[shared, agsls_shared], help="Run repeated seeded AGSLS trials.")
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
        if "no_commit_surrogate" in explicit_dests:
            explicit_dests.add("commit_surrogate_enabled")
        if "no_trust_region" in explicit_dests:
            explicit_dests.add("trust_region_enabled")
        if "no_exploitation_valley_tracking" in explicit_dests:
            explicit_dests.add("exploitation_valley_tracking_enabled")
        for key, value in load_config_file(config_args.config).items():
            if key not in explicit_dests:
                setattr(args, key, value)
    try:
        if args.command == "single":
            payload = run_single(args)
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                _print_single(payload, show_stages=args.show_stages)
            return 0
        if args.command == "agsls":
            payload = run_agsls_command(args)
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                _print_single(payload, show_stages=True)
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


__all__ = ["build_parser", "main", "run_agsls_command", "run_benchmark", "run_simulation", "run_single"]


if __name__ == "__main__":
    raise SystemExit(main())
