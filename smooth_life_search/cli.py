"""Packaged command-line entrypoint for SmoothLife Search and AGSLS."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from . import (
    AGSLSConfig,
    AdaptiveGridSmoothLifeSearch,
    SmoothLifeConfig,
    SmoothLifeSearch,
    StudySpec,
    open_run_viewer,
    run_seeded_trials,
    save_run_animation,
    summarize_results,
    run_tuning_study,
)
from .objectives import DEFAULT_BOUNDS, OBJECTIVES, ObjectiveFn


def _build_bounds(objective_name: str, dimension: int, lower: float | None, upper: float | None) -> list[tuple[float, float]]:
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
    smoothlife = SmoothLifeConfig(
        grid_shape=(args.grid_height, args.grid_width),
        dt=args.dt,
        diffusion=args.diffusion,
        objective_coupling=args.objective_coupling,
        run_mode="simulation" if args.command == "simulate" else "search",
        preset=args.preset,
        maximize=args.maximize,
    )
    agsls = AGSLSConfig(
        max_zoom_cycles=getattr(args, "zoom_cycles", 5),
        initial_steps_per_zoom=getattr(args, "steps_per_zoom", 32),
        min_steps_per_zoom=getattr(args, "min_steps_per_zoom", 8),
        zoom_decay=getattr(args, "zoom_decay", 0.75),
        max_evaluations=args.budget,
    )
    return bounds, smoothlife, agsls


def _snapshot_payload(snapshot: Any) -> dict[str, Any]:
    return {
        "step_index": snapshot.step_index,
        "bounds": snapshot.bounds.tolist(),
        "best_point": snapshot.best_point.tolist(),
        "best_value": snapshot.best_value,
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
    objective = OBJECTIVES[args.objective]
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
    objective = OBJECTIVES[args.objective]
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
        "snapshots": [_snapshot_payload(snapshot) for snapshot in run.snapshots],
        "gif_path": gif_path,
    }


def run_single(args: argparse.Namespace) -> dict[str, Any]:
    objective = OBJECTIVES[args.objective]
    bounds, smoothlife, agsls = _build_configs(args)
    controller = AdaptiveGridSmoothLifeSearch(objective, bounds, smoothlife, agsls)
    controller.reset(seed=args.seed)
    result = controller.run(zoom_cycles=args.zoom_cycles, evaluations=args.budget)
    result.metadata.update({"mode": "single", "objective": args.objective})
    gif_path = _maybe_write_animation(result, args.gif)
    _maybe_show_animation(result, args.show, title="Single Run")
    return {
        "mode": "single",
        "objective": args.objective,
        "dimension": args.dimension,
        "seed": args.seed,
        "budget": args.budget,
        "maximize": args.maximize,
        "evaluations": result.evaluations,
        "best_value": result.best_value,
        "best_point": result.best_point.tolist(),
        "bounds": result.bounds.tolist(),
        "zoom_events": _zoom_payload(result),
        "gif_path": gif_path,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    objective = OBJECTIVES[args.objective]
    bounds, smoothlife, agsls = _build_configs(args)
    seeds = list(range(args.seed_start, args.seed_start + args.trials))

    def search_factory(seed: int) -> AdaptiveGridSmoothLifeSearch:
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
    print(f"best value: {payload['best_value']:.8f}")
    print(f"best point: {payload['best_point']}")
    if show_stages:
        print("zoom events:")
        for event in payload["zoom_events"]:
            print(
                "  "
                f"zoom={event['zoom_index']} steps={event['steps_per_zoom']} "
                f"score={event['selected_basin_score']:.4f} bounds={event['new_bounds']}"
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


def _parse_csv_list(raw: str, *, cast: type = str) -> tuple[Any, ...]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one comma-separated value")
    return tuple(cast(item) for item in values)


def _parse_grid(raw: str) -> tuple[int, int]:
    lowered = raw.lower().replace("x", ",")
    parts = [part.strip() for part in lowered.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError("grid sizes must look like HEIGHTxWIDTH")
    return int(parts[0]), int(parts[1])


def _build_tune_spec(args: argparse.Namespace) -> StudySpec:
    return StudySpec(
        output_dir=Path(args.output_dir),
        objectives=tuple(_parse_csv_list(args.objectives)),
        baseline_budgets=tuple(_parse_csv_list(args.baseline_budgets, cast=int)),
        static_budgets=tuple(_parse_csv_list(args.static_budgets, cast=int)),
        adaptive_screen_budgets=tuple(_parse_csv_list(args.adaptive_screen_budgets, cast=int)),
        adaptive_confirmation_budgets=tuple(_parse_csv_list(args.adaptive_confirmation_budgets, cast=int)),
        interaction_budgets=tuple(_parse_csv_list(args.interaction_budgets, cast=int)),
        final_confirmation_budgets=tuple(_parse_csv_list(args.final_confirmation_budgets, cast=int)),
        seed_start=int(args.seed_start),
        stage1_seeds=int(args.stage1_seeds),
        stage2_seeds=int(args.stage2_seeds),
        stage3_seeds=int(args.stage3_seeds),
        stage4_seeds=int(args.stage4_seeds),
        stage5_seeds=int(args.stage5_seeds),
        stage6_seeds=int(args.stage6_seeds),
        stage1_grid_shape=_parse_grid(args.screen_grid),
        intermediate_grid_shape=_parse_grid(args.screen_grid),
        final_grid_shape=_parse_grid(args.final_grid),
        interaction_top_k=int(args.interaction_top_k),
        finalist_limit=int(args.finalist_limit),
        workers=None if args.workers is None else int(args.workers),
        resume=not bool(args.no_resume),
        keep_snapshots_in_finalists=bool(args.keep_finalist_snapshots),
        parameter_families=None if not args.parameter_families else tuple(_parse_csv_list(args.parameter_families)),
        profile=str(args.profile),
        strict_adaptive_rule=bool(args.strict_adaptive_rule),
        preflight=not bool(args.no_preflight),
    )


def run_tune(args: argparse.Namespace) -> dict[str, Any]:
    summary = run_tuning_study(_build_tune_spec(args))
    return {
        "mode": "tune",
        "output_dir": str(summary.output_dir),
        "total_trials": summary.total_trials,
        "completed_trials": summary.completed_trials,
        "skipped_trials": summary.skipped_trials,
        "trials_path": str(summary.trials_path),
        "per_objective_leaderboard_path": str(summary.per_objective_leaderboard_path),
        "overall_rank_path": str(summary.overall_rank_path),
        "parameter_effects_path": str(summary.parameter_effects_path),
        "adaptive_required_path": str(summary.adaptive_required_path),
        "adaptive_paired_effects_path": str(summary.adaptive_paired_effects_path),
        "runtime_efficiency_path": str(summary.runtime_efficiency_path),
        "family_manifest_path": str(summary.family_manifest_path),
        "finalists_path": str(summary.finalists_path),
        "report_path": str(summary.report_path),
    }


def _print_tune(payload: dict[str, Any]) -> None:
    print(f"output dir: {payload['output_dir']}")
    print(f"total trials: {payload['total_trials']}")
    print(f"completed this run: {payload['completed_trials']}")
    print(f"skipped from resume: {payload['skipped_trials']}")
    print(f"trials: {payload['trials_path']}")
    print(f"per-objective leaderboard: {payload['per_objective_leaderboard_path']}")
    print(f"overall rank: {payload['overall_rank_path']}")
    print(f"parameter effects: {payload['parameter_effects_path']}")
    print(f"adaptive required: {payload['adaptive_required_path']}")
    print(f"adaptive paired effects: {payload['adaptive_paired_effects_path']}")
    print(f"runtime efficiency: {payload['runtime_efficiency_path']}")
    print(f"family manifest: {payload['family_manifest_path']}")
    print(f"finalists: {payload['finalists_path']}")
    print(f"report: {payload['report_path']}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SmoothLife simulation and Adaptive Grid Smooth Life Search jobs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--objective", choices=sorted(OBJECTIVES), required=True, help="Benchmark objective to optimize.")
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
    shared.add_argument("--preset", type=str, default="search", help="SmoothLife preset name.")
    shared.add_argument("--gif", type=str, default=None, help="Optional GIF output path.")
    shared.add_argument("--show", action="store_true", help="Open the animation in a Tk GUI viewer after the run completes.")

    agsls_shared = argparse.ArgumentParser(add_help=False)
    agsls_shared.add_argument("--zoom-cycles", type=int, default=5, help="Maximum number of zoom cycles.")
    agsls_shared.add_argument("--steps-per-zoom", type=int, default=32, help="Initial SmoothLife steps per zoom cycle.")
    agsls_shared.add_argument("--min-steps-per-zoom", type=int, default=8, help="Minimum SmoothLife steps per zoom cycle.")
    agsls_shared.add_argument("--zoom-decay", type=float, default=0.75, help="Decay factor that makes zooms more frequent over time.")

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

    tune = subparsers.add_parser("tune", help="Run the parallel multi-stage tuning study.")
    tune.add_argument("--output-dir", type=str, default="tuning-output", help="Directory for NDJSON results and derived artifacts.")
    tune.add_argument("--objectives", type=str, default="sphere,ackley,rastrigin,griewank,rosenbrock,himmelblau", help="Comma-separated objectives.")
    tune.add_argument("--baseline-budgets", type=str, default="400,800,1600", help="Comma-separated budgets for the baseline stage.")
    tune.add_argument("--static-budgets", type=str, default="400,800", help="Comma-separated budgets for the static ablation stage.")
    tune.add_argument("--adaptive-screen-budgets", type=str, default="400,800,1600", help="Comma-separated budgets for adaptive schedule screening.")
    tune.add_argument("--adaptive-confirmation-budgets", type=str, default="800,1600,3200", help="Comma-separated budgets for adaptive head-to-head confirmation.")
    tune.add_argument("--interaction-budgets", type=str, default="800,1600", help="Comma-separated budgets for the interaction stage.")
    tune.add_argument("--final-confirmation-budgets", type=str, default="3200,6400", help="Comma-separated budgets for final confirmation.")
    tune.add_argument("--screen-grid", type=str, default="64x64", help="Grid shape for stages 1-4 as HEIGHTxWIDTH.")
    tune.add_argument("--final-grid", type=str, default="128x128", help="Grid shape for final confirmation as HEIGHTxWIDTH.")
    tune.add_argument("--seed-start", type=int, default=0, help="First seed used in every stage.")
    tune.add_argument("--stage1-seeds", type=int, default=24, help="Number of paired seeds for the baseline stage.")
    tune.add_argument("--stage2-seeds", type=int, default=16, help="Number of paired seeds for the static ablation stage.")
    tune.add_argument("--stage3-seeds", type=int, default=20, help="Number of paired seeds for the adaptive schedule screen.")
    tune.add_argument("--stage4-seeds", type=int, default=32, help="Number of paired seeds for adaptive confirmation.")
    tune.add_argument("--stage5-seeds", type=int, default=12, help="Number of paired seeds for the interaction stage.")
    tune.add_argument("--stage6-seeds", type=int, default=64, help="Number of paired seeds for final confirmation.")
    tune.add_argument("--interaction-top-k", type=int, default=6, help="How many top families per variant feed the pairwise interaction stage.")
    tune.add_argument("--finalist-limit", type=int, default=8, help="How many finalists per variant advance to final confirmation.")
    tune.add_argument("--workers", type=int, default=None, help="Worker process count. Defaults to cpu_count() - 1.")
    tune.add_argument("--parameter-families", type=str, default=None, help="Optional comma-separated subset of parameter families.")
    tune.add_argument("--profile", type=str, default="maximal", help="Named tune profile to record in the study metadata.")
    tune.add_argument("--relaxed-adaptive-rule", dest="strict_adaptive_rule", action="store_false", help="Use the relaxed adaptive classification rule.")
    tune.set_defaults(strict_adaptive_rule=True)
    tune.add_argument("--keep-finalist-snapshots", action="store_true", help="Retain full snapshots for final confirmation trials.")
    tune.add_argument("--no-resume", action="store_true", help="Ignore existing completed trials in the output directory.")
    tune.add_argument("--no-preflight", action="store_true", help="Skip the serial-vs-parallel smoke preflight check.")
    tune.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
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
        if args.command == "tune":
            payload = run_tune(args)
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                _print_tune(payload)
            return 0
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    return 1


__all__ = ["build_parser", "main", "run_agsls_command", "run_benchmark", "run_simulation", "run_single", "run_tune"]


if __name__ == "__main__":
    raise SystemExit(main())
