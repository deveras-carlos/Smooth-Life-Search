"""Packaged command-line entrypoint for SmoothLife simulation and matrix search."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from .. import (
    MatrixSmoothLifeConfig,
    MatrixSmoothLifeSearch,
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
        raise ValueError("matrix requires --dimension >= 2")
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


def _matrix_shape_from_args(args: argparse.Namespace) -> tuple[int, int] | None:
    height = getattr(args, "matrix_height", None)
    width = getattr(args, "matrix_width", None)
    if height is None and width is None:
        return None
    if height is None or width is None:
        raise ValueError("pass both --matrix-height and --matrix-width together, or neither")
    return int(height), int(width)


def _build_configs(args: argparse.Namespace) -> tuple[list[tuple[float, float]], SmoothLifeConfig, MatrixSmoothLifeConfig]:
    bounds = _build_bounds(args.objective, args.dimension, args.lower, args.upper, command=args.command)
    defaults = MatrixSmoothLifeConfig()
    target_value = getattr(args, "target_value", None)
    early_stop_enabled = getattr(args, "early_stop_enabled", defaults.early_stop_enabled)
    early_stop_value = getattr(args, "early_stop_value", defaults.early_stop_value)
    if target_value is not None:
        early_stop_enabled = True
        early_stop_value = float(target_value)
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
        store_all_snapshots=getattr(args, "store_all_snapshots", True),
    )
    matrix = MatrixSmoothLifeConfig(
        max_evaluations=args.budget,
        max_steps=getattr(args, "max_steps", defaults.max_steps),
        matrix_shape=_matrix_shape_from_args(args),
        decoder=getattr(args, "decoder", defaults.decoder),
        decoder_gain=getattr(args, "decoder_gain", defaults.decoder_gain),
        projection_seed_offset=getattr(args, "projection_seed_offset", defaults.projection_seed_offset),
        cell_alive_threshold=getattr(args, "cell_alive_threshold", defaults.cell_alive_threshold),
        local_decoder_block_size=getattr(args, "local_decoder_block_size", defaults.local_decoder_block_size),
        local_decoder_overlap=getattr(args, "local_decoder_overlap", defaults.local_decoder_overlap),
        local_credit_enabled=getattr(args, "local_credit_enabled", defaults.local_credit_enabled),
        local_credit_strength=getattr(args, "local_credit_strength", defaults.local_credit_strength),
        local_credit_learning_rate=getattr(
            args,
            "local_credit_learning_rate",
            defaults.local_credit_learning_rate,
        ),
        local_credit_decay=getattr(args, "local_credit_decay", defaults.local_credit_decay),
        local_credit_clip=getattr(args, "local_credit_clip", defaults.local_credit_clip),
        patch_probe_enabled=getattr(args, "patch_probe_enabled", defaults.patch_probe_enabled),
        patch_probe_interval_evaluations=getattr(
            args,
            "patch_probe_interval_evaluations",
            defaults.patch_probe_interval_evaluations,
        ),
        patch_probe_count=getattr(args, "patch_probe_count", defaults.patch_probe_count),
        patch_probe_step=getattr(args, "patch_probe_step", defaults.patch_probe_step),
        steps_per_evaluation=getattr(args, "steps_per_evaluation", defaults.steps_per_evaluation),
        elite_pull_strength=getattr(args, "elite_pull_strength", defaults.elite_pull_strength),
        failure_damping=getattr(args, "failure_damping", defaults.failure_damping),
        reward_decay=getattr(args, "reward_decay", defaults.reward_decay),
        reward_boost=getattr(args, "reward_boost", defaults.reward_boost),
        mutation_noise=getattr(args, "mutation_noise", defaults.mutation_noise),
        mutation_decay=getattr(args, "mutation_decay", defaults.mutation_decay),
        advantage_strength=getattr(args, "advantage_strength", defaults.advantage_strength),
        advantage_decay=getattr(args, "advantage_decay", defaults.advantage_decay),
        direction_strength=getattr(args, "direction_strength", defaults.direction_strength),
        direction_decay=getattr(args, "direction_decay", defaults.direction_decay),
        temperature_init=getattr(args, "temperature_init", defaults.temperature_init),
        temperature_decay=getattr(args, "temperature_decay", defaults.temperature_decay),
        temperature_reheat=getattr(args, "temperature_reheat", defaults.temperature_reheat),
        stagnation_reheat_evaluations=getattr(
            args,
            "stagnation_reheat_evaluations",
            defaults.stagnation_reheat_evaluations,
        ),
        matrix_line_search_enabled=getattr(
            args,
            "matrix_line_search_enabled",
            defaults.matrix_line_search_enabled,
        ),
        matrix_line_search_alphas=tuple(
            getattr(args, "matrix_line_search_alphas", defaults.matrix_line_search_alphas)
        ),
        early_stop_enabled=early_stop_enabled,
        early_stop_value=early_stop_value,
        snapshot_interval=getattr(args, "matrix_snapshot_interval", defaults.snapshot_interval),
        store_all_snapshots=getattr(args, "store_all_snapshots", defaults.store_all_snapshots),
        best_improvement_tolerance=args.best_improvement_tolerance,
    )
    return bounds, smoothlife, matrix


def _objective_from_args(args: argparse.Namespace) -> ObjectiveFn:
    if args.objective is None:
        raise ValueError("objective is required")
    return OBJECTIVES[args.objective]


def _snapshot_payload(snapshot: Any) -> dict[str, Any]:
    return {
        "step_index": int(snapshot.step_index),
        "bounds": snapshot.bounds.tolist(),
        "best_point": snapshot.best_point.tolist(),
        "best_value": float(snapshot.best_value),
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
    bounds, smoothlife, _matrix = _build_configs(args)
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


def run_matrix_command(args: argparse.Namespace) -> dict[str, Any]:
    objective = _objective_from_args(args)
    bounds, smoothlife, matrix = _build_configs(args)
    target_value = matrix.early_stop_value if matrix.early_stop_enabled else None
    search = MatrixSmoothLifeSearch(objective, bounds, smoothlife, matrix)
    search.reset(seed=args.seed)
    run = search.run(evaluations=args.budget)
    run.metadata.update({"mode": "matrix", "objective": args.objective})
    gif_path = _maybe_write_animation(run, args.gif)
    _maybe_show_animation(run, args.show, title="Matrix SmoothLife")
    return {
        "mode": "matrix",
        "objective": args.objective,
        "dimension": args.dimension,
        "seed": args.seed,
        "budget": args.budget,
        "evaluations": run.evaluations,
        "best_value": run.best_value,
        "best_point": run.best_point.tolist(),
        "bounds": run.bounds.tolist(),
        "archive_size": run.metadata.get("archive_size", 0),
        "matrix_shape": run.metadata.get("matrix_shape", []),
        "decoder": run.metadata.get("decoder", ""),
        "steps": run.metadata.get("steps", 0),
        "stop_reason": run.metadata.get("stop_reason", ""),
        "target_value": target_value,
        "best_source": run.metadata.get("best_source", ""),
        "local_sparse_block_size": run.metadata.get("local_sparse_block_size", 0),
        "line_search_improvements": run.metadata.get("line_search_improvements", 0),
        "patch_probe_count": run.metadata.get("patch_probe_count", 0),
        "patch_probe_improvements": run.metadata.get("patch_probe_improvements", 0),
        "advantage_norm": run.metadata.get("advantage_norm", 0.0),
        "direction_norm": run.metadata.get("direction_norm", 0.0),
        "local_credit_norm": run.metadata.get("local_credit_norm", 0.0),
        "local_credit_mean": run.metadata.get("local_credit_mean", 0.0),
        "temperature_mean": run.metadata.get("temperature_mean", 0.0),
        "temperature_max": run.metadata.get("temperature_max", 0.0),
        "stagnation_count": run.metadata.get("stagnation_count", 0),
        "snapshots": [_snapshot_payload(snapshot) for snapshot in run.snapshots],
        "gif_path": gif_path,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    objective = _objective_from_args(args)
    bounds, smoothlife, matrix = _build_configs(args)
    seeds = list(range(args.seed_start, args.seed_start + args.trials))

    def search_factory(_seed: int) -> MatrixSmoothLifeSearch:
        return MatrixSmoothLifeSearch(
            objective,
            bounds,
            replace(smoothlife, maximize=args.maximize),
            replace(matrix, max_evaluations=args.budget),
        )

    results = run_seeded_trials(search_factory, seeds)
    summary = summarize_results(results, success_threshold=args.success_threshold, maximize=args.maximize)
    return {
        "mode": "benchmark",
        "engine": "matrix",
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


def _print_matrix(payload: dict[str, Any]) -> None:
    print(f"objective: {payload['objective']}")
    print(f"dimension: {payload['dimension']}")
    print(f"seed: {payload['seed']}")
    print(f"budget: {payload['budget']}")
    print(f"evaluations: {payload['evaluations']}")
    print(f"best value: {payload['best_value']:e}")
    print(f"best point: {payload['best_point']}")
    print(f"archive size: {payload['archive_size']}")
    print(f"matrix shape: {payload['matrix_shape']}")
    print(f"decoder: {payload['decoder']}")
    print(f"best source: {payload['best_source']}")
    print(f"steps: {payload['steps']}")
    print(f"stop reason: {payload['stop_reason']}")
    print(f"line search improvements: {payload['line_search_improvements']}")
    print(f"patch probes: {payload['patch_probe_count']}")
    if payload["target_value"] is not None:
        print(f"target value: {payload['target_value']:e}")
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
    print(f"engine: {payload['engine']}")
    print(f"trials: {payload['trials']}")
    print(f"seed start: {payload['seed_start']}")
    print(f"budget: {payload['budget']}")
    print(f"success threshold: {payload['success_threshold']}")
    print(f"median best value: {payload['median_best_value']:.8f}")
    print(f"iqr best value: [{payload['iqr_best_value'][0]:.8f}, {payload['iqr_best_value'][1]:.8f}]")
    print(f"success rate: {payload['success_rate']:.4f}")


def _add_matrix_args(parser: argparse.ArgumentParser, defaults: MatrixSmoothLifeConfig) -> None:
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps, help="Optional maximum SmoothLife matrix steps.")
    parser.add_argument("--matrix-height", type=int, default=None, help="Genome matrix height; pass with --matrix-width.")
    parser.add_argument("--matrix-width", type=int, default=None, help="Genome matrix width; pass with --matrix-height.")
    parser.add_argument("--decoder", choices=("local_sparse", "random_projection"), default=defaults.decoder, help="Matrix-to-vector decoder.")
    parser.add_argument("--decoder-gain", type=float, default=defaults.decoder_gain, help="Gain before tanh decoder squashing.")
    parser.add_argument("--projection-seed-offset", type=int, default=defaults.projection_seed_offset, help="Seed offset for deterministic random projection.")
    parser.add_argument("--cell-alive-threshold", type=float, default=defaults.cell_alive_threshold, help="Absolute cell value needed to participate in local sparse neighborhoods.")
    parser.add_argument("--local-decoder-block-size", type=int, default=defaults.local_decoder_block_size, help="Coordinate block size for local sparse neighborhoods.")
    parser.add_argument("--local-decoder-overlap", type=int, default=defaults.local_decoder_overlap, help="Coordinate overlap between adjacent local sparse blocks.")
    parser.add_argument("--no-local-credit", dest="local_credit_enabled", action="store_false", default=defaults.local_credit_enabled, help="Disable objective-derived local neighborhood credit.")
    parser.add_argument("--local-credit-strength", type=float, default=defaults.local_credit_strength, help="SmoothLife transition strength for local objective credit.")
    parser.add_argument("--local-credit-learning-rate", type=float, default=defaults.local_credit_learning_rate, help="Learning rate for assigning global improvement to changed neighborhoods.")
    parser.add_argument("--local-credit-decay", type=float, default=defaults.local_credit_decay, help="Per-evaluation local neighborhood credit decay.")
    parser.add_argument("--local-credit-clip", type=float, default=defaults.local_credit_clip, help="Absolute clip for local neighborhood credit.")
    parser.add_argument("--no-patch-probes", dest="patch_probe_enabled", action="store_false", default=defaults.patch_probe_enabled, help="Disable archive-accounted local patch counterfactual probes.")
    parser.add_argument("--patch-probe-interval-evaluations", type=int, default=defaults.patch_probe_interval_evaluations, help="Evaluation interval between local patch probe batches.")
    parser.add_argument("--patch-probe-count", type=int, default=defaults.patch_probe_count, help="Maximum patch probes per probe batch.")
    parser.add_argument("--patch-probe-step", type=float, default=defaults.patch_probe_step, help="Signed field step used for local patch probes.")
    parser.add_argument("--steps-per-evaluation", type=int, default=defaults.steps_per_evaluation, help="SmoothLife matrix steps between objective evaluations.")
    parser.add_argument("--elite-pull-strength", type=float, default=defaults.elite_pull_strength, help="Non-improving field pull toward the best field.")
    parser.add_argument("--failure-damping", type=float, default=defaults.failure_damping, help="Non-improving movement damping.")
    parser.add_argument("--reward-decay", type=float, default=defaults.reward_decay, help="Per-evaluation reward support decay.")
    parser.add_argument("--reward-boost", type=float, default=defaults.reward_boost, help="Reward support boost after improvements.")
    parser.add_argument("--mutation-noise", type=float, default=defaults.mutation_noise, help="Gaussian field mutation added per SmoothLife step.")
    parser.add_argument("--mutation-decay", type=float, default=defaults.mutation_decay, help="Mutation noise multiplier after improvements.")
    parser.add_argument("--advantage-strength", type=float, default=defaults.advantage_strength, help="Signed cell-credit pressure strength.")
    parser.add_argument("--advantage-decay", type=float, default=defaults.advantage_decay, help="Per-evaluation signed cell-credit decay.")
    parser.add_argument("--direction-strength", type=float, default=defaults.direction_strength, help="Decoded-direction pressure strength.")
    parser.add_argument("--direction-decay", type=float, default=defaults.direction_decay, help="Per-evaluation decoded-direction pressure decay.")
    parser.add_argument("--temperature-init", type=float, default=defaults.temperature_init, help="Initial per-cell exploration temperature.")
    parser.add_argument("--temperature-decay", type=float, default=defaults.temperature_decay, help="Per-evaluation temperature decay.")
    parser.add_argument("--temperature-reheat", type=float, default=defaults.temperature_reheat, help="Temperature added in uncertain cells after stagnation.")
    parser.add_argument("--stagnation-reheat-evaluations", type=int, default=defaults.stagnation_reheat_evaluations, help="Non-improving evaluations before temperature reheating.")
    parser.add_argument("--no-matrix-line-search", dest="matrix_line_search_enabled", action="store_false", default=defaults.matrix_line_search_enabled, help="Disable matrix-space line search after improvements.")
    parser.add_argument("--matrix-line-search-alphas", type=float, nargs="+", default=defaults.matrix_line_search_alphas, help="Positive alpha ladder for matrix-space line search.")
    parser.add_argument("--early-stop-enabled", action="store_true", default=defaults.early_stop_enabled, help="Allow explicit value-threshold early stopping.")
    parser.add_argument("--early-stop-value", type=float, default=defaults.early_stop_value, help="Best objective value threshold for explicit early stopping.")
    parser.add_argument("--target-value", type=float, default=None, help="Shorthand target that enables explicit early stopping at this best value.")
    parser.add_argument("--matrix-snapshot-interval", type=int, default=defaults.snapshot_interval, help="Matrix optimizer snapshot interval.")
    parser.add_argument("--no-store-all-snapshots", dest="store_all_snapshots", action="store_false", default=defaults.store_all_snapshots, help="Store only forced snapshots.")


def build_parser() -> argparse.ArgumentParser:
    matrix_defaults = MatrixSmoothLifeConfig()
    parser = argparse.ArgumentParser(
        description="Run SmoothLife simulation and Matrix SmoothLife optimization jobs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Optional JSON/TOML file providing parser defaults.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--objective", choices=sorted(OBJECTIVES), default=None, help="Benchmark objective to optimize.")
    shared.add_argument("--dimension", type=int, default=2, help="Problem dimension. Matrix supports N-D; simulate is 2D.")
    shared.add_argument("--budget", type=int, default=None, help="Objective evaluation budget.")
    shared.add_argument("--lower", type=float, default=None, help="Lower bound for every coordinate.")
    shared.add_argument("--upper", type=float, default=None, help="Upper bound for every coordinate.")
    shared.add_argument("--maximize", action="store_true", help="Treat the objective as a maximization problem.")
    shared.add_argument("--seed", type=int, default=0, help="RNG seed for the run.")
    shared.add_argument("--grid-height", type=int, default=128, help="SmoothLife kernel grid height for simulation.")
    shared.add_argument("--grid-width", type=int, default=128, help="SmoothLife kernel grid width for simulation.")
    shared.add_argument("--dt", type=float, default=0.25, help="SmoothLife time-step.")
    shared.add_argument("--diffusion", type=float, default=0.10, help="Diffusion strength.")
    shared.add_argument("--objective-coupling", type=float, default=0.30, help="How strongly reward support affects the transition function.")
    shared.add_argument("--objective-gamma", type=float, default=1.0, help="Exponent applied to normalized objective values in dense simulation.")
    shared.add_argument("--best-improvement-tolerance", type=float, default=0.0, help="Minimum objective improvement required to update the best record.")
    shared.add_argument("--preset", type=str, default="search", help="SmoothLife preset name.")
    shared.add_argument("--gif", type=str, default=None, help="Optional GIF output path.")
    shared.add_argument("--show", action="store_true", help="Open the animation in a Tk GUI viewer after the run completes.")

    matrix_shared = argparse.ArgumentParser(add_help=False)
    _add_matrix_args(matrix_shared, matrix_defaults)

    matrix = subparsers.add_parser("matrix", parents=[shared, matrix_shared], help="Run Matrix SmoothLife optimization.")
    matrix.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    simulate = subparsers.add_parser("simulate", parents=[shared], help="Run the literal SmoothLife simulator.")
    simulate.add_argument("--steps", type=int, default=24, help="Number of SmoothLife steps to run.")
    simulate.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")

    benchmark = subparsers.add_parser("benchmark", parents=[shared, matrix_shared], help="Run repeated seeded matrix trials.")
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
        if "target_value" in explicit_dests:
            explicit_dests.update({"early_stop_enabled", "early_stop_value"})
        elif "early_stop_value" in explicit_dests:
            explicit_dests.add("target_value")
        if "no_store_all_snapshots" in explicit_dests:
            explicit_dests.add("store_all_snapshots")
        if "no_matrix_line_search" in explicit_dests:
            explicit_dests.add("matrix_line_search_enabled")
        if "no_local_credit" in explicit_dests:
            explicit_dests.add("local_credit_enabled")
        if "no_patch_probes" in explicit_dests:
            explicit_dests.add("patch_probe_enabled")
        for key, value in load_config_file(config_args.config).items():
            if key not in explicit_dests:
                setattr(args, key, value)
    try:
        if args.command == "matrix":
            payload = run_matrix_command(args)
            if args.json:
                print(json.dumps(payload, indent=2))
            else:
                _print_matrix(payload)
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


__all__ = ["build_parser", "main", "run_benchmark", "run_matrix_command", "run_simulation"]


if __name__ == "__main__":
    raise SystemExit(main())
