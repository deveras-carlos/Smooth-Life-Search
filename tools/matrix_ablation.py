"""Small deterministic ablation runner for Matrix SmoothLife search."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smooth_life_search import MatrixSmoothLifeConfig, MatrixSmoothLifeSearch, SmoothLifeConfig
from smooth_life_search.benchmark import ackley, rosenbrock, sphere

Objective = Callable[[np.ndarray], float]


@dataclass(frozen=True, slots=True)
class AblationCase:
    name: str
    objective: Objective
    dimension: int
    seed: int
    budget: int
    bounds: tuple[tuple[float, float], ...]
    group: str


@dataclass(frozen=True, slots=True)
class AblationVariant:
    name: str
    config_overrides: dict[str, object]


VARIANTS: tuple[AblationVariant, ...] = (
    AblationVariant("default", {}),
    AblationVariant("no_reward", {"reward_boost": 0.0}),
    AblationVariant(
        "no_credit_steering",
        {"advantage_strength": 0.0, "direction_strength": 0.0, "temperature_reheat": 0.0},
    ),
    AblationVariant("no_line_search", {"matrix_line_search_enabled": False}),
    AblationVariant("no_elite_pull", {"elite_pull_strength": 0.0}),
    AblationVariant("no_failure_damping", {"failure_damping": 0.0}),
    AblationVariant("low_mutation", {"mutation_noise": 0.0}),
)


def parse_int_list(value: str | None) -> tuple[int, ...] | None:
    if value is None or value.strip() == "":
        return None
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def select_variants(spec: str | None) -> list[AblationVariant]:
    if spec is None or spec == "all":
        return list(VARIANTS)
    requested = [part.strip() for part in spec.split(",") if part.strip()]
    by_name = {variant.name: variant for variant in VARIANTS}
    missing = sorted(set(requested) - set(by_name))
    if missing:
        raise ValueError(f"unknown ablation variants: {', '.join(missing)}")
    return [by_name[name] for name in requested]


def _shifted_sphere(point: np.ndarray) -> float:
    target = np.linspace(-1.0, 1.0, point.size)
    shifted = np.asarray(point, dtype=float) - target
    return float(np.dot(shifted, shifted))


def all_cases(
    case_set: str,
    *,
    dimensions: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    budget: int = 6400,
) -> list[AblationCase]:
    selected_dimensions = tuple(dimensions or (30, 50, 100, 500))
    selected_seeds = tuple(seeds or (7,))
    cases: list[AblationCase] = []
    if case_set in {"primary", "all"}:
        for dimension in selected_dimensions:
            for seed in selected_seeds:
                cases.append(
                    AblationCase(
                        name=f"rosenbrock_{dimension}d_seed{seed}",
                        objective=rosenbrock,
                        dimension=dimension,
                        seed=seed,
                        budget=budget,
                        bounds=tuple([(-10.0, 10.0)] * dimension),
                        group="primary",
                    )
                )
    if case_set in {"guardrail", "all"}:
        for seed in selected_seeds:
            cases.append(
                AblationCase(
                    name=f"shifted_sphere_12d_seed{seed}",
                    objective=_shifted_sphere,
                    dimension=12,
                    seed=seed,
                    budget=min(budget, 1200),
                    bounds=tuple([(-5.0, 5.0)] * 12),
                    group="guardrail",
                )
            )
            cases.append(
                AblationCase(
                    name=f"ackley_5d_seed{seed}",
                    objective=ackley,
                    dimension=5,
                    seed=seed,
                    budget=min(budget, 1200),
                    bounds=tuple([(-10.0, 10.0)] * 5),
                    group="guardrail",
                )
            )
    if not cases:
        raise ValueError("case_set must be primary, guardrail, or all")
    return cases


def run_matrix(
    cases: list[AblationCase],
    variants: list[AblationVariant],
    *,
    progress: bool = False,
    progress_stream=sys.stderr,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    total = len(cases) * len(variants)
    index = 0
    default_by_case: dict[str, float] = {}
    for case in cases:
        for variant in variants:
            index += 1
            if progress:
                print(f"[{index}/{total}] {case.name} :: {variant.name}", file=progress_stream)
            config = MatrixSmoothLifeConfig(
                max_evaluations=case.budget,
                store_all_snapshots=False,
                **variant.config_overrides,
            )
            search = MatrixSmoothLifeSearch(
                case.objective,
                list(case.bounds),
                SmoothLifeConfig(store_all_snapshots=False),
                config,
            )
            search.reset(seed=case.seed)
            run = search.run()
            best_source = str(run.metadata.get("best_source", ""))
            default_value = default_by_case.setdefault(case.name, float(run.best_value))
            if variant.name == "default":
                default_by_case[case.name] = float(run.best_value)
                default_value = float(run.best_value)
            delta = float(run.best_value) - float(default_value)
            ratio = float(run.best_value) / float(default_value) if abs(float(default_value)) > 1e-300 else 1.0
            records.append(
                {
                    "case": case.name,
                    "group": case.group,
                    "variant": variant.name,
                    "objective": getattr(case.objective, "__name__", "objective"),
                    "dimension": case.dimension,
                    "seed": case.seed,
                    "budget": case.budget,
                    "evaluations": int(run.evaluations),
                    "best_value": float(run.best_value),
                    "best_source": best_source,
                    "stop_reason": str(run.metadata.get("stop_reason", "")),
                    "matrix_shape": list(run.metadata.get("matrix_shape", [])),
                    "source_counts": dict(run.metadata.get("source_counts", {})),
                    "delta_vs_default": delta,
                    "ratio_vs_default": ratio,
                    "removed_source_hit": best_source in {"restart:scout", "shade"} or ":cma" in best_source,
                }
            )
    return records


def format_markdown(records: list[dict[str, object]]) -> str:
    lines = [
        "# Matrix SmoothLife Ablation",
        "",
        "| case | variant | best value | delta vs default | best source |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for record in records:
        lines.append(
            "| {case} | {variant} | {best:.6e} | {delta:.6e} | {source} |".format(
                case=record["case"],
                variant=record["variant"],
                best=float(record["best_value"]),
                delta=float(record["delta_vs_default"]),
                source=record["best_source"],
            )
        )
    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-set", choices=("primary", "guardrail", "all"), default="primary")
    parser.add_argument("--dimensions", type=str, default=None)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--budget", type=int, default=6400)
    parser.add_argument("--variants", type=str, default="all")
    parser.add_argument("--format", choices=("markdown", "jsonl"), default="markdown")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cases = all_cases(
        args.case_set,
        dimensions=parse_int_list(args.dimensions),
        seeds=parse_int_list(args.seeds),
        budget=args.budget,
    )
    records = run_matrix(cases, select_variants(args.variants), progress=args.progress)
    if args.format == "markdown":
        output = format_markdown(records)
    else:
        output = "".join(json.dumps(record) + "\n" for record in records)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(output)
    else:
        print(output, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
