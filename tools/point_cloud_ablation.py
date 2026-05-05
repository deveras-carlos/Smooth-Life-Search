"""Deterministic ablation matrix for Point-Cloud SmoothLife search."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from smooth_life_search import PointCloudSearchConfig, PointCloudSmoothLifeSearch, SmoothLifeConfig
from smooth_life_search.benchmark import ackley, rastrigin, rosenbrock, sphere

Objective = Callable[[np.ndarray], float]


@dataclass(frozen=True, slots=True)
class AblationCase:
    """One deterministic objective/bounds/seed/budget case."""

    name: str
    objective: Objective = field(repr=False)
    dimension: int
    seed: int
    budget: int
    bounds: tuple[tuple[float, float], ...]
    group: str = "primary"


@dataclass(frozen=True, slots=True)
class AblationVariant:
    """One optimizer configuration variant."""

    name: str
    overrides: Mapping[str, object]


def _orthogonal_matrix(dimension: int, seed: int = 321) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.sign(np.diag(r))
    signs[signs == 0.0] = 1.0
    return q * signs


def _sphere(point: np.ndarray) -> float:
    z = np.asarray(point, dtype=float)
    return float(np.dot(z, z))


def _ellipsoid(point: np.ndarray) -> float:
    z = np.asarray(point, dtype=float)
    weights = np.geomspace(1.0, 100.0, z.size)
    return float(np.dot(weights, z * z))


def _transformed_objective(
    base: Objective,
    target: np.ndarray,
    rotation: np.ndarray | None = None,
) -> Objective:
    resolved_target = np.asarray(target, dtype=float)
    resolved_rotation = np.eye(resolved_target.size, dtype=float) if rotation is None else np.asarray(rotation, dtype=float)

    def objective(point: np.ndarray) -> float:
        z = np.asarray(point, dtype=float) - resolved_target
        return float(base(resolved_rotation.T @ z))

    return objective


def primary_cases() -> list[AblationCase]:
    """Rosenbrock large-D cases that motivated the ablation pass."""

    return [
        AblationCase(
            name=f"rosenbrock_{dimension}d_seed7",
            objective=rosenbrock,
            dimension=dimension,
            seed=7,
            budget=6400,
            bounds=tuple([(-10.0, 10.0)] * dimension),
            group="primary",
        )
        for dimension in (30, 50, 100, 500)
    ]


def transformed_guardrail_cases() -> list[AblationCase]:
    """Shifted/rotated 12D quality cases used by integration tests."""

    dimension = 12
    target = np.linspace(-2.7, 3.1, dimension)
    rotation = _orthogonal_matrix(dimension)
    cases: list[tuple[str, Objective]] = [
        ("shifted_sphere", _transformed_objective(_sphere, target)),
        ("rotated_sphere", _transformed_objective(_sphere, target, rotation)),
        ("shifted_ellipsoid", _transformed_objective(_ellipsoid, target)),
        ("rotated_ellipsoid", _transformed_objective(_ellipsoid, target, rotation)),
        ("shifted_ackley", _transformed_objective(ackley, target)),
        ("rotated_ackley", _transformed_objective(ackley, target, rotation)),
        ("shifted_rosenbrock", _transformed_objective(rosenbrock, target)),
    ]
    return [
        AblationCase(
            name=f"{name}_12d_seed{seed}",
            objective=objective,
            dimension=dimension,
            seed=seed,
            budget=600,
            bounds=tuple([(-10.0, 10.0)] * dimension),
            group="transformed_12d",
        )
        for name, objective in cases
        for seed in (3, 7, 11)
    ]


def stress_cases() -> list[AblationCase]:
    """Stress and anchor cases that separate search quality from anchor hits."""

    symmetric_bounds = ((-10.0, 10.0), (-10.0, 10.0))
    return [
        AblationCase(
            name="ackley_shifted_bounds_2d_seed0",
            objective=ackley,
            dimension=2,
            seed=0,
            budget=20000,
            bounds=((-7.0, 13.0), (-7.0, 13.0)),
            group="stress",
        ),
        AblationCase(
            name="ackley_origin_anchor_2d_seed7",
            objective=ackley,
            dimension=2,
            seed=7,
            budget=256,
            bounds=symmetric_bounds,
            group="anchor",
        ),
        AblationCase(
            name="sphere_origin_anchor_2d_seed7",
            objective=sphere,
            dimension=2,
            seed=7,
            budget=256,
            bounds=symmetric_bounds,
            group="anchor",
        ),
        AblationCase(
            name="rastrigin_origin_anchor_2d_seed7",
            objective=rastrigin,
            dimension=2,
            seed=7,
            budget=256,
            bounds=((-5.12, 5.12), (-5.12, 5.12)),
            group="anchor",
        ),
    ]


def all_cases(case_set: str = "all") -> list[AblationCase]:
    if case_set == "primary":
        return primary_cases()
    if case_set == "guardrails":
        return [*transformed_guardrail_cases(), *stress_cases()]
    if case_set == "stress":
        return stress_cases()
    if case_set == "all":
        return [*primary_cases(), *transformed_guardrail_cases(), *stress_cases()]
    raise ValueError(f"unknown case set: {case_set}")


def default_variants() -> list[AblationVariant]:
    """Configuration variants for source contribution measurement."""

    return [
        AblationVariant("default", {}),
        AblationVariant("no_cooperative_refinement", {"cooperative_refinement_enabled": False}),
        AblationVariant("no_coherent_probes", {"coherent_probes_enabled": False}),
        AblationVariant("no_direction_refinement", {"direction_refinement_enabled": False}),
        AblationVariant("no_shade", {"shade_enabled": False}),
        AblationVariant("no_cma_region", {"cma_region_enabled": False}),
        AblationVariant("no_surrogate_ranking", {"surrogate_ranking_enabled": False}),
        AblationVariant("no_restart_strategy", {"restart_strategy_enabled": False}),
        AblationVariant("no_local_refinement", {"local_refinement_enabled": False}),
        AblationVariant(
            "evolution_only",
            {
                "local_refinement_enabled": False,
                "direction_refinement_enabled": False,
                "cooperative_refinement_enabled": False,
                "basin_polishing_enabled": False,
                "linkage_blocks_enabled": False,
                "cross_block_lbfgs_enabled": False,
            },
        ),
        AblationVariant(
            "polishing_only",
            {
                "source_adaptation_enabled": False,
                "coherent_probes_enabled": False,
                "shade_enabled": False,
                "cma_region_enabled": False,
                "restart_strategy_enabled": False,
                "surrogate_ranking_enabled": False,
            },
        ),
    ]


def select_variants(names: str | Sequence[str] = "all") -> list[AblationVariant]:
    variants = default_variants()
    if names == "all":
        return variants
    requested = [name.strip() for name in names.split(",")] if isinstance(names, str) else list(names)
    by_name = {variant.name: variant for variant in variants}
    unknown = sorted(set(requested) - set(by_name))
    if unknown:
        raise ValueError(f"unknown ablation variants: {', '.join(unknown)}")
    return [by_name[name] for name in requested]


def _source_counts_from_events(events: Iterable[Mapping[str, object]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for event in events:
        source_counts = event.get("source_counts", {})
        if isinstance(source_counts, Mapping):
            counts.update({str(source): int(count) for source, count in source_counts.items()})
    return dict(sorted(counts.items()))


def _source_improvements_from_stats(stats: Mapping[str, object]) -> dict[str, int]:
    improvements: dict[str, int] = {}
    for source, payload in stats.items():
        if isinstance(payload, Mapping):
            improvements[str(source)] = int(payload.get("improvements", 0))
    return dict(sorted(improvements.items()))


def run_case_variant(case: AblationCase, variant: AblationVariant) -> dict[str, object]:
    config_kwargs = {
        "max_evaluations": case.budget,
        "batch_size": 32,
        "snapshot_interval_batches": 999,
        **dict(variant.overrides),
    }
    search = PointCloudSmoothLifeSearch(
        case.objective,
        case.bounds,
        SmoothLifeConfig(store_all_snapshots=False),
        PointCloudSearchConfig(**config_kwargs),
    )
    search.reset(seed=case.seed)
    run = search.run()
    best_sample = min(search.archive.samples, key=lambda sample: sample.value)
    metadata = run.metadata
    source_stats = metadata.get("source_stats", {})
    batch_events = metadata.get("batch_events", [])

    return {
        "case": case.name,
        "group": case.group,
        "variant": variant.name,
        "dimension": int(case.dimension),
        "seed": int(case.seed),
        "budget": int(case.budget),
        "evaluations": int(run.evaluations),
        "best_value": float(run.best_value),
        "best_source": str(best_sample.source),
        "best_sample_value": float(best_sample.value),
        "best_point": np.asarray(run.best_point, dtype=float).tolist(),
        "stop_reason": str(metadata.get("stop_reason", "")),
        "active_regions": int(metadata.get("active_region_count", 0)),
        "sleeping_regions": int(metadata.get("sleeping_region_count", 0)),
        "cooperative_refinement_active": bool(metadata.get("cooperative_refinement_active", False)),
        "active_set_size": int(metadata.get("active_set_size", 0)),
        "cooperative_improvements": int(
            sum(
                int(event.get("cooperative_improvements", 0))
                for event in batch_events
                if isinstance(event, Mapping)
            )
        ),
        "source_counts": _source_counts_from_events(batch_events if isinstance(batch_events, Sequence) else []),
        "source_improvements": _source_improvements_from_stats(source_stats if isinstance(source_stats, Mapping) else {}),
        "exact_diagonal_scout": bool(
            best_sample.source == "restart:scout"
            and abs(float(best_sample.value)) <= 1e-14
            and np.allclose(best_sample.point, np.ones(case.dimension))
        ),
    }


def run_matrix(cases: Sequence[AblationCase], variants: Sequence[AblationVariant]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for case in cases:
        for variant in variants:
            records.append(run_case_variant(case, variant))
    default_values = {
        (record["case"], record["seed"]): float(record["best_value"])
        for record in records
        if record["variant"] == "default"
    }
    for record in records:
        baseline = default_values.get((record["case"], record["seed"]))
        if baseline is None:
            record["delta_vs_default"] = None
            record["ratio_vs_default"] = None
            continue
        value = float(record["best_value"])
        record["delta_vs_default"] = float(value - baseline)
        record["ratio_vs_default"] = None if abs(baseline) <= 1e-300 else float(value / baseline)
    return records


def _format_float(value: object) -> str:
    if value is None:
        return ""
    return f"{float(value):.6e}"


def _top_improvements(record: Mapping[str, object], limit: int = 3) -> str:
    improvements = record.get("source_improvements", {})
    if not isinstance(improvements, Mapping):
        return ""
    ranked = sorted(
        ((str(source), int(count)) for source, count in improvements.items() if int(count) > 0),
        key=lambda item: (-item[1], item[0]),
    )
    return ", ".join(f"{source}:{count}" for source, count in ranked[:limit])


def format_markdown(records: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Point-Cloud Ablation Report",
        "",
        "Deterministic no-GIF ablation output. Lower best values are better.",
        "",
        "## Primary Rosenbrock Cases",
        "",
        "| case | variant | best value | delta vs default | ratio | evals | best source | active regions | coop active | improvements |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- |",
    ]
    for record in records:
        if record.get("group") != "primary":
            continue
        lines.append(
            "| {case} | {variant} | {best} | {delta} | {ratio} | {evals} | {source} | {regions} | {coop} | {improvements} |".format(
                case=record["case"],
                variant=record["variant"],
                best=_format_float(record["best_value"]),
                delta=_format_float(record.get("delta_vs_default")),
                ratio="" if record.get("ratio_vs_default") is None else f"{float(record['ratio_vs_default']):.3f}",
                evals=int(record["evaluations"]),
                source=record["best_source"],
                regions=int(record["active_regions"]),
                coop="yes" if bool(record["cooperative_refinement_active"]) else "no",
                improvements=_top_improvements(record),
            )
        )

    guardrail_records = [record for record in records if record.get("group") != "primary"]
    if guardrail_records:
        lines.extend(
            [
                "",
                "## Guardrail Summary",
                "",
                "| group | variant | cases | median ratio vs default | exact diagonal scout hits |",
                "| --- | --- | ---: | ---: | ---: |",
            ]
        )
        groups = sorted({str(record["group"]) for record in guardrail_records})
        variants = [variant.name for variant in default_variants()]
        for group in groups:
            for variant in variants:
                subset = [
                    record
                    for record in guardrail_records
                    if record.get("group") == group
                    and record.get("variant") == variant
                ]
                if not subset:
                    continue
                ratios = np.asarray(
                    [
                        float(record["ratio_vs_default"])
                        for record in subset
                        if record.get("ratio_vs_default") is not None
                    ],
                    dtype=float,
                )
                median_ratio = "n/a" if ratios.size == 0 else f"{float(np.median(ratios)):.3f}"
                scout_hits = sum(1 for record in subset if bool(record.get("exact_diagonal_scout")))
                lines.append(
                    f"| {group} | {variant} | {len(subset)} | {median_ratio} | {scout_hits} |"
                )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `delta vs default` is positive when the ablated run is worse than the default for the same case and seed.",
            "- Symmetric origin-anchor cases are guardrails only; exact zero there confirms anchor behavior, not search quality.",
            "- `exact diagonal scout hits` must stay zero for high-D Rosenbrock-style shortcut protection.",
            "",
        ]
    )
    return "\n".join(lines)


def format_csv(records: Sequence[Mapping[str, object]]) -> str:
    output = StringIO()
    fieldnames = [
        "case",
        "group",
        "variant",
        "dimension",
        "seed",
        "budget",
        "evaluations",
        "best_value",
        "delta_vs_default",
        "ratio_vs_default",
        "best_source",
        "stop_reason",
        "active_regions",
        "sleeping_regions",
        "cooperative_refinement_active",
        "active_set_size",
        "cooperative_improvements",
        "exact_diagonal_scout",
        "source_counts",
        "source_improvements",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        row = {key: record.get(key) for key in fieldnames}
        row["source_counts"] = json.dumps(record.get("source_counts", {}), sort_keys=True)
        row["source_improvements"] = json.dumps(record.get("source_improvements", {}), sort_keys=True)
        writer.writerow(row)
    return output.getvalue()


def format_records(records: Sequence[Mapping[str, object]], output_format: str) -> str:
    if output_format == "markdown":
        return format_markdown(records)
    if output_format == "csv":
        return format_csv(records)
    if output_format == "json":
        return json.dumps(list(records), indent=2, sort_keys=True) + "\n"
    if output_format == "jsonl":
        return "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    raise ValueError(f"unknown format: {output_format}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case-set",
        choices=("primary", "guardrails", "stress", "all"),
        default="all",
        help="Benchmark case set to run.",
    )
    parser.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variant names, or 'all'.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json", "jsonl", "csv"),
        default="markdown",
        help="Output format.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output path.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    records = run_matrix(all_cases(args.case_set), select_variants(args.variants))
    payload = format_records(records, args.format)
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
