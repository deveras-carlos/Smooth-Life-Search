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


def primary_cases(
    *,
    dimensions: Sequence[int] = (30, 50, 100, 500),
    seeds: Sequence[int] = (3, 7, 11),
    budget: int = 6400,
) -> list[AblationCase]:
    """Rosenbrock large-D cases that motivated the ablation pass."""

    return [
        AblationCase(
            name=f"rosenbrock_{dimension}d_seed{seed}",
            objective=rosenbrock,
            dimension=dimension,
            seed=seed,
            budget=budget,
            bounds=tuple([(-10.0, 10.0)] * dimension),
            group="primary",
        )
        for dimension in dimensions
        for seed in seeds
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


def all_cases(
    case_set: str = "all",
    *,
    dimensions: Sequence[int] | None = None,
    seeds: Sequence[int] | None = None,
    budget: int | None = None,
) -> list[AblationCase]:
    primary_dimensions = (30, 50, 100, 500) if dimensions is None else tuple(int(dimension) for dimension in dimensions)
    primary_seeds = (3, 7, 11) if seeds is None else tuple(int(seed) for seed in seeds)
    primary_budget = 6400 if budget is None else int(budget)
    if case_set == "primary":
        return primary_cases(dimensions=primary_dimensions, seeds=primary_seeds, budget=primary_budget)
    if case_set == "guardrails":
        cases = [*transformed_guardrail_cases(), *stress_cases()]
        return _filter_cases(cases, dimensions=dimensions, seeds=seeds, budget=budget)
    if case_set == "stress":
        return _filter_cases(stress_cases(), dimensions=dimensions, seeds=seeds, budget=budget)
    if case_set == "all":
        cases = [
            *primary_cases(dimensions=primary_dimensions, seeds=primary_seeds, budget=primary_budget),
            *transformed_guardrail_cases(),
            *stress_cases(),
        ]
        return _filter_cases(cases, dimensions=dimensions, seeds=seeds, budget=budget, preserve_primary=True)
    raise ValueError(f"unknown case set: {case_set}")


def _filter_cases(
    cases: Sequence[AblationCase],
    *,
    dimensions: Sequence[int] | None,
    seeds: Sequence[int] | None,
    budget: int | None,
    preserve_primary: bool = False,
) -> list[AblationCase]:
    dimension_filter = None if dimensions is None else {int(dimension) for dimension in dimensions}
    seed_filter = None if seeds is None else {int(seed) for seed in seeds}
    filtered: list[AblationCase] = []
    for case in cases:
        if dimension_filter is not None and int(case.dimension) not in dimension_filter:
            continue
        if seed_filter is not None and int(case.seed) not in seed_filter:
            continue
        if budget is not None and (not preserve_primary or case.group != "primary"):
            case = AblationCase(
                name=f"{case.name}_budget{int(budget)}",
                objective=case.objective,
                dimension=case.dimension,
                seed=case.seed,
                budget=int(budget),
                bounds=case.bounds,
                group=case.group,
            )
        filtered.append(case)
    return filtered


def default_variants() -> list[AblationVariant]:
    """Configuration variants for source contribution measurement."""

    return [
        AblationVariant("default", {}),
        AblationVariant("no_projection_ensemble", {"projection_ensemble_enabled": False}),
        AblationVariant("no_cooperative_refinement", {"cooperative_refinement_enabled": False}),
        AblationVariant("no_cooperative_frontier", {"cooperative_frontier_enabled": False}),
        AblationVariant("no_coherent_probes", {"coherent_probes_enabled": False}),
        AblationVariant("no_direction_refinement", {"direction_refinement_enabled": False}),
        AblationVariant("opportunistic_direction_search", {"direction_line_search_mode": "opportunistic"}),
        AblationVariant("no_surrogate_ranking", {"surrogate_ranking_enabled": False}),
        AblationVariant("no_surrogate_reliability", {"surrogate_reliability_enabled": False}),
        AblationVariant("no_local_refinement", {"local_refinement_enabled": False}),
        AblationVariant("no_basin_polishing", {"basin_polishing_enabled": False}),
        AblationVariant("no_trust_regions", {"trust_regions_enabled": False}),
        AblationVariant(
            "global_density_only",
            {
                "local_refinement_enabled": False,
                "direction_refinement_enabled": False,
                "cooperative_refinement_enabled": False,
                "basin_polishing_enabled": False,
                "coherent_probes_enabled": False,
                "projection_ensemble_enabled": False,
                "trust_regions_enabled": False,
                "linkage_blocks_enabled": False,
                "cross_block_lbfgs_enabled": False,
                "cooperative_frontier_enabled": False,
            },
        ),
        AblationVariant(
            "polishing_only",
            {
                "global_candidate_fraction": 0.0,
                "density_candidate_fraction": 0.0,
                "coherent_probes_enabled": True,
                "trust_regions_enabled": True,
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


def parse_int_list(raw: str | None) -> tuple[int, ...] | None:
    if raw is None or raw.strip() == "":
        return None
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    return values or None


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


def _source_metric_from_stats(stats: Mapping[str, object], metric: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for source, payload in stats.items():
        if isinstance(payload, Mapping):
            values[str(source)] = float(payload.get(metric, 0.0))
    return dict(sorted(values.items()))


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
        "source_improvement_rate": _source_metric_from_stats(
            source_stats if isinstance(source_stats, Mapping) else {},
            "improvement_rate",
        ),
        "removed_source_hit": bool(
            best_sample.source == "shade"
            or best_sample.source == "restart:scout"
            or ":cma" in best_sample.source
        ),
    }


def run_matrix(
    cases: Sequence[AblationCase],
    variants: Sequence[AblationVariant],
    *,
    progress: bool = False,
    progress_stream=sys.stderr,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    total = max(len(cases) * len(variants), 1)
    index = 0
    for case in cases:
        for variant in variants:
            index += 1
            if progress:
                print(
                    f"[{index}/{total}] {case.name} :: {variant.name}",
                    file=progress_stream,
                    flush=True,
                )
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


def _top_float_metric(record: Mapping[str, object], field: str, limit: int = 3) -> str:
    values = record.get(field, {})
    if not isinstance(values, Mapping):
        return ""
    ranked = sorted(
        ((str(source), float(value)) for source, value in values.items() if float(value) > 0.0),
        key=lambda item: (-item[1], item[0]),
    )
    return ", ".join(f"{source}:{value:.3f}" for source, value in ranked[:limit])


def format_markdown(records: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Point-Cloud Ablation Report",
        "",
        "Deterministic no-GIF ablation output. Lower best values are better.",
        "",
        "## Primary Rosenbrock Cases",
        "",
        "| case | variant | best value | delta vs default | ratio | evals | best source | active regions | coop active | improvement rates | improvements |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- | --- | --- |",
    ]
    for record in records:
        if record.get("group") != "primary":
            continue
        lines.append(
            "| {case} | {variant} | {best} | {delta} | {ratio} | {evals} | {source} | {regions} | {coop} | {rates} | {improvements} |".format(
                case=record["case"],
                variant=record["variant"],
                best=_format_float(record["best_value"]),
                delta=_format_float(record.get("delta_vs_default")),
                ratio="" if record.get("ratio_vs_default") is None else f"{float(record['ratio_vs_default']):.3f}",
                evals=int(record["evaluations"]),
                source=record["best_source"],
                regions=int(record["active_regions"]),
                coop="yes" if bool(record["cooperative_refinement_active"]) else "no",
                rates=_top_float_metric(record, "source_improvement_rate"),
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
                "| group | variant | cases | median ratio vs default | removed source hits |",
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
                source_hits = sum(1 for record in subset if bool(record.get("removed_source_hit")))
                lines.append(
                    f"| {group} | {variant} | {len(subset)} | {median_ratio} | {source_hits} |"
                )

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `delta vs default` is positive when the ablated run is worse than the default for the same case and seed.",
            "- Symmetric origin-anchor cases are guardrails only; exact zero there confirms anchor behavior, not search quality.",
            "- `removed source hits` must stay zero after the SHADE/CMA/restart cleanup.",
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
        "removed_source_hit",
        "source_counts",
        "source_improvements",
        "source_improvement_rate",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for record in records:
        row = {key: record.get(key) for key in fieldnames}
        row["source_counts"] = json.dumps(record.get("source_counts", {}), sort_keys=True)
        row["source_improvements"] = json.dumps(record.get("source_improvements", {}), sort_keys=True)
        row["source_improvement_rate"] = json.dumps(record.get("source_improvement_rate", {}), sort_keys=True)
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
        "--dimensions",
        default=None,
        help="Comma-separated dimensions to include, for example '100,500'.",
    )
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated seeds to include; primary defaults to 3,7,11.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Override budget for selected cases.",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print case/variant progress to stderr during long matrices.",
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
    records = run_matrix(
        all_cases(
            args.case_set,
            dimensions=parse_int_list(args.dimensions),
            seeds=parse_int_list(args.seeds),
            budget=args.budget,
        ),
        select_variants(args.variants),
        progress=bool(args.progress),
    )
    payload = format_records(records, args.format)
    if args.output is None:
        sys.stdout.write(payload)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
