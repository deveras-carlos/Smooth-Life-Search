# Point-Cloud Ablation Report

Deterministic no-GIF ablation output. Lower best values are better.

## Lean Cleanup Checkpoint

The point-cloud optimizer now uses a smaller set of default engines:

- archive-centered evaluation and de-duplication,
- SmoothLife density/projection proposals,
- adaptive trust regions and coherent probes,
- projection-ensemble SmoothLife density views,
- surrogate-ranked candidate pools,
- surrogate reliability gating,
- local, block, and bracketed direction-set polishing,
- cooperative active-set frontier propagation for large dimensions.

The removed experimental engines are SHADE-style proposals, CMA-region
proposals, restart/scout proposals, live populations, and source-efficiency
governor tuning. The ablation runner now measures only the lean systems that
remain.

## Seed-7 Rosenbrock Smoke

Current no-GIF smoke values at budget `6400`:

| dimension | best value | lean target | stop reason |
| ---: | ---: | ---: | --- |
| 30 | 6.772577e-02 | < 1.0 | budget_exhausted |
| 50 | 5.708452e-01 | < 1.0 | budget_exhausted |
| 100 | 5.545930e-02 | < 5.0 | budget_exhausted |
| 500 | 2.181981e+01 | < 75.0 | budget_exhausted |

These are intentionally looser than the pre-cleanup stretch metrics. The lean
pass favors understandable behavior and correctness guardrails over preserving
every previous benchmark number.

## Lean Variants

The supported ablation variants are:

- `default`
- `no_projection_ensemble`
- `no_cooperative_refinement`
- `no_cooperative_frontier`
- `no_coherent_probes`
- `no_direction_refinement`
- `opportunistic_direction_search`
- `no_surrogate_ranking`
- `no_surrogate_reliability`
- `no_local_refinement`
- `no_basin_polishing`
- `no_trust_regions`
- `global_density_only`
- `polishing_only`

Run a primary matrix with:

```bash
python tools/point_cloud_ablation.py --case-set primary --dimensions 30,50,100,500 --seeds 7 --budget 6400 --variants all --format markdown
```

## Interpretation

The cleanup keeps the optimizer honest: there are no exact diagonal scout
shortcuts and no hidden known-optimum injections. Symmetric origin-anchor cases
can still reach exact zero because the center/origin anchor is deliberately
evaluated and is the real optimum for those objectives.

The main tradeoff is that high-dimensional polishing is less aggressive than
the previous many-engine version. The next useful measurement is a lean ablation
matrix that identifies whether projection ensembles, cooperative frontier
propagation, coherent probes, bracketed direction refinement, or surrogate
reliability should receive more deterministic budget in 100D and 500D runs.
