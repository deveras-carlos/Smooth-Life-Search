# Matrix SmoothLife Algorithm

This branch replaces public point-cloud optimization with `MatrixSmoothLifeSearch`.

## Core Idea

The optimizer keeps one 2D SmoothLife matrix as a genome. Every objective evaluation decodes the entire matrix into one N-D candidate point, evaluates that point, and stores the result in an archive. SmoothLife dynamics then continues changing matrix-cell values under reward/support feedback from the global objective evaluations.

The matrix is not a dense N-D objective grid. It is a living 2D state whose cells collectively encode one candidate solution.

## State

- `field`: signed SmoothLife genome matrix in `[-1, 1]`.
- `reward_field`: objective/support feedback on matrix cells.
- `advantage_field`: signed cell credit from matrix changes that helped or hurt.
- `direction_field`: decoded successful directions mapped back into matrix space.
- `local_credit_field`: objective-derived pressure assigned back to local neighborhoods.
- `neighborhood_credit`: persistent signed credit for each Moore-neighborhood subproblem.
- `temperature_field`: per-cell exploration temperature for local reheating.
- `best_field`: elite matrix snapshot.
- `archive`: evaluated decoded points, field hashes, values, and source labels.
- `best_point` and `best_value`: best decoded candidate seen so far.
- `snapshots`: typed `MatrixSmoothLifeSnapshot` records for rendering and diagnostics.

## Matrix Shape

If no matrix shape is provided, the optimizer chooses the smallest near-square shape with at least `max(256, 4 * dimension)` cells. Explicit matrix shapes must have both axes at least `16` and must contain at least `dimension` cells.

## Decoder

The default decoder is `local_sparse`. Each cell owns a tiny local subproblem made from its eight surrounding Moore-neighborhood cells. Neighbor values whose absolute value is below `cell_alive_threshold` are treated as dead and contribute zero.

For every cell:

1. Collect the eight non-wrapping neighbor values, excluding the center cell.
2. Map that local vector through a deterministic sparse projection into a small coordinate block.
3. Assign coordinate blocks by deterministic overlapping sliding windows over the N-D vector.
4. Sum all neighborhood contributions into an N-D raw vector.
5. Normalize each coordinate by its number of contributing neighborhoods.
6. Squash with `normalized = 0.5 + 0.5 * tanh(decoder_gain * raw)`.
7. Map each coordinate into its original box bounds.

The neutral zero field still decodes exactly to the domain center. `random_projection` remains available through `--decoder random_projection` for ablation and fallback comparisons.

## Evaluation Loop

1. Evaluate the neutral zero field.
2. Evaluate a small deterministic matrix warmup made of decoder raw pulses and random whole-matrix fields.
3. Initialize SmoothLife noise around the best field.
4. Repeatedly evolve the matrix for `steps_per_evaluation` SmoothLife steps.
5. Decode and evaluate the current matrix if its field hash is new.
6. Update the archive, best field, reward field, signed credit fields, mutation level, and snapshots.
7. After improvements, run a small archive-accounted matrix-space line search along the successful matrix direction.
8. Periodically run archive-accounted local patch probes that perturb only one selected neighborhood.
9. Stop only on budget exhaustion, `max_steps`, or explicit target-value early stopping.

All real objective calls go through the archive path and count against the exact budget.

## Feedback

On improvement:

- store the current matrix as `best_field`;
- boost `reward_field` where matrix vitality and movement are high;
- add signed cell credit to `advantage_field`;
- map successful decoded-space movement back to matrix cells through the active decoder;
- assign local neighborhood credit to changed Moore-neighborhood patterns;
- cool successful structures in `temperature_field`;
- decay mutation noise slowly.

On non-improvement:

- decay `reward_field`;
- decay signed advantage, decoded-direction pressure, and local neighborhood credit;
- pull the field slightly toward `best_field`;
- damp unstable movement;
- reheat uncertain cells after sustained stagnation;
- keep evolving unless the field becomes non-finite.

Local patch probes provide sharper divide-and-conquer credit. At a fixed evaluation interval, the optimizer selects high-temperature/high-change cells, perturbs only their eight neighbors, evaluates the full decoded candidate once, and assigns the measured global delta directly to that patch.

## CLI

```bash
smooth-life-search matrix --objective ackley --dimension 5 --seed 7 --budget 20000
smooth-life-search matrix --objective rosenbrock --dimension 100 --seed 7 --budget 6400
smooth-life-search matrix --objective sphere --dimension 5 --budget 1000 --gif matrix.gif
```

Divide-and-conquer controls include `--cell-alive-threshold`,
`--local-decoder-block-size`, `--local-decoder-overlap`,
`--no-local-credit`, `--local-credit-strength`, `--no-patch-probes`,
and `--patch-probe-step`. Credit-steering controls also include
`--advantage-strength`, `--direction-strength`, `--temperature-reheat`,
and `--no-matrix-line-search`. The default line-search alpha ladder is
`0.5 1.0 1.5 2.0`.

`simulate` remains the literal 2D SmoothLife simulator. The removed `point-cloud` command is intentionally not compatible with this branch.

## V1 Expectations

This version prioritizes SmoothLife-native representation, deterministic decoding, bounded archive accounting, and honest shifted/rotated guardrails. It is not expected to match the previous point-cloud optimizer on high-dimensional Rosenbrock polishing.
