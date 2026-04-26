# Reading Results

This page explains how to read the outputs of a run or sweep.

Results show what happened during training, what solution was reached, and whether the run or sweep is suitable for comparison.

## Run results

A single run result shows the outcome of one configuration.

Use it to inspect:

- final metrics
- training curves
- configuration
- warnings
- saved artifacts
- run metadata

A run result is useful for checking whether training behaved normally and whether the recorded outputs match the experiment you intended to run.

## Sweep results

A sweep result shows how outputs changed across a group of related runs.

Use **Results → Sweeps** to inspect how metrics vary across the fields selected in the sweep. This is the main place to read optimizer effects, learning-rate effects, initialization effects, parameterization effects, regularization effects, or seed effects.

A sweep result should make the varied fields visible. If you cannot tell what changed across the runs, the sweep is hard to interpret.

## Metrics

Metrics are recorded values used to describe training and final outcomes.

Some metrics describe performance, such as loss, error, RMSE, or objective value. Other metrics describe the solution itself, such as norm, distance, effective rank, stable rank, reconstruction error, or structure-specific quantities.

For implicit-regularization experiments, performance metrics are often not enough. Two runs can reach similar performance while ending at different solutions.

When reading metrics, check both:

- how well the model fit the task
- what kind of solution training produced

## Training curves

Training curves show how values changed during optimization.

Use them to see whether a run converged, plateaued, diverged, oscillated, or stopped before the behavior became clear.

Curves are especially useful when final values hide important differences. Two runs can end with similar loss while following different training paths.

## Comparing conditions

When comparing selected runs, focus first on the intended difference.

Ask:

- Which field changed?
- Which fields stayed fixed?
- Are the same metrics available for each run?
- Did any runs fail, diverge, or stop early?
- Are differences consistent across seeds?

The result should be read in the context of the experiment design. A metric difference is more meaningful when the compared runs differ only in the intended way.

## Seeds and variability

Seeded runs should be read both individually and as a group.

Individual seeds can reveal instability, outliers, or failure cases. Aggregates can show whether the overall pattern is stable.

A result that appears for one seed but disappears across several seeds should be treated differently from a result that repeats across conditions.

## Warnings and failed runs

Warnings are part of the result.

A warning may indicate that a run diverged, stopped early, produced invalid values, missed an artifact, or cannot be compared cleanly with other runs.

Do not treat failed or warned runs as normal rows unless the warning has been checked.

For sweeps, failed runs can also reveal useful information. A condition that frequently fails may be an important part of the result.

## Exports

Export results after deciding what needs to be saved or reported.

Exports may include plots, tables, summaries, configurations, metadata, or run artifacts.

Use exports to preserve the evidence behind a result, not just the final number.

## Next steps

Go to **Fields & Metrics Reference** for definitions of recorded values.

Go to **Reproducibility & Provenance** for details about saved configs, seeds, metadata, and artifacts.
