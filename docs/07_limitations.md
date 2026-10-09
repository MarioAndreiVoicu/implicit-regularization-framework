# Limitations

This page explains the main limits to keep in mind when using the lab.

The app can make experiments easier to run, inspect, compare, and reproduce. The interpretation still depends on the design of the experiment and the evidence in the results.

## A run is one outcome

A single run shows what happened for one configuration and one random setup.

It can be useful for debugging, inspection, or demonstration. It is usually weaker evidence than a comparison across conditions or seeds.

## Comparisons depend on what changed

A comparison is only as clear as the differences between the runs.

If the intended factor changed and everything else stayed fixed, the result is easier to interpret.

If several settings changed at once, the result may still be useful, but it is harder to attribute the difference to one cause.

## Seeds do not remove variability

Multiple seeds help show whether a pattern is stable, but they do not remove randomness from the experiment.

A seed sweep can reveal sensitivity to initialization, sampling, or noise. It should not be read as a proof that the same pattern will appear in every related setting.

## Metrics are partial descriptions

Metrics describe selected aspects of the run.

A metric can show loss, reconstruction error, norm, effective rank, stable rank, alignment, RMSE, or another measured quantity. It does not describe every property of the learned solution.

When a claim depends on solution structure, use metrics that actually measure that structure.

## Plots are selected views

The app shows a set of plots that are useful for common runs and sweeps, but it does not try to include every possible visualization.

For more specialized analysis, use the saved metrics and exported artifacts outside the app. The recorded metric histories and summary values are intended to support additional plotting, filtering, aggregation, and statistical analysis when the built-in views are not enough.

## Synthetic and real data answer different questions

Synthetic data is useful because the target structure is known and controlled.

Real data is useful because it reflects an observed dataset rather than a generated one.

The two settings support different kinds of conclusions. A pattern found on synthetic data may not carry over to real data, and a pattern found on real data may be harder to explain mechanistically.

## Sweep coverage is finite

A sweep only covers the values that were selected.

If a sweep tests three learning rates, the result describes those three learning rates. It does not automatically describe the whole learning-rate range.

For 2D sweeps, the heatmap only describes the selected full grid of axis values. It should not be read as a continuous surface unless the grid is dense enough and the experiment justifies that interpretation.

Use the sweep definition to check exactly what was covered.

## Hardware and numeric behavior can matter

Device, dtype, and implementation details can affect numerical behavior.

Small differences may appear across CPU, GPU, precision settings, or library versions. These differences matter most when runs are unstable, ill-conditioned, or close to a threshold.

Use recorded metadata when exact reproducibility is important.

## 2D heatmaps are simple summaries

The app intentionally keeps 2D sweep support simple. A 2D sweep is shown as a full-grid heatmap of one selected aggregate metric. This is useful for inspecting interactions between two settings, but it does not replace deeper statistical analysis.

Heatmaps should be interpreted together with seed counts, run counts, failures, and warnings. Missing or failed cells can change the meaning of the visual pattern.

## Failed runs are still information

A failed or divergent run should not be silently treated as a normal result.

Failure can indicate that a configuration is unstable, outside the useful range, or sensitive to numeric details.

When failures occur in a sweep, inspect which conditions failed before summarizing the sweep.

## Claims should match the experiment

The conclusion should stay close to what was actually tested.

A sweep over optimizers on one problem supports a statement about those optimizers on that problem. A seed sweep supports a statement about variability under the tested seed policy. A comparison of parameterizations supports a statement about the selected parameterizations under the selected setup.

Use the saved configuration, sweep definition, metrics, and artifacts to keep the claim tied to the experiment.

The lab helps organize evidence; it does not replace experimental judgment.
