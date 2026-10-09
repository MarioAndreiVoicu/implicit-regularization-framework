# Reproducibility & Provenance

This page explains what the app records so that runs, sweeps, and exported results can be traced back to the configuration that produced them.

Reproducibility in the lab means being able to answer:

> What was run, with which settings, from which starting conditions, and what outputs did it produce?

## Run configuration

Each run is created from a configuration.

The configuration records the selected problem, data source, model, parameterization, optimizer, training settings, regularization settings, metric/logging options, and runtime choices.

When reviewing a result, the configuration is the first place to check what the run actually did.

## Seeds

The seed records the random state used for the run.

Seeds can affect initialization, synthetic data generation, sampling, noise, and other randomized parts of the experiment.

When comparing conditions, use a consistent seed policy. For example, a sweep may run the same list of seeds for each optimizer or initialization scale.

The seed should be treated as part of the experiment, not as an incidental detail.

## Data-source records

The Data Sources page records and displays information for reusable observed-rating data sources.

For MovieLens 100K, the app uses fixed built-in dataset, split, and preprocessing labels. For custom matrix-completion data, a triplets CSV can be uploaded or referenced by path, validated, previewed, fingerprinted, and registered. Registered custom datasets receive a dataset identifier and version label so later runs can refer to the same source.

Synthetic PSD and synthetic matrix-completion data are not registered as reusable data-source records on this page; they are generated from the experiment configuration and seed policy.

## Sweep definition

A sweep records how multiple runs were generated from a shared setup.

The sweep definition should show:

- the base configuration
- the fields that varied
- the values used for each varied field
- whether the sweep was 1D or full-grid 2D
- the runs created from the sweep
- the status of each run

This makes it possible to inspect what changed across the sweep and which settings were shared. For 2D sweeps, the saved definition also determines the heatmap axes and their declared value order.

## Run metadata

Run metadata records information about execution.

This can include:

- run identifier
- run name
- group identifier
- seed group identifier
- creation time
- completion time
- run status
- device information
- dtype or precision choice
- stopping reason
- warnings or failure messages

Metadata helps explain how the run was executed and whether anything unusual happened.

## Artifacts

Artifacts are files produced by a run or sweep.

They may include:

- plots
- metric histories
- final metric summaries
- configuration files
- reconstructed matrices or learned factors
- comparison tables
- sweep aggregate tables
- exported bundles or downloaded outputs

Artifacts should be interpreted together with the configuration and metadata that produced them.

## Saved metrics

Metrics are saved during training and at the end of a run.

Trajectory metrics describe how values changed over time. Final metrics summarize the last recorded state or the best recorded state, depending on the metric.

For reproducibility, the important point is that metrics are connected back to the run configuration and seed.

## Warnings and stopping reasons

Warnings and stopping reasons are part of the provenance of a run.

They may indicate divergence, early stopping, invalid values, missing outputs, failed artifact generation, or a comparison issue.

When a run has warnings, keep them attached to the result. They explain how the result was produced and whether it should be included in later comparisons.

## Exports

Exports preserve selected results outside the app.

Before exporting, check that the selected runs or sweeps match the experiment you want to save.

An export is most useful when it includes enough context to reconstruct the result:

- selected runs or sweep
- plotted metrics
- summary table
- configuration
- seed information
- warnings or run status
- relevant artifacts

## Re-running experiments

To reproduce a result, start from the saved configuration and use the same seed policy.

Exact reproducibility may also depend on device, dtype, library versions, and hardware behavior. When those details matter, use the recorded metadata to check whether the rerun environment matches the original run. The clean thesis archive does not include local runtime folders such as saved runs, sweeps, or registered custom datasets; those are regenerated locally when the app is used.

The app aims to make reruns traceable, not to guarantee bit-for-bit identical results across all hardware.

## Next steps

Go to **Fields & Metrics Reference** for definitions of recorded fields and metric names.

Go to **Reading Results** for guidance on interpreting saved metrics, warnings, and artifacts.
