# Workflow

This page explains the usual path through the app: create a run, inspect the result, create a sweep, then inspect the sweep in Results.

The exact workflow depends on the experiment, but most work follows the same sequence.

## 1. Check data sources when needed

If the experiment uses MovieLens 100K or a custom triplets CSV file, start in **Data Sources**.

Use this page to inspect available data sources, validate a custom CSV file, preview its rows and statistics, and register it for later matrix-completion runs. A custom CSV should represent observed triplets such as `row,col,value` or equivalent aliases like `user,item,rating`.

Synthetic PSD and synthetic matrix-completion problems do not need to be registered here; they are configured directly from the experiment controls.

## 2. Start with a single run

Use a single run to check that the configuration behaves as expected.

Choose the problem, model, optimizer, training settings, seed, and metric/logging options. The goal is to make sure the setup is valid before launching a larger sweep.

After the run finishes, inspect the result. Check the training curve, final metrics, configuration, warnings, and saved artifacts.

A single run is useful for debugging and orientation. It is usually not enough to support a comparison claim by itself.

## 3. Review the run configuration

Every run should be traceable back to the configuration that produced it.

Before using a run as part of a comparison, check that the important settings match the experiment you intended to run.

Look especially at:

- problem family
- data source
- parameterization
- optimizer
- learning rate
- initialization
- regularization
- training budget / stopping
- seed policy
- metric/logging settings

If a setting differs unexpectedly, treat that difference as part of the result until the run is corrected or excluded.

## 4. Create a sweep

Once the single-run setup looks correct, create a sweep from the same base configuration.

A sweep varies selected fields across multiple runs. Use it when you want to test how results change across optimizers, learning rates, seeds, initialization scales, parameterizations, training budgets, regularization settings, or other controlled choices.

Start small. A narrow sweep is easier to interpret than a broad one.

A 1D sweep varies one field and is read as a curve. A full-grid 2D sweep varies two fields and is read as a heatmap, where each cell represents one pair of selected axis values. The app intentionally keeps 2D support simple: it is for full-grid interaction checks, not arbitrary coupled-pair trajectories.

The sweep should make the varied fields clear. Everything else should remain fixed unless it is intentionally part of the sweep.

## 5. Inspect the sweep in Results

After the sweep finishes, inspect it in Results.

Use the sweep view to see how metrics changed across the varied fields. This is where patterns become visible across multiple runs.

For 1D sweeps, read the selected aggregate metric as a curve across the varied field. For full-grid 2D sweeps, read the selected aggregate metric as a heatmap. Check the seed-balance and run-status summaries before interpreting either plot.

Depending on the sweep, you may look for:

- consistent differences between conditions
- sensitivity to seed
- unstable or divergent runs
- similar performance with different solution structure
- differences that appear only under certain settings

The goal is to understand how the sweep behaved, not just to find the best row.

## 6. Use Compare for individual runs

Use Compare when you want to inspect selected runs directly.

Compare is for run-level comparison. It is useful when you want to place two or more specific runs side by side and inspect their configurations, metrics, and artifacts.

When exactly two runs are selected, Compare can audit whether the comparison is controlled. Use **Auto-detect** for the expected change in most cases. Common axes such as optimizer, learning rate, parameterization, regularization, initialization, training budget, and random seed are shown first; more technical axes are available under advanced controls.

For sweep-level inspection, use Results first. For selected run-level details, use Compare.

## 7. Export artifacts

After inspecting the run or sweep, export the artifacts you need from Results or Compare.

Exports may include plots, summaries, configuration files, run metadata, or other saved outputs.

Export after deciding which results are relevant. That way the exported material matches the experiment you actually want to report or archive.

## Common workflow patterns

### Single-run check

Use this when setting up a new problem or model.

Run once, inspect the output, adjust the configuration if needed.

### One-factor sweep

Use this when testing one main factor.

Keep everything fixed except the field being tested. The Results page shows this as a curve over the varied field.

### Full-grid 2D sweep

Use this when testing whether one factor changes the effect of another factor.

Keep everything fixed except the two selected sweep fields. The Results page shows this as a heatmap. Interpret it as an interaction view, not as two independent one-factor sweeps.

### Seeded sweep

Use this when checking whether a pattern is stable.

Repeat each condition across multiple seeds.

### Run-level comparison

Use this when inspecting selected runs in detail.

Choose specific runs and compare their configurations, metrics, and artifacts side by side.

## Next steps

Go to **Reading Results** to understand the plots, metrics, and sweep outputs.

Go to **Fields & Metrics Reference** for definitions of configuration fields and recorded values.
