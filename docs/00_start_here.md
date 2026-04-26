# Start Here

This lab is for running and comparing experiments about implicit regularization.

Implicit regularization refers to the way training choices can shape the solution a model finds, even when those choices are not written as explicit penalty terms in the objective.

In this lab, those choices might include the optimizer, learning rate, initialization, parameterization, training budget, data regime, or explicit regularization settings.

The main purpose of the lab is to make these choices easy to vary, track, compare, and inspect.

## What you can do here

You can use the lab to:

- run a single experiment
- launch a sweep over selected settings
- compare selected runs
- inspect training curves and final metrics
- review the full configuration behind each result
- export plots, summaries, and run artifacts

The app is organized around experiments and comparisons. A single run shows what happened under one configuration. A sweep shows what changes across a family of related configurations. A comparison helps inspect differences between selected runs.

## Typical workflow

Most work starts with a single run.

Use the run page to choose the problem, model, optimizer, training settings, and metric/logging options. After the run finishes, inspect the outputs before expanding into a larger sweep.

Once the basic setup looks correct, create a sweep. A sweep lets you vary one or more fields across multiple runs while keeping the rest of the configuration fixed.

After the sweep completes, inspect the sweep in Results to look at how the results changed across settings. This is where optimizer effects, initialization effects, parameterization effects, or regularization effects become easier to see.

## What to read next

Go to **Experiment Design** if you are deciding what to vary and what to keep fixed.

Go to **Core Theory** if you want the conceptual background behind implicit regularization.

Go to **Workflow** if you want a step-by-step guide to running experiments in the app.

Go to **Reading Results** if you already have runs and want to understand the plots, metrics, and comparisons.

Go to **Fields & Metrics Reference** if you need definitions for configuration fields or output values.

Go to **Reproducibility & Provenance** if you need to understand saved configs, seeds, metadata, and exported artifacts.
