# Core Theory

This page explains the main ideas behind the experiments in the lab.

The lab focuses on implicit regularization: the way training choices influence which solution is found, even when those choices are not written as explicit penalty terms in the objective.

In many problems, there is more than one solution that fits the data well. Training does not choose among those solutions randomly. The optimizer, parameterization, initialization, learning rate, training budget, data regime, and explicit regularization settings can all affect the solution that training reaches.

## Objective vs. training process

The objective defines what training is trying to minimize.

The training process defines how the model moves through parameter space while minimizing it.

Two runs can have the same objective but different training processes. They may reach similar loss values while ending at different solutions.

This is the central reason implicit regularization is worth studying.

## Explicit regularization

Explicit regularization is regularization that appears directly in the training objective or update rule.

Examples include:

- weight decay
- L1 or L2 penalties
- norm constraints

Explicit regularization is visible in the configuration. It is something the user intentionally adds.

Early stopping can act like a regularizer in some contexts, but in this lab it is mainly treated as a stopping choice.

## Implicit regularization

Implicit regularization is not added as a separate penalty term.

Instead, it comes from the behavior of the training setup itself.

Examples include effects from:

- optimizer choice
- learning rate
- initialization
- parameterization
- batch size or sampling noise
- training budget
- data geometry

These choices can bias training toward some solutions over others, even when the objective function is unchanged.

## Parameterization

Parameterization is the way the model represents the solution being optimized.

Two parameterizations can represent similar or identical solutions but make optimization behave differently. A change in parameterization can change the path taken during training and the solution selected at the end.

This is why parameterization is treated as an experimental factor, not just an implementation detail.

## Optimization bias

Optimization bias refers to the tendency of a training procedure to prefer certain solutions.

This does not necessarily mean the optimizer is doing anything wrong. It means that when many solutions are possible, the path taken by optimization can make some outcomes more likely than others.

Different optimizers, learning rates, and initialization scales can create different optimization biases.

## Data regime

The data regime describes the conditions under which the model is trained.

This can include dataset size, noise level, sampling process, feature structure, or train/test split.

Implicit-regularization effects can depend strongly on the data regime. A training choice that matters in one regime may matter less in another.

## Why comparisons matter

Implicit regularization is usually studied through comparisons.

A single run can show what happened under one configuration. It usually cannot show whether a training choice changed the solution.

To study that, compare runs where the intended factor changes and everything else stays fixed.

For example:

> Same objective, same data, same model, same training budget, different optimizer.

If the final solutions differ in a consistent way, the optimizer may be part of the explanation.

## What the lab records

The lab is designed to keep the theory connected to the actual run data.

For each run, the app records the configuration, metrics, artifacts, and metadata needed to inspect what happened.

For sweeps, the app records which fields varied across runs.

For results, the app shows the metrics and plots used to inspect the outcome.

The goal is to make the comparison traceable: when two runs differ, the docs and UI should make it clear what changed.

## Next steps

Go to **Workflow** to see how these ideas map onto running experiments in the app.

Go to **Reading Results** to see how to interpret plots, metrics, and sweep outputs.
