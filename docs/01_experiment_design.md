# Experiment Design

This page explains how to set up experiments so that the results are comparable.

The lab is built around a simple structure:

> choose a question, choose what to vary, keep everything else fixed, then compare the resulting runs.

This matters because implicit regularization is usually not visible from a single run. It appears when two or more training setups solve the same task differently.

## Start with a question

Before creating a sweep, decide what comparison you want to make.

Good experiment questions are specific:

- How does optimizer choice affect the final solution?
- Does initialization scale change the solution found by training?
- Does explicit regularization reduce the difference between parameterizations?
- Does the effect of learning rate depend on the training budget?

A vague question like “which setup works best?” can still be useful for exploration, but it is harder to interpret as an implicit-regularization experiment.

## Choose the factor to vary

The factor you vary is the main thing the experiment is testing.

Common factors include:

- optimizer
- learning rate
- initialization scale
- parameterization
- batch size
- training budget
- explicit regularization strength
- dataset size or noise level

For a first experiment, vary one factor at a time. This makes the comparison easier to read.

For example, an optimizer comparison changes the optimizer and keeps everything else fixed.

## Keep the comparison context fixed

Everything outside the tested factor should stay fixed unless it is intentionally part of the experiment.

If other settings change accidentally, the comparison may still run, but the result becomes harder to interpret.

This is the main thing to check before reading a result as evidence.

## Use sweeps carefully

A sweep creates multiple related runs from one base configuration.

Use sweeps when you want to test how results change across selected fields. The base configuration should represent the shared setup, and the sweep fields should represent the intended variation.

A small focused sweep is usually easier to interpret than a sweep where many unrelated settings change at once.

For example:

**Good:**

Same task, same model, same training budget, three optimizers, five seeds each.

**Harder to interpret:**

Different optimizers, different learning rates, different training budgets, different initialization scales, all changed at once.

## Seeds

Seeds control random choices such as initialization, sampling, or noise.

A single seed can show one example of behavior, but it may not represent the full pattern. Multiple seeds help distinguish a stable effect from a lucky or unlucky run.

For comparisons, seed policy should be deliberate.

Common choices:

- use the same list of seeds for each condition
- use multiple seeds per condition
- report both individual runs and aggregate summaries
- avoid comparing one seeded run against another as if it proves a general effect

Seeded results are still experimental results. They are not automatically general conclusions.

## Training budget

Training budget includes the number of steps or optimization updates allowed for a run.

Budget matters because two configurations may look different early in training but similar later, or similar early and different later.

When comparing runs, make sure the budget is either held fixed or intentionally varied.

If budget is the tested factor, make that explicit.

## Metrics

Choose metrics based on the question.

Some metrics describe performance, such as loss, error, RMSE, or objective value. Others describe the solution itself, such as norm, distance, effective rank, stable rank, reconstruction error, or structure-specific measurements.

For implicit regularization experiments, final performance is often not enough. Two runs may achieve similar loss while finding different kinds of solutions.

A useful comparison usually looks at both:

- what performance the model achieved
- what kind of solution the training process selected

## Before trusting a comparison

Before interpreting results, check that the compared runs really belong together.

Ask:

- Are they solving the same problem?
- Was the intended factor actually the main difference?
- Were seeds handled consistently?
- Was the training budget comparable?
- Are the same metrics available for all runs?
- Did any run fail, diverge, stop early, or produce warnings?

A comparison does not need to be perfect to be useful, but the limits should be visible before drawing conclusions.

## Next steps

After designing the comparison, go to **Workflow** to run it in the app.

After runs are complete, go to **Reading Results** to interpret plots, metrics, and comparison tables.
