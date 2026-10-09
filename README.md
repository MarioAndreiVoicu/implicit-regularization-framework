# Implicit Regularization Framework

**Experimental code accompanying the 2026 Bachelor's thesis _Implicit Bias in Overparameterized Learning_.**

This repository is a local research prototype for exploring **solution selection in overparameterized learning**. It combines configurable numerical experiments in matrix recovery with tools for comparing optimization regimes. It is not a production ML library.

## Research question

When a learning problem admits many solutions with similarly low or zero training loss, the objective alone does not explain **which solution training selects**. Parameterization, initialization, and optimization dynamics may favor solutions with different structural properties, even without an explicit regularization term.

The thesis studies this question primarily through **matrix factorization**, a setting in which the relationship between optimization dynamics and implicit regularization can be analyzed mathematically. In particular, it examines a restricted commuting-measurement result concerning nuclear-norm minimization under factorized gradient flow, and the conditions and limitations of that result.

**The purpose of the code is experimental:** to vary relevant factors, inspect the resulting training trajectories and solution properties, and compare outcomes under controlled configurations. The mathematical analysis is developed in the accompanying Bachelor's thesis, while this repository provides the experimental framework used to investigate the problem computationally.

## What the framework supports

- **Problem settings:** positive-semidefinite (PSD) matrix sensing and matrix completion, including synthetic cases and supported custom matrix-completion datasets.
- **Experimental controls:** parameterization, initialization, optimizer, learning rate and schedule, seed, training budget, and explicit regularization settings.
- **Experiment design:** individual runs, one-dimensional sweeps, and two-dimensional full-grid sweeps.
- **Analysis:** training histories, reconstruction metrics, selected structural metrics, saved configurations, comparisons of runs, and exports for further analysis.
- **Execution:** an interactive Streamlit interface and headless execution from JSON requests.

A typical experiment compares configurations that fit the observed measurements, then examines whether their recovered solutions differ in properties such as nuclear norm, rank-related metrics, or reconstruction error. The interpretation depends on the specific problem, parameterization, and numerical regime.

## Quick start

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

A small command-line demo is also available:

```bash
python demo_run.py
```

For headless requests, use the included request format and documentation:

```bash
python run_headless.py new-run path/to/request.json
python run_headless.py sweep path/to/sweep_request.json
```

Run the test suite with:

```bash
python -m pytest -q
```

Experiments create local files in `runs/`, `sweeps/`, and `datasets_registry/`. These are excluded from version control. A small `custom_dataset_example.csv` is provided as sample input.

## Using the app

- **Experiments:** configure individual runs and parameter sweeps.
- **Results:** inspect runs, sweeps, charts, metrics, and saved artifacts.
- **Compare:** compare selected individual runs under specified configurations.
- **Data Sources:** inspect supported external or custom matrix-completion data sources.
- **Validation:** run selected checks on experiment paths.
- **Docs / Theory:** read experimental design guidance, core theory, metric definitions, and limitations.

## Scope and limitations

The framework focuses on controlled experiments in PSD matrix sensing and matrix completion. Results depend on the chosen data, parameterization, initialization, optimizer, numerical precision, and stopping conditions.

Experiments complement the mathematical analysis in the thesis; observations in specific numerical settings should not be interpreted as general theoretical guarantees. The framework records experiment configurations and provenance to support reproducibility, although results may vary across computational environments.

For further context, see [`docs/02_core_theory.md`](docs/02_core_theory.md), [`docs/06_reproducibility_provenance.md`](docs/06_reproducibility_provenance.md), and [`docs/07_limitations.md`](docs/07_limitations.md).

## Academic context

This project was developed as the practical component of the Bachelor's thesis **_Implicit Bias in Overparameterized Learning_** (B.Sc. Computer Science, Transilvania University of Brașov, 2026). The thesis investigates implicit regularization through mathematical analysis of matrix factorization and an accompanying configurable experimental framework.
