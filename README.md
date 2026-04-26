# Implicit Regularization Framework

A Streamlit-based local framework for running experiments on implicit regularization in PSD sensing and matrix completion.

## Project Overview

This project provides a local app for creating, running, inspecting, and comparing controlled implicit-regularization experiments.

The framework is centered around two problem families:

- **PSD sensing**
- **Matrix completion**

It supports single runs, sweeps, result inspection, custom datasets, artifact export, and headless execution.

Each run stores its configuration, metrics, metadata, warnings, and generated artifacts so that results can be inspected later.

## Features

- Single-run experiment execution.
- Sweep execution over selected configuration fields.
- PSD sensing experiments.
- Matrix completion experiments.
- Synthetic matrix-completion datasets.
- Registered custom matrix-completion datasets.
- Experiments over optimizer, learning rate, initialization, parameterization, seed, and regularization settings.
- Results page for inspecting completed runs and sweeps.
- Compare page for selected individual runs.
- Saved metric histories and final summaries.
- Run metadata, warnings, stopping reasons, and provenance records.
- Export support for plots, summaries, configurations, and artifacts.
- Built-in documentation for theory, workflow, fields, metrics, reproducibility, and limitations.
- Headless execution from JSON request files.

## App Pages

### Experiments

Used to create and launch new runs or sweeps.

This page contains the main experiment configuration controls: problem family, data source, model settings, optimizer settings, training settings, seed settings, measurement options, and sweep fields.

### Results

Used to inspect completed runs and sweeps.

This page shows metrics, plots, artifacts, configurations, warnings, sweep summaries, and export options.

### Compare

Used to compare selected individual runs.

Compare is run-level only. Sweep-level inspection is done in Results.

### Datasets

Used to inspect available datasets and register custom matrix-completion datasets.

### Validation

Used to run framework checks for selected experiment paths.

### Docs / Theory

Contains the in-app documentation pages:

- Start Here
- Experiment Design
- Core Theory
- Workflow
- Reading Results
- Fields & Metrics Reference
- Reproducibility & Provenance
- Limitations

## Installation

Python 3.10 or newer is recommended.

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on macOS/Linux:

```bash
source .venv/bin/activate
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

Start the Streamlit app with:

```bash
streamlit run app.py
```

The app writes local runtime outputs to directories such as:

```text
runs/
sweeps/
datasets_registry/
```

These directories are generated locally and are ignored by git.

## Headless Usage

Run a single experiment request:

```bash
python run_headless.py new-run path/to/request.json
```

Run a sweep request:

```bash
python run_headless.py sweep path/to/sweep_request.json
```

Run the demo:

```bash
python demo_run.py
```

## Notes

This project runs experiments locally. Cloud deployment is planned but not currently the main execution mode.

The built-in plots are selected views of the saved metrics. For additional analysis, use the exported metric histories, summaries, configurations, and artifacts outside the app.

Exact reproducibility can depend on device, dtype, library versions, and hardware behavior. The framework records run metadata to make those details easier to inspect.
