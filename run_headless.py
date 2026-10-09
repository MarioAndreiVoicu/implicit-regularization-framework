"""Command-line launcher for headless experiments.

The implementation lives in framework.headless. This wrapper exists so users can
run experiments with `python run_headless.py ...` instead of importing the module.
"""

from framework.headless import main


if __name__ == "__main__":
    raise SystemExit(main())