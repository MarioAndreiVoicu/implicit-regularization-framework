from __future__ import annotations

import os
import sys
from pathlib import Path

# Keep the test suite fast and stable on small CPU-only machines. The experiments
# in the tests use tiny tensors, where PyTorch/BLAS thread-pool overhead can
# dominate runtime and make smoke tests appear to hang.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
# Force a non-GUI Matplotlib backend during tests. This avoids Tk/Tcl failures
# on Windows and headless machines when export/report tests generate PNGs.
os.environ.setdefault("MPLBACKEND", "Agg")

try:  # pragma: no cover - defensive for environments without torch installed yet.
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch can reject changing interop threads after parallel work started.
        pass
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
