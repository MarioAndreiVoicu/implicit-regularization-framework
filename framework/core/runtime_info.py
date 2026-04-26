from __future__ import annotations

import platform
import sys
from importlib import import_module
from importlib import metadata as importlib_metadata
from typing import Any


RUNTIME_PROVENANCE_SCHEMA_VERSION = 1


def _package_version(package_name: str) -> str | None:
    try:
        return str(importlib_metadata.version(package_name))
    except Exception:
        return None


def _torch_version() -> str | None:
    try:
        torch = import_module("torch")
    except Exception:
        return None
    version = getattr(torch, "__version__", None)
    return str(version) if version is not None else None


def build_runtime_snapshot() -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_PROVENANCE_SCHEMA_VERSION,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "platform_version": platform.version(),
        "executable": sys.executable,
        "numpy_version": _package_version("numpy"),
        "pandas_version": _package_version("pandas"),
        "torch_version": _torch_version(),
    }


def missing_runtime_snapshot() -> dict[str, Any]:
    return {
        "schema_version": RUNTIME_PROVENANCE_SCHEMA_VERSION,
        "capture_status": "missing_legacy",
        "python_version": None,
        "python_implementation": None,
        "platform_system": None,
        "platform_release": None,
        "platform_machine": None,
        "platform_version": None,
        "executable": None,
        "numpy_version": None,
        "pandas_version": None,
        "torch_version": None,
    }
