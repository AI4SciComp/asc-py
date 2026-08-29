# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Import-safe environment and capability diagnostics."""

from __future__ import annotations

import importlib.metadata
import platform
import sys

from asc import _version as version_module
from asc.backends.capabilities import backend_info


def _distribution_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _backend_document(name: str) -> dict[str, object]:
    """Convert immutable backend metadata to JSON-safe primitives."""
    info = backend_info(name)
    return {
        "name": info.name,
        "installed": info.installed,
        "version": info.version,
        "array_api_version": info.array_api_version,
        "devices": info.devices,
        "dtypes": info.dtypes,
        "dtype_families": info.dtype_families,
        "capabilities": tuple(
            sorted(capability.value for capability in info.capabilities)
        ),
    }


def diagnostics() -> dict[str, object]:
    """Return metadata without importing or initializing optional backends."""
    return {
        "asc_version": version_module.__version__,
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "array_api_version": "2024.12",
        "backends": {
            name: _backend_document(name) for name in ("numpy", "torch", "jax")
        },
        "optional_dependencies": {
            "h5py": _distribution_version("h5py"),
            "scipy": _distribution_version("scipy"),
            "array_api_strict": _distribution_version("array-api-strict"),
        },
    }


__all__ = ["diagnostics"]
