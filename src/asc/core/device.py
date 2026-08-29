# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Backend-neutral CPU device identity checks."""

from __future__ import annotations

import sys


def _has_exported_native_type(device: object) -> bool:
    """Authenticate a device class through its already-loaded owner module."""
    native_type = type(device)
    owner = sys.modules.get(native_type.__module__)
    if owner is None or "<locals>" in native_type.__qualname__:
        return False
    candidate: object = owner
    try:
        for component in native_type.__qualname__.split("."):
            candidate = getattr(candidate, component)
    except AttributeError:
        return False
    return candidate is native_type


def is_cpu_device(device: object | None) -> bool:
    """Return whether a device is an exact recognized CPU identity."""
    if device is None:
        return True
    if type(device) is str:
        return device == "cpu"
    if not _has_exported_native_type(device):
        return False
    module = type(device).__module__
    platform = getattr(device, "platform", None)
    if module.startswith("jaxlib.") and platform is not None:
        return platform == "cpu"
    device_type = getattr(device, "type", None)
    if module == "torch" and device_type is not None:
        return device_type == "cpu" and getattr(device, "index", None) is None
    return module.startswith("array_api_strict.") and (
        getattr(device, "_device", None) == "CPU_DEVICE"
    )


__all__ = ["is_cpu_device"]
