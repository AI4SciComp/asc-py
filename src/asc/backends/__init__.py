# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Import-safe backend capability metadata."""

from asc.backends.capabilities import (
    BackendInfo,
    Capability,
    backend_info,
    has_capability,
    require_capability,
)

__all__ = [
    "BackendInfo",
    "Capability",
    "backend_info",
    "has_capability",
    "require_capability",
]
