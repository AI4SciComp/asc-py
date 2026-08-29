# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Runtime version derived from installed distribution metadata."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("asc-py")
except importlib.metadata.PackageNotFoundError:
    # Source-tree fallback used before an editable/build installation exists.
    __version__ = "0.1.0"

__all__ = ["__version__"]
