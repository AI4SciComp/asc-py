# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0

import importlib.metadata
import importlib.util

import asc


def test_version_matches_distribution_metadata() -> None:
    assert asc.__version__ == importlib.metadata.version("asc-py")
    assert importlib.util.find_spec("asc_py") is None


def test_public_exports_are_intentional() -> None:
    assert (
        tuple(sorted(name for name in asc.__all__ if name != "PUBLIC_EXPORTS"))
        == asc.PUBLIC_EXPORTS
    )
    assert all(hasattr(asc, name) for name in asc.PUBLIC_EXPORTS)
