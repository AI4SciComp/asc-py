# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Global test configuration."""

import os
import typing

import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_PROFILE_BACKENDS = {
    "base": frozenset({"numpy"}),
    "torch": frozenset({"numpy", "torch"}),
    "jax": frozenset({"numpy", "jax"}),
    "all": frozenset({"numpy", "torch", "jax"}),
}


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Deselect only cases requiring an intentionally absent backend."""
    profile = os.environ.get("ASC_TEST_PROFILE", "all")
    allowed = _PROFILE_BACKENDS.get(profile)
    if allowed is None:
        raise pytest.UsageError(f"unsupported ASC_TEST_PROFILE {profile!r}")
    if profile == "all":
        return
    selected: list[pytest.Item] = []
    deselected: list[pytest.Item] = []
    for item in items:
        required = {
            typing.cast(str, argument)
            for marker in item.iter_markers("backend")
            for argument in marker.args
        }
        callspec = getattr(item, "callspec", None)
        if callspec is not None:
            required.update(
                value
                for value in callspec.params.values()
                if isinstance(value, str) and value in {"torch", "jax"}
            )
        if required <= allowed:
            selected.append(item)
        else:
            deselected.append(item)
    items[:] = selected
    if deselected:
        config.hook.pytest_deselected(items=deselected)
