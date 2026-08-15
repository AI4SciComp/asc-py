# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Execute every documented example under warning-as-error policy."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys

import pytest

_EXAMPLES = tuple(
    pytest.param(path, marks=pytest.mark.backend("jax"))
    if path.stem == "random_autodiff"
    else pytest.param(path)
    for path in sorted(
        (pathlib.Path(__file__).resolve().parents[2] / "examples").glob("*.py")
    )
)


@pytest.mark.parametrize("path", _EXAMPLES, ids=lambda path: path.stem)
def test_example(path: pathlib.Path) -> None:
    environment = dict(os.environ)
    environment["PYTHONWARNINGS"] = "error"
    environment["JAX_PLATFORMS"] = "cpu"
    subprocess.run(
        [sys.executable, str(path)],
        check=True,
        cwd=path.parents[1],
        env=environment,
    )
