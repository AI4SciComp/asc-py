# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Build and test documentation in a NumPy-only installed environment."""

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _run(*arguments: str) -> None:
    """Run one documentation gate with the active isolated interpreter."""
    subprocess.run(
        [sys.executable, *arguments],
        check=True,
        cwd=_ROOT,
    )


def main() -> None:
    """Prove optional backends are absent and execute every base docs gate."""
    for package in ("torch", "jax"):
        if importlib.util.find_spec(package) is not None:
            raise RuntimeError(
                "base documentation environment unexpectedly contains "
                f"{package}"
            )
    _run(
        "-m",
        "sphinx",
        "-W",
        "--keep-going",
        "-n",
        "-b",
        "html",
        "docs",
        "docs/_build/html",
    )
    _run(
        "-m",
        "sphinx",
        "-W",
        "--keep-going",
        "-n",
        "-b",
        "doctest",
        "docs",
        "docs/_build/doctest",
    )
    _run(
        "-m",
        "pytest",
        "tests/docs",
        "-k",
        "not torch_example and not jax_example",
        "--no-cov",
    )


if __name__ == "__main__":
    main()
