# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Run the complete suite in an isolated minimum-dependency environment."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile

_FLOORS = (
    "array-api-compat==1.13.0",
    "array-api-strict==2.3.0",
    "h5py==3.11.0",
    "jax==0.6.0",
    "jaxlib==0.6.0",
    "myst-parser==5.1.0",
    "numpy==1.26.4",
    "pydata-sphinx-theme==0.19.0",
    "scipy==1.13.0",
    "sphinx==9.1.0",
    "torch==2.13.0",
)
_TEST_TOOLS = (
    "hypothesis==6.165.2",
    "pytest==9.1.1",
    "pytest-benchmark==5.2.3",
    "pytest-cov==7.1.0",
)


def _run(command: list[str], *, cwd: pathlib.Path | None = None) -> None:
    subprocess.run(command, check=True, cwd=cwd)


def _python(environment: pathlib.Path) -> pathlib.Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def main() -> int:
    """Install exact floors, then run all tests with branch coverage."""
    root = pathlib.Path(__file__).resolve().parents[1]
    uv = os.environ.get("ASC_PY_UV", "uv")
    with tempfile.TemporaryDirectory(prefix="asc-py-floor-") as directory:
        environment = pathlib.Path(directory)
        _run([uv, "venv", "--python", sys.executable, str(environment)])
        python = _python(environment)
        _run(
            [
                uv,
                "pip",
                "install",
                "--python",
                str(python),
                "--torch-backend",
                "cpu",
                "--editable",
                f"{root}[all,docs]",
                *_FLOORS,
                *_TEST_TOOLS,
            ]
        )
        child_environment = dict(os.environ)
        child_environment["JAX_PLATFORMS"] = "cpu"
        child_environment["JAX_ENABLE_X64"] = "0"
        subprocess.run(
            [str(python), "-m", "pytest"],
            check=True,
            cwd=root,
            env=child_environment,
        )
    print("dependency floor: complete suite passed at every direct minimum")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
