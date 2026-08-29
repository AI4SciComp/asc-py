# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Install release artifacts into isolated optional-dependency profiles."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import tempfile

_PROBE = r"""
import importlib.resources
import importlib.util
import pathlib
import sys
import tempfile

profile = sys.argv[1]
expected = {
    "base": (),
    "torch": ("torch",),
    "jax": ("jax",),
    "hdf5": ("h5py",),
    "mat": ("scipy",),
    "all": ("torch", "jax", "h5py", "scipy"),
}[profile]
absent = {
    "base": ("torch", "jax", "h5py", "scipy"),
    "torch": ("jax", "h5py", "scipy"),
    "jax": ("torch", "h5py"),
    "hdf5": ("torch", "jax", "scipy"),
    "mat": ("torch", "jax", "h5py"),
    "all": (),
}[profile]
for distribution in ("numpy", "array_api_compat", *expected):
    assert importlib.util.find_spec(distribution) is not None, distribution
for distribution in absent:
    assert importlib.util.find_spec(distribution) is None, distribution
assert importlib.util.find_spec("array_api_strict") is None
assert importlib.util.find_spec("asc_py") is None

import asc

assert asc.__version__ == "0.1.0"
assert not ({"torch", "jax", "h5py", "scipy"} & set(sys.modules))
assert importlib.resources.files("asc").joinpath("py.typed").is_file()

import numpy

value = numpy.asarray([1.0, 2.0], dtype=numpy.float32)
numpy.testing.assert_allclose(asc.sum_of_squares(value), 5.0)

if profile in {"torch", "all"}:
    import torch

    native = torch.asarray([1.0, 2.0], dtype=torch.float32)
    torch.testing.assert_close(asc.sum_of_squares(native), torch.tensor(5.0))
if profile in {"jax", "all"}:
    import jax.numpy as jnp

    native = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    result = numpy.asarray(asc.sum_of_squares(native))
    numpy.testing.assert_allclose(result, 5.0)
if profile in {"hdf5", "all"}:
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "data.h5"
        asc.data.save_hdf5(path, {"x": value})
        restored = asc.data.load_hdf5(path)
        numpy.testing.assert_array_equal(restored.tree["x"], value)
if profile in {"mat", "all"}:
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "data.mat"
        asc.data.save_mat(path, {"x": value})
        restored = asc.data.load_mat(path)
        numpy.testing.assert_array_equal(restored.tree["x"], value)
"""

_PROFILES: dict[str, tuple[str, ...]] = {
    "base": (),
    "torch": ("torch",),
    "jax": ("jax",),
    "hdf5": ("io-hdf5",),
    "mat": ("io-mat",),
    "all": ("all",),
}


def _run(command: list[str]) -> None:
    subprocess.run(command, check=True)


def _python(environment: pathlib.Path) -> pathlib.Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _requirement(artifact: pathlib.Path, extras: tuple[str, ...]) -> str:
    suffix = f"[{','.join(extras)}]" if extras else ""
    return f"{artifact}{suffix}"


def _check(
    uv: str,
    artifact: pathlib.Path,
    label: str,
    extras: tuple[str, ...],
) -> None:
    with tempfile.TemporaryDirectory(prefix=f"asc-py-{label}-") as directory:
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
                _requirement(artifact, extras),
            ]
        )
        _run([str(python), "-I", "-c", _PROBE, label])


def main() -> int:
    """Verify base artifacts and every independent wheel extra profile."""
    root = pathlib.Path(__file__).resolve().parents[1]
    uv = os.environ.get("ASC_PY_UV", "uv")
    wheel_matches = tuple((root / "dist").glob("asc_py-0.1.0-*.whl"))
    sdist = root / "dist" / "asc_py-0.1.0.tar.gz"
    if len(wheel_matches) != 1 or not sdist.is_file():
        print(
            "clean install: build exactly one wheel and one sdist first",
            file=sys.stderr,
        )
        return 1
    wheel = wheel_matches[0]
    for profile, extras in _PROFILES.items():
        _check(uv, wheel, profile, extras)
    _check(uv, sdist, "base", ())
    print(
        "clean installs: base wheel/sdist and torch, jax, HDF5, MAT, "
        "all wheel profiles passed"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
