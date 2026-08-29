# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Inspect wheel and source-distribution release contracts."""

from __future__ import annotations

import email.parser
import pathlib
import sys
import tarfile
import zipfile


def _one_artifact(dist: pathlib.Path, pattern: str) -> pathlib.Path:
    matches = tuple(dist.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"expected one {pattern!r} artifact, found {len(matches)}"
        )
    return matches[0]


def _check_wheel(wheel: pathlib.Path) -> list[str]:
    errors: list[str] = []
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        if "asc/py.typed" not in names:
            errors.append("wheel: missing asc/py.typed")
        if any(name.startswith("asc_py/") for name in names):
            errors.append("wheel: contains obsolete asc_py import package")
        if any(
            "__pycache__" in name or name.endswith(".pyc") for name in names
        ):
            errors.append("wheel: contains generated Python cache files")
        metadata_names = tuple(
            name for name in names if name.endswith(".dist-info/METADATA")
        )
        if len(metadata_names) != 1:
            errors.append("wheel: expected one METADATA file")
            return errors
        metadata_text = archive.read(metadata_names[0]).decode("utf-8")
    metadata = email.parser.Parser().parsestr(metadata_text)
    if metadata["Metadata-Version"] != "2.4":
        errors.append("wheel: expected interoperable Core Metadata 2.4")
    if metadata["Name"] != "asc-py" or metadata["Version"] != "0.1.0":
        errors.append("wheel: unexpected distribution identity or version")
    requirements = metadata.get_all("Requires-Dist", [])
    if metadata["License-Expression"] != "Apache-2.0":
        errors.append("wheel: missing Apache-2.0 license expression")
    if "LICENSE" not in metadata.get_all("License-File", []):
        errors.append("wheel: missing embedded LICENSE declaration")
    if not any(
        requirement.startswith("array-api-compat")
        for requirement in requirements
    ):
        errors.append("wheel: missing array-api-compat runtime requirement")
    if not any(requirement.startswith("numpy") for requirement in requirements):
        errors.append("wheel: missing NumPy runtime requirement")
    optional_names = ("torch", "jax", "h5py", "scipy")
    for name in optional_names:
        matching = [
            requirement
            for requirement in requirements
            if requirement.startswith(name)
        ]
        if not matching or any("extra ==" not in item for item in matching):
            errors.append(
                f"wheel: optional dependency {name} is missing or unconditional"
            )
    for name in ("myst-parser", "pydata-sphinx-theme", "sphinx"):
        matching = [
            requirement
            for requirement in requirements
            if requirement.lower().startswith(name)
        ]
        if not matching or any("extra ==" not in item for item in matching):
            errors.append(
                f"wheel: docs dependency {name} is missing or unconditional"
            )
    expected_extras = {
        "all",
        "docs",
        "io-hdf5",
        "io-mat",
        "jax",
        "torch",
    }
    if set(metadata.get_all("Provides-Extra", [])) != expected_extras:
        errors.append("wheel: optional extra declarations are incomplete")
    return errors


def _check_sdist(sdist: pathlib.Path) -> list[str]:
    required_suffixes = {
        "/AGENTS.md",
        "/asc-py-automatic-documentation-runbook.md",
        "/asc-py-comprehensive-development-runbook.md",
        "/LICENSE",
        "/Makefile",
        "/README.md",
        "/docs/conf.py",
        "/docs/index.md",
        "/pyproject.toml",
        "/scripts/audit_portable_core.py",
        "/src/asc/py.typed",
        "/uv.lock",
    }
    with tarfile.open(sdist, mode="r:gz") as archive:
        names = set(archive.getnames())
    errors = [
        f"sdist: missing required path ending {suffix!r}"
        for suffix in sorted(required_suffixes)
        if not any(name.endswith(suffix) for name in names)
    ]
    if any("__pycache__" in name or name.endswith(".pyc") for name in names):
        errors.append("sdist: contains generated Python cache files")
    if any(
        "/docs/_build/" in name or "/docs/api/generated/" in name
        for name in names
    ):
        errors.append("sdist: contains generated documentation output")
    return errors


def main() -> int:
    """Return nonzero when a built artifact violates the release contract."""
    root = pathlib.Path(__file__).resolve().parents[1]
    dist = root / "dist"
    try:
        wheel = _one_artifact(dist, "asc_py-0.1.0-*.whl")
        sdist = _one_artifact(dist, "asc_py-0.1.0.tar.gz")
    except ValueError as exception:
        print(str(exception), file=sys.stderr)
        return 1
    errors = _check_wheel(wheel) + _check_sdist(sdist)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("artifact contracts: wheel and sdist contents are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
