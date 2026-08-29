# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Static graph-integrity checks for promised portable numerical paths."""

from __future__ import annotations

import ast
import pathlib


def test_portable_numerical_paths_have_no_hidden_host_boundary() -> None:
    root = pathlib.Path(__file__).resolve().parents[2] / "src" / "asc"
    paths = tuple((root / name).rglob("*.py") for name in ("ops", "metrics"))
    files = tuple(path for group in paths for path in group)
    assert files
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = (
                    [alias.name for alias in node.names]
                    if isinstance(node, ast.Import)
                    else [node.module or ""]
                )
                assert not any(
                    name.partition(".")[0] in {"jax", "numpy", "torch"}
                    for name in names
                ), f"hidden backend import in {path}:{node.lineno}"
            if isinstance(node, ast.Call) and isinstance(
                node.func, ast.Attribute
            ):
                assert node.func.attr not in {"item", "numpy", "tolist"}, (
                    f"host scalarization in {path}:{node.lineno}"
                )
