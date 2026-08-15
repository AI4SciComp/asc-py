# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Verify that generated API documentation exactly matches public exports."""

from __future__ import annotations

import pathlib
import re
import runpy
import typing

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_FUNCTIONALITY_ID = re.compile(r"^\| ([A-Z]+-?[0-9]+) \|", re.MULTILINE)
_TRACE_PATH = re.compile(
    r"`((?:\.github|examples|src|tests|docs)/[^`]+|AGENTS\.md|"
    r"CHANGELOG\.md|CITATION\.cff|Makefile|pyproject\.toml)`"
)


def _inventory() -> dict[str, object]:
    return runpy.run_path(str(_ROOT / "docs" / "_inventory.py"))


def test_every_public_export_has_one_documentation_target(
    tmp_path: pathlib.Path,
) -> None:
    inventory = _inventory()
    modules = typing.cast(tuple[str, ...], inventory["PUBLIC_MODULES"])
    targets = typing.cast(
        typing.Callable[[], tuple[str, ...]],
        inventory["documentation_targets"],
    )()
    aliases = typing.cast(
        typing.Callable[[], tuple[tuple[str, str], ...]],
        inventory["documentation_aliases"],
    )()
    write = typing.cast(
        typing.Callable[[pathlib.Path], pathlib.Path],
        inventory["write_autosummary_inventory"],
    )

    assert len(targets) == len(set(targets))
    assert not any("._" in module for module in modules)
    assert not any(
        component.startswith("_")
        for target in targets
        for component in target.split(".")[1:-1]
    )
    assert not any(
        target.startswith(
            ("asc.backends.numpy", "asc.backends.torch", "asc.backends.jax")
        )
        for target in targets
    )

    generated = write(tmp_path).read_text(encoding="utf-8")
    for target in targets[len(modules) :]:
        assert generated.splitlines().count(f"   {target}") == 1
    for alias, canonical in aliases:
        assert generated.splitlines().count(f"   * - ``{alias}``") == 1
        assert f":doc:`{canonical} <{canonical}>`" in generated

    exports = typing.cast(
        typing.Callable[[], tuple[object, ...]],
        inventory["public_exports"],
    )()
    represented = set(targets) | {alias for alias, _ in aliases}
    assert represented == {
        *modules,
        *(export.qualified_name for export in exports),
    }

    api_index = (_ROOT / "docs" / "api" / "index.rst").read_text(
        encoding="utf-8"
    )
    for module in modules:
        assert api_index.splitlines().count(f"   {module}") == 1


def test_functionality_traceability_combines_both_runbooks() -> None:
    runbooks = (
        _ROOT / "asc-py-comprehensive-development-runbook.md",
        _ROOT / "asc-py-automatic-documentation-runbook.md",
    )
    expected = [
        functionality_id
        for runbook in runbooks
        for functionality_id in _FUNCTIONALITY_ID.findall(
            runbook.read_text(encoding="utf-8")
        )
    ]
    ledger = (
        _ROOT / "docs" / "specification" / "functionality-matrix.md"
    ).read_text(encoding="utf-8")
    observed = _FUNCTIONALITY_ID.findall(ledger)
    assert observed == expected
    assert len(observed) == len(set(observed))

    rows = [
        line for line in ledger.splitlines() if _FUNCTIONALITY_ID.match(line)
    ]
    assert all(row.endswith("| Complete |") for row in rows)
    for row in rows:
        documentation_cell = row.split("|")[5]
        paths = _TRACE_PATH.findall(documentation_cell)
        assert paths, row
        assert all((_ROOT / path).exists() for path in paths), row
        if row.startswith("| DOC-"):
            assert ".github/workflows/docs.yml" in row.split("|")[7]


def test_public_modules_define_intentional_all() -> None:
    exports = typing.cast(
        typing.Callable[[], tuple[object, ...]],
        _inventory()["public_exports"],
    )()
    assert exports
    assert all(export.name for export in exports)
    assert all(export.value is not None for export in exports)
