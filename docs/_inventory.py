# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Build the intentional installed-package documentation inventory."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
import pathlib
import shutil
import typing

PUBLIC_MODULES: typing.Final = (
    "asc",
    "asc.autodiff",
    "asc.backends",
    "asc.compilation",
    "asc.config",
    "asc.conversion",
    "asc.data",
    "asc.diagnostics",
    "asc.errors",
    "asc.extensions",
    "asc.extensions.autodiff",
    "asc.extensions.compilation",
    "asc.extensions.indexing",
    "asc.extensions.random",
    "asc.fft",
    "asc.linalg",
    "asc.logging",
    "asc.metrics",
    "asc.ops",
    "asc.random",
    "asc.tree",
    "asc.typing",
    "asc.updates",
)


@dataclasses.dataclass(frozen=True, slots=True)
class PublicExport:
    """Describe one intentional qualified export from a public module."""

    module: str
    name: str
    qualified_name: str
    kind: typing.Literal["class", "data", "function", "module"]
    value: object = dataclasses.field(compare=False, repr=False)


def _kind(
    value: object,
) -> typing.Literal["class", "data", "function", "module"]:
    if inspect.ismodule(value):
        return "module"
    if inspect.isclass(value):
        return "class"
    if callable(value) and typing.get_origin(value) is None:
        return "function"
    return "data"


def public_exports() -> tuple[PublicExport, ...]:
    """Return every intentional export from every documented public module."""
    exports: list[PublicExport] = []
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        declared = getattr(module, "__all__", None)
        if not isinstance(declared, (list, tuple)):
            raise TypeError(
                f"{module_name} must define a list or tuple __all__"
            )
        if len(declared) != len(set(declared)):
            raise ValueError(f"{module_name}.__all__ contains duplicates")
        for name in declared:
            if not isinstance(name, str) or not name:
                raise TypeError(
                    f"{module_name}.__all__ contains an invalid name"
                )
            value = getattr(module, name)
            exports.append(
                PublicExport(
                    module=module_name,
                    name=name,
                    qualified_name=f"{module_name}.{name}",
                    kind=_kind(value),
                    value=value,
                )
            )
    qualified_names = tuple(export.qualified_name for export in exports)
    if len(qualified_names) != len(set(qualified_names)):
        raise ValueError("public documentation inventory contains duplicates")
    return tuple(exports)


def documentation_targets() -> tuple[str, ...]:
    """Return one canonical target for each intentional public object."""
    module_targets = set(PUBLIC_MODULES)
    grouped: dict[int, list[PublicExport]] = {}
    for export in public_exports():
        if export.qualified_name not in module_targets:
            grouped.setdefault(id(export.value), []).append(export)

    export_targets: list[str] = []
    for aliases in grouped.values():
        exact = [
            export
            for export in aliases
            if export.module == getattr(export.value, "__module__", None)
        ]
        candidates = exact or aliases
        canonical = max(
            candidates,
            key=lambda export: (
                not export.module.startswith("asc.extensions"),
                export.module.count("."),
                export.qualified_name,
            ),
        )
        export_targets.append(canonical.qualified_name)
    return (*PUBLIC_MODULES, *export_targets)


def documentation_aliases() -> tuple[tuple[str, str], ...]:
    """Return public access paths mapped to their canonical documentation."""
    targets = set(documentation_targets())
    canonical_by_identity: dict[int, str] = {}
    for export in public_exports():
        if export.qualified_name in targets:
            canonical_by_identity[id(export.value)] = export.qualified_name
    for module_name in PUBLIC_MODULES:
        module = importlib.import_module(module_name)
        canonical_by_identity[id(module)] = module_name
    return tuple(
        (export.qualified_name, canonical_by_identity[id(export.value)])
        for export in public_exports()
        if export.qualified_name not in targets
    )


def write_autosummary_inventory(directory: pathlib.Path) -> pathlib.Path:
    """Regenerate the ignored autosummary source from installed exports."""
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    destination = directory / "public-symbols.rst"
    symbols = documentation_targets()[len(PUBLIC_MODULES) :]
    aliases = documentation_aliases()
    lines = [
        "Public symbols",
        "==============",
        "",
        "This file is regenerated from installed ``__all__`` declarations.",
        "",
        ".. autosummary::",
        "   :toctree: .",
        "",
        *(f"   {symbol}" for symbol in symbols),
        "",
        "Public aliases",
        "--------------",
        "",
        "Every intentional alias resolves to the single canonical page below.",
        "",
        ".. list-table::",
        "   :header-rows: 1",
        "",
        "   * - Public access path",
        "     - Canonical documentation",
        *(
            line
            for alias, canonical in aliases
            for line in (
                f"   * - ``{alias}``",
                f"     - :doc:`{canonical} <{canonical}>`",
            )
        ),
        "",
    ]
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination


__all__ = [
    "PUBLIC_MODULES",
    "PublicExport",
    "documentation_aliases",
    "documentation_targets",
    "public_exports",
    "write_autosummary_inventory",
]
