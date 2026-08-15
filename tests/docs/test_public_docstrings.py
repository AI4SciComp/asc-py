# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Enforce typed public APIs and Google-style documentation summaries."""

from __future__ import annotations

import inspect
import pathlib
import runpy
import textwrap
import typing

from sphinx.ext.napoleon.docstring import GoogleDocstring

_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _exports() -> tuple[object, ...]:
    inventory = runpy.run_path(str(_ROOT / "docs" / "_inventory.py"))
    return typing.cast(
        typing.Callable[[], tuple[object, ...]],
        inventory["public_exports"],
    )()


def _assert_google_summary(qualified_name: str, value: object) -> None:
    docstring = inspect.getdoc(value)
    assert docstring, f"{qualified_name} lacks a docstring"
    summary = docstring.splitlines()[0]
    assert len(summary) <= 80, f"{qualified_name} summary exceeds 80 characters"
    assert summary.endswith((".", "?", "!")), (
        f"{qualified_name} summary must end with punctuation"
    )
    if len(docstring.splitlines()) > 1:
        source_docstring = value.__doc__ or ""
        assert "\n\n" in source_docstring, (
            f"{qualified_name} needs a blank line after its summary"
        )


def _assert_typed(qualified_name: str, value: object) -> None:
    signature = inspect.signature(value)
    for parameter in signature.parameters.values():
        if parameter.name in {"self", "cls"}:
            continue
        assert parameter.annotation is not inspect.Signature.empty, (
            f"{qualified_name}.{parameter.name} lacks an annotation"
        )
    assert signature.return_annotation is not inspect.Signature.empty, (
        f"{qualified_name} lacks a return annotation"
    )


def test_public_objects_have_google_summaries_and_types() -> None:
    for export in _exports():
        kind = export.kind
        value = export.value
        qualified_name = export.qualified_name
        if kind in {"module", "class", "function"}:
            _assert_google_summary(qualified_name, value)
        if kind == "function" and inspect.isfunction(value):
            _assert_typed(qualified_name, value)


def test_public_class_members_are_documented_and_typed() -> None:
    visited: set[type[object]] = set()
    for export in _exports():
        value = export.value
        if export.kind != "class" or value in visited:
            continue
        visited.add(value)
        for name, member in vars(value).items():
            if name.startswith("_"):
                continue
            target = member.fget if isinstance(member, property) else member
            if not inspect.isfunction(target):
                continue
            qualified_name = f"{value.__module__}.{value.__qualname__}.{name}"
            _assert_google_summary(qualified_name, target)
            _assert_typed(qualified_name, target)


def test_napoleon_parses_every_google_semantic_section() -> None:
    standard_sections = "\n".join(
        GoogleDocstring(
            textwrap.dedent(
                """
                Demonstrates every supported semantic section.

                Args:
                    value: Input value.

                Returns:
                    The transformed value.

                Raises:
                    ValueError: If the value is invalid.

                Attributes:
                    state: Current immutable state.

                Notes:
                    Types come from annotations.

                Examples:
                    ``example(1)``
                """
            ).strip()
        ).lines()
    )
    yield_section = "\n".join(
        GoogleDocstring(
            textwrap.dedent(
                """
                Demonstrates generator documentation.

                Yields:
                    Streamed values.
                """
            ).strip()
        ).lines()
    )
    rendered = f"{standard_sections}\n{yield_section}"
    for directive in (
        ":param value:",
        ":returns:",
        ":Yields:",
        ":raises ValueError:",
        ".. attribute:: state",
        ".. rubric:: Notes",
        ".. rubric:: Examples",
    ):
        assert directive in rendered


def test_scientific_semantics_are_documented() -> None:
    pages = (
        _ROOT / "docs" / "user-guide" / "array-api.md",
        _ROOT / "docs" / "user-guide" / "dtype-device.md",
        _ROOT / "docs" / "user-guide" / "conversion.md",
        _ROOT / "docs" / "user-guide" / "random.md",
        _ROOT / "docs" / "user-guide" / "autodiff-jit.md",
        _ROOT / "docs" / "user-guide" / "data.md",
    )
    documentation = "\n".join(
        page.read_text(encoding="utf-8").lower() for page in pages
    )
    for semantic in (
        "shape",
        "dtype",
        "device",
        "copy",
        "mutat",
        "backend",
        "random",
        "autodiff",
        "jit",
        "optional",
        "error",
    ):
        assert semantic in documentation
