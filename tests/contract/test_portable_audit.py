# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the portable numerical-package source audit."""

from __future__ import annotations

import pathlib

from scripts import audit_portable_core


def test_audit_visits_every_portable_numerical_package(
    tmp_path: pathlib.Path,
) -> None:
    packages = ("core", "fft", "linalg", "metrics", "ops")
    for package in packages:
        directory = tmp_path / "src" / "example" / package
        directory.mkdir(parents=True)
        (directory / "__init__.py").write_text(
            "from __future__ import annotations\n", encoding="utf-8"
        )
    (tmp_path / "src" / "example" / "ops" / "bad.py").write_text(
        "import numpy\n", encoding="utf-8"
    )

    file_count, findings = audit_portable_core.audit(tmp_path)

    assert file_count == len(packages) + 1
    assert len(findings) == 1
    assert findings[0].path.name == "bad.py"
    assert findings[0].message == "direct optional-backend import: numpy"
