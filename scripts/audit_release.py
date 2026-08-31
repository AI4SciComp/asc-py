# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Audit repository-owned release structure and metadata."""

from __future__ import annotations

import pathlib
import re
import subprocess
import sys
import tomllib

_REQUIRED = (
    ".github/workflows/ci.yml",
    ".github/workflows/release.yml",
    "AGENTS.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "Makefile",
    "README.md",
    "SECURITY.md",
    ".github/workflows/docs.yml",
    "docs/api/index.rst",
    "docs/architecture/downstream-public-api.md",
    "docs/conf.py",
    "docs/index.md",
    "src/asc/py.typed",
    "uv.lock",
)
_FUNCTIONALITY_ID = re.compile(r"^\| ([A-Z]+-?[0-9]+) \|", re.MULTILINE)
_TRACE_PATH = re.compile(
    r"`((?:\.github|examples|src|tests|docs)/[^`]+|AGENTS\.md|"
    r"CHANGELOG\.md|CITATION\.cff|Makefile|pyproject\.toml)`"
)


def main() -> int:
    """Return nonzero when repository release invariants are incomplete."""
    root = pathlib.Path(__file__).resolve().parents[1]
    errors = [path for path in _REQUIRED if not (root / path).is_file()]
    project = tomllib.loads(
        (root / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    if project["name"] != "asc-py" or project["version"] != "0.1.0":
        errors.append("pyproject.toml: unexpected name or version")
    runbooks = (
        root / "asc-py-comprehensive-development-runbook.md",
        root / "asc-py-automatic-documentation-runbook.md",
    )
    ledger_path = root / "docs/specification/functionality-matrix.md"
    ledger = ledger_path.read_text(encoding="utf-8")
    runbook_ids = [
        functionality_id
        for runbook in runbooks
        for functionality_id in _FUNCTIONALITY_ID.findall(
            runbook.read_text(encoding="utf-8")
        )
    ]
    ledger_ids = _FUNCTIONALITY_ID.findall(ledger)
    if runbook_ids != ledger_ids or len(ledger_ids) != len(set(ledger_ids)):
        errors.append(
            "functionality matrix: IDs differ from normative runbook order"
        )
    incomplete = [
        line
        for line in ledger.splitlines()
        if _FUNCTIONALITY_ID.match(line) and not line.endswith("| Complete |")
    ]
    if incomplete:
        errors.append(
            f"functionality matrix: {len(incomplete)} rows are incomplete"
        )
    errors.extend(
        f"functionality matrix: missing trace path {trace}"
        for trace in _TRACE_PATH.findall(ledger)
        if not (root / trace).exists()
    )
    for test in (root / "tests").rglob("*.py"):
        source = test.read_text(encoding="utf-8")
        if "pytest.skip" in source or "pytest.mark.skip" in source:
            errors.append(
                f"{test.relative_to(root)}: required suite contains a skip"
            )
    for source in (root / "src" / "asc").rglob("*.py"):
        lines = source.read_text(encoding="utf-8").splitlines()
        if not lines or "Copyright 2026 AI4SciComp" not in lines[0]:
            errors.append(
                f"{source.relative_to(root)}: missing copyright header"
            )
    if (root / ".git").exists():
        diff_check = subprocess.run(
            ["git", "diff", "--check"],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )
        if diff_check.returncode:
            errors.append(diff_check.stdout or diff_check.stderr)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("release audit: repository structure and metadata are valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
