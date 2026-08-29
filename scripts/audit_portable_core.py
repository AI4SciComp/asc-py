# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Audit asc-py's portable core for prohibited backend coupling.

The audit is intentionally conservative. It complements tests and type checks;
it does not prove semantic portability by itself.
"""

from __future__ import annotations

import ast
import dataclasses
import pathlib
import sys
from collections.abc import Iterable

_BACKEND_IMPORTS = frozenset({"jax", "numpy", "torch"})
_CONVERSION_METHODS = frozenset(
    {
        "cpu",
        "cuda",
        "detach",
        "device_put",
        "numpy",
        "to_dlpack",
        "to_numpy",
    }
)
_SELECTOR_FRAGMENTS = (
    "active_backend",
    "backend_selector",
    "current_backend",
    "default_backend",
)
_PORTABLE_PACKAGES = ("core", "fft", "linalg", "metrics", "ops")


@dataclasses.dataclass(frozen=True, slots=True)
class Finding:
    """A static portable-core audit finding."""

    path: pathlib.Path
    line: int
    message: str


def _import_root(node: ast.Import | ast.ImportFrom) -> Iterable[str]:
    """Yield top-level import names from an import statement."""
    if isinstance(node, ast.Import):
        for alias in node.names:
            yield alias.name.partition(".")[0]
        return
    if node.module is not None:
        yield node.module.partition(".")[0]


def _assignment_names(node: ast.Assign | ast.AnnAssign) -> Iterable[str]:
    """Yield simple names targeted by a module-level assignment."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    for target in targets:
        if isinstance(target, ast.Name):
            yield target.id.lower()


def _audit_file(path: pathlib.Path) -> tuple[Finding, ...]:
    """Return prohibited constructs found in one portable-core module."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    findings: list[Finding] = []

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            findings.extend(
                Finding(
                    path,
                    node.lineno,
                    f"direct optional-backend import: {root_name}",
                )
                for root_name in _import_root(node)
                if root_name in _BACKEND_IMPORTS
            )
            if (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module
                in {
                    "asc.backends.jax",
                    "asc.backends.numpy",
                    "asc.backends.torch",
                }
            ):
                findings.append(
                    Finding(
                        path,
                        node.lineno,
                        "portable core imports a backend adapter",
                    )
                )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in _CONVERSION_METHODS
        ):
            findings.append(
                Finding(
                    path,
                    node.lineno,
                    (f"suspicious conversion or transfer: {node.func.attr}()"),
                )
            )
        if isinstance(node, ast.Assign):
            findings.extend(
                Finding(
                    path,
                    node.lineno,
                    "subscript mutation in portable core",
                )
                for target in node.targets
                if isinstance(target, ast.Subscript)
            )

    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            findings.extend(
                Finding(
                    path,
                    node.lineno,
                    f"module-level backend selector: {name}",
                )
                for name in _assignment_names(node)
                if any(fragment in name for fragment in _SELECTOR_FRAGMENTS)
            )
    return tuple(findings)


def audit(repository: pathlib.Path) -> tuple[int, tuple[Finding, ...]]:
    """Audit every Python file in the portable numerical packages.

    Args:
        repository: Repository root containing the `src` directory.

    Returns:
        The number of files audited and all error-level findings.
    """
    paths = (
        path
        for package in _PORTABLE_PACKAGES
        for path in repository.glob(f"src/*/{package}/**/*.py")
    )
    core_files = tuple(sorted(paths, key=str))
    findings = tuple(
        finding for path in core_files for finding in _audit_file(path)
    )
    return len(core_files), findings


def main() -> int:
    """Run the audit for the repository containing this script."""
    repository = pathlib.Path(__file__).resolve().parents[1]
    file_count, findings = audit(repository)
    if file_count == 0:
        print("NOTICE: no portable-core Python files exist yet; baseline only")
    for finding in findings:
        relative_path = finding.path.relative_to(repository)
        print(
            f"ERROR: {relative_path}:{finding.line}: {finding.message}",
            file=sys.stderr,
        )
    print(f"portable-core audit: {file_count} files, {len(findings)} errors")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
