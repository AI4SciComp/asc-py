# ADR 0001: Use `asc_py` as the Import Package

Status: Superseded by ADR 0006. Date: 2026-08-10.

## Context

The distribution name is fixed as `asc-py`, but the repository did not select
an import package. A short `asc` package could align visually with AI4SciComp's
project prefix, while `asc_py` is explicit and follows Python package naming.

PyPI already contains an unrelated `asc` 0.3.1 distribution. Inspection of its
wheel on 2026-08-10 confirmed that it installs `asc/__init__.py` and other
`asc/` modules. The `asc-py` distribution identity was unclaimed. PyPI name
normalization treats `asc-py` and `asc_py` as the same distribution name.

The public AI4SciComp siblings `asc-cpp`, `asc-devtools`, and `asc-cmake` use
the ASC project prefix but provide no established Python namespace-package
policy. A future `asc-xde-py` can depend directly on an explicit foundational
package without sharing its import tree.

## Decision

Use distribution `asc-py` and import package `asc_py`. Do not claim or extend
the unrelated `asc` package, and do not create an implicit namespace package.

The user explicitly approved the `asc-py` / `asc_py` pair on 2026-08-10.

## Consequences

Users will install `asc-py` and write `import asc_py`. Future AI4SciComp Python
projects retain independent import names and declare ordinary dependencies on
`asc-py`. Documentation must explain the distribution/import spelling once.
The longer name avoids collision and makes ownership clear.

## Evidence

- [Unrelated `asc` project on PyPI](https://pypi.org/project/asc/)
- [PyPA name normalization](https://packaging.python.org/en/latest/specifications/name-normalization/)
- [Google Python naming rules](https://google.github.io/styleguide/pyguide.html#3162-naming-conventions)

Sources accessed 2026-08-10.
