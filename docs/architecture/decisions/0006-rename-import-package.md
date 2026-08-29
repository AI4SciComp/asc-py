# ADR 0006: Rename the Import Package to `asc`

Status: Accepted by explicit user direction. Date: 2026-08-10.

## Context

ADR 0001 selected `asc_py` after identifying an unrelated PyPI distribution
that also installs an `asc` package. Before the first release, the user
explicitly changed the required import spelling to `asc`.

## Decision

Keep the distribution name `asc-py`, but install the regular import package
`asc`. Supersede ADR 0001 and migrate production code, tests, documentation,
type markers, artifact audits, and examples atomically. Do not provide an
`asc_py` compatibility package because 0.1.0 has not been released.

## Consequences

Users install `asc-py` and write `import asc`. Installing the unrelated `asc`
distribution in the same environment creates a file-level import collision;
dependency guidance must state that the two distributions cannot safely
coexist. Clean-install tests verify `asc` ownership for this distribution.
