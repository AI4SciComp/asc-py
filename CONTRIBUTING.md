# Contributing to asc-py

Read `AGENTS.md`, the portability contract, support matrix, and relevant ADRs
before changing code. Contract, dependency, API, numerical, or compatibility
changes require an ADR and matching documentation and tests.

## Setup and Checks

Install uv 0.12, then run:

```bash
uv sync --frozen --all-groups --all-extras
make check
```

Use `make format` before review. `make lint`, `make typecheck`, `make test`,
`make docs`, `make docs-linkcheck`, `make build`, and `make audit` expose the
individual gates. Do not weaken or skip a gate to accept a change.

## Architecture and Tests

Portable core code uses only array namespaces and narrow protocols. Optional
backend imports belong in lazy adapters. Never add global backend selection,
implicit conversion, hidden graph detach, or backend-wide default mutation.

Every public portable operation needs a shared behavioral suite over NumPy,
PyTorch, JAX, and array-api-strict. Record operation- and dtype-specific
tolerances. A bug fix includes a regression test.

## Changes and Pull Requests

Use a focused branch and imperative commit subjects, such as
`Add immutable creation context`. Pull requests must explain motivation and
approach, link issues and ADRs, identify API/dependency/numerical changes, and
list exact validation commands, versions, failures, and justified unrun jobs.
Resolve every review conversation before merge.

Do not commit environments, caches, coverage output, built artifacts, secrets,
or benchmark noise. Do not publish, tag, or release as part of a contribution.
