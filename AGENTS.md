# Repository Guidelines

Read the [portability contract](docs/architecture/portability-contract.md),
[support matrix](docs/architecture/support-matrix.md), and
[execution plan](.agent/execplans/asc-py-v0.1.0.md) before changing behavior.

## Project Structure & Module Organization

Production code uses a `src/asc/` layout. Portable numerical code belongs in
`core/`, `ops/`, `linalg/`, `fft/`, and `metrics/`; lazy NumPy, Torch, and JAX
adapters live in `backends/`. Keep conversion, functional updates, random,
autodiff, compilation, trees, and data concerns in their named modules. Tests
are grouped by purpose under `tests/contract/`, `conformance/`, `parity/`,
`properties/`, and `data/`. Documentation, examples, benchmarks, and release
checks live in `docs/`, `examples/`, `benchmarks/`, and `scripts/`.

## Build, Test & Development Commands

Use the locked environment and Make facade:

```bash
uv sync --frozen --all-groups --all-extras  # provision development tools
make format       # apply Ruff-compatible formatting and safe fixes
make lint         # formatter, Ruff, Pylint, and portability audits
make typecheck    # strict Pyright
make test         # pytest, properties, parity, and branch coverage
make docs         # strict Sphinx HTML and doctest builds
make docs-base    # prove docs with NumPy only in an isolated environment
make docs-linkcheck  # scheduled/manual external-link validation
make examples     # execute every documented example
make build        # build, inspect, and clean-install all profiles
make check        # complete local release gate
```

## Coding Style & Naming Conventions

Follow the Google Python Style Guide, four-space indentation, an 80-column
target, typed public APIs, and Google-style docstrings. Use descriptive
`snake_case` functions, `PascalCase` classes, and immutable records for
configuration or state. Avoid broad suppressions; explain narrow ones inline.

Portable code must not import optional backends, mutate inputs, extract host
scalars, or silently copy, convert, transfer, cast, detach, or fall back through
NumPy. Native arrays, explicit contexts, and explicit random state are core
invariants.

## Testing Guidelines

Name tests `test_<behavior>`. Parametrize portable behavior across NumPy,
Torch, and JAX CPU; include `array-api-strict` for standard-surface checks.
Cover dtype, shape, 0-D/empty, error, side-effect, gradient, and JIT semantics
where applicable. Required skips and weakened gates are prohibited. Update the
[functionality ledger](docs/specification/functionality-matrix.md) whenever a
contract ID changes.

## Commit & Pull Request Guidelines

History establishes no prefix convention; use focused, imperative subjects.
Pull requests should describe contract impact, link issues or ADRs, and list
exact validation commands. Include benchmark evidence only for performance
claims. Never commit secrets or generated caches. Do not push, publish, tag,
release, or alter remotes without explicit authorization.
