# ADR 0004: Use a Strict Google-compatible Python Toolchain

Status: Amended by ADR 0008. Date: 2026-08-10.

## Context

The greenfield repository has no formatter, linter, type checker, documentation
builder, environment manager, or command facade. The project requires an
80-column Google Python style and reproducible local/CI commands.

## Decision

Use:

- Ruff's Pyink-compatible Google-style formatter with line length 80;
- Ruff for imports, syntax, bug risks, modernization, simplification,
  performance, project rules, and Google pydocstyle;
- Pylint for semantic checks and symbolic, explained suppressions;
- Pyright in strict mode for maintained source and public APIs;
- pytest, coverage, and Hypothesis for semantic tests;
- Sphinx, MyST, and PyData Sphinx Theme for narrative and generated reference
  documentation, as amended by ADR 0008;
- Hatchling for standards-based builds;
- uv and a committed lock for reproducible development;
- pre-commit for local hooks and Make as the documented command facade.

Ruff is used as the compatible formatter because running two formatters over
the same source produced contradictory layouts. CI calls the same underlying
commands as Make. Documentation warnings and
broken internal links fail. Public/nontrivial code uses three-double-quoted
Google-style docstrings and annotations.

## Consequences

Contributor setup has one reproducible path and CI drift is testable. Tool
configuration lives in `pyproject.toml` except where a tool requires a dedicated
file. Tooling dependencies remain development-only. A suppression must be
narrow, symbolic, and explained; a formatter or gate is not weakened to accept
incorrect code.

## Sources

Accessed 2026-08-10:

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [pyproject.toml specification](https://packaging.python.org/en/latest/specifications/pyproject-toml/)
- [Packaging tutorial](https://packaging.python.org/en/latest/tutorials/packaging-projects/)
