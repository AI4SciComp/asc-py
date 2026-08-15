# Contributing

Read `AGENTS.md`, the portability contract, support matrix, and relevant ADRs
before editing. Install the locked environment and run:

```bash
uv sync --frozen --all-groups --all-extras
make check
```

Public changes require typed signatures, Google-style docstrings, API and
narrative documentation, exact functionality-matrix traceability, and tests
over every declared backend. Bug fixes include a regression. Pull requests
explain contract impact, link issues or ADRs, and list exact validation
commands. See the repository-level
{download}`contribution policy <../../CONTRIBUTING.md>`.
