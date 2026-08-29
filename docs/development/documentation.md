# Documentation development

Sphinx is the only site generator. MyST renders narrative Markdown; autodoc and
autosummary render the installed package; Napoleon parses Google docstrings.
Intersphinx resolves external Python objects, MathJax renders equations,
viewcode links public objects to source, and the PyData theme provides the HTML
shell.

Install and build with:

```bash
python -m pip install -e ".[docs]"
python -m sphinx -W --keep-going -n -b html docs docs/_build/html
python -m sphinx -W --keep-going -n -b doctest docs docs/_build/doctest
python -m sphinx -W --keep-going -b linkcheck docs docs/_build/linkcheck
```

Run `make docs-base` to create a temporary NumPy-only environment, assert that
Torch and JAX are absent, and execute the HTML, doctest, inventory, docstring,
and NumPy-example gates together.

`docs/conf.py` does not alter `sys.path` or mock optional packages. It imports
the installed distribution and regenerates ignored autosummary sources from
the public module list and each module's `__all__`. Rendered HTML and generated
stubs must not be committed.

The documentation workflow uploads the tested HTML artifact on every applicable
build. It deploys only after a successful push to `main` when the repository
variable `ASC_PY_PAGES_ENABLED` is explicitly set to `true`. Before the first
authorized deployment, a repository administrator must also select
**Settings → Pages → Source → GitHub Actions**. Neither opt-in is part of the
foundation merge; publication requires separate authorization.

## Google style comparison

The repository snapshot is the documentation runbook's Section 7 together
with ADR 0004. It was compared on 2026-08-11 with the current official Google
Python Style Guide. Both require triple-double-quoted docstrings, a punctuated
one-line summary no longer than 80 characters, a blank line before detail, and
semantic `Args`, `Returns`/`Yields`, `Raises`, and `Attributes` sections when
needed. Type annotations remain authoritative for Python types. The current
guide additionally clarifies overridden-method and mathematical-notation
rules; neither changes asc's public contract.

Numerical and data docstrings or their linked task guide must state observable
shape, dtype, device, mutation/copy, backend, random, autodiff/JIT,
optional-dependency, and error behavior wherever those concerns apply.
