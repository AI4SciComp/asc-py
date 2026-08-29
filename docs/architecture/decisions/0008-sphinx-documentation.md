# ADR 0008: Generate Documentation with Sphinx

Status: Accepted. Date: 2026-08-11.

## Context

The initial release candidate used MkDocs and one hand-written API page. The
automatic-documentation addendum requires installed-object reference pages,
strict cross-references, doctests, and GitHub Pages deployment without exposing
lazy backend adapters.

## Decision

Sphinx 9.1, MyST Parser 5.1, and PyData Sphinx Theme 0.19 are the bounded docs
extra for Python 3.12–3.14. Built-in autodoc, autosummary, doctest,
intersphinx, MathJax, Napoleon, viewcode, and GitHub Pages extensions form one
canonical site. The public module allowlist and each installed module's
`__all__` generate ignored autosummary sources on every build. Internal modules
are forbidden by tests.

Re-exported access paths resolve to one canonical object page. Internal
ParamSpecs and private signature aliases are rendered as unlinked type text
because they intentionally have no public object target; no `nitpick_ignore`
suppression is used. Private asc base classes are omitted from public
inheritance lists.

The current official Google guide agrees with the runbook snapshot on
docstring quoting, summary layout, semantic sections, and typed APIs. The
official guide's newer clarification for overrides and mathematical notation is
adopted without weakening the snapshot.

## Consequences

Public API drift, missing docstrings, duplicate inventory entries, broken
cross-references, warnings, and failing examples block CI. The NumPy-only docs
environment cannot import Torch or JAX. Published HTML is a CI artifact and is
never committed. Enabling GitHub Actions as the Pages source remains a one-time
administrator action.

## Sources

Accessed 2026-08-11:

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Sphinx installation](https://www.sphinx-doc.org/en/master/usage/installation.html)
- [MyST Parser](https://myst-parser.readthedocs.io/)
- [PyData Sphinx Theme](https://pydata-sphinx-theme.readthedocs.io/)
- [GitHub Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
