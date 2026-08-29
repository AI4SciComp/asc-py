# Automatic Documentation Runbook for `asc-py`

**Purpose:** Add a standard, automatically generated and continuously verified
documentation system to `AI4SciComp/asc-py`.

**Status:** Normative addendum to
`asc-py-comprehensive-development-runbook.md`.

## 1. Required outcome

Use **Sphinx** as the documentation engine and the **PyData Sphinx Theme** as the
HTML theme. Use Sphinx's built-in extensions to generate the API reference from
the package's public Python objects and Google-style docstrings.

The system must provide two kinds of documentation:

1. **Automatically generated reference documentation**
   - modules, classes, functions, methods, attributes, signatures, type hints,
     inheritance, and source links;
   - generated from the installed package, public `__all__` declarations, type
     annotations, and Google-style docstrings;
   - regenerated on every documentation build.
2. **Curated narrative documentation**
   - installation, concepts, tutorials, backend policies, data workflows,
     examples, architecture, contribution, and release notes;
   - initially written by Codex and then reviewed like source code;
   - never replaced by low-quality generated prose.

Do not commit rendered HTML or generated build directories. Build and publish
them in CI.

## 2. Documentation functionality matrix

Add these rows to the main `asc-py` functionality matrix.

| ID | Required functionality | Acceptance evidence |
|---|---|---|
| DOC-001 | Sphinx documentation project | `docs/conf.py` and `docs/index.md` build successfully |
| DOC-002 | Separate `docs` dependency extra | `pip install -e ".[docs]"` installs documentation tools without Torch or JAX |
| DOC-003 | Google-style docstring parsing | Napoleon renders `Args`, `Returns`, `Yields`, `Raises`, `Attributes`, `Notes`, and `Examples` |
| DOC-004 | Automatic public API reference | Autosummary generates pages for every supported public module and symbol |
| DOC-005 | Public API completeness check | CI fails if a public export lacks a docstring or generated API entry |
| DOC-006 | Signature and type rendering | Public signatures and type annotations appear in API pages without duplicating types in docstrings |
| DOC-007 | Cross-reference validation | Missing Python-object references fail the strict documentation build |
| DOC-008 | Executable examples | User-facing examples and doctests run in CI |
| DOC-009 | Optional-backend safety | Core documentation builds with NumPy only; optional backend pages state installation requirements |
| DOC-010 | Scientific semantics | Public numerical APIs document shape, dtype, device, mutation/copy, randomness, autodiff, JIT, and errors where applicable |
| DOC-011 | Data-module documentation | Dataset, sampler, loader, transform, statistics, and I/O APIs have reference and task-oriented guides |
| DOC-012 | Support and compatibility pages | Backend, Python, Array API, dtype, device, and optional-extra matrices are published |
| DOC-013 | Local documentation commands | One documented command builds HTML; separate commands run doctests and link checks |
| DOC-014 | Pull-request documentation gate | HTML build with warnings as errors and documentation tests run on pull requests |
| DOC-015 | Automatic publication | A successful default-branch build deploys the HTML site to GitHub Pages |
| DOC-016 | External-link validation | A scheduled or manually triggered job checks external links without making ordinary PR checks flaky |
| DOC-017 | Release documentation | Changelog, migration notes, deprecations, compatibility statement, citation, and release notes are linked from the site |
| DOC-018 | Traceability | Every main functionality ID links to its API page or user guide in the functionality matrix |

An item is complete only when its source, test, documentation page, and CI
evidence are recorded.

## 3. Required toolchain

Use these documentation dependencies in `pyproject.toml` under an independent
extra. Codex must select compatible version bounds from current primary
documentation and the repository's supported Python range.

```toml
[project.optional-dependencies]
docs = [
    "myst-parser",
    "pydata-sphinx-theme",
    "sphinx",
]
```

Use these Sphinx extensions:

```python
extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]
```

The required responsibilities are:

| Component | Responsibility |
|---|---|
| `autodoc` | Read signatures, annotations, and docstrings from installed `asc-py` objects |
| `autosummary` | Generate navigable module, class, and function pages |
| `napoleon` | Parse the Google docstring convention |
| MyST Parser | Allow narrative documentation to be written in Markdown |
| `doctest` | Execute documented examples and verify shown results |
| `intersphinx` | Link Python, NumPy, PyTorch, JAX, and related API objects |
| `mathjax` | Render mathematical definitions and equations |
| `viewcode` | Link documented objects to their source code |
| PyData Sphinx Theme | Provide a searchable, responsive scientific-Python documentation site |

Do not introduce a second documentation generator such as MkDocs for the same
site. One canonical build avoids duplicated configuration and inconsistent API
pages.

## 4. Required repository layout

```text
asc-py/
├── docs/
│   ├── conf.py
│   ├── index.md
│   ├── getting-started/
│   │   ├── installation.md
│   │   └── quickstart.md
│   ├── user-guide/
│   │   ├── backend-selection.md
│   │   ├── array-api.md
│   │   ├── dtype-device.md
│   │   ├── conversion.md
│   │   ├── random.md
│   │   ├── autodiff-jit.md
│   │   └── data.md
│   ├── tutorials/
│   │   ├── portable-computation.md
│   │   └── data-pipeline.md
│   ├── api/
│   │   ├── index.rst
│   │   └── generated/          # generated during the build
│   ├── reference/
│   │   ├── support-matrix.md
│   │   ├── functionality-matrix.md
│   │   ├── exceptions.md
│   │   └── configuration.md
│   ├── development/
│   │   ├── architecture.md
│   │   ├── contributing.md
│   │   ├── testing.md
│   │   └── documentation.md
│   ├── release/
│   │   ├── changelog.md
│   │   ├── migration.md
│   │   └── release-notes.md
│   ├── _static/
│   └── _templates/
├── tests/docs/
│   ├── test_public_docstrings.py
│   ├── test_public_api_inventory.py
│   └── test_documented_examples.py
├── examples/
├── .github/workflows/docs.yml
└── pyproject.toml
```

The generated `docs/api/generated/` directory and `docs/_build/` directory must
be ignored by Git unless the selected Sphinx workflow proves that a small
generated source stub must be tracked. Rendered HTML must never be committed.

## 5. Sphinx configuration contract

Configure at least the following behavior in `docs/conf.py`:

```python
from importlib import metadata

project = "asc-py"
release = metadata.version("asc-py")

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autosummary_generate = True
autosummary_ignore_module_all = False
autosummary_imported_members = True

autodoc_typehints = "description"
autodoc_typehints_format = "short"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_attr_annotations = True

nitpicky = True
html_theme = "pydata_sphinx_theme"
source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}
```

Add project metadata, copyright, repository links, source links, navigation,
logo, favicon, and `intersphinx_mapping` using current valid configuration.
Suppress a warning only when it is proven unavoidable, and document each narrow
suppression. Do not use a broad `nitpick_ignore` list to make the build pass.

Install `asc-py` normally before running Sphinx. Do not modify `sys.path` in
`conf.py`, and do not mock NumPy, Torch, JAX, or package modules merely to hide
import failures.

## 6. Public API generation policy

Every public package and module must define an intentional `__all__`. Internal
modules beginning with `_` must not appear in the public reference.

Create `docs/api/index.rst` with an explicit list of supported public modules,
for example:

```rst
API reference
=============

.. autosummary::
   :toctree: generated
   :recursive:

   asc
   asc.backends
   asc.config
   asc.conversion
   asc.data
   asc.errors
   asc.extensions
   asc.logging
   asc.random
```

Codex must replace this illustrative list with the actual public-module
inventory. It must not expose implementation-only adapters or symbols merely
because they are importable.

Add a test that compares:

- all names exported by each public module's `__all__`;
- all objects with a required Google-style docstring;
- all objects represented in the generated API inventory;
- all public symbols in the functionality traceability matrix.

Any mismatch must fail CI. This is what keeps the generated documentation in
sync when the API changes.

## 7. Google-style docstring contract

Follow the current official Google Python Style Guide and the supplied snapshot.
Use `"""` for every docstring. The first line must be a complete summary of at
most 80 characters, followed by a blank line before additional text.

Every public or nontrivial function, class, method, property, and module must be
documented. Type annotations are the authoritative location for Python types;
docstrings document semantics.

A portable numerical function should use this pattern:

```python
def root_mean_square(x: Array, /, *, axis: Axis | None = None) -> Array:
    """Computes the root mean square of an array.

    The result remains in the namespace, dtype family, and device prescribed by
    the portability contract. The operation does not mutate ``x``.

    Args:
        x: Input array from one supported backend.
        axis: Axis or axes over which to compute the result. If ``None``, uses
            every axis.

    Returns:
        An array containing the root-mean-square values.

    Raises:
        MixedBackendError: If inputs from incompatible backends are combined.

    Notes:
        State promotion, precision, autodiff, and JIT behavior when those details
        are part of the public contract.

    Examples:
        Provide a short executable example.
    """
```

Document the following whenever they affect observable behavior:

- accepted shapes and returned shape;
- dtype creation, promotion, and precision;
- device placement and synchronization;
- copy, aliasing, and mutation behavior;
- backend discovery and mixed-backend rejection;
- random-state consumption and reproducibility scope;
- differentiable arguments and gradient limitations;
- eager, JIT, or compiled behavior;
- optional dependencies and actionable installation commands;
- stable public exceptions and unsupported capability behavior.

Do not repeat obvious implementation details or claim support not verified by
tests.

## 8. Build, test, and publication automation

Provide one documented local entry point for each underlying command:

```bash
python -m sphinx -W --keep-going -n -b html docs docs/_build/html
python -m sphinx -W --keep-going -n -b doctest docs docs/_build/doctest
python -m sphinx -W --keep-going -b linkcheck docs docs/_build/linkcheck
```

The HTML and doctest commands must run on every pull request that changes source,
examples, packaging, or documentation. Run external link checking on a schedule
or manual trigger so transient external outages do not block ordinary changes.

The documentation workflow must test at least:

1. a NumPy-only environment, proving that documentation does not require Torch
   or JAX;
2. the documentation environment used for the complete published reference;
3. executable NumPy examples;
4. separately capability-gated Torch and JAX examples when those extras are
   available in their own CI jobs.

On a successful push to the default branch, build once and deploy that exact
artifact to GitHub Pages. Use official GitHub Pages actions, least-privilege
workflow permissions, concurrency control, and the repository's action-pinning
policy. Pull requests must build and test documentation but must never deploy.

Enabling **Settings → Pages → Source → GitHub Actions** is a one-time repository
administrator action and requires the user's authorization. Codex may prepare
the workflow locally but must not change repository settings or deploy without
authorization.

## 9. Codex implementation directive

Run this at the root of the current `asc-py` checkout:

```text
$port-scientific-python

Implement the automatic documentation system specified in
asc-py-automatic-documentation-runbook.md and integrate DOC-001 through DOC-018
into the existing asc-py comprehensive functionality matrix and ExecPlan.

Use Sphinx, PyData Sphinx Theme, MyST, autodoc, autosummary, Napoleon, doctest,
intersphinx, MathJax, viewcode, and the GitHub Pages extension. Generate the API
reference from the installed package, intentional public __all__ declarations,
type annotations, and Google-style docstrings. Do not expose internal modules.

Follow the current official Google Python Style Guide and compare it with the
supplied snapshot. Require typed public APIs and complete Google-style docstrings.
For numerical and data APIs, document shape, dtype, device, copy/mutation,
backend, random, autodiff/JIT, optional-dependency, and error semantics whenever
applicable.

Write the complete narrative documentation listed in this runbook. Add a strict
public-API documentation inventory test, executable examples, warnings-as-errors
HTML builds, doctests, scheduled link checking, and a least-privilege GitHub Pages
workflow. The base documentation build must work with NumPy and the docs extra
without installing Torch or JAX. Test Torch and JAX examples independently.

Do not mock optional backends to conceal import defects, commit rendered HTML,
publish documentation, change GitHub Pages settings, push, or release without
explicit authorization. You may prepare and fully validate all local source and
workflow files.

Finish only when DOC-001 through DOC-018 are traceable to source, tests,
documentation, and CI; all strict HTML and doctest builds pass; every public
symbol is represented exactly once; and the final report records commands,
results, skips, limitations, and the one-time Pages action still requiring user
authorization.
```

## 10. Definition of done

The documentation system is complete only when:

- changing a public signature or docstring changes the generated API site;
- adding an undocumented public symbol fails CI;
- broken cross-references and Sphinx warnings fail CI;
- executable examples agree with the installed package;
- the NumPy-only build succeeds without Torch or JAX;
- optional backend documentation states exact installation and capability rules;
- each functionality-matrix row links to its documentation;
- the default-branch workflow can publish the tested artifact after Pages is
  explicitly enabled;
- no generated HTML is tracked in the source branch.

## 11. Primary references

- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [Sphinx autodoc](https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html)
- [Sphinx autosummary](https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html)
- [Sphinx Napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html)
- [Sphinx doctest](https://www.sphinx-doc.org/en/master/usage/extensions/doctest.html)
- [GitHub Pages custom workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)
