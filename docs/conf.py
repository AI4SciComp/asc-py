# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Sphinx configuration for the installed asc-py package."""

# Sphinx discovers configuration through these required lowercase names.
# pylint: disable=invalid-name

from __future__ import annotations

import importlib.metadata
import pathlib
import runpy
import typing

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.util.nodes import make_refnode

project = "asc-py"
author = "AI4SciComp contributors"
copyright = "2026, AI4SciComp contributors"  # pylint: disable=redefined-builtin
release = importlib.metadata.version("asc-py")
version = release
needs_sphinx = "9.1"

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
    "show-inheritance": True,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_attr_annotations = True

nitpicky = True
root_doc = "index"
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build"]
templates_path = ["_templates"]
myst_enable_extensions = ["colon_fence", "deflist", "fieldlist"]
doctest_global_setup = "import numpy as np"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
    "jax": ("https://docs.jax.dev/en/latest", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master", None),
}
intersphinx_timeout = 15

html_theme = "pydata_sphinx_theme"
html_title = "asc-py documentation"
html_baseurl = "https://ai4scicomp.github.io/asc-py/"
html_static_path = ["_static"]
html_logo = "_static/asc-logo.svg"
html_favicon = "_static/favicon.svg"
html_theme_options = {
    "github_url": "https://github.com/AI4SciComp/asc-py",
    "header_links_before_dropdown": 5,
    "show_toc_level": 2,
    "use_edit_page_button": True,
}
html_context = {
    "github_user": "AI4SciComp",
    "github_repo": "asc-py",
    "github_version": "main",
    "doc_path": "docs",
}

_DOCS = pathlib.Path(__file__).resolve().parent
_inventory = runpy.run_path(str(_DOCS / "_inventory.py"))
_public_exports = typing.cast(
    typing.Callable[[], tuple[object, ...]],
    _inventory["public_exports"],
)()
_documentation_targets = typing.cast(
    typing.Callable[[], tuple[str, ...]],
    _inventory["documentation_targets"],
)()
_documentation_aliases = typing.cast(
    typing.Callable[[], tuple[tuple[str, str], ...]],
    _inventory["documentation_aliases"],
)()
_write_inventory = typing.cast(
    typing.Callable[[pathlib.Path], pathlib.Path],
    _inventory["write_autosummary_inventory"],
)
_write_inventory(_DOCS / "api" / "generated")

_ALIASES = dict(_documentation_aliases)
_ALIASES.update((target, target) for target in _documentation_targets)
_suffixes: dict[str, set[str]] = {}
for _alias, _canonical in _ALIASES.items():
    _suffixes.setdefault(_alias.rsplit(".", maxsplit=1)[-1], set()).add(
        _canonical
    )
for _suffix, _canonicals in _suffixes.items():
    if len(_canonicals) == 1:
        _ALIASES[_suffix] = next(iter(_canonicals))

_PUBLIC_CLASS_IDENTITIES = {
    id(export.value) for export in _public_exports if export.kind == "class"
}


def _resolve_public_alias(
    app: Sphinx,
    _environment: BuildEnvironment,
    node: addnodes.pending_xref,
    content: nodes.Element,
) -> nodes.Element | None:
    """Resolve an intentional public alias to its one canonical API page."""
    if node.get("refdomain") != "py":
        return None
    target = typing.cast(str, node["reftarget"])
    if target.startswith("asc_typing."):
        target = target.replace("asc_typing.", "asc.typing.", 1)
    canonical = _ALIASES.get(target)
    if canonical is not None:
        return make_refnode(
            app.builder,
            node["refdoc"],
            f"api/generated/{canonical}",
            "",
            content,
            canonical,
        )
    # ParamSpecs and private implementation aliases are meaningful signature
    # text but intentionally have no public object page.
    if target in {
        "Mode",
        "P",
        "typing.P",
        "Sampler",
        "asc.data.sampler.Sampler",
    }:
        return content
    return None


def _hide_internal_bases(
    _app: Sphinx,
    _name: str,
    _value: object,
    _options: object,
    bases: list[type[object]],
) -> None:
    """Keep implementation-only asc base classes out of public inheritance."""
    bases[:] = [
        base
        for base in bases
        if not (
            base.__module__.startswith("asc.")
            and id(base) not in _PUBLIC_CLASS_IDENTITIES
        )
    ]


def setup(app: Sphinx) -> dict[str, bool]:
    """Register strict public-alias and inheritance handling."""
    app.connect("missing-reference", _resolve_public_alias)
    app.connect("autodoc-process-bases", _hide_internal_bases)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
