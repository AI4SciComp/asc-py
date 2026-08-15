# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Verify the canonical Sphinx toolchain, narratives, and CI policies."""

from __future__ import annotations

import pathlib
import re
import runpy
import tomllib
import typing

from docutils import nodes
from sphinx import addnodes

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_REQUIRED_EXTENSIONS = {
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
}
_NARRATIVES = (
    "getting-started/installation.md",
    "getting-started/quickstart.md",
    "user-guide/backend-selection.md",
    "user-guide/array-api.md",
    "user-guide/dtype-device.md",
    "user-guide/conversion.md",
    "user-guide/random.md",
    "user-guide/autodiff-jit.md",
    "user-guide/data.md",
    "tutorials/portable-computation.md",
    "tutorials/data-pipeline.md",
    "reference/support-matrix.md",
    "reference/functionality-matrix.md",
    "reference/exceptions.md",
    "reference/configuration.md",
    "development/architecture.md",
    "development/contributing.md",
    "development/testing.md",
    "development/documentation.md",
    "release/changelog.md",
    "release/migration.md",
    "release/release-notes.md",
)


def test_docs_extra_is_independent_and_canonical() -> None:
    project = tomllib.loads(
        (_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )["project"]
    docs = project["optional-dependencies"]["docs"]
    assert {requirement.split(">=")[0] for requirement in docs} == {
        "myst-parser",
        "pydata-sphinx-theme",
        "sphinx",
    }
    assert not any(
        "torch" in requirement or "jax" in requirement for requirement in docs
    )
    assert not (_ROOT / "mkdocs.yml").exists()


def test_sphinx_configuration_is_strict_and_import_honest() -> None:
    source = (_ROOT / "docs" / "conf.py").read_text(encoding="utf-8")
    namespace = runpy.run_path(str(_ROOT / "docs" / "conf.py"))
    assert set(namespace["extensions"]) == _REQUIRED_EXTENSIONS
    assert namespace["nitpicky"] is True
    assert namespace["autosummary_generate"] is True
    assert namespace["autosummary_ignore_module_all"] is False
    assert namespace["autodoc_typehints"] == "description"
    assert namespace["html_theme"] == "pydata_sphinx_theme"
    assert "sys.path" not in source
    assert "autodoc_mock_imports" not in source
    assert "nitpick_ignore" not in source


def test_private_qualified_annotations_are_resolved_as_signature_text() -> None:
    namespace = runpy.run_path(str(_ROOT / "docs" / "conf.py"))
    resolver = typing.cast(
        typing.Callable[..., nodes.Element | None],
        namespace["_resolve_public_alias"],
    )
    reference = addnodes.pending_xref()
    reference["refdomain"] = "py"
    reference["reftarget"] = "asc.data.sampler.Sampler"
    reference["refdoc"] = "api/generated/asc.data.DataLoader"
    content = nodes.inline("", "Sampler")

    assert resolver(None, None, reference, content) is content


def test_required_narratives_and_local_commands_exist() -> None:
    index = (_ROOT / "docs" / "index.md").read_text(encoding="utf-8")
    for relative in _NARRATIVES:
        assert (_ROOT / "docs" / relative).is_file()
        assert relative.rsplit(".", maxsplit=1)[0] in index
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "-W --keep-going -n" in makefile
    assert "-b html docs docs/_build/html" in makefile
    assert "-b doctest docs docs/_build/doctest" in makefile
    assert "-b linkcheck docs docs/_build/linkcheck" in makefile
    assert "scripts/check_docs_base.py" in makefile


def test_pages_workflow_is_gated_and_least_privilege() -> None:
    workflow = (_ROOT / ".github" / "workflows" / "docs.yml").read_text(
        encoding="utf-8"
    )
    assert "permissions:\n  contents: read" in workflow
    assert "pages: write" in workflow
    assert "id-token: write" in workflow
    assert "github.event_name == 'push'" in workflow
    assert "refs/heads/main" in workflow
    assert "github.event_name == 'schedule'" in workflow
    assert "workflow_dispatch:" in workflow
    assert "deploy-pages@" in workflow
    assert "configure-pages@" in workflow
    assert "upload-pages-artifact@" in workflow
    assert "make docs-linkcheck" in workflow
    assert "torch-examples:" in workflow
    assert "jax-examples:" in workflow
    assert "python scripts/check_docs_base.py" in workflow
    assert "cancel-in-progress: false" in workflow
    assert "path: docs/_build/html" in workflow
    actions = re.findall(r"uses: [^@\s]+@([^\s]+)", workflow)
    assert actions
    assert all(re.fullmatch(r"[0-9a-f]{40}", revision) for revision in actions)


def test_generated_documentation_is_ignored() -> None:
    ignored = (_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "docs/_build/" in ignored
    assert "docs/api/generated/" in ignored
    assert not tuple(
        path
        for path in (_ROOT / "docs").rglob("*.html")
        if "_build" not in path.parts
    )
