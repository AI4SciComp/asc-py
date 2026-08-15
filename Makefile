# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0

.DEFAULT_GOAL := help
.ONESHELL:
SHELL := /bin/bash
.SHELLFLAGS := -eu -o pipefail -c

UV ?= uv
PYTHON ?= python

.PHONY: help
help:
	@echo "asc-py development commands"
	@echo "  make env          Sync every locked group and backend extra"
	@echo "  make format       Apply the Ruff-compatible formatter"
	@echo "  make lint         Run format, Ruff, Pylint, and core audit"
	@echo "  make typecheck    Run strict Pyright"
	@echo "  make test         Run pytest with branch coverage"
	@echo "  make docs         Build strict HTML and run doctests"
	@echo "  make docs-html    Build strict nitpicky Sphinx HTML"
	@echo "  make docs-doctest Execute Sphinx documentation examples"
	@echo "  make docs-base    Prove the docs in an isolated NumPy-only env"
	@echo "  make docs-linkcheck  Check external links separately"
	@echo "  make examples     Execute every documented example"
	@echo "  make floor        Test every direct dependency minimum"
	@echo "  make build        Build and inspect wheel and sdist"
	@echo "  make audit        Run repository contract audits"
	@echo "  make benchmark    Run reproducible benchmarks"
	@echo "  make check        Run the complete local release gate"

.PHONY: env
env:
	"$(UV)" sync --frozen --all-groups --all-extras

.PHONY: format
format:
	"$(UV)" run ruff check --fix src tests scripts benchmarks docs
	"$(UV)" run ruff format src tests scripts benchmarks docs

.PHONY: format-check
format-check:
	"$(UV)" run ruff format --check src tests scripts benchmarks docs

.PHONY: ruff
ruff:
	"$(UV)" run ruff check src tests scripts benchmarks docs

.PHONY: pylint
pylint:
	"$(UV)" run pylint src/asc scripts docs/conf.py docs/_inventory.py

.PHONY: lint
lint: format-check ruff pylint audit

.PHONY: typecheck
typecheck:
	"$(UV)" run pyright

.PHONY: test
test:
	"$(UV)" run pytest

.PHONY: docs docs-html docs-doctest docs-linkcheck
docs: docs-html docs-doctest

docs-html:
	"$(UV)" run --extra docs python -m sphinx -W --keep-going -n \
		-b html docs docs/_build/html

docs-doctest:
	"$(UV)" run --extra docs python -m sphinx -W --keep-going -n \
		-b doctest docs docs/_build/doctest

docs-linkcheck:
	"$(UV)" run --extra docs python -m sphinx -W --keep-going \
		-b linkcheck docs docs/_build/linkcheck

.PHONY: docs-base
docs-base:
	"$(UV)" run --isolated --python 3.12 --frozen --extra docs \
		--group test --no-default-groups python scripts/check_docs_base.py

.PHONY: examples
examples:
	"$(UV)" run pytest tests/contract/test_examples.py --no-cov

.PHONY: floor
floor:
	ASC_PY_UV="$(UV)" "$(UV)" run python scripts/check_dependency_floor.py

.PHONY: build
build:
	"$(UV)" run python -m build
	"$(UV)" run twine check dist/*
	"$(UV)" run check-wheel-contents dist/*.whl
	"$(UV)" run python scripts/check_artifacts.py
	ASC_PY_UV="$(UV)" "$(UV)" run python scripts/check_clean_install.py

.PHONY: audit
audit:
	"$(PYTHON)" scripts/audit_portable_core.py
	"$(UV)" run python scripts/check_docs_links.py
	"$(UV)" run python scripts/audit_release.py

.PHONY: benchmark
benchmark:
	"$(UV)" run pytest benchmarks --benchmark-only --no-cov

.PHONY: check
check: lint typecheck test docs docs-base examples build floor
