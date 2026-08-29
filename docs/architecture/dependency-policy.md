# Dependency Policy

Status: Frozen for 0.1.0 implementation. Last reviewed: 2026-08-11.

Runtime dependencies must implement a required public contract, remain
license-compatible with Apache-2.0 distribution, import safely, and stay inside
tested bounds. Runtime metadata uses bounded ranges; `uv.lock` records the
reproducible development environment.

## Installation Sets

- Base: NumPy `>=1.26.4,<2.6` and array-api-compat `>=1.13,<1.14`.
- `torch`: PyTorch `>=2.13,<2.14`, independent of JAX.
- `jax`: JAX/JAXlib `>=0.6,<0.11`, independent of PyTorch.
- `io-hdf5`: h5py, imported only by HDF5 functions.
- `io-mat`: SciPy, imported only by MATLAB functions.
- `docs`: Sphinx 9.1, MyST Parser 5.1, and PyData Sphinx Theme 0.19;
  independent of Torch and JAX.
- `all`: union of backend and I/O extras, with no additional behavior.

NumPy is always available. `import asc` must not import PyTorch, JAX, h5py,
SciPy, or array-api-strict. Missing optional features raise
`BackendUnavailableError` or `CapabilityNotSupportedError` and name the exact
installation extra.

array-api-compat 1.11 first targeted Array API 2024.12. The semantic floor is
1.13 because earlier PyTorch behavior failed the package's graph contract. The
range is capped below 1.14 because later compatibility namespaces target a
newer standard. NumPy 1.26 is supported through the compatibility layer; NumPy
2.x uses the same normalized consumer surface.

`array-api-strict`, pytest/Hypothesis, Ruff, Pylint, Pyright, benchmark, build,
and release-inspection tools are development-only. Sphinx, MyST, and the PyData
theme are isolated in the `docs` extra. Optional backend wheels must never leak
into base or documentation-only environments.

The `asc` import path is also owned by an unrelated PyPI distribution named
`asc`; the two distributions cannot safely coexist. This is an explicit
pre-release decision recorded in ADR 0006.

## Change Control

Adding or widening a dependency requires an ADR, minimum/maximum behavior and
license evidence, lazy-import tests, independent installation coverage, lock
regeneration, and wheel/sdist metadata inspection. Security constraints may
narrow a bound immediately with a changelog entry.

Primary evidence: the
[array-api-compat changelog](https://data-apis.org/array-api-compat/changelog.html),
[NumPy Array API statement](https://numpy.org/doc/stable/reference/array_api.html),
and standard [version specifier rules](https://packaging.python.org/en/latest/specifications/version-specifiers/).

## Direct dependency license review

Installed distribution metadata was inspected on 2026-08-11. The direct
runtime and optional dependencies are compatible with Apache-2.0 distribution;
asc-py does not vendor their source or binaries.

| Distribution | Current metadata finding |
|---|---|
| array-api-compat | MIT |
| NumPy | BSD-3-Clause plus listed permissive bundled notices |
| PyTorch | Apache-2.0 plus LLVM/BSD/Boost/MIT bundled notices |
| JAX/JAXlib | Apache-2.0 |
| h5py | BSD-3-Clause |
| SciPy | BSD-style license with bundled third-party notices |
| Sphinx | BSD-2-Clause |
| MyST Parser | MIT |
| PyData Sphinx Theme | BSD |

The wheel embeds asc-py's Apache-2.0 `LICENSE` and records the SPDX license
expression. Artifact inspection rejects a missing expression or license file.
