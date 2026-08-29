# Release Readiness: 0.1.0

Status: Local release candidate; remote CI and publication require separate
authorization. Evidence date: 2026-08-12.

## Delivered contract

The `asc-py` distribution installs the typed `asc` package. All 168 Required
IDs from the comprehensive and automatic-documentation runbooks are `Complete` in the
[functionality matrix](../specification/functionality-matrix.md), with existing
source, test, documentation, backend, and export evidence. NumPy is required;
Torch, JAX, HDF5, and MATLAB support remain lazy independent extras.

The implementation uses native arrays, an immutable explicit context, explicit
random state, functional updates, and named conversion/I/O boundaries. It has
no tensor wrapper, mutable global backend, global random generator, hidden
NumPy fallback, implicit device transfer, or automatic graph detachment. The
complete backend-neutral data package is included. Domain solvers, physics data
generation, models, training, optimization, and AutoML remain excluded.

## Exact local evidence

| Command/gate | Observed result |
|---|---|
| `uv run pytest` on CPython 3.12.4 | 703 passed; 90.34% branch coverage |
| `python scripts/check_dependency_floor.py` | 703 passed; 90.42% branch coverage |
| isolated CPython 3.13.15 suite | 366 passed |
| isolated CPython 3.14.7 suite | 366 passed |
| `JAX_ENABLE_X64=1 uv run pytest --no-cov` | 703 passed |
| isolated full backend profiles | Base 206, Torch-without-JAX 314, JAX-without-Torch 305; no skips |
| Ruff formatter and lint | Passed on 159 Python files |
| `pylint src/asc scripts docs/conf.py docs/_inventory.py` | Passed, 10.00/10 |
| strict-mode Pyright and pre-commit | 0 errors; all five hooks passed |
| `make docs` | Strict HTML passed with 0 warnings; all 35 doctests passed |
| `make docs-linkcheck` | Passed with 0 warnings |
| `make docs-base` | NumPy-only HTML/doctest/inventory tests passed with Torch/JAX absent |
| isolated Torch and JAX examples | One passed in each independent environment; opposite backend absent |
| CPython 3.13 and 3.14 docs profiles | 13 base docs tests passed on each interpreter |
| public API inventory | 23 module pages, 224 canonical symbol pages, and 86 documented aliases |
| Actionlint 1.7.7 and YAML parse | Documentation and CI workflows passed |
| portability and release audits | Passed; 16 portable-core files, 0 errors |
| wheel/sdist metadata and content inspection | Passed |
| isolated artifact installation profiles | Base wheel/sdist, Torch, JAX, HDF5, MAT, and all passed |
| `make benchmark` | Four backend cases passed |

No required test was unavailable or skipped. The NumPy-only docs selection
deselected two optional examples, which then passed in independent Torch-only
and JAX-only environments. The independent full-suite profiles deselect only
tests explicitly requiring an absent backend; the all-extras gate runs every
case. The release audit compares both runbooks with the ledger, rejects
incomplete/missing trace paths, and rejects any `pytest.skip` or skip marker in
the required suite.

The floor environment pins NumPy 1.26.4, array-api-compat 1.13.0,
array-api-strict 2.3.0, Torch 2.13.0+cpu, JAX/JAXlib 0.6.0, h5py 3.11.0, and
SciPy 1.13.0. The current environment uses NumPy 2.5.2, Torch 2.13.0+cpu,
JAX/JAXlib 0.10.2, h5py 3.16.0, and SciPy 1.18.0. Floor testing found and
regressed NumPy 1.26 DLPack keyword, SciPy 1.13 corrupt-MAT, and strict-oracle
scalar-kind compatibility defects.

## Benchmark observation

One steady-state smoke run over 4,096 float32 values observed mean
`sum_of_squares` times of 9.47 us (NumPy), 13.67 us (Torch), 50.01 us
(array-api-strict), and 66.32 us (JAX). Each case used 20 rounds of 10
iterations after warm-up; JAX was synchronized. These values validate the
benchmark harness only and make no comparative performance claim.

## Dependency and license findings

The built wheel uses interoperable Core Metadata 2.4, records `Apache-2.0`,
embeds `LICENSE`, includes `asc/py.typed`, and contains no obsolete `asc_py`
import tree. Optional dependencies are extra-marked in wheel metadata.
Installed direct dependency metadata reports
MIT for array-api-compat; BSD-3-Clause/permissive bundled notices for NumPy;
Apache/LLVM/BSD/Boost/MIT notices for Torch; Apache-2.0 for JAX/JAXlib;
BSD-3-Clause for h5py; and SciPy's BSD-style license plus bundled notices. No
dependency source or binary is vendored. The docs extra resolves Sphinx 9.1.0
(BSD-2-Clause), MyST Parser 5.1.0 (MIT), and PyData Sphinx Theme 0.19.0 (BSD).

## Known limitations

- The supported execution claim is dense CPU on Linux x86-64. Sparse, masked,
  nested, distributed, quantized, and accelerator arrays fail explicitly.
- Windows x86-64 and macOS arm64 workflows are provisional and have not run on
  hosted CI. Local evidence covers Linux only.
- JAX is the only JIT backend in 0.1.0; Torch JIT raises a capability error.
  JAX int64/uint64/float64/complex128 require x64 mode.
- Loading is single-process. Cross-backend random bitstream identity and
  cross-backend graph conversion are not promised.
- The unrelated PyPI distribution `asc` owns the same import path and cannot
  coexist safely with `asc-py` in one environment.
- Pyright runs in strict mode; dynamic Array API and optional-backend calls use
  documented diagnostic overrides because their runtime structural interfaces
  are not fully expressible to Pyright.

## External actions still requiring authorization

No commit, push, pull request, tag, publication, release, repository-setting
change, or DeepMLT modification was performed. Hosted CI, provisional-platform
evidence, signing, tagging, and package publication remain external authorized
actions. Documentation publication additionally requires the repository
variable `ASC_PY_PAGES_ENABLED=true` and the one-time administrator action
**Settings → Pages → Source → GitHub Actions**; neither opt-in was performed.
