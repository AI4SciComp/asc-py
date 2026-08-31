# asc-py 0.1.0 ExecPlan

This is a living plan. Check boxes and the validation log must reflect observed
results, not intent.

## Goal and boundaries

Deliver a locally release-ready `asc-py` 0.1.0 implementing every Required row
in the comprehensive runbook and its automatic-documentation addendum. Use
native NumPy, PyTorch, and JAX
arrays; NumPy is required and the other backends are independent lazy extras.
Include the complete backend-neutral data package. Exclude PDE/domain solvers,
physics-specific generation, models, neural operators, training, optimizers,
and AutoML.

Commit and push operations require explicit user authorization. The current
CI-remediation commit and push to the existing pull-request branch were
authorized on 2026-08-16. No tag, release, package publication,
remote/repository setting change, or DeepMLT modification was authorized in
that work session. On 2026-08-31, the user separately authorized repairing and
publishing the missing tag, GitHub Release, and PyPI release.

## Frozen decisions

- Distribution/import/version: `asc-py` / `asc` / `0.1.0`.
- License: Apache-2.0, based on repository and AI4SciComp sibling evidence.
- Standard: Python Array API 2024.12 (ADR 0002, revalidated 2026-08-11).
- Architecture: native arrays, immutable explicit contexts, no global backend,
  no hidden NumPy fallback, explicit conversion/random/update boundaries.
- Execution claim: dense CPU on Linux x86-64; other platforms provisional.
- Toolchain: Ruff's Pyink-compatible formatter, Pylint, strict Pyright,
  pytest/Hypothesis, Sphinx/PyData/MyST,
  Hatchling, uv, pre-commit, and Make.

## Milestones

- [x] Discovery and contract reconciliation.
  - [x] Read `AGENTS.md` and both normative runbooks in full.
  - [x] Inspect Git state, repository layout, existing implementation/tests,
    packaging, docs, CI, and prior local changes.
  - [x] Recreate a read-only DeepMLT inventory at commit `7dab7906`.
  - [x] Revalidate Array API 2024.12 at dependency floors and ceilings.
  - [x] Run the discovery portability audit (3 files, 0 errors).
  - [x] Add every Section 4 ID to the functionality ledger.
  - [x] Update architecture, portability, public API, dependency, support,
    DeepMLT audit, ADR, `AGENTS.md`, and this plan.
- [x] Repository foundation and backend/context contract (B*, A*, Q*).
- [x] Linalg, FFT, portable operations, and metrics (L*, F*, E*).
- [x] Conversion, updates, random, autodiff, and compilation (C*, U*, R*,
  D*, J*).
- [x] PyTrees and array-tree helpers (T*).
- [x] Datasets, schemas, splits, samplers, collation, and loaders (DS*, S*,
  CO*, DL*).
- [x] CombinedLoader, DataModule, transforms, statistics, and I/O (CL*, DM*,
  P*, IO*).
- [x] Complete conformance/parity/property/integration tests, examples,
  documentation, benchmarks, CI, and public traceability.
- [x] Independent base/Torch/JAX/HDF5/MAT/all installs, floor/ceiling and Python
  matrix, artifacts, final audit, and adversarial release report.
- [x] Automatic documentation addendum.
  - [x] DOC-001 Sphinx project and DOC-002 independent docs extra.
  - [x] DOC-003 Google docstrings and DOC-006 typed signature rendering.
  - [x] DOC-004 generated API and DOC-005 exact public inventory.
  - [x] DOC-007 strict references and DOC-010 scientific semantics.
  - [x] DOC-008 executable examples and DOC-009 optional-backend safety.
  - [x] DOC-011 data guides and DOC-012 compatibility matrices.
  - [x] DOC-013 local commands and DOC-014 pull-request gates.
  - [x] DOC-015 Pages workflow and DOC-016 scheduled link checking.
  - [x] DOC-017 release narratives and DOC-018 complete traceability.

Milestones are sequencing only. Work continues to the next incomplete ID after
each internal gate.

## Discovery evidence

- Initial repository has one commit (`4af647e`) and extensive uncommitted
  user-directed work; it is preserved and not committed.
- AI4SciComp sibling license evidence and the `asc` import collision are
  retained in ADRs 0005 and 0006.
- Array API floor probes: array-api-compat 1.13.0 with NumPy 1.26.4 and JAX
  0.6.0 returned 2024.12 namespaces and exposed every required main/linalg/FFT
  symbol. Current probes passed with NumPy 2.5.2, PyTorch 2.13.0, and JAX 0.10.2.
- DeepMLT's data concepts are adopted through a clean-room redesign; its
  mutable dispatch, wrappers, implicit conversion, global RNG, and domain
  modules are rejected.
- The automatic-documentation runbook's style snapshot was compared with the
  current official Google Python Style Guide on 2026-08-11. Both require typed
  semantic docstrings with punctuated summaries and standard sections.
- Current primary package metadata resolves Sphinx 9.1.0, MyST Parser 5.1.0,
  and PyData Sphinx Theme 0.19.0 across Python 3.12-3.14. Official Pages action
  tags were resolved to immutable commit SHAs before writing the workflow.

## Validation log

| Date | Command | Result |
|---|---|---|
| 2026-08-11 | `git status --short --branch` | One modified README and repository implementation untracked; preserved. |
| 2026-08-11 | `python scripts/audit_portable_core.py` | Pass: 3 files, 0 errors. |
| 2026-08-11 | isolated NumPy 1.26.4 / JAX 0.6.0 symbol probes | Pass: 2024.12 main, linalg, FFT inventory present. |
| 2026-08-11 | current NumPy/Torch/JAX symbol probe | Pass at NumPy 2.5.2, Torch 2.13.0, JAX 0.10.2. |
| 2026-08-11 | `uv run pytest` | 366 passed; no skips; 90.99% branch coverage. |
| 2026-08-11 | isolated direct dependency floors | 381 passed; 91.09% branch coverage, including docs floors. |
| 2026-08-11 | isolated CPython 3.13.15 / 3.14.7 | 366 passed on each interpreter. |
| 2026-08-11 | `JAX_ENABLE_X64=1 uv run pytest --no-cov` | 366 passed. |
| 2026-08-11 | Ruff, Pylint, Pyright, pre-commit | Passed; Pylint 10.00/10 and Pyright 0 errors. |
| 2026-08-11 | initial strict Sphinx HTML/linkcheck | Failed: 517 unresolved duplicate/alias member references; replaced duplicate rendering with canonical pages and explicit public-alias resolution. |
| 2026-08-11 | `make docs-html` | Passed with warnings as errors and nitpicky references; 0 warnings. |
| 2026-08-11 | `make docs-doctest` | Passed: 35 examples, 0 failures. |
| 2026-08-11 | `make docs-linkcheck` | Passed after strict reference repair: 0 warnings. |
| 2026-08-11 | `make docs-base` | Passed in an isolated NumPy/docs/test environment with Torch and JAX absent: HTML, 35 doctests, and 13 selected tests. |
| 2026-08-11 | isolated Torch/JAX docs examples | Each passed independently; the opposite backend was absent. |
| 2026-08-11 | isolated CPython 3.13/3.14 docs tests | 13 base docs tests passed on each; two backend examples were selected into their independent jobs. |
| 2026-08-11 | `uv run pytest` | 381 passed; no skips; 91.01% branch coverage. |
| 2026-08-11 | Ruff, Pylint, Pyright, pre-commit | 155 files formatted; Ruff passed; Pylint 10.00/10; Pyright 0 errors; five hooks passed. |
| 2026-08-11 | Actionlint 1.7.7 and Ruby YAML parse | Both workflow validation gates passed. |
| 2026-08-11 | `make build` | Wheel/sdist inspection and base wheel/sdist plus Torch/JAX/HDF5/MAT/all clean installs passed. |
| 2026-08-11 | `make benchmark` | Four backend smoke benchmarks passed; no performance claim. |
| 2026-08-11 | isolated artifact profiles | Base wheel/sdist plus Torch, JAX, HDF5, MAT, and all wheel profiles passed. |
| 2026-08-11 | focused adversarial regressions | 64 passed for transformation, conversion, data, random, padding, and CI findings. |
| 2026-08-11 | `make lint` and `uv run pyright` | Ruff passed; Pylint 10.00/10; audits passed; Pyright 0 errors. |
| 2026-08-11 | `make test` | 389 passed; no skips; 90.82% branch coverage. |
| 2026-08-11 | `make docs` | Strict HTML and all 35 doctests passed with warnings as errors. |
| 2026-08-11 | initial post-review `make build` | Failed because Hatchling 1.32 emitted Core Metadata 2.5, unsupported by the locked Twine inspector. |
| 2026-08-11 | `make build` with Core Metadata 2.4 | Artifact inspection and base wheel/sdist plus Torch/JAX/HDF5/MAT/all clean installs passed. |
| 2026-08-11 | `make floor` | 389 passed at every direct minimum; no skips; 90.89% branch coverage. |
| 2026-08-11 | Torch/capability/dataclass/loader regressions | 49 focused tests passed for all six review findings. |
| 2026-08-11 | `make lint` and `uv run pyright` | Ruff passed; Pylint 10.00/10; audits passed; Pyright 0 errors. |
| 2026-08-11 | `make test` | 396 passed; no skips; 90.85% branch coverage. |
| 2026-08-11 | `make docs` | Strict HTML and all 35 doctests passed with warnings as errors. |
| 2026-08-11 | `make floor` | 396 passed at every direct minimum; no skips; 90.93% branch coverage. |
| 2026-08-11 | `make build` | Core Metadata 2.4 wheel/sdist inspection and all isolated install profiles passed. |
| 2026-08-11 | nine-review-finding focused regressions | 130 backend, conversion, FFT, transform, sampler, and persistence tests passed. |
| 2026-08-11 | `make lint` and `uv run pyright` | Ruff passed; Pylint 10.00/10; all three audits passed; Pyright 0 errors. |
| 2026-08-11 | `make test` | 405 passed; no skips; 90.98% branch coverage. |
| 2026-08-11 | initial post-review `make floor` | Failed only because the new regression used the unavailable NumPy 1.26 `np.bool` alias; replaced it with stable `np.bool_`. |
| 2026-08-11 | `make floor` after test compatibility repair | 405 passed at every direct minimum; no skips; 91.05% branch coverage. |
| 2026-08-11 | `make docs` and `make docs-base` | Strict HTML and 35 doctests passed; isolated NumPy-only docs and 13 selected tests passed. |
| 2026-08-11 | `make build` | Core Metadata 2.4 wheel/sdist inspection and base wheel/sdist plus Torch/JAX/HDF5/MAT/all clean installs passed. |
| 2026-08-11 | `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run pytest --no-cov` | 405 passed; no skips. |
| 2026-08-11 | `uv run pre-commit run --all-files` | All five format, Ruff, Pylint, strict Pyright, and portability-audit hooks passed. |
| 2026-08-11 | seven-review-finding regression suite | FFT device overrides, Torch lazy DLPack views, linearithmic JAX duplicate checks, exact integer/empty CSV policy, mapping-type preservation, and CI profile assertions passed. |
| 2026-08-11 | isolated base/Torch/JAX full test profiles | Base: 206 passed/130 backend cases deselected; Torch without JAX: 314 passed/64 deselected; JAX without Torch: 305 passed/73 deselected; no skips. |
| 2026-08-11 | `make lint typecheck` | Ruff and formatting passed; Pylint 10.00/10; three audits passed; strict Pyright reported 0 errors. |
| 2026-08-11 | `make test` | 422 passed; no skips; 90.86% branch coverage. |
| 2026-08-11 | `make floor` | 422 passed at every direct minimum; no skips; 90.94% branch coverage. |
| 2026-08-11 | `make docs` | Strict HTML passed with no warnings; all 35 doctests passed. |
| 2026-08-11 | Actionlint 1.7.7 and YAML parsers | Updated independent-backend CI workflow passed. |
| 2026-08-11 | `make build` | Core Metadata 2.4 wheel/sdist inspection and base wheel/sdist plus Torch/JAX/HDF5/MAT/all clean installs passed. |
| 2026-08-11 | `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run pytest --no-cov` | 422 passed; no skips. |
| 2026-08-11 | final `uv run pre-commit run --all-files` | All five hooks passed. |
| 2026-08-12 | seven-finding adversarial review regressions | 138 focused random, sampler, persistence, activation, and PyTree tests passed. |
| 2026-08-12 | `make lint typecheck` | Formatting and Ruff passed; Pylint 10.00/10; portability, documentation-link, and release audits passed; strict Pyright reported 0 errors. |
| 2026-08-12 | `make test` | 482 passed; no skips; 90.79% branch coverage. |
| 2026-08-12 | `make docs` | Strict warnings-as-errors HTML passed; all 35 doctests passed. |
| 2026-08-12 | `make build` | Wheel/sdist metadata and contents passed; isolated base wheel/sdist and Torch/JAX/HDF5/MAT/all wheel installs passed. |
| 2026-08-12 | eight-finding review regression suite | 160 focused conversion, PyTree, dataset, collation, HDF5, and random tests passed. |
| 2026-08-12 | `make lint typecheck` | 156 files formatted; Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors. |
| 2026-08-12 | `make test` | 495 passed; no skips; 90.48% branch coverage. |
| 2026-08-12 | `make docs` | Strict warnings-as-errors HTML passed; all 35 doctests passed. |
| 2026-08-12 | `make build` | Wheel/sdist inspection and isolated base wheel/sdist plus Torch/JAX/HDF5/MAT/all wheel installs passed. |
| 2026-08-12 | `make floor` | 495 passed at every direct minimum; no skips; 90.55% branch coverage. |
| 2026-08-12 | eight-review-finding regression suite | 71 focused scalar-representability, legacy-array, probability, Torch-vmap, collation, and schema tests passed. |
| 2026-08-12 | `make lint typecheck` | 157 files formatted; Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors. |
| 2026-08-12 | `make test` | 507 passed; no skips; 90.17% branch coverage. |
| 2026-08-12 | `make docs` | Strict warnings-as-errors HTML passed; all 35 doctests passed. |
| 2026-08-12 | `make build` | Wheel/sdist inspection and isolated base wheel/sdist plus Torch/JAX/HDF5/MAT/all wheel installs passed. |
| 2026-08-12 | `make floor` | 507 passed at every direct minimum; no skips; 90.23% branch coverage. |
| 2026-08-12 | nine-review-finding regression cycle | Added coverage for scalar-only random/activation parameters, scaler shape preservation, complete uncollation extent checks, JAX JVP/VJP operands, duplicate NPZ names, counter exhaustion, custom-node paths, stable zero-slope activation limits, and non-numeric array boundaries. |
| 2026-08-12 | `make lint typecheck` | 158 files formatted; Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors. |
| 2026-08-12 | `make test` | 523 passed; no skips; 90.10% branch coverage. |
| 2026-08-12 | `make docs` | Strict warnings-as-errors HTML passed; all 35 doctests passed. |
| 2026-08-12 | `make build` | Wheel/sdist inspection and isolated base wheel/sdist plus Torch/JAX/HDF5/MAT/all wheel installs passed. |
| 2026-08-12 | `make floor` | 523 passed at every direct minimum; no skips; 90.16% branch coverage. |
| 2026-08-12 | four-finding review regression cycle | Added exact NumPy-scalar collation, discrete fill, portable weighted-probability, complete-gate, scalar-overflow, and exact CPU-device coverage. |
| 2026-08-12 | `make lint typecheck` | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors. |
| 2026-08-12 | `make test` | 534 passed; no skips; 90.09% branch coverage. |
| 2026-08-12 | `make docs` and `make docs-base` | Strict HTML and all 35 doctests passed; isolated NumPy-only docs and 13 selected tests passed. |
| 2026-08-12 | `make build` | Wheel/sdist inspection and isolated base wheel/sdist plus Torch/JAX/HDF5/MAT/all wheel installs passed. |
| 2026-08-12 | `make floor` | 534 passed at every direct minimum; no skips; 90.16% branch coverage. |
| 2026-08-12 | second independent review | Found nine P2 boundary defects in scalar numerical parameters, raw DLPack provenance, distribution parameter representability, splits, collation, scalers, and CSV delimiters; all received focused source and regression fixes. |
| 2026-08-12 | second-review focused regressions | 153 numerical, conversion, autodiff, data, and persistence tests passed without skips. |
| 2026-08-12 | `make lint typecheck test` | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors; 549 tests passed with 90.09% branch coverage. |
| 2026-08-12 | `make docs docs-base` | Strict warnings-as-errors HTML and all 35 doctests passed; isolated NumPy-only documentation and 13 selected tests passed. |
| 2026-08-12 | `make build` | Wheel/sdist inspection and isolated base wheel/sdist plus Torch/JAX/HDF5/MAT/all wheel installs passed. |
| 2026-08-12 | `make floor` | 549 passed at every direct minimum; no skips; 90.15% branch coverage. |
| 2026-08-12 | third independent review | Found one P1 and nine P2 defects in DLPack no-copy enforcement, numerical scalar representability, tree-spec validation, and long-kernel valid convolution; all received source and regression fixes. |
| 2026-08-12 | third-review focused regressions | 192 conversion, numerical, random, transform, tree, and parity tests passed without skips. |
| 2026-08-12 | `make lint typecheck test` after third review | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors; 588 tests passed with 90.11% branch coverage. |
| 2026-08-12 | `make docs docs-base` after third review | Strict warnings-as-errors HTML and all 35 doctests passed; isolated NumPy-only documentation and 13 selected tests passed. |
| 2026-08-12 | `make floor` after DLPack floor repair | 588 tests passed against every direct minimum, including NumPy 1.26.4, JAX 0.6.0, and Torch 2.13.0; no skips; 90.02% branch coverage. |
| 2026-08-12 | `make build` after third review | Wheel/sdist inspection and isolated base wheel/sdist plus Torch/JAX/HDF5/MAT/all wheel installs passed. |
| 2026-08-12 | fourth independent review | Found two P1 and four P2 defects in JAX trace-safe scalar validation, floating fill underflow, FFT reciprocal overflow, negative-step Torch dataset slices, and leafless collation; all received source and regression fixes. |
| 2026-08-12 | fourth-review focused regressions | 208 random, FFT, activation, dataset, collation, and parity tests passed without skips. |
| 2026-08-12 | `make lint typecheck test` after fourth review | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors; 605 tests passed with 90.10% branch coverage. |
| 2026-08-12 | `make docs docs-base` after fourth review | Strict warnings-as-errors HTML and all 35 doctests passed; isolated NumPy-only documentation and 13 selected tests passed. |
| 2026-08-12 | `make floor` after fourth review | 605 tests passed against every direct minimum, including NumPy 1.26.4, JAX 0.6.0, and Torch 2.13.0; no skips; 90.19% branch coverage. |
| 2026-08-12 | `make build` and all-files pre-commit after fourth review | Wheel/sdist inspection and isolated base wheel/sdist plus Torch/JAX/HDF5/MAT/all wheel installs passed; all five pre-commit hooks passed. |
| 2026-08-12 | fifth independent review | Found three P1 and eight P2 defects in Torch CPU random allocation, explicit array conversion and DLPack graph boundaries, random splitting, FFT/metric stability, inferred dtypes, numeric persistence, dataset-axis validation, and `TreeSpec` equality; all received source, regression, and documentation fixes. |
| 2026-08-12 | fifth-review focused regressions | 251 backend, conversion, random, FFT, metric, data, persistence, and PyTree tests passed without skips. |
| 2026-08-12 | `make test` after fifth review | 629 tests passed without skips and with 90.14% branch coverage. |
| 2026-08-12 | `make lint typecheck docs` after fifth review | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors; warnings-as-errors HTML and all 35 doctests passed. |
| 2026-08-12 | `make docs-base` after fifth review | Isolated NumPy-only documentation HTML, doctests, and selected tests passed without Torch or JAX installed. |
| 2026-08-12 | `make floor` after fifth review | 629 tests passed against every direct minimum, including NumPy 1.26.4, JAX 0.6.0, and Torch 2.13.0; no skips; 90.23% branch coverage. |
| 2026-08-12 | `make build` after fifth review | Wheel/sdist inspection and isolated base wheel/sdist plus Torch/JAX/HDF5/MAT/all wheel installs passed. |
| 2026-08-12 | all-files pre-commit after fifth review | All five format, Ruff, Pylint, strict Pyright, and portability-audit hooks passed. |
| 2026-08-12 | `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu uv run pytest --no-cov -q` after fifth review | All 629 tests passed with JAX 64-bit support active on CPU. |
| 2026-08-12 | sixth independent review pair | Two concurrent reviewers found three P1 and thirteen P2 defects in construction boundaries, integer convolution, compatibility random bounds, schema/transform structure, NPZ names, and extreme-value scaler/statistics stability; all received source, regression, and documentation fixes. |
| 2026-08-12 | sixth-review focused regressions | 172 backend, signal, random, schema, dataset, transform, statistics, and persistence tests passed without skips. |
| 2026-08-12 | `make lint typecheck test` after sixth review | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors; 647 tests passed without skips and with 90.06% branch coverage. |
| 2026-08-12 | `make docs docs-base` after sixth review | Strict warnings-as-errors HTML and all 35 doctests passed; isolated NumPy-only documentation and selected tests passed. |
| 2026-08-12 | `make floor` after NumPy 1.26 scalar-promotion repair | 647 tests passed against every direct minimum; no skips; 90.15% branch coverage. Scale state and count arithmetic preserve float32 under NumPy 1.26.4. |
| 2026-08-12 | `make build` after sixth review | Wheel/sdist inspection and isolated base wheel/sdist plus Torch/JAX/HDF5/MAT/all wheel installs passed. |
| 2026-08-12 | all-files pre-commit and JAX x64 after sixth review | All five hooks passed; all 647 tests passed with JAX 64-bit support active on CPU. |
| 2026-08-12 | seventh independent review | Found one P1 and four P2 defects in low-precision streaming counts, `eigh` triangle selection, JAX `vmap` error translation, random-state validation order, and portable-audit scope; all received source, regression, and documentation fixes. |
| 2026-08-12 | seventh-review focused regressions | Seven triangle-selection, float16-statistics, JAX-vmap, random-state, and audit-scope tests passed without skips. The expanded audit checks 16 files across `core`, `fft`, `linalg`, `metrics`, and `ops`. |
| 2026-08-12 | `make lint typecheck test` after seventh review | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors; 654 tests passed without skips and with 90.15% branch coverage. |
| 2026-08-12 | `make docs docs-base` after seventh review | Strict warnings-as-errors HTML and all 35 doctests passed; isolated NumPy-only documentation and 13 selected tests passed. |
| 2026-08-12 | `make floor` after seventh review | 654 tests passed against every direct minimum, including NumPy 1.26.4, JAX 0.6.0, and Torch 2.13.0; no skips; 90.23% branch coverage. |
| 2026-08-12 | `make build` after seventh review | Wheel/sdist inspection and isolated base wheel/sdist plus Torch/JAX/HDF5/MAT/all wheel installs passed. |
| 2026-08-12 | all-files pre-commit and JAX x64 after seventh review | All five hooks passed; all 654 tests passed with JAX 64-bit support active on CPU. |
| 2026-08-12 | eighth independent review | Found two P1 and six P2 defects in linalg operand coercion, Boolean padding, stable metrics/comparisons, HDF5 error translation, and orthogonal CPU QR dtypes; all received source, regression, and documentation fixes. |
| 2026-08-12 | eighth-review focused regressions | 186 linalg, padding, metric, comparison, initializer, HDF5, and numerical parity tests passed without skips. |
| 2026-08-12 | `make lint typecheck test` after eighth review | Formatting and Ruff passed; Pylint 10.00/10; expanded portability and release audits passed; strict Pyright reported 0 errors; 668 tests passed without skips and with 90.06% branch coverage. |
| 2026-08-12 | `make docs docs-base` after eighth review | Strict warnings-as-errors HTML and all 35 doctests passed; isolated NumPy-only documentation and 13 selected tests passed without Torch or JAX. |
| 2026-08-12 | `make floor` after eighth review | 668 tests passed against every direct minimum, including NumPy 1.26.4, JAX 0.6.0, and Torch 2.13.0; no skips; 90.14% branch coverage. |
| 2026-08-12 | `make build` after eighth review | Wheel/sdist inspection and isolated base wheel/sdist plus Torch/JAX/HDF5/MAT/all wheel installs passed. |
| 2026-08-12 | all-files pre-commit and JAX x64 after eighth review | All five hooks passed; all 668 tests passed with JAX 64-bit support active on CPU. |
| 2026-08-12 | ninth independent review | Found four P2 defects in extreme residual/comparison arithmetic, serialized-schema validation, and nested empty collation; all received source, regression, and documentation fixes. |
| 2026-08-12 | ninth-review focused regressions | 225 metric, comparison, schema, collation, loader, data, and parity tests passed without skips. |
| 2026-08-12 | `make lint typecheck test` after ninth review | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors; 670 tests passed without skips and with 90.08% branch coverage. |
| 2026-08-12 | `make docs docs-base` after ninth review | Strict warnings-as-errors HTML and all 35 doctests passed; isolated NumPy-only documentation and 13 selected tests passed without Torch or JAX. |
| 2026-08-12 | `make floor` after ninth review | 670 tests passed against every direct minimum, including NumPy 1.26.4, JAX 0.6.0, and Torch 2.13.0; no skips; 90.16% branch coverage. |
| 2026-08-12 | `make build` after ninth review | Wheel/sdist inspection and isolated base wheel/sdist plus Torch/JAX/HDF5/MAT/all wheel installs passed. |
| 2026-08-12 | all-files pre-commit and JAX x64 after ninth review | All five hooks passed; all 670 tests passed with JAX 64-bit support active on CPU. |
| 2026-08-12 | tenth independent review | Found three P1 and four P2 defects in extreme R2/scaler arithmetic, numeric CSV delimiters, NumPy byte-order conversion, NPZ names, and option-record validation; all received source, regression, and documentation fixes. |
| 2026-08-12 | tenth-review focused regressions | 17 extreme metric/scaler, conversion, persistence, and configuration tests passed without skips. |
| 2026-08-12 | `make lint typecheck test` after tenth review | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors; 686 tests passed without skips and with 90.14% branch coverage. |
| 2026-08-12 | `make docs docs-base examples` after tenth review | Strict warnings-as-errors HTML and all 35 doctests passed; isolated NumPy-only documentation and 13 selected tests passed; all five executable examples passed. |
| 2026-08-12 | `make floor` after tenth review | 686 tests passed against every direct minimum, including NumPy 1.26.4, JAX 0.6.0, and Torch 2.13.0; no skips; 90.22% branch coverage. |
| 2026-08-12 | `make build` after tenth review | Wheel/sdist inspection and isolated base wheel/sdist plus Torch/JAX/HDF5/MAT/all wheel installs passed. |
| 2026-08-12 | all-files pre-commit and JAX x64 after tenth review | All five hooks passed; all 686 tests passed with JAX 64-bit support active on CPU. |
| 2026-08-12 | eleventh independent review | Found one P1 and two P2 defects in float16 metric accumulation, closeness-boundary precision, and nested empty-list uncollation; all received source, regression, and documentation fixes. |
| 2026-08-12 | eleventh-review focused regressions | All 194 affected metric, comparison, collation, and data tests passed without skips. |
| 2026-08-12 | `make lint typecheck test` after eleventh review | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors; 692 tests passed without skips and with 90.32% branch coverage. |
| 2026-08-12 | `make docs docs-base examples` after eleventh review | Strict warnings-as-errors HTML and all 35 doctests passed; isolated NumPy-only documentation and 13 selected tests passed; all five executable examples passed. |
| 2026-08-12 | `make floor` and JAX x64 after eleventh review | All 692 tests passed at every direct minimum with 90.40% branch coverage and with JAX 64-bit support active on CPU. |
| 2026-08-12 | twelfth independent review | Found three P2 defects in mixed-tolerance comparison boundaries, sparse float16 metric reductions, and sparse float16 scaler moments; all received source, regression, and documentation fixes. |
| 2026-08-12 | twelfth-review focused regressions and adjacent probes | Exact stored-value comparison fuzzing found no mismatches across float16/float32/float64; sparse 33,554,432-element metric/scaler cases passed; mixed-dtype and Boolean `kron` plus unrepresentable FFT-bin tests passed on NumPy/Torch/JAX. |
| 2026-08-12 | `make lint typecheck test` after twelfth review | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors; 703 tests passed without skips and with 90.34% branch coverage. |
| 2026-08-12 | `make docs docs-base examples` after twelfth review | Strict warnings-as-errors HTML and all 35 doctests passed; isolated NumPy-only documentation and 13 selected tests passed; all five executable examples passed. |
| 2026-08-12 | `make floor`, `make build`, pre-commit, and JAX x64 after twelfth review | All 703 tests passed at every direct minimum with 90.42% branch coverage and with JAX 64-bit support active on CPU; wheel/sdist inspection, all isolated install profiles, and all five hooks passed. |
| 2026-08-12 | thirteenth independent review | Found four P2 defects in low-precision moving means, schema-inference inspection limits, Boolean signal axes, and schema string metadata; all received source and regression fixes. |
| 2026-08-12 | thirteenth-review focused regressions | All 87 affected signal, schema, dataset, and parity tests passed without skips. |
| 2026-08-12 | `make lint typecheck test` after thirteenth review | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors; 731 tests passed without skips and with 90.27% branch coverage. |
| 2026-08-12 | fourteenth independent review | Found five P2 defects in NPZ member aliases, scalar padding, metric overflow reconstruction, and probability-total validation; all received source and regression fixes. |
| 2026-08-12 | fourteenth-review focused regressions | All 236 affected persistence, padding, metric, and random tests passed without skips. |
| 2026-08-12 | `make lint typecheck test` after fourteenth review | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors; 739 tests passed without skips and with 90.30% branch coverage. |
| 2026-08-12 | fifteenth independent review | Found three P2 defects in exact tolerance-boundary comparisons, Boolean Torch `einsum`, and MAT shape metadata; all received source and regression fixes. Adjacent MAT dtype metadata validation was tightened as well. |
| 2026-08-12 | fifteenth-review focused regressions | All 251 affected comparison, linalg, persistence, and parity tests passed without skips; strict Pyright reported 0 errors for the affected files. |
| 2026-08-12 | `make lint typecheck test` after fifteenth review | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors; 752 tests passed without skips and with 90.35% branch coverage. |
| 2026-08-12 | sixteenth independent review | Found five P2 defects in MAT lossless restoration/schema errors, JAX max-plus-one integer bounds, and truncated NPY/NPZ parser errors; all received source and regression fixes. Adjacent probability-entry and malformed-stream resource validation was tightened as well. |
| 2026-08-12 | sixteenth-review focused regressions | All 258 affected persistence and random tests passed without skips; strict Pyright reported 0 errors for the affected files. JAX endpoint regressions also passed with 64-bit support enabled. |
| 2026-08-12 | `make lint typecheck test` after sixteenth review | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors; 764 tests passed without skips and with 90.36% branch coverage. |
| 2026-08-12 | seventeenth independent review | Found one P1 and two P2 defects in JAX-traced `ArrayDataset` bounds, case-insensitive numeric CSV delimiters, and empty metric reductions; all received source, regression, and documentation fixes. Adjacent empty CSV-header and complex MAT component round-trip validation was tightened as well. |
| 2026-08-12 | seventeenth-review focused regressions | All 252 affected dataset, persistence, configuration, metric, and review-regression tests passed without skips; strict Pyright reported 0 errors for the affected files. Invalid JAX JIT and vmap dataset indices both translated to stable `IndexError`s. |
| 2026-08-12 | `make lint typecheck test` after seventeenth review | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors; 784 tests passed without skips and with 90.58% branch coverage. |
| 2026-08-12 | eighteenth independent review | Found one P1 and three P2 defects in direct JAX namespace dtype validation, complex comparison residuals, Torch max-plus-one `int64` sampling, and persistence input coercion; all received source, regression, and documentation fixes. |
| 2026-08-12 | eighteenth-review focused regressions and stress probes | All seven focused cases passed; strict Pyright reported 0 errors for affected files. A randomized 360-case-per-backend complex64 comparison probe matched direct stored-value inequalities without warnings. |
| 2026-08-12 | `make lint typecheck test` after eighteenth review | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors; 790 tests passed without skips and with 90.67% branch coverage. |
| 2026-08-12 | nineteenth independent review | Found two P1 and two P2 defects in direct JAX reduction dtypes, linalg keyword arrays, Boolean metric axes, and `DataSpec` path validation; all received source, regression, and documentation fixes. Adjacent JAX `argsort` output-dtype narrowing was closed as well. |
| 2026-08-12 | nineteenth-review focused regressions | All 257 affected linalg, metric, schema, parity, and boundary tests passed without skips; targeted Ruff and strict Pyright reported no issues. |
| 2026-08-12 | `make lint typecheck test` after nineteenth review | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors; 791 tests passed without skips and with 90.70% branch coverage. |
| 2026-08-12 | twentieth independent review | Found four P2 defects in exact backend dtype boundaries, Torch negative-low max-endpoint sampling, named-tuple integer-path replacement, and public record validation; all received source, regression, and documentation fixes. NumPy namespace discovery and persistence dtype enforcement were tightened on the same boundary. |
| 2026-08-12 | twentieth-review adjacent probes and final review | Statistical probing found and repaired the analogous JAX `int32` negative-low endpoint bias with JIT-compatible rejection sampling. Focused dtype, namespace, persistence, tree, record, and random tests passed, and the final scoped `/review` returned no findings. |
| 2026-08-12 | `make lint typecheck test` after twentieth review | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors; 796 tests passed without skips and with 90.80% branch coverage. |
| 2026-08-12 | twenty-first iterative review | Repaired active JAX-x64 array/context discovery, exact random dtype provenance, portable reduction controls, genuine-native array discovery, and explicit NumPy/DLPack conversion ownership, normalization, dtype, device, hostile-metadata, and pre-consumption validation. Every follow-up finding received a focused regression before the affected scope was reviewed again. |
| 2026-08-12 | twenty-first final scoped reviews | Conversion/DLPack, reduction controls, dtype/random portability, and the final NumPy-scalar/impostor boundary each returned `No actionable issues were found`; focused conversion, namespace, reduction, random, data, and autodiff regressions passed. |
| 2026-08-12 | `make lint typecheck` and full test after twenty-first review | Formatting and Ruff passed; Pylint 10.00/10; all audits passed; strict Pyright reported 0 errors; 852 tests passed without skips and with 90.80% branch coverage. |
| 2026-08-12 | JAX x64 after twenty-first review | All 852 tests passed without skips with JAX 64-bit support active on CPU. |
| 2026-08-13 | iterative review remediation | Repaired namespace ownership, CPU-bound creation, non-finite diagonal construction, fitted-rank checks, frozen dtype semantics, FFT/linalg domains, standard array operands, random-state decoding, HDF5 membership validation, dataclass reconstruction, strict-namespace numerical helpers, and hostile protocol containment. Every finding received a focused regression. |
| 2026-08-13 | focused regressions after iterative review | All 364 affected contract, tree, collation, strict-conformance, metric, comparison, activation, linalg, FFT, and parity tests passed without skips. |
| 2026-08-13 | `make lint typecheck test` after iterative review | Formatting and Ruff passed; Pylint completed at 9.94/10; all audits passed; strict Pyright reported 0 errors; 958 tests passed without skips and with 90.23% branch coverage. |
| 2026-08-13 | promotion and rank follow-up review | Repaired portable pre-dispatch promotion, Torch wide-unsigned kernels and reductions, JAX active-surface result dtypes, linalg promotion/rank/control validation, mixed-dtype clipping, triangular rank checks, and hostile public namespace discovery. Every finding received a focused regression. |
| 2026-08-13 | focused regressions after promotion follow-up | All 253 review-regression tests and 189 adjacent namespace, operations, conformance, and linalg tests passed without skips. |
| 2026-08-13 | `make format lint typecheck test` after promotion follow-up | Formatting and Ruff passed; Pylint completed at 9.94/10; all audits passed; strict Pyright reported 0 errors; 977 tests passed without skips and with 90.01% branch coverage. |
| 2026-08-13 | wide-unsigned and data-state follow-up review | Repaired the remaining Torch CPU gaps for advertised wide-unsigned standard operations and functional index updates, upcast low-precision gamma calculations, froze validated `DataLoader` configuration without changing its tree-leaf semantics, and detached first-sample statistics extrema from dataset storage. Adjacent wide-unsigned indexed reads, triangular operations, nonzero, and search behavior were covered as well. |
| 2026-08-13 | focused regressions after data-state follow-up | All 261 review-regression tests passed; after correcting the loader tree-leaf interaction, all 268 combined-loader and review-regression tests passed without skips. |
| 2026-08-13 | `make format lint typecheck test` after data-state follow-up | Formatting and Ruff passed; Pylint completed at 9.94/10; all audits passed; strict Pyright reported 0 errors; 985 tests passed without skips and with 90.19% branch coverage. |
| 2026-08-13 | graph-safe namespace and linalg follow-up review | Repaired graph-safe out-of-bounds `take`/`take_along_axis` rejection, negative integer power/shift/repeat validation, oversized Torch `uint64` shifts, and redundant same-dtype standard-linalg casts. Every finding received eager and transformed-path regressions where applicable. |
| 2026-08-13 | eighth independent review | The full no-coverage suite passed with 1,195 tests; semantic review then found two P2 defects in same-dtype extension-linalg copies and mutable nested custom PyTree metadata. Both received shared source fixes and focused regressions. |
| 2026-08-13 | eighth-review affected validation | All 540 tree, linalg, parity, and review-regression tests passed without skips; formatting and Ruff passed; Pylint completed at 9.94/10; all audits passed; strict Pyright reported 0 errors. |
| 2026-08-13 | `make test` after eighth review | All 1,196 tests passed without skips and with 90.65% branch coverage. |
| 2026-08-13 | ninth independent review | Found three P2 control-validation defects in non-positive FFT lengths, complex linalg tolerance arrays, and unsupported matrix norm orders. All received shared pre-dispatch validation and cross-backend regressions; tolerance dtype and batch-shape validation now follows the 2024.12 standard. |
| 2026-08-13 | ninth-review affected validation | All 546 tree, FFT, linalg, parity, and review-regression tests passed without skips; formatting and Ruff passed; Pylint completed at 9.94/10; all audits passed; strict Pyright reported 0 errors. |
| 2026-08-13 | `make test` after ninth review | All 1,202 tests passed without skips and with 90.60% branch coverage. |
| 2026-08-13 | persistence host-boundary follow-up | Reproduced and repaired implicit CPU Torch/JAX conversion in all five save formats. Non-NumPy persistence now requires `allow_transfer=True`, active graphs independently require `allow_detach=True`, and NumPy arrays/scalars remain transfer-free. Cross-backend regressions cover NPY, NPZ, CSV, HDF5, and MAT. An adjacent weak-float overflow probe was checked against the frozen 2024.12 Array API and correctly left unchanged because over-precision scalar conversion is implementation-defined. |
| 2026-08-13 | scoped persistence review and affected validation | A commit-scoped `/review` found and drove fixes for NumPy scalar ownership and combined backend markers. All 104 affected conversion and persistence tests passed; the five new format cases passed independently in both Torch-only and JAX-only profiles; formatting and Ruff passed; Pylint completed at 9.94/10; all audits passed; strict Pyright reported 0 errors. |
| 2026-08-13 | `make test` after persistence review | All 1,213 tests passed without skips and with 90.69% branch coverage. |
| 2026-08-13 | final scoped persistence review | The follow-up commit-scoped `/review` returned no remaining findings after tracing NumPy scalar classification through persistence, probing supported and unsupported scalar dtypes, and confirming independent Torch/JAX profile collection. |
| 2026-08-16 | GitHub Actions failure diagnosis | The dependency-floor job had 16 failures from NumPy 1.26 DLPack controls, unsafe opaque Torch DLPack layouts, NumPy/JAX `bool_` dtype spelling, JAX 0.6 x64 context location, and JAX 0.6 PyTree sentinels. The clean documentation job had 12 unresolved `typing.P` references. |
| 2026-08-16 | affected current and exact floor regressions | All 627 affected current-environment tests passed; all 18 originally failing cases passed under NumPy 1.26.4, JAX/JAXlib 0.6.0, and Torch 2.13.0. |
| 2026-08-16 | `make lint` and `make typecheck` after CI remediation | Formatting and Ruff passed; Pylint completed at 9.94/10; portability, documentation-link, and release audits passed; strict Pyright reported 0 errors. |
| 2026-08-16 | `make docs-base` after CI remediation | Isolated NumPy-only warnings-as-errors HTML, all 35 doctests, and 14 selected documentation tests passed. |
| 2026-08-16 | `make floor` after CI remediation | All 1,213 tests passed against every direct minimum with 90.57% branch coverage. |
| 2026-08-16 | `make docs` after CI remediation | All-extras warnings-as-errors HTML and all 35 doctests passed. |
| 2026-08-16 | `make test` after CI remediation | All 1,213 tests passed without skips and with 90.51% branch coverage. |
| 2026-08-31 | release infrastructure audit | Confirmed no tags, GitHub Releases, release workflow, PyPI project, PyPI credential, GitHub environment, or repository secret; default-branch CI had passed all 12 jobs at `6350540`. |
| 2026-08-31 | repository release settings | Enabled private vulnerability reporting and created a tag-restricted `pypi` environment for short-lived Trusted Publishing credentials. |
| 2026-08-31 | release-workflow regressions and YAML parse | Two focused tests passed; all actions are immutable-SHA pinned; only the PyPI job receives `id-token: write`; the GitHub Release depends on successful PyPI publication. |
| 2026-08-31 | `make check` | Formatting and Ruff passed on 164 files; Pylint passed at 9.94/10; all audits passed; strict Pyright reported 0 errors; 1,215 tests passed without skips and with 90.51% branch coverage; strict HTML, 35 doctests, NumPy-only docs, five examples, artifact inspection, and all isolated install profiles passed. |
| 2026-08-31 | dependency-floor portion of `make check` | All 1,215 tests passed against every direct minimum without skips and with 90.57% branch coverage. |
| 2026-08-31 | `uv publish --dry-run` | The wheel and source distribution passed the PyPI upload plan. |

The functionality ledger contains all 168 normative IDs in runbook order. The
release audit verifies every row is Complete with existing trace paths.

## Risks and recovery

- The requested `port-scientific-python` skill is unavailable. Use the runbook,
  repository audit scripts, strict gates, and explicit final limitation note.
- JAX x64 and abstract tracer placement have enabled/disabled and compiled-path
  tests; retain the explicit CPU proof when backends evolve.
- The data/I/O surface has lazy optional imports, unsafe-object rejection, and
  atomic failure tests; new formats require equivalent security review.
- If a backend cannot satisfy a capability over the frozen version range,
  document and test a capability error rather than emulate or skip it.

## Handoff criteria

Every ledger row is `Complete`; no required backend is skipped; quality, type,
unit, property, conformance, parity, data, I/O, docs, packaging, benchmark, and
audit gates pass; isolated wheel/sdist installs pass for every installation set;
and the final report records exact commands, versions, skips, limitations,
benchmarks, licenses, and unauthorized external actions left undone.
