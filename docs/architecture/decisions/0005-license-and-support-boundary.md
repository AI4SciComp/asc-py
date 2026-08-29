# ADR 0005: Apache-2.0 and a CPU-first Support Boundary

Status: Accepted. Date: 2026-08-10.

## Context

The repository includes the full Apache License 2.0 but no explicit ADR. Public
AI4SciComp siblings could have established a conflicting policy. The package
also needs a support claim that can be exercised in bounded local and hosted
environments.

## Decision

Release asc-py under Apache-2.0. The current `LICENSE` byte-matches the licenses
in `AI4SciComp/asc-cpp`, `asc-devtools`, and `asc-cmake` as inspected on
2026-08-10.

Version 0.1.0 supports dense CPU arrays on Linux x86-64 across the full semantic
matrix. Windows x86-64 and macOS arm64 receive installation and smoke coverage
but remain provisional. Accelerators, sparse arrays, and distributed arrays are
unsupported. ADR 0007 supersedes this decision's earlier blanket complex-dtype
exclusion with capability-based complex support.

## Consequences

Source headers, metadata, artifacts, and notices must identify Apache-2.0.
Dependency licenses are recorded separately and inspected in built artifacts.
No CUDA/ROCm/MPS/TPU dependency is downloaded merely to create a passing job.
An accelerator or additional platform support claim needs actual runners,
contract/parity coverage, documentation, and a new ADR.
