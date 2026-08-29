# asc-py 0.1.0 Specification

Status: normative release contract. Last reviewed: 2026-08-11.

## Scope

`asc-py` is standards-first scientific infrastructure for code that must retain
native NumPy, PyTorch, or JAX arrays. Version 0.1.0 freezes Python Array API
revision 2024.12 and adds explicit, narrowly scoped facilities for:

- backend selection, capability discovery, immutable contexts, linalg, and FFT;
- portable operations, comparisons, activations, signals, and metrics;
- explicit conversion, DLPack, device policy, detach, and functional updates;
- backend-native random state, distributions, initializers, autodiff, JIT, and
  vectorization where declared in the support matrix;
- PyTrees, datasets, schemas, deterministic splits and samplers, collation,
  loaders, combined loaders, and lifecycle-managed data modules;
- transforms, streaming statistics, and safe atomic NPY, NPZ, CSV, HDF5, and
  MATLAB persistence; and
- typed configuration, stable errors, library-safe logging, diagnostics,
  packaging, documentation, benchmarks, and CI.

The complete per-ID contract and evidence ledger is the
[functionality matrix](specification/functionality-matrix.md).

## Normative Semantics

Arrays remain backend-native. Mathematical APIs select a namespace only from
array inputs, ignore accompanying Python scalars and `None`, and reject mixed
backends before computation. Creation without an array uses an immutable
context. No ordinary operation silently converts, transfers, copies, detaches,
mutates, or falls back through NumPy.

Dense CPU is the execution boundary. Sparse, masked, nested, distributed,
quantized, and accelerator arrays fail explicitly. Standard promotion is used;
real-only APIs reject complex inputs. Explicit JAX 64-bit requests fail rather
than narrow when x64 is disabled.

Random generation consumes explicit state and returns progressed state.
Functional updates return new logical values. Cross-backend conversion and I/O
are named boundaries with explicit ownership, graph, transfer, and dtype
policies. Data APIs preserve tree structure, backend, dtype, device, and graphs
where native stacking supports them.

## Support and Non-goals

NumPy is a runtime dependency. PyTorch and JAX are independent extras and do
not require one another. HDF5 and MATLAB support are independent format extras.
Exact bounds and conditional capabilities are recorded in the
[support matrix](architecture/support-matrix.md).

PDE/physics solvers, domain-specific data generation, neural operators, models,
training, inference, optimizers, AutoML, multiprocessing loaders, plotting, and
accelerator promises are outside this repository. Cross-backend random
bitstreams and cross-backend gradient preservation are not promised.

## Compatibility Policy

Public API is the documented export manifest. Patch releases may correct
contract violations without preserving erroneous behavior. Other incompatible
changes require an ADR, changelog entry, migration documentation, and a
targeted deprecation period unless an immediate security or corruption fix is
necessary. A capability is supported only when its declared matrix job passes
without a required skip.
