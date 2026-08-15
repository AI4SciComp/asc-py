# Architecture Overview

Status: Frozen for 0.1.0. Last reviewed: 2026-08-11.

`asc-py` is a standards-first scientific foundation using native NumPy,
PyTorch, and JAX arrays. It is not an array wrapper and has no active global
backend. Standard operations are reached through an inferred or explicit
Python Array API 2024.12 namespace.

```text
public API -> portable core -> narrow capability protocols
                                  ^
                                  |
                         lazy backend adapters

data API -> trees + portable core + explicit conversion/I/O boundaries
```

## Dependency Direction

NumPy is the required base backend. Portable mathematical modules depend on
array namespaces and structural protocols, never on optional PyTorch or JAX
imports. Backend adapters supply nonstandard random, functional update,
autodiff, compilation, and conversion behavior. Importing `asc` must not import
or initialize either optional backend.

The data package is backend-neutral: datasets, samplers, loaders, schemas, and
tree utilities preserve native leaves. Collation uses the selected namespace.
Persistence is an explicit host boundary and loads NumPy by default; callers
must request another backend explicitly.

## Package Layout

```text
src/asc/
  backends/ core/ ops/ linalg/ fft/ metrics/
  conversion/ updates/ random/ autodiff/ compilation/
  tree/ data/
  config.py errors.py logging.py diagnostics.py typing.py
```

## Data and Control Flow

Input-driven operations validate every native array, reject mixed backends, and
select one namespace. Creation starts from a frozen `ArrayContext`. Explicit
conversion is the only mathematical boundary that may change backend, dtype,
device, ownership, or graph state. Named I/O functions may transfer to host
only under their documented policy.

## Design Invariants

- Native arrays remain native; no universal `Tensor` exists.
- Backend, dtype, device, copy, precision, and random state are explicit values.
- Ordinary operations are functional and never mutate caller-owned inputs.
- Unsupported layouts, devices, dtypes, and capabilities fail before work.
- No portable extension falls back through NumPy or extracts host scalars.
- JIT and gradients remain backend graphs on every promised path.
- Every required functionality ID is traceable in the functionality matrix.

PDEs, physics-specific generators, solvers, neural operators, training,
optimizers, and AutoML belong in downstream projects.
