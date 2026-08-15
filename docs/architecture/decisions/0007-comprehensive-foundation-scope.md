# ADR 0007: Implement the Comprehensive Scientific Foundation Contract

Status: Accepted by explicit user direction. Date: 2026-08-11.

## Context

The initial implementation deliberately covered a small portability slice and
excluded complex numbers, FFT, datasets, and persistence. The comprehensive
development runbook now makes those facilities normative for 0.1.0 and requires
NumPy in the base installation.

## Decision

Supersede the narrower scope statements in ADR 0005 and the earlier public API
documents. NumPy is required. PyTorch and JAX remain independent lazy extras.
The package supports the runbook's complete standard Array API, linalg, FFT,
portable operation, conversion, update, random, autodiff, compilation, PyTree,
data, transform, statistics, and persistence matrix.

Complex64 and complex128 are supported where the selected backend and operation
declare that capability; APIs whose mathematical contract is real-only reject
complex inputs explicitly. CPU remains the only supported execution device for
0.1.0. HDF5 and MATLAB support are lazy optional extras. PDE solvers, physics
data generation, neural operators, models, training, optimizers, and AutoML
remain out of scope.

## Consequences

The functionality ledger is the release-completion authority. Old statements
that data, FFT, serialization, or all complex values are unsupported are no
longer valid. No capability may be simulated through a hidden NumPy or host
conversion, but named persistence boundaries convert to host data explicitly.
