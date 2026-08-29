# ADR 0003: Accept Native Arrays and Dispatch by Namespace

Status: Accepted. Date: 2026-08-10.

## Context

A universal tensor wrapper or dynamically replaced Tensor subclass would make
ownership, dispatch, graph, device, and typing behavior part of asc-py. A
mutable global backend selector would prevent independent backends in one
process and make imports order-dependent. The Array API provides a consumer
protocol designed to avoid both designs.

## Decision

Accept and return native arrays. Discover a namespace through
`__array_namespace__` or `array_api_compat.array_namespace`, always requesting
revision 2024.12. Reject mixed backends. Require an immutable creation context
when no input array selects a namespace.

The portable core imports no NumPy, PyTorch, or JAX modules. It depends on
narrow protocols. Lazy backend adapters implement only explicit extensions.
There is no mutable active-backend state and no universal Tensor class.

## Consequences

Array identity, dtype, device, gradients, and compilation remain owned by the
selected backend. Portable functions can be tested against array-api-strict.
Backend-specific behavior is visible at extension boundaries. Return typing is
necessarily structural or generic rather than one concrete Tensor type.

Any future wrapper requires a new ADR plus user approval because it changes the
core public design.

## Sources

Accessed 2026-08-10:

- [Array API purpose and scope](https://data-apis.org/array-api/2024.12/purpose_and_scope.html)
- [array-api-compat helper functions](https://data-apis.org/array-api-compat/helper-functions.html)
- [JAX native Array API entry point](https://docs.jax.dev/en/latest/jax.numpy.html#python-array-api-standard)
