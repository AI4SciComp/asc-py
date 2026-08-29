# ADR 0002: Freeze Python Array API Revision 2024.12

Status: Accepted and revalidated. Date: 2026-08-11.

## Context

The public Array API has continued beyond 2024.12. Using an unqualified
"latest" contract would allow semantic drift without a package release. The
contract must be supported by every minimum and maximum dependency entry.

array-api-compat 1.11.0 first targets revision 2024.12, but the 0.1 semantic
floor is 1.13.0 because earlier PyTorch wrapper behavior failed the required
functional-gradient test. Version 1.13.0 is the newest release that retains
2024.12. Version 1.14.0 changes wrapped namespaces to 2025.12;
requesting 2024.12 from 1.15.0 was verified to return a 2025.12 namespace with a
warning. array-api-strict 2.3.0 uses 2024.12 by default and current versions can
select it explicitly. JAX 0.6.0 is the first supported release that exposes
revision 2024.12 without an experimental import. JAX 0.11.0 exposes only
2025.12, so the 0.1 dependency ceiling is `<0.11`. NumPy 2.3 and newer document
main-namespace compatibility with 2024.12; older supported NumPy is reached
through the compatibility layer.

## Decision

Freeze revision `2024.12` for asc-py 0.1.0 and cap array-api-compat below 1.14.
Namespace discovery passes `api_version="2024.12"` explicitly, conformance tests
configure strict to that revision, and the constant `ARRAY_API_VERSION` exposes
the contract.

## Consequences

Features added by later standards are unavailable unless implemented as a
named asc-py extension. Raising or lowering the revision requires a new ADR,
support-matrix evidence, contract and conformance updates, and a changelog.

## Verification

On 2026-08-11, isolated floor probes used array-api-compat 1.13.0 with
NumPy 1.26.4 and JAX 0.6.0. Both returned a namespace whose
`__array_api_version__` was `2024.12`; every operation required by the runbook's
main, linalg, and FFT inventories was present. The same inventory passed with
NumPy 2.5.2, PyTorch 2.13.0, and JAX 0.10.2. The public 2025.12 standard is
newer, but NumPy's current compatibility statement and this package's frozen
compatibility-helper range establish 2024.12 as the newest common promise.

## Sources

Accessed or revalidated 2026-08-11:

- [Array API revision 2024.12](https://data-apis.org/array-api/2024.12/)
- [array-api-compat changelog](https://data-apis.org/array-api-compat/changelog.html)
- [array-api-strict changelog](https://data-apis.org/array-api-strict/changelog.html)
- [NumPy compatibility](https://numpy.org/doc/stable/reference/array_api.html)
- [JAX compatibility](https://docs.jax.dev/en/latest/jax.numpy.html#python-array-api-standard)
