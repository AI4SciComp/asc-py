# Testing

The test suite separates standard conformance, public contracts, backend
parity, properties, data behavior, documentation, and packaging. Required
tests may not be skipped.

```bash
make test          # unit, contract, parity, property, and docs inventory tests
make docs-html     # strict, nitpicky HTML build with warnings as errors
make docs-doctest  # execute Sphinx doctests
make docs-linkcheck  # validate external links separately
make examples      # execute repository examples
```

Portable numerical tests cover NumPy, Torch, and JAX CPU; standard-only checks
also use array-api-strict. Documentation examples use NumPy in the base build,
while Torch and JAX examples run in independent optional-extra environments.
CI additionally runs the complete applicable test tree in three isolated array
profiles: NumPy-only, NumPy plus Torch without JAX, and NumPy plus JAX without
Torch. Backend-marked cases are deselected only when their declared extra is
absent; the all-extras semantic job remains the complete parity and coverage
gate.
