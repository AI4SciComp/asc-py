# Examples

Run every executable example from an all-extras development environment:

```bash
make examples
```

`portable_core.py` and `arrays_and_updates.py` require only the base install.
`random_autodiff.py` also requires the JAX extra. `data_pipeline.py` and
`persistence.py` demonstrate the backend-neutral data package and safe base I/O.

Executable backend-neutral examples live here. Every example must run in CI or
be marked clearly as a non-executable illustration. Mathematical code must not
be duplicated merely to change backend imports.
