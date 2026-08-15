# asc-py

`asc-py` is the portable Python foundation for AI4SciComp. It exposes native
NumPy, PyTorch, and JAX arrays through a validated Python Array API 2024.12
contract, plus backend-neutral numerical and data utilities. There is no tensor
wrapper, mutable global backend, hidden NumPy fallback, or implicit host/device
transfer.

Version 0.1.0 supports dense CPU arrays. NumPy is required; PyTorch, JAX, HDF5,
and MATLAB I/O are independent lazy extras. PDE solvers, physics data
generation, models, training, optimization, AutoML, and visualization are
intentionally out of scope.

## Installation

```bash
python -m pip install asc-py
python -m pip install "asc-py[torch]"
python -m pip install "asc-py[jax]"
python -m pip install "asc-py[io-hdf5]"
python -m pip install "asc-py[io-mat]"
python -m pip install "asc-py[all]"
python -m pip install "asc-py[docs]"  # documentation tools, no Torch/JAX
```

The distribution installs the `asc` import package. It cannot coexist in one
environment with the unrelated PyPI distribution named `asc`, which owns the
same import path.

## Quick Start

```python
import asc

backend = asc.backend("numpy", dtype=None)
xp = backend.xp
x = xp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=xp.float32)

energy = asc.sum_of_squares(x, axis=1)
updated = asc.index_add(
    x,
    xp.asarray([0], dtype=xp.int16),
    xp.asarray([[10.0, 20.0]], dtype=xp.float32),
)
state = asc.random_state(7, backend="numpy")
noise, next_state = asc.random.random(
    (2, 2), state=state, dtype=xp.float32
)
dataset = asc.data.ArrayDataset(x)
loader = asc.data.DataLoader(dataset, batch_size=2)
```

Standard operations are available through `backend.xp`. `backend.linalg` and
`backend.fft` provide normalized numerical namespaces. Explicit APIs cover
conversion/DLPack, functional updates, random state, Torch/JAX autodiff, JAX
JIT, PyTrees, datasets, samplers, collation, transforms, statistics, and safe
atomic persistence.

See the [specification](docs/specification.md),
[portability contract](docs/architecture/portability-contract.md),
[support matrix](docs/architecture/support-matrix.md), and
[API reference](docs/api/index.rst).

## Development

Install uv 0.12 and run:

```bash
uv sync --frozen --all-groups --all-extras
make check
```

The local gate formats, lints, type-checks, tests with branch coverage, builds
strict documentation, audits portability, builds artifacts, and tests isolated
installs. See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under Apache-2.0.
