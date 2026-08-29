# Installation

asc-py requires CPython 3.12–3.14. Install the NumPy-only base package with:

```bash
python -m pip install asc-py
```

Install capabilities independently; Torch never installs JAX and JAX never
installs Torch:

```bash
python -m pip install "asc-py[torch]"
python -m pip install "asc-py[jax]"
python -m pip install "asc-py[io-hdf5]"
python -m pip install "asc-py[io-mat]"
python -m pip install "asc-py[all]"
```

The distribution is named `asc-py`, but the import is `asc`. It cannot coexist
with the unrelated PyPI distribution named `asc` in one environment. Optional
modules are imported only after their backend is selected; a missing extra
raises {py:class}`asc.BackendUnavailableError` with an installation hint.

For documentation work, install only NumPy and the independent docs extra:

```bash
python -m pip install -e ".[docs]"
make docs
```

No documentation build requires Torch or JAX.
