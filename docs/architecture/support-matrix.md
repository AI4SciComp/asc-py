# Support Matrix

Status: 0.1.0 release baseline. The hosted Linux matrix and provisional Windows
x86-64 and macOS arm64 smoke jobs passed on 2026-08-30. Last reviewed:
2026-08-31.

The frozen contract is Python Array API 2024.12. A version or capability is
release-supported only after its required job passes without unexpected skips.

## Version Bounds

| Component | Minimum probed | Maximum probed | Published bound |
|---|---:|---:|---|
| CPython | 3.12 | 3.14 | `>=3.12,<3.15` |
| NumPy | 1.26.4 | 2.5.2 | `>=1.26.4,<2.6` |
| PyTorch | 2.13.0 | 2.13.0 | `>=2.13,<2.14` |
| JAX/JAXlib | 0.6.0 | 0.10.2 | `>=0.6,<0.11` |
| array-api-compat | 1.13.0 | 1.13.0 | `>=1.13,<1.14` |
| array-api-strict | 2.3.0 | 2.6.1 | development only |
| Sphinx | 9.1.0 | 9.1.0 | `>=9.1,<10` in `docs` |
| MyST Parser | 5.1.0 | 5.1.0 | `>=5.1,<6` in `docs` |
| PyData Sphinx Theme | 0.19.0 | 0.19.0 | `>=0.19,<0.20` in `docs` |

Isolated 2026-08-11 suites verified revision 2024.12 and the runbook's required
main, linalg, FFT, extension, and data behavior with NumPy 1.26.4/JAX 0.6.0
and the current ceilings. NumPy 2.3+ officially documents 2024.12 compatibility;
NumPy 1.x is normalized by array-api-compat. Complete behavior tests, rather
than symbol presence alone, are the gate.

## Runtime and dtype capabilities

| Capability | NumPy | PyTorch | JAX | strict |
|---|---|---|---|---|
| Standard namespace | Required | Required | Required | Conformance oracle |
| Linalg/FFT | Required | Required | Required | Standard checks only |
| bool/int8/int16/int32 | Required | Required | Required | Required |
| int64 | Required | Required | Requires JAX x64 | Required |
| uint8/uint16/uint32 | Native subset | Native subset | Native subset | Native subset |
| uint64 | Required | Required | Requires JAX x64 | By strict capability |
| float16 | Required | Required | Required | Not exposed |
| bfloat16 | Not exposed | Required | Required | Not exposed |
| float32 | Required | Required | Required | Required |
| float64 | Required | Required | Requires JAX x64 | Required |
| complex64 | Required | Required | Required | By strict capability |
| complex128 | Required | Required | Requires JAX x64 | By strict capability |
| explicit conversion | Required | Required | Required | Test boundary only |
| random/update | Required | Required | Required | Explicit capability error |
| autodiff | Capability error | Required | Required | Capability error |
| JIT | Capability error | Capability error | Required | Capability error |
| vmap | Capability error | Frozen capability | Required | Capability error |
| HDF5/MAT | Named host boundary | Named host boundary | Named host boundary | N/A |

Complex support is operation-specific: standard, linalg, and FFT paths honor
backend capabilities, while real-only metrics, distributions, and initializers
raise `DTypeError`. JAX int64/uint64/float64/complex128 requests fail when x64
is disabled. Dtypes outside the exact table, including NumPy extended-precision
and PyTorch float8 types, are rejected at selection, array-discovery, creation,
and persistence boundaries rather than admitted as backend-specific extensions.
Low-precision linalg and real-input FFT support is also operation- and
backend-specific. An unavailable CPU kernel raises
`CapabilityNotSupportedError` before native dispatch.

## Devices, layouts, and platforms

Dense CPU arrays are the 0.1.0 execution boundary. Sparse, masked, nested,
distributed, quantized, and accelerator arrays are rejected before computation.
Explicit `to_device` exists, but an unsupported destination raises
`CapabilityNotSupportedError`; it does not broaden the release claim.

Linux x86-64 receives the full semantic matrix. Windows x86-64 and macOS arm64
receive install and smoke jobs and remain provisional until hosted evidence
exists. CPU-only PyTorch wheels are used for local validation.

## Required independent jobs

- base (NumPy only), Torch without JAX, and JAX without Torch;
- the complete applicable pytest tree in each independent array profile, with
  the all-extras job retaining full cross-backend parity and coverage;
- HDF5, MATLAB, and all-extras installs;
- dependency floors and ceilings on supported Python endpoints;
- JAX x64 enabled and disabled;
- full standard conformance, parity, gradients/JIT, data, and I/O tests;
- strict docs, wheel/sdist inspection, and isolated artifact installs.
- NumPy-only strict HTML/doctest inventory, plus independent Torch and JAX
  documentation examples and scheduled external-link validation.

Primary evidence: [Array API 2024.12](https://data-apis.org/array-api/2024.12/),
[array-api-compat changelog](https://data-apis.org/array-api-compat/changelog.html),
and [NumPy compatibility](https://numpy.org/doc/stable/reference/array_api.html).
