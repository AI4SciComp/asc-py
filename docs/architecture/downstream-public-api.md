# Released Public API Boundary for Downstream Repositories

Status: normative downstream boundary for the planned `asc-no` repository.

`asc-py` is a domain-neutral scientific-computing foundation. A future
`asc-no` may depend on a released `asc-py` distribution and use only the public
interfaces documented here and in the [0.1 public API](public-api.md). It must
not import underscore-prefixed modules, objects absent from documented
`__all__` lists, test helpers, or repository scripts.

## Supported boundary

- Backend discovery and capabilities: `asc.backend`, `asc.backend_of`,
  `asc.available_backends`, `asc.backend_info`, `asc.array_namespace`,
  `asc.has_capability`, `asc.require_capability`, and the public `Backend`,
  `BackendInfo`, `Capability`, and `NamespaceInfo` records.
- Native arrays and explicit conversion: public `Backend.xp`, `Backend.linalg`,
  and `Backend.fft` namespaces; `asc.conversion`; and the top-level conversion,
  copy, device, DLPack, detach, and stop-gradient functions. Downstream code
  must preserve native backend arrays and make every conversion, transfer,
  cast, copy, and graph boundary explicit.
- Reproducibility: `asc.RandomState`, `asc.random_state`, and the documented
  `asc.random` functions. Reproducibility claims remain bounded to the same
  backend, dependency version, device, dtype, and configuration.
- Data and persistence: documented exports from `asc.data`, including dataset,
  loader, sampler, schema, split, transform, statistics, and safe persistence
  interfaces. Optional HDF5 and MATLAB functionality requires its separately
  declared extra.
- Configuration, logging, and errors: documented immutable values from
  `asc.config`, `asc.logging.get_logger`, and the exception hierarchy rooted at
  `asc.errors.AscError`. A downstream library must not configure the root
  logger or depend on private exception details.

## Compatibility policy

Consumers must declare a released `asc-py` version range rather than a Git
branch, commit, source-tree path, or private module. Public compatibility is
governed by `asc.__version__`, the support matrix, changelog, release notes,
and migration guidance. A deprecation must be documented before removal and
remain within the versioning policy; downstream code must not treat private
implementation behavior as a compatibility promise.

The dependency direction is one way:

```text
asc-no -> released public asc-py APIs
```

`asc-py` must never depend on or import `asc-no`, and neural-operator models,
training, evaluation, and benchmarks remain outside this repository. `asc-os`
may be used through authored manifests and a separately installed development
sidecar, but it is not an `asc-py` runtime dependency and importing `asc` must
not import `asc_os`.
