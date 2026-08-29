# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Standards-first scientific infrastructure using native arrays."""

from asc import (
    _version,
    autodiff,
    compilation,
    config,
    conversion,
    data,
    errors,
    fft,
    linalg,
    metrics,
    ops,
    random,
    tree,
    typing,
    updates,
)
from asc.autodiff import grad, hessian, jacobian, jvp, value_and_grad, vjp
from asc.backends.capabilities import (
    BackendInfo,
    Capability,
    backend_info,
    has_capability,
    require_capability,
)
from asc.compilation import jit, vmap
from asc.config import (
    ArrayContext,
    CopyPolicy,
    CreationContext,
    CsvOptions,
    DataLoaderConfig,
    ExtensionHandle,
    Hdf5Options,
    MatOptions,
    NpyOptions,
    NpzOptions,
    PrecisionPolicy,
)
from asc.conversion import (
    convert_array,
    copy_array,
    detach,
    from_dlpack,
    from_numpy,
    stop_gradient,
    to_device,
    to_dlpack,
    to_numpy,
)
from asc.core.backend import (
    Backend,
    available_backends,
    backend,
    backend_of,
    is_array,
)
from asc.core.namespace import (
    ARRAY_API_VERSION,
    NamespaceInfo,
    array_namespace,
    namespace_info,
)
from asc.core.operations import create_full, sum_of_squares
from asc.diagnostics import diagnostics
from asc.errors import *  # noqa: F403
from asc.random import RandomState, random_state
from asc.updates import (
    index_add,
    index_max,
    index_min,
    index_multiply,
    index_set,
    scatter_add,
    scatter_max,
    scatter_min,
    scatter_multiply,
    scatter_set,
)

__version__ = _version.__version__
BackendName = typing.BackendName

__all__ = [
    "ARRAY_API_VERSION",
    "ArrayContext",
    "Backend",
    "BackendInfo",
    "BackendName",
    "Capability",
    "CopyPolicy",
    "CreationContext",
    "CsvOptions",
    "DataLoaderConfig",
    "ExtensionHandle",
    "Hdf5Options",
    "MatOptions",
    "NamespaceInfo",
    "NpyOptions",
    "NpzOptions",
    "PrecisionPolicy",
    "RandomState",
    "__version__",
    "array_namespace",
    "autodiff",
    "available_backends",
    "backend",
    "backend_info",
    "backend_of",
    "compilation",
    "config",
    "conversion",
    "convert_array",
    "copy_array",
    "create_full",
    "data",
    "detach",
    "diagnostics",
    "errors",
    "fft",
    "from_dlpack",
    "from_numpy",
    "grad",
    "has_capability",
    "hessian",
    "index_add",
    "index_max",
    "index_min",
    "index_multiply",
    "index_set",
    "is_array",
    "jacobian",
    "jit",
    "jvp",
    "linalg",
    "metrics",
    "namespace_info",
    "ops",
    "random",
    "random_state",
    "require_capability",
    "scatter_add",
    "scatter_max",
    "scatter_min",
    "scatter_multiply",
    "scatter_set",
    "stop_gradient",
    "sum_of_squares",
    "to_device",
    "to_dlpack",
    "to_numpy",
    "tree",
    "typing",
    "updates",
    "value_and_grad",
    "vjp",
    "vmap",
]
__all__.extend(errors.__all__)
PUBLIC_EXPORTS = tuple(sorted(set(__all__)))
__all__.append("PUBLIC_EXPORTS")
