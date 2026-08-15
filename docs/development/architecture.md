# Architecture

asc uses a `src/asc` layout and passes native arrays across public APIs. The
portable core selects one Array API namespace, while lazy adapters contain
backend-specific random, update, conversion, autodiff, and compilation code.
Immutable contexts replace global backend state.

The data package operates on ordinary Python trees and native leaves. Named
conversion and persistence APIs are the only host/backend boundaries. Internal
modules beginning with `_` and backend adapter modules are excluded from the
generated reference.

Read the [architecture overview](../architecture/overview.md), [portability
contract](../architecture/portability-contract.md), and [public API
policy](../architecture/public-api.md) before changing behavior.
