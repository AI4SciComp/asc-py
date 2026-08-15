# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Safe atomic persistence with explicit host and backend boundaries."""

from __future__ import annotations

import collections.abc
import contextlib
import json
import os
import pathlib
import tempfile
import typing
import warnings
import zipfile

import numpy

from asc import _array_api_compat, errors
from asc.core import _dtype
from asc.tree import (
    TreeSpec,
    tree_flatten,
    tree_to_backend,
    tree_to_numpy,
    tree_unflatten,
)

PathLike = str | os.PathLike[str]
_AMBIGUOUS_CSV_DELIMITERS = frozenset("0123456789+-.eEjJ()% nNaAiIfFtTyY")


class NPZData(typing.NamedTuple):
    """Named NPZ arrays and decoded safe JSON metadata."""

    arrays: dict[str, object]
    metadata: object | None


class CSVData(typing.NamedTuple):
    """Two-dimensional numeric CSV array and optional header."""

    array: object
    header: tuple[str, ...] | None


class TreeData(typing.NamedTuple):
    """Restored numeric tree and decoded safe JSON metadata."""

    tree: object
    metadata: object | None


def _json(value: object, operation: str) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exception:
        raise errors.DataFormatError(
            f"{operation}: metadata must be safe JSON values"
        ) from exception


def _strict_json(document: str, operation: str) -> object:
    """Decode JSON while rejecting JavaScript-style non-finite constants."""

    def reject_constant(value: str) -> typing.NoReturn:
        raise ValueError(f"non-finite JSON constant {value!r}")

    try:
        return json.loads(document, parse_constant=reject_constant)
    except (TypeError, ValueError, json.JSONDecodeError) as exception:
        raise errors.DataFormatError(
            f"{operation}: metadata is not strict JSON"
        ) from exception


def _require_schema_one(value: object) -> None:
    """Require an exactly integral scalar version-one marker."""
    schema = numpy.asarray(value).squeeze()
    if (
        schema.ndim != 0
        or schema.dtype.kind not in {"i", "u"}
        or schema.item() != 1
    ):
        raise ValueError("unsupported schema")


def _atomic(
    path: PathLike,
    writer: typing.Callable[[pathlib.Path], None],
    operation: str,
) -> pathlib.Path:
    temporary: pathlib.Path | None = None
    try:
        destination = pathlib.Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = pathlib.Path(handle.name)
        writer(temporary)
        os.replace(temporary, destination)
    except OSError as exception:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)
        raise errors.DataFormatError(
            f"{operation}: atomic filesystem write failed"
        ) from exception
    except Exception:
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)
        raise
    return destination


def _host(
    data: object,
    operation: str,
    *,
    allow_detach: bool,
    allow_transfer: bool,
) -> object:
    native_arrays: tuple[object, ...]
    if _array_api_compat.compat.is_array_api_obj(data):
        native_arrays = (data,)
    else:
        leaves, _ = tree_flatten(data)
        native_arrays = tuple(
            leaf
            for leaf in leaves
            if _array_api_compat.compat.is_array_api_obj(leaf)
        )
    if not allow_transfer and any(
        not isinstance(array, (numpy.ndarray, numpy.generic))
        for array in native_arrays
    ):
        raise errors.ConversionError(
            f"{operation}: non-NumPy arrays require allow_transfer=True "
            "for persistence host conversion"
        )
    try:
        if _array_api_compat.compat.is_array_api_obj(data):
            from asc.conversion import to_numpy

            return to_numpy(
                data,
                allow_detach=allow_detach,
                allow_transfer=allow_transfer,
                copy=True,
            )
        return tree_to_numpy(
            data,
            allow_detach=allow_detach,
            allow_transfer=allow_transfer,
            copy=True,
        )
    except errors.UnsupportedCapabilityError as exception:
        raise errors.DataFormatError(
            f"{operation}: every array must use a supported numeric dtype"
        ) from exception


def _destination(data: object, destination: object | None) -> object:
    return data if destination is None else tree_to_backend(data, destination)


def _reject_object(array: numpy.ndarray, operation: str) -> None:
    if array.dtype.hasobject:
        raise errors.DataFormatError(
            f"{operation}: object arrays are unsafe and unsupported by default"
        )


def _reject_non_numeric(array: numpy.ndarray, operation: str) -> None:
    """Require one safe release-supported numeric array."""
    _reject_object(array, operation)
    if array.dtype.kind not in "biufc" or not _dtype.is_supported_dtype(
        "numpy", array.dtype
    ):
        raise errors.DataFormatError(
            f"{operation}: every array must have a supported numeric dtype"
        )


def _require_native_array(value: object, operation: str) -> None:
    """Reject persistence inputs that would require implicit coercion."""
    if not _array_api_compat.compat.is_array_api_obj(value):
        raise errors.DataFormatError(
            f"{operation}: input must be a native Boolean or numeric array"
        )


def _require_booleans(operation: str, **values: object) -> None:
    if any(not isinstance(value, bool) for value in values.values()):
        raise errors.DataFormatError(
            f"{operation}: policy flags must be explicit Booleans"
        )


def save_npy(
    path: PathLike,
    array: object,
    *,
    allow_detach: bool = False,
    allow_transfer: bool = False,
) -> pathlib.Path:
    """Atomically save one numeric array without pickle."""
    _require_booleans(
        "save_npy", allow_detach=allow_detach, allow_transfer=allow_transfer
    )
    _require_native_array(array, "save_npy")
    if isinstance(array, numpy.ndarray):
        _reject_non_numeric(array, "save_npy")
    native = numpy.asarray(
        _host(
            array,
            "save_npy",
            allow_detach=allow_detach,
            allow_transfer=allow_transfer,
        )
    )
    _reject_non_numeric(native, "save_npy")

    def write(temporary: pathlib.Path) -> None:
        with temporary.open("wb") as stream:
            numpy.save(stream, native, allow_pickle=False)

    return _atomic(path, write, "save_npy")


def load_npy(
    path: PathLike,
    *,
    mmap_mode: typing.Literal["r", "r+", "w+", "c"] | None = None,
    destination: object | None = None,
    allow_unsafe_pickle: bool = False,
) -> object:
    """Load NPY to NumPy by default and convert only when requested."""
    _require_booleans("load_npy", allow_unsafe_pickle=allow_unsafe_pickle)
    if mmap_mode not in {None, "r", "r+", "w+", "c"}:
        raise errors.DataFormatError("load_npy: invalid mmap_mode")
    if allow_unsafe_pickle:
        warnings.warn(
            "load_npy: unsafe pickle loading can execute arbitrary code; "
            "only load trusted files",
            UserWarning,
            stacklevel=2,
        )
    try:
        if mmap_mode is None:
            with pathlib.Path(path).open("rb") as stream:
                array = numpy.load(stream, allow_pickle=allow_unsafe_pickle)
        else:
            array = numpy.load(
                path,
                allow_pickle=allow_unsafe_pickle,
                mmap_mode=mmap_mode,
            )
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as exception:
        raise errors.DataFormatError(
            f"load_npy: unable to load safe NPY file {str(path)!r}"
        ) from exception
    if not isinstance(array, numpy.ndarray):
        array.close()
        raise errors.DataFormatError(
            f"load_npy: file {str(path)!r} is an NPZ archive, not an NPY array"
        )
    if not (allow_unsafe_pickle and array.dtype.hasobject):
        _reject_non_numeric(array, "load_npy")
    return _destination(array, destination)


def save_npz(
    path: PathLike,
    arrays: collections.abc.Mapping[str, object],
    *,
    metadata: object | None = None,
    compressed: bool = True,
    allow_detach: bool = False,
    allow_transfer: bool = False,
) -> pathlib.Path:
    """Atomically save named numeric arrays plus safe JSON metadata."""
    _require_booleans(
        "save_npz",
        compressed=compressed,
        allow_detach=allow_detach,
        allow_transfer=allow_transfer,
    )
    if not arrays or "__asc_metadata__" in arrays:
        raise errors.DataFormatError(
            "save_npz: arrays must be non-empty and not use reserved names"
        )
    names = set(arrays)
    names.add("__asc_metadata__")
    if any(
        name.endswith(".npy") and name.removesuffix(".npy") in names
        for name in names
        if isinstance(name, str)
    ):
        raise errors.DataFormatError(
            "save_npz: array names must not alias another .npy archive member"
        )
    native: dict[str, numpy.ndarray] = {}
    for name, value in arrays.items():
        if not isinstance(name, str) or not name or "\0" in name:
            raise errors.DataFormatError(
                "save_npz: array names must be non-empty strings without NUL"
            )
        _require_native_array(value, "save_npz")
        item = numpy.asarray(
            _host(
                value,
                "save_npz",
                allow_detach=allow_detach,
                allow_transfer=allow_transfer,
            )
        )
        _reject_non_numeric(item, "save_npz")
        native[name] = item
    encoded = numpy.frombuffer(
        _json(metadata, "save_npz").encode(), dtype=numpy.uint8
    )
    native["__asc_metadata__"] = encoded

    def write(temporary: pathlib.Path) -> None:
        with temporary.open("wb") as stream:
            function = numpy.savez_compressed if compressed else numpy.savez
            function(stream, **native)

    return _atomic(path, write, "save_npz")


def load_npz(path: PathLike, *, destination: object | None = None) -> NPZData:
    """Load named numeric NPZ arrays with pickle disabled."""
    try:
        with pathlib.Path(path).open("rb") as stream:
            loaded = numpy.load(stream, allow_pickle=False)
            if isinstance(loaded, numpy.ndarray):
                raise errors.DataFormatError(
                    f"load_npz: file {str(path)!r} is an NPY array, not an "
                    "NPZ archive"
                )
            with loaded as archive:
                if len(set(archive.files)) != len(archive.files):
                    raise errors.DataFormatError(
                        "load_npz: archive contains duplicate logical names"
                    )
                names = set(archive.files)
                if any(
                    name.endswith(".npy") and name.removesuffix(".npy") in names
                    for name in names
                ):
                    raise errors.DataFormatError(
                        "load_npz: logical names alias another .npy archive "
                        "member"
                    )
                arrays = {
                    name: numpy.array(archive[name], copy=True)
                    for name in archive.files
                    if name != "__asc_metadata__"
                }
                if "__asc_metadata__" in archive.files:
                    metadata_array = numpy.asarray(archive["__asc_metadata__"])
                    if (
                        metadata_array.ndim != 1
                        or metadata_array.dtype != numpy.dtype(numpy.uint8)
                    ):
                        raise errors.DataFormatError(
                            "load_npz: metadata must be a one-dimensional "
                            "uint8 array"
                        )
                    encoded = metadata_array.tobytes()
                else:
                    encoded = b"null"
    except (
        EOFError,
        OSError,
        ValueError,
        KeyError,
        zipfile.BadZipFile,
    ) as exception:
        raise errors.DataFormatError(
            f"load_npz: unable to load safe NPZ file {str(path)!r}"
        ) from exception
    for array in arrays.values():
        _reject_non_numeric(array, "load_npz")
    try:
        metadata = _strict_json(encoded.decode(), "load_npz")
    except UnicodeDecodeError as exception:
        raise errors.DataFormatError(
            "load_npz: metadata is not valid UTF-8 JSON"
        ) from exception
    converted = {
        name: _destination(array, destination) for name, array in arrays.items()
    }
    return NPZData(converted, metadata)


def save_csv(
    path: PathLike,
    array: object,
    *,
    header: collections.abc.Sequence[str] | None = None,
    delimiter: str = ",",
    allow_detach: bool = False,
    allow_transfer: bool = False,
) -> pathlib.Path:
    """Atomically save the documented two-dimensional numeric CSV subset."""
    _require_booleans(
        "save_csv", allow_detach=allow_detach, allow_transfer=allow_transfer
    )
    _require_native_array(array, "save_csv")
    native = numpy.asarray(
        _host(
            array,
            "save_csv",
            allow_detach=allow_detach,
            allow_transfer=allow_transfer,
        )
    )
    _reject_non_numeric(native, "save_csv")
    if (
        native.ndim != 2
        or 0 in native.shape
        or not numpy.issubdtype(native.dtype, numpy.number)
    ):
        raise errors.DataFormatError(
            "save_csv: array must be non-empty, two-dimensional, and numeric"
        )
    if (
        not isinstance(delimiter, str)
        or len(delimiter) != 1
        or delimiter in {"\r", "\n"}
        or delimiter in _AMBIGUOUS_CSV_DELIMITERS
    ):
        raise errors.DataFormatError(
            "save_csv: delimiter must be one unambiguous character"
        )
    if header is not None and (
        len(header) != native.shape[1]
        or any(
            not isinstance(name, str)
            or not name
            or delimiter in name
            or "\n" in name
            or "\r" in name
            for name in header
        )
    ):
        raise errors.DataFormatError(
            "save_csv: header must match columns and contain no delimiter "
            "or newline"
        )

    def write(temporary: pathlib.Path) -> None:
        number_format = (
            "%d" if numpy.issubdtype(native.dtype, numpy.integer) else "%.18e"
        )
        numpy.savetxt(
            temporary,
            native,
            fmt=number_format,
            delimiter=delimiter,
            header="" if header is None else delimiter.join(header),
            comments="",
        )

    return _atomic(path, write, "save_csv")


def load_csv(
    path: PathLike,
    *,
    header: bool = False,
    delimiter: str = ",",
    dtype: object = numpy.float64,
    destination: object | None = None,
) -> CSVData:
    """Load the numeric CSV subset with an explicit header/dtype policy."""
    _require_booleans("load_csv", header=header)
    if (
        not isinstance(delimiter, str)
        or len(delimiter) != 1
        or delimiter in {"\r", "\n"}
        or delimiter in _AMBIGUOUS_CSV_DELIMITERS
    ):
        raise errors.DataFormatError(
            "load_csv: delimiter must be one unambiguous character"
        )
    try:
        native_dtype = numpy.dtype(dtype)
    except TypeError as exception:
        raise errors.DataFormatError(
            "load_csv: invalid numeric dtype"
        ) from exception
    if not numpy.issubdtype(
        native_dtype, numpy.number
    ) or not _dtype.is_supported_dtype("numpy", native_dtype):
        raise errors.DataFormatError(
            "load_csv: dtype must be a supported numeric dtype"
        )
    names: tuple[str, ...] | None = None
    try:
        if header:
            with pathlib.Path(path).open(
                encoding="utf-8", newline=""
            ) as stream:
                names = tuple(stream.readline().rstrip("\r\n").split(delimiter))
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            array = numpy.loadtxt(
                path,
                delimiter=delimiter,
                comments=None,
                dtype=native_dtype,
                skiprows=1 if header else 0,
                ndmin=2,
            )
    except (OSError, TypeError, UnicodeError, ValueError, Warning) as exception:
        raise errors.DataFormatError(
            f"load_csv: unable to load numeric CSV file {str(path)!r}"
        ) from exception
    if 0 in array.shape:
        raise errors.DataFormatError("load_csv: empty CSV data is unsupported")
    _reject_non_numeric(array, "load_csv")
    if names is not None and (
        len(names) != array.shape[1] or any(not name for name in names)
    ):
        raise errors.DataFormatError(
            "load_csv: header fields must be non-empty and match numeric data"
        )
    return CSVData(_destination(array, destination), names)


def _numeric_tree(
    tree: object,
    operation: str,
    *,
    allow_detach: bool,
    allow_transfer: bool,
) -> tuple[list[numpy.ndarray], TreeSpec]:
    host = _host(
        tree,
        operation,
        allow_detach=allow_detach,
        allow_transfer=allow_transfer,
    )
    leaves, spec = tree_flatten(host)
    arrays: list[numpy.ndarray] = []
    for leaf in leaves:
        if not isinstance(leaf, numpy.ndarray):
            raise errors.DataFormatError(
                f"{operation}: every tree leaf must be a native numeric array"
            )
        _reject_non_numeric(leaf, operation)
        arrays.append(leaf)
    return arrays, spec


def _reject_custom_tree_spec(spec: TreeSpec, operation: str) -> None:
    """Reject persisted nodes whose reconstruction can execute callbacks."""
    if spec.kind == "custom":
        raise errors.DataFormatError(
            f"{operation}: custom PyTree nodes are unsafe for persistence"
        )
    for child in spec.children:
        _reject_custom_tree_spec(child, operation)


def save_hdf5(
    path: PathLike,
    tree: object,
    *,
    metadata: object | None = None,
    compression: str | None = None,
    chunks: bool | tuple[int, ...] | None = None,
    allow_detach: bool = False,
    allow_transfer: bool = False,
) -> pathlib.Path:
    """Atomically save a nested numeric tree through lazy h5py."""
    _require_booleans(
        "save_hdf5",
        allow_detach=allow_detach,
        allow_transfer=allow_transfer,
    )
    if compression is not None and (
        not isinstance(compression, str) or not compression
    ):
        raise errors.DataFormatError(
            "save_hdf5: compression must be a non-empty string or None"
        )
    if isinstance(chunks, tuple) and (
        not chunks
        or any(
            isinstance(extent, bool)
            or not isinstance(extent, int)
            or extent <= 0
            for extent in chunks
        )
    ):
        raise errors.DataFormatError(
            "save_hdf5: chunk extents must be positive integers"
        )
    if not isinstance(chunks, (bool, tuple, type(None))):
        raise errors.DataFormatError("save_hdf5: invalid chunks policy")
    if chunks is False and compression is not None:
        raise errors.DataFormatError(
            "save_hdf5: compression requires chunked storage and is "
            "incompatible with chunks=False"
        )
    try:
        import h5py
    except ModuleNotFoundError as exception:
        raise errors.BackendUnavailableError(
            "save_hdf5: install the asc-py[io-hdf5] extra"
        ) from exception
    arrays, spec = _numeric_tree(
        tree,
        "save_hdf5",
        allow_detach=allow_detach,
        allow_transfer=allow_transfer,
    )
    _reject_custom_tree_spec(spec, "save_hdf5")
    metadata_json = _json(metadata, "save_hdf5")

    def write(temporary: pathlib.Path) -> None:
        with h5py.File(temporary, "w") as file:
            file.attrs["asc_schema"] = 1
            file.attrs["tree_spec"] = spec.to_json()
            file.attrs["metadata"] = metadata_json
            group = file.create_group("leaves")
            for index, array in enumerate(arrays):
                dataset_options = (
                    {}
                    if array.ndim == 0
                    else {
                        "compression": compression,
                        "chunks": None if chunks is False else chunks,
                    }
                )
                group.create_dataset(
                    str(index),
                    data=array,
                    **dataset_options,
                )

    try:
        return _atomic(path, write, "save_hdf5")
    except errors.AscError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as exception:
        raise errors.DataFormatError(
            "save_hdf5: unable to create datasets with the requested "
            "compression and chunk policy"
        ) from exception


def load_hdf5(path: PathLike, *, destination: object | None = None) -> TreeData:
    """Load a safe numeric HDF5 tree through lazy h5py."""
    try:
        import h5py
    except ModuleNotFoundError as exception:
        raise errors.BackendUnavailableError(
            "load_hdf5: install the asc-py[io-hdf5] extra"
        ) from exception
    try:
        with h5py.File(path, "r") as file:
            _require_schema_one(file.attrs.get("asc_schema"))
            spec = TreeSpec.from_json(str(file.attrs["tree_spec"]))
            _reject_custom_tree_spec(spec, "load_hdf5")
            metadata = _strict_json(str(file.attrs["metadata"]), "load_hdf5")
            if not isinstance(file.get("leaves", getlink=True), h5py.HardLink):
                raise ValueError(
                    "leaves must be stored inside the supplied file"
                )
            group = file["leaves"]
            if not isinstance(group, h5py.Group):
                raise TypeError("leaves must be an HDF5 group")
            expected_names = {str(index) for index in range(spec.num_leaves)}
            if set(group.keys()) != expected_names:
                raise ValueError(
                    "leaf members must match the serialized tree exactly"
                )
            leaves: list[numpy.ndarray] = []
            for index in range(spec.num_leaves):
                name = str(index)
                if not isinstance(group.get(name, getlink=True), h5py.HardLink):
                    raise ValueError(
                        "leaf datasets must use self-contained hard links"
                    )
                dataset = group[name]
                if not isinstance(dataset, h5py.Dataset):
                    raise TypeError("every leaf must be an HDF5 dataset")
                creation = dataset.id.get_create_plist()
                if (
                    getattr(dataset, "is_virtual", False)
                    or creation.get_external_count() != 0
                ):
                    raise ValueError(
                        "virtual and externally stored leaf datasets are unsafe"
                    )
                leaves.append(numpy.asarray(dataset))
            for leaf in leaves:
                _reject_non_numeric(leaf, "load_hdf5")
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as exception:
        raise errors.DataFormatError(
            f"load_hdf5: invalid asc HDF5 file {str(path)!r}"
        ) from exception
    try:
        tree = tree_unflatten(spec, leaves)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        raise errors.DataFormatError(
            f"load_hdf5: invalid asc HDF5 file {str(path)!r}"
        ) from exception
    return TreeData(_destination(tree, destination), metadata)


def save_mat(
    path: PathLike,
    tree: object,
    *,
    metadata: object | None = None,
    do_compression: bool = False,
    allow_detach: bool = False,
    allow_transfer: bool = False,
) -> pathlib.Path:
    """Atomically save a documented numeric tree through lazy SciPy."""
    _require_booleans(
        "save_mat",
        do_compression=do_compression,
        allow_detach=allow_detach,
        allow_transfer=allow_transfer,
    )
    try:
        import scipy.io
    except ModuleNotFoundError as exception:
        raise errors.BackendUnavailableError(
            "save_mat: install the asc-py[io-mat] extra"
        ) from exception
    arrays, spec = _numeric_tree(
        tree,
        "save_mat",
        allow_detach=allow_detach,
        allow_transfer=allow_transfer,
    )
    _reject_custom_tree_spec(spec, "save_mat")
    payload: dict[str, object] = {
        "asc_schema": numpy.asarray([[1]], dtype=numpy.int64),
        "asc_tree_spec": numpy.asarray(spec.to_json()),
        "asc_metadata": numpy.asarray(_json(metadata, "save_mat")),
        "asc_shapes": numpy.asarray(
            _json([array.shape for array in arrays], "save_mat")
        ),
        "asc_dtypes": numpy.asarray(
            _json([array.dtype.str for array in arrays], "save_mat")
        ),
    }
    payload.update(
        {f"asc_leaf_{index}": array for index, array in enumerate(arrays)}
    )

    def write(temporary: pathlib.Path) -> None:
        scipy.io.savemat(
            temporary,
            payload,
            appendmat=False,
            do_compression=do_compression,
            long_field_names=True,
        )

    return _atomic(path, write, "save_mat")


def _mat_string(value: object) -> str:
    array = numpy.asarray(value).squeeze()
    if array.ndim == 0:
        return str(array.item())
    return "".join(str(item) for item in array.tolist())


def _restore_mat_dtype(
    leaf: numpy.ndarray, dtype: numpy.dtype[typing.Any]
) -> numpy.ndarray:
    """Restore recorded MAT dtype only when the stored values round-trip."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            restored = leaf.astype(dtype, copy=False)
            round_trip = restored.astype(leaf.dtype, copy=False)
    except (OverflowError, TypeError, ValueError, Warning) as exception:
        raise errors.DataFormatError(
            "load_mat: stored leaf does not match its declared dtype"
        ) from exception
    if leaf.dtype.kind == "c":
        real_equal = (leaf.real == round_trip.real) | (
            numpy.isnan(leaf.real) & numpy.isnan(round_trip.real)
        )
        imaginary_equal = (leaf.imag == round_trip.imag) | (
            numpy.isnan(leaf.imag) & numpy.isnan(round_trip.imag)
        )
        values_equal = bool(numpy.all(real_equal & imaginary_equal))
    elif leaf.dtype.kind == "f":
        equal = (leaf == round_trip) | (
            numpy.isnan(leaf) & numpy.isnan(round_trip)
        )
        values_equal = bool(numpy.all(equal))
    else:
        values_equal = bool(numpy.array_equal(leaf, round_trip))
    if not values_equal:
        raise errors.DataFormatError(
            "load_mat: stored leaf does not match its declared dtype"
        )
    return restored


def load_mat(path: PathLike, *, destination: object | None = None) -> TreeData:
    """Load the safe asc numeric MATLAB subset and remove MATLAB metadata."""
    try:
        import scipy.io
    except ModuleNotFoundError as exception:
        raise errors.BackendUnavailableError(
            "load_mat: install the asc-py[io-mat] extra"
        ) from exception
    try:
        payload = scipy.io.loadmat(
            path, appendmat=False, struct_as_record=False, squeeze_me=False
        )
        _require_schema_one(payload["asc_schema"])
        spec = TreeSpec.from_json(_mat_string(payload["asc_tree_spec"]))
        _reject_custom_tree_spec(spec, "load_mat")
        metadata = _strict_json(
            _mat_string(payload["asc_metadata"]), "load_mat"
        )
        shapes = _strict_json(_mat_string(payload["asc_shapes"]), "load_mat")
        dtypes = _strict_json(_mat_string(payload["asc_dtypes"]), "load_mat")
        if (
            not isinstance(shapes, list)
            or not isinstance(dtypes, list)
            or len(shapes) != spec.num_leaves
            or len(dtypes) != spec.num_leaves
            or any(
                not isinstance(shape, list)
                or any(
                    isinstance(extent, bool)
                    or not isinstance(extent, int)
                    or extent < 0
                    for extent in shape
                )
                for shape in shapes
            )
            or any(not isinstance(dtype, str) or not dtype for dtype in dtypes)
        ):
            raise ValueError("leaf metadata does not match the tree")
        leaves = []
        for index in range(spec.num_leaves):
            leaf = numpy.asarray(payload[f"asc_leaf_{index}"])
            _reject_non_numeric(leaf, "load_mat")
            restored_dtype = numpy.dtype(dtypes[index])
            if restored_dtype.kind not in "biufc" or not (
                _dtype.is_supported_dtype("numpy", restored_dtype)
            ):
                raise errors.DataFormatError(
                    "load_mat: every array must have a supported numeric dtype"
                )
            restored = _restore_mat_dtype(leaf, restored_dtype)
            _reject_non_numeric(restored, "load_mat")
            leaves.append(restored.reshape(tuple(shapes[index])))
    except (
        OSError,
        OverflowError,
        IndexError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        scipy.io.matlab.MatReadError,
    ) as exception:
        raise errors.DataFormatError(
            f"load_mat: invalid asc MATLAB file {str(path)!r}"
        ) from exception
    try:
        tree = tree_unflatten(spec, leaves)
    except Exception as exception:  # pylint: disable=broad-exception-caught
        raise errors.DataFormatError(
            f"load_mat: invalid asc MATLAB file {str(path)!r}"
        ) from exception
    return TreeData(_destination(tree, destination), metadata)


__all__ = [
    "CSVData",
    "NPZData",
    "TreeData",
    "load_csv",
    "load_hdf5",
    "load_mat",
    "load_npy",
    "load_npz",
    "save_csv",
    "save_hdf5",
    "save_mat",
    "save_npy",
    "save_npz",
]
