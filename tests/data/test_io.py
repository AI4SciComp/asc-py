# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Safe array and tree persistence contracts."""

from __future__ import annotations

import io
import json
import pathlib
import typing
import warnings
import zipfile

import numpy
import pytest

import asc
from asc import data


@pytest.mark.parametrize(
    "options",
    ({"compression": "bogus"}, {"chunks": (3, 3)}),
)
def test_hdf5_dataset_creation_errors_are_translated(
    tmp_path: pathlib.Path,
    options: dict[str, object],
) -> None:
    with pytest.raises(asc.DataFormatError, match="compression and chunk"):
        data.save_hdf5(
            tmp_path / "invalid-options.h5",
            {"value": numpy.ones((2,), dtype=numpy.float32)},
            **options,
        )


@pytest.mark.parametrize("destination", (None, "torch", "jax"))
def test_npy_npz_csv_round_trips(
    tmp_path: pathlib.Path, destination: str | None
) -> None:
    array = numpy.arange(6, dtype=numpy.float32).reshape(3, 2)
    npy = data.save_npy(tmp_path / "values.npy", array)
    loaded = data.load_npy(npy, destination=destination)
    numpy.testing.assert_allclose(numpy.asarray(loaded), array)
    if destination is not None:
        assert asc.backend_of(loaded) == destination
    archive = data.save_npz(
        tmp_path / "values.npz",
        {"x": array, "y": array + 1},
        metadata={"unit": "m", "version": 1},
        compressed=False,
    )
    restored = data.load_npz(archive, destination=destination)
    assert tuple(restored.arrays) == ("x", "y")
    assert restored.metadata == {"unit": "m", "version": 1}
    csv = data.save_csv(
        tmp_path / "values.csv",
        array,
        header=("first", "second"),
        delimiter=";",
    )
    table = data.load_csv(
        csv,
        header=True,
        delimiter=";",
        dtype=numpy.float32,
        destination=destination,
    )
    assert table.header == ("first", "second")
    numpy.testing.assert_allclose(numpy.asarray(table.array), array)


def test_npy_mmap_graph_and_safety_policies(tmp_path: pathlib.Path) -> None:
    array = numpy.arange(4, dtype=numpy.float32)
    path = data.save_npy(tmp_path / "array.npy", array)
    mapped = data.load_npy(path, mmap_mode="r")
    assert isinstance(mapped, numpy.memmap)
    with pytest.raises(asc.DataFormatError, match="mmap"):
        data.load_npy(path, mmap_mode="bad")  # type: ignore[arg-type]
    with pytest.raises(asc.DataFormatError, match="Boolean"):
        data.save_npy(path, array, allow_detach=1)  # type: ignore[arg-type]
    with pytest.raises(asc.DataFormatError, match="object"):
        data.save_npy(tmp_path / "unsafe.npy", numpy.asarray([object()]))
    numpy.save(tmp_path / "pickle.npy", numpy.asarray([{"x": 1}], dtype=object))
    with pytest.raises(asc.DataFormatError):
        data.load_npy(tmp_path / "pickle.npy")


def test_array_persistence_rejects_implicit_python_coercion(
    tmp_path: pathlib.Path,
) -> None:
    for call in (
        lambda: data.save_npy(tmp_path / "scalar.npy", 1),
        lambda: data.save_npy(tmp_path / "list.npy", [1, 2]),
        lambda: data.save_npz(tmp_path / "list.npz", {"x": [1, 2]}),
        lambda: data.save_csv(tmp_path / "list.csv", [[1, 2], [3, 4]]),
    ):
        with pytest.raises(asc.DataFormatError, match="native"):
            call()


@pytest.mark.backend("torch")
def test_npy_graph_policy(tmp_path: pathlib.Path) -> None:
    import torch

    graph = torch.tensor([1.0], requires_grad=True)
    with pytest.raises(asc.ConversionError, match="allow_transfer"):
        data.save_npy(tmp_path / "graph.npy", graph)
    with pytest.raises(asc.ConversionError, match="graph"):
        data.save_npy(tmp_path / "graph.npy", graph, allow_transfer=True)
    data.save_npy(
        tmp_path / "detached.npy",
        graph,
        allow_detach=True,
        allow_transfer=True,
    )


@pytest.mark.parametrize("backend", ("torch", "jax"))
@pytest.mark.parametrize(
    ("format_name", "suffix"),
    (
        ("npy", ".npy"),
        ("npz", ".npz"),
        ("csv", ".csv"),
        ("hdf5", ".h5"),
        ("mat", ".mat"),
    ),
)
def test_save_requires_explicit_host_permission_for_non_numpy_arrays(
    tmp_path: pathlib.Path,
    backend: str,
    format_name: str,
    suffix: str,
) -> None:
    xp = asc.backend(backend).xp
    array = xp.asarray([[1.0, 2.0]], dtype=xp.float32)
    path = tmp_path / f"values{suffix}"

    def save(**kwargs: bool) -> pathlib.Path:
        if format_name == "npy":
            return data.save_npy(path, array, **kwargs)
        if format_name == "npz":
            return data.save_npz(path, {"values": array}, **kwargs)
        if format_name == "csv":
            return data.save_csv(path, array, **kwargs)
        return getattr(data, f"save_{format_name}")(
            path, {"values": array}, **kwargs
        )

    with pytest.raises(asc.ConversionError, match="allow_transfer=True"):
        save()
    assert save(allow_transfer=True) == path


def test_numpy_scalar_persistence_does_not_require_transfer_permission(
    tmp_path: pathlib.Path,
) -> None:
    scalar = numpy.float32(1.25)
    calls = (
        lambda: data.save_npy(tmp_path / "scalar.npy", scalar),
        lambda: data.save_npz(tmp_path / "scalar.npz", {"value": scalar}),
        lambda: data.save_hdf5(tmp_path / "scalar.h5", {"value": scalar}),
        lambda: data.save_mat(tmp_path / "scalar.mat", {"value": scalar}),
    )

    for save in calls:
        assert save().is_file()


def test_atomic_write_failure_preserves_destination(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "values.npy"
    destination.write_bytes(b"previous release data")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated storage failure")

    monkeypatch.setattr(numpy, "save", fail)
    with pytest.raises(asc.DataFormatError, match=r"save_npy.*filesystem"):
        data.save_npy(destination, numpy.arange(3))
    assert destination.read_bytes() == b"previous release data"
    assert tuple(tmp_path.glob(".values.npy.*.tmp")) == ()


def test_atomic_setup_failure_uses_public_data_error(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_args: object, **_kwargs: object) -> typing.NoReturn:
        raise OSError("simulated temporary-file failure")

    monkeypatch.setattr("asc.data.io.tempfile.NamedTemporaryFile", fail)
    with pytest.raises(asc.DataFormatError, match=r"save_npy.*filesystem"):
        data.save_npy(tmp_path / "values.npy", numpy.arange(3))


def test_unsafe_pickle_requires_an_explicit_warning_acknowledgement(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "object.npy"
    numpy.save(path, numpy.asarray([{"value": 1}], dtype=object))
    with pytest.warns(UserWarning, match="execute arbitrary code"):
        restored = data.load_npy(path, allow_unsafe_pickle=True)
    assert restored[0] == {"value": 1}


def test_npz_csv_validation_and_corruption(tmp_path: pathlib.Path) -> None:
    array = numpy.ones((2, 2), dtype=numpy.float32)
    for arrays in ({}, {"__asc_metadata__": array}, {"": array}):
        with pytest.raises(asc.DataFormatError):
            data.save_npz(tmp_path / "bad.npz", arrays)
    with pytest.raises(asc.DataFormatError, match="JSON"):
        data.save_npz(tmp_path / "bad.npz", {"x": array}, metadata=object())
    for invalid in (
        lambda: data.save_csv(tmp_path / "bad.csv", numpy.ones((2,))),
        lambda: data.save_csv(tmp_path / "bad.csv", array, delimiter="xx"),
        lambda: data.save_csv(tmp_path / "bad.csv", array, header=("x",)),
        lambda: data.load_csv(tmp_path / "missing.csv"),
        lambda: data.load_csv(tmp_path / "bad.csv", dtype=object),
    ):
        with pytest.raises(asc.DataFormatError):
            invalid()
    corrupt = tmp_path / "corrupt.npz"
    corrupt.write_bytes(b"not an archive")
    with pytest.raises(asc.DataFormatError):
        data.load_npz(corrupt)


@pytest.mark.parametrize("loader", (data.load_npy, data.load_npz))
@pytest.mark.parametrize("payload", (b"", b"PK\x03\x04"))
def test_array_loaders_wrap_truncated_parser_failures(
    tmp_path: pathlib.Path,
    loader: typing.Callable[[pathlib.Path], object],
    payload: bytes,
) -> None:
    path = tmp_path / "truncated"
    path.write_bytes(payload)

    with pytest.raises(asc.DataFormatError, match="unable to load safe"):
        loader(path)


def test_npz_rejects_duplicate_logical_names(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "duplicate.npz"
    payload = io.BytesIO()
    numpy.save(payload, numpy.asarray([1.0]))
    with warnings.catch_warnings(), zipfile.ZipFile(path, "w") as archive:
        warnings.simplefilter("ignore", UserWarning)
        archive.writestr("value.npy", payload.getvalue())
        archive.writestr("value.npy", payload.getvalue())

    with pytest.raises(asc.DataFormatError, match="duplicate"):
        data.load_npz(path)


def test_npz_rejects_names_that_alias_physical_members(
    tmp_path: pathlib.Path,
) -> None:
    path = tmp_path / "alias.npz"
    first = io.BytesIO()
    second = io.BytesIO()
    numpy.save(first, numpy.asarray([1]))
    numpy.save(second, numpy.asarray([2]))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("value.npy", first.getvalue())
        archive.writestr("value.npy.npy", second.getvalue())

    with pytest.raises(asc.DataFormatError, match="alias"):
        data.load_npz(path)


def test_npz_rejects_nul_member_names(tmp_path: pathlib.Path) -> None:
    with pytest.raises(asc.DataFormatError, match="NUL"):
        data.save_npz(
            tmp_path / "nul-name.npz",
            {"value\0truncated": numpy.asarray([1])},
        )


def test_array_loaders_normalize_valid_wrong_format_files(
    tmp_path: pathlib.Path,
) -> None:
    array = numpy.arange(3, dtype=numpy.float32)
    npy_path = data.save_npy(tmp_path / "array.npy", array)
    npz_path = data.save_npz(tmp_path / "archive.npz", {"array": array})

    with pytest.raises(asc.DataFormatError, match="not an NPY"):
        data.load_npy(npz_path)
    with pytest.raises(asc.DataFormatError, match="not an NPZ"):
        data.load_npz(npy_path)


@pytest.mark.parametrize(
    ("dtype", "values"),
    (
        (numpy.int32, [[1, -2], [3, 4]]),
        (
            numpy.int64,
            [[numpy.iinfo(numpy.int64).min, numpy.iinfo(numpy.int64).max]],
        ),
        (numpy.uint64, [[0, numpy.iinfo(numpy.uint64).max]]),
    ),
)
def test_csv_integer_round_trip_is_exact(
    tmp_path: pathlib.Path,
    dtype: object,
    values: object,
) -> None:
    source = numpy.asarray(values, dtype=dtype)

    path = data.save_csv(tmp_path / "integers.csv", source)
    restored = data.load_csv(path, dtype=dtype).array

    assert restored.dtype == source.dtype
    numpy.testing.assert_array_equal(restored, source)


@pytest.mark.parametrize("shape", ((0, 2), (2, 0)))
def test_csv_rejects_shapes_that_cannot_round_trip(
    tmp_path: pathlib.Path,
    shape: tuple[int, int],
) -> None:
    with pytest.raises(asc.DataFormatError, match="non-empty"):
        data.save_csv(tmp_path / "empty.csv", numpy.empty(shape))


def test_csv_rejects_an_external_empty_file(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(asc.DataFormatError, match="unable to load"):
        data.load_csv(path)


def test_csv_comment_marker_delimiter_round_trip(
    tmp_path: pathlib.Path,
) -> None:
    source = numpy.asarray([[1.25, 2.5], [3.75, 4.0]], dtype=numpy.float64)

    path = data.save_csv(
        tmp_path / "comment-delimiter.csv", source, delimiter="#"
    )
    restored = data.load_csv(path, delimiter="#").array

    numpy.testing.assert_array_equal(restored, source)


@pytest.mark.parametrize(
    "delimiter",
    (*"eEnNaAiIfFtTyYjJ", ".", "-", "+", "0", " "),
)
def test_csv_rejects_delimiters_in_numeric_grammar(
    tmp_path: pathlib.Path,
    delimiter: str,
) -> None:
    source = numpy.asarray([[-1.5, 2.25]], dtype=numpy.float64)

    with pytest.raises(asc.DataFormatError, match="unambiguous"):
        data.save_csv(tmp_path / "ambiguous.csv", source, delimiter=delimiter)
    with pytest.raises(asc.DataFormatError, match="unambiguous"):
        data.load_csv(tmp_path / "ambiguous.csv", delimiter=delimiter)


def test_csv_percent_delimiter_is_rejected_eagerly(
    tmp_path: pathlib.Path,
) -> None:
    values = numpy.asarray([[1.0, 2.0]])

    with pytest.raises(asc.DataFormatError, match="delimiter"):
        data.save_csv(tmp_path / "percent.csv", values, delimiter="%")
    with pytest.raises(asc.ContextError, match="delimiter"):
        asc.CsvOptions(delimiter="%")


def test_csv_rejects_empty_loaded_header_fields(tmp_path: pathlib.Path) -> None:
    path = tmp_path / "empty-header.csv"
    path.write_text(",value\n1,2\n", encoding="utf-8")

    with pytest.raises(asc.DataFormatError, match="non-empty"):
        data.load_csv(path, header=True)


@pytest.mark.parametrize("format_name", ("hdf5", "mat"))
@pytest.mark.parametrize("destination", (None, "torch", "jax"))
def test_optional_tree_format_round_trip(
    tmp_path: pathlib.Path, format_name: str, destination: str | None
) -> None:
    tree = {
        "x": numpy.arange(3, dtype=numpy.float32),
        "nested": (numpy.asarray([[2]], dtype=numpy.int32),),
    }
    suffix = ".h5" if format_name == "hdf5" else ".mat"
    save = getattr(data, f"save_{format_name}")
    load = getattr(data, f"load_{format_name}")
    path = save(
        tmp_path / f"tree{suffix}",
        tree,
        metadata={"format": format_name},
    )
    restored = load(path, destination=destination)
    assert restored.metadata == {"format": format_name}
    numpy.testing.assert_array_equal(
        numpy.asarray(restored.tree["x"]), tree["x"]
    )
    if destination is not None:
        assert asc.backend_of(restored.tree["x"]) == destination


def test_optional_tree_format_validation(tmp_path: pathlib.Path) -> None:
    tree = {"x": numpy.ones((2,), dtype=numpy.float32)}
    for factory in (
        lambda: data.save_hdf5(tmp_path / "bad.h5", tree, chunks=(0,)),
        lambda: data.save_hdf5(tmp_path / "bad.h5", tree, compression=""),
        lambda: data.save_hdf5(tmp_path / "bad.h5", {"x": "not-array"}),
        lambda: data.save_mat(
            tmp_path / "bad.mat",
            tree,
            do_compression=1,  # type: ignore[arg-type]
        ),
        lambda: data.save_mat(tmp_path / "bad.mat", {"x": "not-array"}),
    ):
        with pytest.raises(asc.DataError):
            factory()
    invalid = tmp_path / "invalid.bin"
    invalid.write_bytes(b"invalid")
    with pytest.raises(asc.DataFormatError):
        data.load_hdf5(invalid)
    with pytest.raises(asc.DataFormatError):
        data.load_mat(invalid)


@pytest.mark.parametrize("format_name", ("hdf5", "mat"))
@pytest.mark.parametrize(
    "array",
    (
        numpy.asarray(["text"]),
        numpy.asarray(["2026-08-12"], dtype="datetime64[D]"),
        numpy.asarray([(1, 2)], dtype=[("x", "i4"), ("y", "i4")]),
    ),
)
def test_tree_persistence_rejects_non_numeric_dtypes(
    tmp_path: pathlib.Path, format_name: str, array: numpy.ndarray
) -> None:
    with pytest.raises(asc.DataFormatError, match="numeric dtype"):
        getattr(data, f"save_{format_name}")(
            tmp_path / f"non-numeric.{format_name}", array
        )


@pytest.mark.parametrize("format_name", ("npy", "npz"))
@pytest.mark.parametrize(
    "array",
    (
        numpy.asarray(["text"]),
        numpy.asarray(["2026-08-12"], dtype="datetime64[D]"),
        numpy.asarray([(1, 2)], dtype=[("x", "i4"), ("y", "i4")]),
    ),
)
def test_npy_npz_reject_non_numeric_dtypes_on_save_and_load(
    tmp_path: pathlib.Path,
    format_name: str,
    array: numpy.ndarray,
) -> None:
    path = tmp_path / f"non-numeric.{format_name}"
    save = getattr(data, f"save_{format_name}")
    with pytest.raises(asc.DataFormatError, match="numeric dtype"):
        save(path, array if format_name == "npy" else {"value": array})

    if format_name == "npy":
        numpy.save(path, array, allow_pickle=False)
    else:
        numpy.savez(path, value=array)
    with pytest.raises(asc.DataFormatError, match="numeric dtype"):
        getattr(data, f"load_{format_name}")(path)


@pytest.mark.parametrize("format_name", ("hdf5", "mat"))
def test_tree_persistence_rejects_non_finite_metadata(
    tmp_path: pathlib.Path, format_name: str
) -> None:
    with pytest.raises(asc.DataFormatError, match="safe JSON"):
        getattr(data, f"save_{format_name}")(
            tmp_path / f"metadata.{format_name}",
            numpy.asarray([1.0]),
            metadata={"value": float("nan")},
        )


def test_mat_round_trip_preserves_bool_and_float16_dtypes(
    tmp_path: pathlib.Path,
) -> None:
    source = {
        "flags": numpy.asarray([True, False], dtype=numpy.bool_),
        "half": numpy.asarray([1.25, -2.5], dtype=numpy.float16),
    }

    path = data.save_mat(tmp_path / "dtypes.mat", source)
    restored = data.load_mat(path).tree

    for name, expected in source.items():
        assert restored[name].dtype == expected.dtype
        numpy.testing.assert_array_equal(restored[name], expected)


@pytest.mark.parametrize("shape", ([-1], [True], [6.0], "6"))
def test_mat_rejects_malformed_shape_metadata(
    tmp_path: pathlib.Path, shape: object
) -> None:
    import scipy.io

    path = data.save_mat(tmp_path / "bad-shape.mat", numpy.arange(6))
    payload = {
        name: value
        for name, value in scipy.io.loadmat(path).items()
        if not name.startswith("__")
    }
    payload["asc_shapes"] = numpy.asarray(json.dumps([shape]))
    scipy.io.savemat(path, payload, appendmat=False)

    with pytest.raises(asc.DataFormatError, match="invalid asc MATLAB"):
        data.load_mat(path)


def test_mat_rejects_non_string_dtype_metadata(tmp_path: pathlib.Path) -> None:
    import scipy.io

    path = data.save_mat(tmp_path / "bad-dtype.mat", numpy.arange(6))
    payload = {
        name: value
        for name, value in scipy.io.loadmat(path).items()
        if not name.startswith("__")
    }
    payload["asc_dtypes"] = numpy.asarray(json.dumps([None]))
    scipy.io.savemat(path, payload, appendmat=False)

    with pytest.raises(asc.DataFormatError, match="invalid asc MATLAB"):
        data.load_mat(path)


def test_mat_rejects_lossy_dtype_restoration(tmp_path: pathlib.Path) -> None:
    import scipy.io

    path = data.save_mat(
        tmp_path / "lossy-dtype.mat", numpy.asarray([1], dtype=numpy.int8)
    )
    payload = {
        name: value
        for name, value in scipy.io.loadmat(path).items()
        if not name.startswith("__")
    }
    payload["asc_leaf_0"] = numpy.asarray([300], dtype=numpy.int64)
    scipy.io.savemat(path, payload, appendmat=False)

    with pytest.raises(asc.DataFormatError, match="declared dtype"):
        data.load_mat(path)


def test_mat_rejects_lossy_complex_nan_component_restoration(
    tmp_path: pathlib.Path,
) -> None:
    import scipy.io

    path = data.save_mat(
        tmp_path / "lossy-complex.mat",
        numpy.asarray([0j], dtype=numpy.complex64),
    )
    payload = {
        name: value
        for name, value in scipy.io.loadmat(path).items()
        if not name.startswith("__")
    }
    payload["asc_leaf_0"] = numpy.asarray(
        [complex(float("nan"), 16_777_217.0)], dtype=numpy.complex128
    )
    scipy.io.savemat(path, payload, appendmat=False)

    with pytest.raises(asc.DataFormatError, match="declared dtype"):
        data.load_mat(path)


def test_mat_wraps_overflowing_schema_metadata(tmp_path: pathlib.Path) -> None:
    import scipy.io

    path = data.save_mat(tmp_path / "schema.mat", numpy.asarray([1]))
    payload = {
        name: value
        for name, value in scipy.io.loadmat(path).items()
        if not name.startswith("__")
    }
    payload["asc_schema"] = numpy.asarray([[numpy.inf]])
    scipy.io.savemat(path, payload, appendmat=False)

    with pytest.raises(asc.DataFormatError, match="invalid asc MATLAB"):
        data.load_mat(path)


@pytest.mark.parametrize("schema", (1.0, 1.5))
def test_mat_rejects_non_integral_schema_storage(
    tmp_path: pathlib.Path, schema: float
) -> None:
    import scipy.io

    path = data.save_mat(tmp_path / "schema-float.mat", numpy.asarray([1]))
    payload = {
        name: value
        for name, value in scipy.io.loadmat(path).items()
        if not name.startswith("__")
    }
    payload["asc_schema"] = numpy.asarray([[schema]])
    scipy.io.savemat(path, payload, appendmat=False)

    with pytest.raises(asc.DataFormatError, match="invalid asc MATLAB"):
        data.load_mat(path)


def test_hdf5_rejects_non_integral_schema_storage(
    tmp_path: pathlib.Path,
) -> None:
    import h5py

    path = data.save_hdf5(tmp_path / "schema-float.h5", numpy.asarray([1]))
    with h5py.File(path, "r+") as file:
        file.attrs["asc_schema"] = 1.0

    with pytest.raises(asc.DataFormatError, match="invalid asc HDF5"):
        data.load_hdf5(path)


@pytest.mark.parametrize(
    ("compression", "chunks"),
    (("gzip", True), (None, (1,))),
)
def test_hdf5_scalar_leaves_ignore_unsupported_storage_options(
    tmp_path: pathlib.Path,
    compression: str | None,
    chunks: bool | tuple[int, ...],
) -> None:
    source = {"scalar": numpy.asarray(3.5, dtype=numpy.float32)}

    path = data.save_hdf5(
        tmp_path / "scalar.h5",
        source,
        compression=compression,
        chunks=chunks,
    )
    restored = data.load_hdf5(path).tree

    assert restored["scalar"].shape == ()
    assert restored["scalar"].dtype == numpy.float32
    assert restored["scalar"].item() == pytest.approx(3.5)


def test_hdf5_false_chunk_policy_writes_contiguous_arrays(
    tmp_path: pathlib.Path,
) -> None:
    import h5py

    source = numpy.arange(4, dtype=numpy.float32)
    path = data.save_hdf5(tmp_path / "contiguous.h5", source, chunks=False)

    with h5py.File(path, "r") as file:
        assert file["leaves"]["0"].chunks is None
    numpy.testing.assert_array_equal(data.load_hdf5(path).tree, source)


def test_hdf5_rejects_compression_with_contiguous_storage(
    tmp_path: pathlib.Path,
) -> None:
    with pytest.raises(asc.DataFormatError, match="incompatible"):
        data.save_hdf5(
            tmp_path / "invalid-storage.h5",
            numpy.arange(4),
            compression="gzip",
            chunks=False,
        )


@pytest.mark.parametrize("storage", ("external-link", "virtual", "external"))
def test_hdf5_safe_loader_rejects_externally_backed_leaves(
    tmp_path: pathlib.Path,
    storage: str,
) -> None:
    import h5py

    source_path = tmp_path / "outside.h5"
    with h5py.File(source_path, "w") as source:
        source.create_dataset(
            "secret", data=numpy.asarray([41.0], dtype=numpy.float32)
        )
    path = data.save_hdf5(
        tmp_path / f"{storage}.h5",
        numpy.asarray([0.0], dtype=numpy.float32),
    )
    with h5py.File(path, "r+") as file:
        group = file["leaves"]
        del group["0"]
        if storage == "external-link":
            group["0"] = h5py.ExternalLink(str(source_path), "/secret")
        elif storage == "virtual":
            layout = h5py.VirtualLayout(shape=(1,), dtype=numpy.float32)
            layout[:] = h5py.VirtualSource(
                str(source_path), "/secret", shape=(1,)
            )
            group.create_virtual_dataset("0", layout)
        else:
            external_path = tmp_path / "outside.raw"
            dataset = group.create_dataset(
                "0",
                shape=(1,),
                dtype=numpy.float32,
                external=[(str(external_path), 0, h5py.h5f.UNLIMITED)],
            )
            dataset[...] = numpy.asarray([42.0], dtype=numpy.float32)

    with pytest.raises(asc.DataFormatError, match="invalid asc HDF5"):
        data.load_hdf5(path)


@pytest.mark.parametrize("storage", ("external-link", "virtual"))
def test_hdf5_safe_loader_rejects_extra_unsafe_members(
    tmp_path: pathlib.Path,
    storage: str,
) -> None:
    import h5py

    source_path = tmp_path / "outside-extra.h5"
    with h5py.File(source_path, "w") as source:
        source.create_dataset(
            "secret", data=numpy.asarray([41.0], dtype=numpy.float32)
        )
    path = data.save_hdf5(
        tmp_path / f"extra-{storage}.h5",
        numpy.asarray([0.0], dtype=numpy.float32),
    )
    with h5py.File(path, "r+") as file:
        group = file["leaves"]
        if storage == "external-link":
            group["extra"] = h5py.ExternalLink(str(source_path), "/secret")
        else:
            layout = h5py.VirtualLayout(shape=(1,), dtype=numpy.float32)
            layout[:] = h5py.VirtualSource(
                str(source_path), "/secret", shape=(1,)
            )
            group.create_virtual_dataset("extra", layout)

    with pytest.raises(asc.DataFormatError, match="invalid asc HDF5"):
        data.load_hdf5(path)
