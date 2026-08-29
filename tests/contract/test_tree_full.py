# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Stable PyTree structure, path, registry, JSON, and array helpers."""

from __future__ import annotations

import collections
import collections.abc
import dataclasses
import operator
import threading

import numpy
import pytest

import asc
from asc import tree


class Pair(collections.namedtuple("PairBase", "left right")):
    """Named tuple fixture."""

    __slots__ = ()


@dataclasses.dataclass(frozen=True)
class Record:
    """Dataclass fixture."""

    value: object
    label: str


def test_flatten_unflatten_order_and_structures() -> None:
    value = {
        "tuple": (1, [2, 3]),
        "named": Pair(4, 5),
        "record": Record(6, "x"),
    }
    leaves, spec = tree.tree_flatten(value)
    assert leaves == [1, 2, 3, 4, 5, 6, "x"]
    assert tree.tree_leaves(value) == leaves
    assert tree.tree_structure(value) == spec
    assert tree.tree_unflatten(spec, leaves) == value
    assert spec.num_leaves == 7
    with pytest.raises(ValueError, match="too few"):
        tree.tree_unflatten(spec, leaves[:-1])
    with pytest.raises(ValueError, match="too many"):
        tree.tree_unflatten(spec, [*leaves, 9])


def test_tree_map_paths_predicates_get_and_replace() -> None:
    value = {"a": [1, 2], "b": Pair(3, 4)}
    assert tree.tree_map(lambda item: item * 2, value) == {
        "a": [2, 4],
        "b": Pair(6, 8),
    }
    paths = tree.tree_map_with_path(lambda path, item: (path, item), value)
    assert paths["a"][1] == (("a", 1), 2)
    assert tree.tree_all(lambda item: item > 0, value)
    assert tree.tree_any(lambda item: item == 4, value)
    assert tree.tree_get(value, ("b", "right")) == 4
    assert tree.tree_get(value, ("b", 1)) == 4
    assert tree.tree_replace(value, ("b", 1), 9) == {
        "a": [1, 2],
        "b": Pair(3, 9),
    }
    assert tree.tree_replace(value, ("a", 1), 9) == {
        "a": [1, 9],
        "b": Pair(3, 4),
    }
    assert tree.tree_replace(value, (), "root") == "root"
    with pytest.raises(ValueError, match="structure"):
        tree.tree_map(lambda first, second: first + second, value, [1, 2])
    with pytest.raises(KeyError, match="prefix"):
        tree.tree_get(value, ("missing",))
    for path in (("a", True), ("b", True)):
        with pytest.raises(KeyError):
            tree.tree_get(value, path)
        with pytest.raises(KeyError):
            tree.tree_replace(value, path, 9)
    for path in ((True,), (False,)):
        integer_mapping = {1: "one", 0: "zero"}
        with pytest.raises(KeyError):
            tree.tree_get(integer_mapping, path)
        with pytest.raises(KeyError):
            tree.tree_replace(integer_mapping, path, "changed")
    for leaf, path in (
        (numpy.asarray([1]), (0,)),
        ("x", ("upper",)),
        (1, ("real",)),
    ):
        with pytest.raises(KeyError, match="prefix"):
            tree.tree_get(leaf, path)


def test_dataclass_replace_changes_exactly_one_field() -> None:
    @dataclasses.dataclass(frozen=True)
    class DerivedRecord:
        value: int
        derived: int = dataclasses.field(init=False)

        def __post_init__(self) -> None:
            object.__setattr__(self, "derived", self.value * 2)

    source = DerivedRecord(3)

    replaced_value = tree.tree_replace(source, ("value",), 4)
    replaced_derived = tree.tree_replace(source, ("derived",), 99)

    assert (replaced_value.value, replaced_value.derived) == (4, 6)
    assert (replaced_derived.value, replaced_derived.derived) == (3, 99)


def test_mapping_nodes_preserve_native_types_and_behavior() -> None:
    mappings: tuple[collections.abc.Mapping[str, int], ...] = (
        collections.OrderedDict((("a", 1), ("b", 2))),
        collections.defaultdict(int, {"a": 1, "b": 2}),
        collections.Counter({"a": 1, "b": 2}),
        collections.UserDict({"a": 1, "b": 2}),
    )

    for source in mappings:
        leaves, spec = tree.tree_flatten(source)
        restored = tree.tree_unflatten(spec, leaves)
        mapped = tree.tree_map(lambda item: item + 1, source)
        replaced = tree.tree_replace(source, ("a",), 9)
        converted = tree.tree_to_backend(source, "numpy")

        assert spec.node_type is type(source)
        assert type(restored) is type(source)
        assert type(mapped) is type(source)
        assert type(replaced) is type(source)
        assert type(converted) is type(source)
        assert tuple(mapped.items()) == (("a", 2), ("b", 3))
        assert replaced["a"] == 9
        if isinstance(source, collections.defaultdict):
            assert restored.default_factory is int


def test_required_dataclass_init_variables_are_rejected_eagerly() -> None:
    @dataclasses.dataclass
    class RequiredInitVariable:
        seed: dataclasses.InitVar[int]
        value: int
        cached: int = dataclasses.field(init=False)

        def __post_init__(self, seed: int) -> None:
            self.cached = seed

    source = RequiredInitVariable(3, 4)

    with pytest.raises(TypeError, match="required InitVar"):
        tree.tree_flatten(source)
    with pytest.raises(asc.CollationError, match="required InitVar"):
        asc.data.default_convert(source)
    with pytest.raises(asc.CollationError, match="required InitVar"):
        asc.data.default_collate((source, source))


def test_combined_loader_preserves_ordered_mapping_batches() -> None:
    loaders = collections.OrderedDict(
        (("first", range(1, 3)), ("second", range(3, 5)))
    )

    batches = list(asc.data.CombinedLoader(loaders))

    assert all(
        isinstance(batch.data, collections.OrderedDict) for batch in batches
    )
    assert tuple(batches[0].data.items()) == (("first", 1), ("second", 3))


def test_tree_spec_safe_json_roundtrip() -> None:
    value = {"a": [1, Pair(2, 3)], 4: Record(5, "six")}
    spec = tree.tree_structure(value)
    restored = tree.TreeSpec.from_json(spec.to_json())
    assert restored != spec
    assert restored.is_compatible(spec)
    rebuilt = tree.tree_unflatten(restored, tree.tree_leaves(value))
    assert tree.tree_leaves(rebuilt) == tree.tree_leaves(value)
    for document in (
        "bad",
        '{"schema":2}',
        '{"schema":2,"tree":{}}',
        '{"schema":1,"tree":{"kind":"tuple","metadata":[],"children":[1]}}',
        '{"schema":1,"tree":{"kind":"tuple","metadata":[],'
        '"children":[],"unexpected":true}}',
        '{"schema":1,"tree":{"kind":1,"metadata":[],"children":[]}}',
        '{"schema":1,"tree":{"kind":"bad","metadata":[],"children":[]}}',
        '{"schema":true,"tree":{"kind":"leaf","metadata":[],"children":[]}}',
        '{"schema":1.0,"tree":{"kind":"leaf","metadata":[],"children":[]}}',
        '{"schema":1,"tree":{"kind":"leaf","metadata":[NaN],"children":[]}}',
        '{"schema":1,"tree":{"kind":"mapping","metadata":["x","x"],'
        '"children":[{"kind":"leaf","metadata":[],"children":[]},'
        '{"kind":"leaf","metadata":[],"children":[]}]}}',
    ):
        with pytest.raises(ValueError, match="TreeSpec"):
            tree.TreeSpec.from_json(document)

    with pytest.raises(ValueError):
        tree.TreeSpec("custom", ("test", float("nan"))).to_json()


def test_tree_spec_json_roundtrip_preserves_renamed_namedtuple_fields() -> None:
    renamed_type = collections.namedtuple(
        "RenamedFields", ("class", "value"), rename=True
    )
    value = renamed_type(1, 2)
    leaves, spec = tree.tree_flatten(value)

    restored_spec = tree.TreeSpec.from_json(spec.to_json())
    restored = tree.tree_unflatten(restored_spec, leaves)

    assert restored._fields == value._fields
    assert restored == value


@pytest.mark.parametrize(
    "spec",
    (
        tree.TreeSpec("leaf", (), (tree.TreeSpec("leaf"),)),
        tree.TreeSpec("tuple", ("unexpected",), ()),
        tree.TreeSpec("mapping", ("only-key",), ()),
        tree.TreeSpec("namedtuple", ("Pair", "left"), ()),
        tree.TreeSpec("dataclass", (), ()),
        tree.TreeSpec("custom", ("tag",), ()),
    ),
)
def test_tree_spec_rejects_structurally_malformed_nodes(
    spec: tree.TreeSpec,
) -> None:
    with pytest.raises(ValueError):
        spec.to_json()
    with pytest.raises(ValueError):
        tree.tree_unflatten(spec, ())


@pytest.mark.parametrize("field_name", ("x-y", "class"))
def test_tree_spec_rejects_invalid_serialized_dataclass_fields(
    field_name: str,
) -> None:
    document = (
        '{"schema":1,"tree":{"kind":"dataclass","metadata":'
        f'["Record","{field_name}"],"children":'
        '[{"kind":"leaf","metadata":[],"children":[]}]}}'
    )

    with pytest.raises(ValueError, match="TreeSpec"):
        tree.TreeSpec.from_json(document)


def test_tree_map_requires_matching_live_container_types() -> None:
    ordered = collections.OrderedDict((("value", 1),))

    with pytest.raises(ValueError, match="structure"):
        tree.tree_map(lambda left, right: left + right, ordered, {"value": 2})


def test_tree_spec_equality_is_strict_and_transitive() -> None:
    ordered = tree.tree_structure(collections.OrderedDict(value=1))
    plain = tree.tree_structure({"value": 1})
    serialized = tree.TreeSpec.from_json(ordered.to_json())

    assert ordered != plain
    assert ordered != serialized
    assert serialized != plain
    assert len({ordered, plain, serialized}) == 3
    assert serialized.is_compatible(ordered)
    assert serialized.is_compatible(plain)


def test_custom_node_registration_and_thread_safety() -> None:
    @dataclasses.dataclass(frozen=True)
    class Box:
        values: tuple[object, ...]
        label: str

    tree.register_pytree_node(
        Box,
        lambda box: (box.values, {"label": box.label, "tags": (1, 2)}),
        lambda metadata, values: Box(tuple(values), metadata["label"]),
        name="tests.Box",
    )
    value = Box((1, 2), "sample")
    leaves, spec = tree.tree_flatten(value)
    assert leaves == [1, 2]
    assert tree.tree_unflatten(spec, leaves) == value
    restored_spec = tree.TreeSpec.from_json(spec.to_json())
    assert restored_spec != spec
    assert restored_spec.is_compatible(spec)
    assert tree.tree_get(value, (1,)) == 2
    assert tree.tree_replace(value, (0,), 9) == Box((9, 2), "sample")
    with pytest.raises(ValueError, match="duplicate"):
        tree.register_pytree_node(
            Box, lambda value: ((), None), lambda metadata, values: value
        )

    failures: list[Exception] = []

    def duplicate() -> None:
        try:
            tree.register_pytree_node(
                Box, lambda value: ((), None), lambda metadata, values: value
            )
        except ValueError:
            return
        except (
            Exception
        ) as exception:  # pragma: no cover - diagnostic safeguard
            failures.append(exception)

    threads = [threading.Thread(target=duplicate) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert failures == []


def test_custom_node_metadata_is_deeply_immutable() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tagged:
        value: object
        label: str

    tree.register_pytree_node(
        Tagged,
        lambda node: (
            (node.value,),
            {"labels": [node.label], "nested": {"version": 1}},
        ),
        lambda metadata, values: Tagged(
            next(iter(values)), metadata["labels"][0]
        ),
        name="tests.ImmutableMetadataTagged",
    )
    value = Tagged(3, "sample")
    leaves, live = tree.tree_flatten(value)
    restored = tree.TreeSpec.from_json(live.to_json())
    replaced = dataclasses.replace(live)

    assert replaced == live

    for spec in (live, restored):
        document = spec.to_json()
        assert isinstance(spec.metadata[1], collections.abc.Mapping)
        custom_metadata = spec.metadata[1]
        labels = custom_metadata["labels"]
        nested = custom_metadata["nested"]
        assert isinstance(nested, collections.abc.Mapping)
        with pytest.raises(AttributeError):
            operator.methodcaller("append", "changed")(labels)
        with pytest.raises(TypeError):
            operator.setitem(nested, "version", 2)
        assert spec.to_json() == document
        assert tree.tree_unflatten(spec, leaves) == value


@pytest.mark.parametrize("backend", ("numpy", "torch", "jax"))
def test_array_tree_helpers(backend: str) -> None:
    selected = asc.backend(backend)
    value = {
        "x": selected.xp.asarray([1.0, 2.0], dtype=selected.xp.float32),
        "metadata": "kept",
    }
    assert asc.backend_info(tree.tree_array_namespace(value)).name == backend
    converted = tree.tree_to_backend(value, "numpy")
    assert isinstance(converted["x"], numpy.ndarray)
    assert converted["metadata"] == "kept"
    moved = tree.tree_to_device(value, "cpu")
    assert asc.backend_of(moved["x"]) == backend
    host = tree.tree_to_numpy(value)
    assert isinstance(host["x"], numpy.ndarray)
    with pytest.raises(asc.NamespaceError, match="no native"):
        tree.tree_array_namespace({"a": 1})


@pytest.mark.backend("torch")
def test_array_tree_helpers_reject_mixed_backends() -> None:
    with pytest.raises(asc.MixedBackendError):
        tree.tree_array_namespace(
            {"a": numpy.asarray([1]), "b": asc.backend("torch").xp.asarray([1])}
        )


def test_tree_rejects_unsafe_keys_metadata_and_unknown_custom() -> None:
    with pytest.raises(TypeError, match="mapping keys"):
        tree.tree_flatten({object(): 1})

    class Unsafe:
        pass

    tree.register_pytree_node(
        Unsafe,
        lambda value: ((), object()),
        lambda metadata, values: Unsafe(),
        name="tests.Unsafe",
    )
    with pytest.raises(TypeError, match="JSON-safe"):
        tree.tree_flatten(Unsafe())
    spec = tree.TreeSpec("custom", ("missing", None), ())
    with pytest.raises(ValueError, match="not registered"):
        tree.tree_unflatten(spec, ())
