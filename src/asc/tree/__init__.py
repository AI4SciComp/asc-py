# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Stable backend-neutral PyTree utilities."""

from __future__ import annotations

import collections
import collections.abc
import dataclasses
import json
import keyword
import threading
import typing

from asc import _array_api_compat, errors

PathEntry = int | str
Path = tuple[PathEntry, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class _FrozenJsonObject(collections.abc.Mapping[str, object]):
    """Immutable mapping representation for JSON object metadata."""

    entries: tuple[tuple[str, object], ...]

    def __getitem__(self, key: str) -> object:
        for candidate, value in self.entries:
            if candidate == key:
                return value
        raise KeyError(key)

    def __iter__(self) -> typing.Iterator[str]:
        return (key for key, _value in self.entries)

    def __len__(self) -> int:
        return len(self.entries)


def _freeze_json_value(value: object) -> object:
    """Deep-freeze a value already normalized through the JSON decoder."""
    if isinstance(value, dict):
        return _FrozenJsonObject(
            tuple(
                (key, _freeze_json_value(item)) for key, item in value.items()
            )
        )
    if isinstance(value, list):
        return tuple(_freeze_json_value(item) for item in value)
    return value


def _thaw_json_value(value: object) -> object:
    """Return fresh JSON-native containers from immutable metadata."""
    if isinstance(value, _FrozenJsonObject):
        return {key: _thaw_json_value(item) for key, item in value.entries}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


def _normalize_custom_metadata(value: object) -> object:
    """Canonicalize custom metadata and deep-freeze all containers."""
    normalized = json.loads(json.dumps(value, allow_nan=False))
    return _freeze_json_value(normalized)


@dataclasses.dataclass(frozen=True, slots=True, eq=False)
class TreeSpec:
    """Immutable tree structure with safe JSON metadata."""

    kind: str
    metadata: tuple[object, ...] = ()
    children: tuple[TreeSpec, ...] = ()
    node_type: type[object] | None = dataclasses.field(
        default=None, compare=False, repr=False
    )
    node_data: object | None = dataclasses.field(
        default=None, compare=False, repr=False
    )

    def __post_init__(self) -> None:
        """Defensively freeze custom metadata supplied to the public record."""
        if self.kind != "custom" or len(self.metadata) != 2:
            return
        try:
            frozen = _normalize_custom_metadata(
                _thaw_json_value(self.metadata[1])
            )
        except (TypeError, ValueError) as exception:
            raise ValueError(
                "TreeSpec: custom metadata must be JSON-safe"
            ) from exception
        object.__setattr__(self, "metadata", (self.metadata[0], frozen))

    def __eq__(self, other: object) -> bool:
        """Compare complete structure and live container identity strictly."""
        if not isinstance(other, TreeSpec):
            return NotImplemented
        return (
            self.kind == other.kind
            and self.metadata == other.metadata
            and self.children == other.children
            and self.node_type is other.node_type
            and self.node_data == other.node_data
        )

    def __hash__(self) -> int:
        """Return a hash consistent with live/serialized equality."""
        return hash(self.kind)

    def is_compatible(self, other: object) -> bool:
        """Return whether live and serialized structures can share leaves."""
        if not isinstance(other, TreeSpec):
            return False
        if (
            self.kind != other.kind
            or self.metadata != other.metadata
            or len(self.children) != len(other.children)
        ):
            return False
        if (
            self.node_type is not None
            and other.node_type is not None
            and (
                self.node_type is not other.node_type
                or self.node_data != other.node_data
            )
        ):
            return False
        return all(
            first.is_compatible(second)
            for first, second in zip(self.children, other.children, strict=True)
        )

    @property
    def num_leaves(self) -> int:
        """Return the number of leaves represented by this specification."""
        if self.kind == "leaf":
            return 1
        return sum(child.num_leaves for child in self.children)

    def to_json(self) -> str:
        """Serialize structure metadata without pickle or dynamic imports."""
        _validate_tree_spec(self)

        def encode(spec: TreeSpec) -> dict[str, object]:
            metadata: object = spec.metadata
            if spec.kind == "custom":
                metadata = (
                    spec.metadata[0],
                    _thaw_json_value(spec.metadata[1]),
                )
            return {
                "kind": spec.kind,
                "metadata": metadata,
                "children": [encode(child) for child in spec.children],
            }

        return json.dumps(
            {"schema": 1, "tree": encode(self)},
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, document: str) -> TreeSpec:
        """Load versioned structure JSON without importing referenced code."""

        def reject_constant(value: str) -> typing.NoReturn:
            raise ValueError(f"non-finite JSON constant {value!r}")

        def decode(payload: object) -> TreeSpec:
            if not isinstance(payload, dict):
                raise ValueError("tree must be an object")
            if set(payload) != {"kind", "metadata", "children"}:
                raise ValueError("tree fields are missing or unexpected")
            kind = payload["kind"]
            metadata = payload["metadata"]
            children = payload["children"]
            if (
                not isinstance(kind, str)
                or not isinstance(metadata, list)
                or not isinstance(children, list)
            ):
                raise ValueError("invalid tree fields")
            return cls(
                kind,
                tuple(metadata),
                tuple(decode(child) for child in children),
            )

        try:
            payload = json.loads(document, parse_constant=reject_constant)
            if not isinstance(payload, dict) or set(payload) != {
                "schema",
                "tree",
            }:
                raise ValueError("document fields are missing or unexpected")
            if (
                not isinstance(payload["schema"], int)
                or isinstance(payload["schema"], bool)
                or payload["schema"] != 1
            ):
                raise ValueError("unsupported schema")
            result = decode(payload["tree"])
            _validate_tree_spec(result)
            return result
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as exception:
            raise ValueError(
                "TreeSpec.from_json: invalid safe tree specification"
            ) from exception


def _validate_tree_spec(spec: TreeSpec) -> None:
    """Reject malformed structures before traversal or serialization."""
    if not isinstance(spec, TreeSpec):
        raise ValueError("tree specification nodes must be TreeSpec values")
    if not isinstance(spec.metadata, tuple) or not isinstance(
        spec.children, tuple
    ):
        raise ValueError("tree specification fields must be tuples")
    if any(not isinstance(child, TreeSpec) for child in spec.children):
        raise ValueError("tree specification children must be TreeSpec values")
    if spec.kind == "leaf":
        if spec.metadata or spec.children:
            raise ValueError("leaf nodes cannot have metadata or children")
    elif spec.kind in {"tuple", "list"}:
        if spec.metadata:
            raise ValueError("sequence nodes cannot have metadata")
    elif spec.kind == "mapping":
        keys = spec.metadata
        valid_keys = all(
            isinstance(key, str)
            or (isinstance(key, int) and not isinstance(key, bool))
            for key in keys
        )
        if (
            len(keys) != len(spec.children)
            or not valid_keys
            or len(set(keys)) != len(keys)
        ):
            raise ValueError(
                "mapping keys must be unique strings or integers and match "
                "the child count"
            )
    elif spec.kind in {"namedtuple", "dataclass"}:
        names = spec.metadata
        if (
            not names
            or any(not isinstance(name, str) or not name for name in names)
            or len(names[1:]) != len(spec.children)
            or len(set(names[1:])) != len(names[1:])
        ):
            raise ValueError(
                f"{spec.kind} metadata must name the type and every child"
            )
        if spec.kind == "namedtuple":
            try:
                regenerated = collections.namedtuple(
                    typing.cast(str, names[0]), names[1:], rename=True
                )
            except (TypeError, ValueError) as exception:
                raise ValueError(
                    "namedtuple metadata must contain valid generated names"
                ) from exception
            if regenerated._fields != names[1:]:
                raise ValueError(
                    "namedtuple metadata must preserve generated field names"
                )
        elif any(
            not typing.cast(str, name).isidentifier()
            or keyword.iskeyword(typing.cast(str, name))
            for name in names
        ):
            raise ValueError(
                "dataclass metadata must contain valid Python identifiers"
            )
    elif spec.kind == "custom":
        if (
            len(spec.metadata) != 2
            or not isinstance(spec.metadata[0], str)
            or not spec.metadata[0]
        ):
            raise ValueError("custom nodes require a non-empty tag and data")
    else:
        raise ValueError(f"unknown tree kind {spec.kind!r}")
    for child in spec.children:
        _validate_tree_spec(child)


@dataclasses.dataclass(frozen=True, slots=True)
class _CustomNode:
    tag: str
    flatten: typing.Callable[[object], tuple[typing.Iterable[object], object]]
    unflatten: typing.Callable[[object, typing.Iterable[object]], object]


_REGISTRY_LOCK = threading.RLock()
_REGISTRY: dict[type[object], _CustomNode] = {}
_TAGS: dict[str, _CustomNode] = {}


def _rebuild_mapping(
    node_type: type[object] | None,
    node_data: object | None,
    items: tuple[tuple[object, object], ...],
) -> object:
    """Reconstruct a supported mapping without degrading its native type."""
    if node_type is None or node_type is dict:
        return dict(items)
    try:
        if issubclass(node_type, collections.defaultdict):
            return node_type(node_data, items)
        try:
            return node_type(dict(items))
        except TypeError:
            return node_type(items)
    except (TypeError, ValueError) as exception:
        raise TypeError(
            "tree_unflatten: mapping type must reconstruct from key-value "
            "pairs; register a custom PyTree node otherwise"
        ) from exception


def _rebuild_dataclass(
    node_type: type[object], values: collections.abc.Mapping[str, object]
) -> object:
    """Restore persistent dataclass state without rerunning initialization."""
    result = object.__new__(node_type)
    for field in dataclasses.fields(node_type):
        object.__setattr__(result, field.name, values[field.name])
    return result


def register_pytree_node(
    node_type: type[object],
    flatten_func: typing.Callable[
        [object], tuple[typing.Iterable[object], object]
    ],
    unflatten_func: typing.Callable[[object, typing.Iterable[object]], object],
    *,
    name: str | None = None,
) -> None:
    """Register one immutable custom-node contract thread-safely."""
    tag = (
        f"{node_type.__module__}.{node_type.__qualname__}"
        if name is None
        else name
    )
    if not tag or tag.strip() != tag:
        raise ValueError(
            "register_pytree_node: name must be non-empty and trimmed"
        )
    with _REGISTRY_LOCK:
        if node_type in _REGISTRY or tag in _TAGS:
            raise ValueError(
                f"register_pytree_node: duplicate type or name {tag!r}"
            )
        node = _CustomNode(tag, flatten_func, unflatten_func)
        _REGISTRY[node_type] = node
        _TAGS[tag] = node


def tree_flatten(tree: object) -> tuple[list[object], TreeSpec]:
    """Flatten a supported tree in deterministic left-to-right order."""
    leaves: list[object] = []

    def visit(value: object) -> TreeSpec:
        with _REGISTRY_LOCK:
            custom = _REGISTRY.get(type(value))
        if custom is not None:
            children, metadata = custom.flatten(value)
            materialized = tuple(children)
            try:
                metadata = json.loads(json.dumps(metadata, allow_nan=False))
            except (TypeError, ValueError) as exception:
                raise TypeError(
                    "tree_flatten: custom-node metadata must be JSON-safe"
                ) from exception
            return TreeSpec(
                "custom",
                (custom.tag, metadata),
                tuple(visit(child) for child in materialized),
                type(value),
            )
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            persistent_fields = {
                field.name for field in dataclasses.fields(value)
            }
            required_init_variables = tuple(
                field.name
                for field in value.__dataclass_fields__.values()
                if field.name not in persistent_fields
                and field.init
                and field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            )
            if required_init_variables:
                raise TypeError(
                    "tree_flatten: dataclasses with required InitVar "
                    "parameters are unsupported; observed "
                    f"{required_init_variables!r}"
                )
            fields = tuple(field.name for field in dataclasses.fields(value))
            children = tuple(visit(getattr(value, field)) for field in fields)
            return TreeSpec(
                "dataclass",
                (type(value).__name__, *fields),
                children,
                type(value),
            )
        if isinstance(value, tuple) and hasattr(type(value), "_fields"):
            fields = tuple(type(value)._fields)
            return TreeSpec(
                "namedtuple",
                (type(value).__name__, *fields),
                tuple(visit(child) for child in value),
                type(value),
            )
        if isinstance(value, tuple):
            return TreeSpec("tuple", (), tuple(visit(child) for child in value))
        if isinstance(value, list):
            return TreeSpec("list", (), tuple(visit(child) for child in value))
        if isinstance(value, collections.abc.Mapping):
            keys = tuple(value.keys())
            if not all(
                isinstance(key, str)
                or (isinstance(key, int) and not isinstance(key, bool))
                for key in keys
            ):
                raise TypeError(
                    "tree_flatten: mapping keys must be strings or integers"
                )
            return TreeSpec(
                "mapping",
                keys,
                tuple(visit(value[key]) for key in keys),
                type(value),
                (
                    value.default_factory
                    if isinstance(value, collections.defaultdict)
                    else None
                ),
            )
        leaves.append(value)
        return TreeSpec("leaf")

    return leaves, visit(tree)


def tree_unflatten(spec: TreeSpec, leaves: typing.Iterable[object]) -> object:
    """Reconstruct a tree and reject missing or excess leaves."""
    _validate_tree_spec(spec)
    iterator = iter(leaves)

    def build(node: TreeSpec) -> object:
        if node.kind == "leaf":
            try:
                return next(iterator)
            except StopIteration as exception:
                raise ValueError(
                    "tree_unflatten: too few leaves"
                ) from exception
        children = [build(child) for child in node.children]
        if node.kind == "tuple":
            return tuple(children)
        if node.kind == "list":
            return children
        if node.kind == "mapping":
            items = tuple(zip(node.metadata, children, strict=True))
            return _rebuild_mapping(node.node_type, node.node_data, items)
        if node.kind == "namedtuple":
            node_type = node.node_type
            if node_type is None:
                node_type = collections.namedtuple(
                    typing.cast(str, node.metadata[0]),
                    node.metadata[1:],
                    rename=True,
                )
            return node_type(*children)
        if node.kind == "dataclass":
            node_type = node.node_type
            if node_type is None:
                node_type = dataclasses.make_dataclass(
                    typing.cast(str, node.metadata[0]),
                    [
                        (typing.cast(str, name), object)
                        for name in node.metadata[1:]
                    ],
                    frozen=True,
                )
            values = dict(zip(node.metadata[1:], children, strict=True))
            return _rebuild_dataclass(node_type, values)
        if node.kind == "custom":
            tag = typing.cast(str, node.metadata[0])
            with _REGISTRY_LOCK:
                custom = _TAGS.get(tag)
            if custom is None:
                raise ValueError(
                    f"tree_unflatten: custom node {tag!r} is not registered"
                )
            metadata = _thaw_json_value(node.metadata[1])
            return custom.unflatten(metadata, children)
        raise ValueError(f"tree_unflatten: unknown tree kind {node.kind!r}")

    result = build(spec)
    try:
        next(iterator)
    except StopIteration:
        return result
    raise ValueError("tree_unflatten: too many leaves")


def tree_leaves(tree: object) -> list[object]:
    """Return leaves without modifying the input tree."""
    return tree_flatten(tree)[0]


def tree_structure(tree: object) -> TreeSpec:
    """Return only immutable structure metadata."""
    return tree_flatten(tree)[1]


def tree_map(
    function: typing.Callable[..., object],
    tree: object,
    *rest: object,
) -> object:
    """Map one function over strictly matching tree structures."""
    leaves, spec = tree_flatten(tree)
    other_leaves: list[list[object]] = []
    for position, other in enumerate(rest, start=1):
        flat, other_spec = tree_flatten(other)
        if other_spec != spec:
            raise ValueError(
                f"tree_map: tree {position} structure does not match the first"
            )
        other_leaves.append(flat)
    mapped = [
        function(leaf, *(flat[index] for flat in other_leaves))
        for index, leaf in enumerate(leaves)
    ]
    return tree_unflatten(spec, mapped)


def _paths(spec: TreeSpec, prefix: Path = ()) -> list[Path]:
    if spec.kind == "leaf":
        return [prefix]
    if spec.kind in {"mapping", "dataclass", "namedtuple"}:
        entries = spec.metadata if spec.kind == "mapping" else spec.metadata[1:]
    else:
        entries = tuple(range(len(spec.children)))
    result: list[Path] = []
    for entry, child in zip(entries, spec.children, strict=True):
        result.extend(_paths(child, (*prefix, typing.cast(PathEntry, entry))))
    return result


def tree_map_with_path(
    function: typing.Callable[..., object],
    tree: object,
    *rest: object,
) -> object:
    """Map a function receiving a stable path before each leaf."""
    leaves, spec = tree_flatten(tree)
    other_leaves: list[list[object]] = []
    for position, other in enumerate(rest, start=1):
        flat, other_spec = tree_flatten(other)
        if other_spec != spec:
            raise ValueError(
                f"tree_map_with_path: tree {position} structure mismatch"
            )
        other_leaves.append(flat)
    mapped = [
        function(path, leaf, *(flat[index] for flat in other_leaves))
        for index, (path, leaf) in enumerate(
            zip(_paths(spec), leaves, strict=True)
        )
    ]
    return tree_unflatten(spec, mapped)


def tree_all(predicate: typing.Callable[[object], bool], tree: object) -> bool:
    """Return whether a predicate accepts every leaf."""
    return all(predicate(leaf) for leaf in tree_leaves(tree))


def tree_any(predicate: typing.Callable[[object], bool], tree: object) -> bool:
    """Return whether a predicate accepts at least one leaf."""
    return any(predicate(leaf) for leaf in tree_leaves(tree))


def tree_get(tree: object, path: Path) -> object:
    """Get one node by an integer/string path."""
    value = tree
    for position, entry in enumerate(path):
        try:
            if isinstance(entry, bool) or not isinstance(entry, (int, str)):
                raise TypeError(
                    "tree paths use strings or non-Boolean integers"
                )
            with _REGISTRY_LOCK:
                custom = _REGISTRY.get(type(value))
            if custom is not None:
                if isinstance(entry, bool) or not isinstance(entry, int):
                    raise TypeError("custom-node paths use integer positions")
                children, _ = custom.flatten(value)
                value = tuple(children)[entry]
            elif dataclasses.is_dataclass(value) and not isinstance(
                value, type
            ):
                fields = {field.name for field in dataclasses.fields(value)}
                if not isinstance(entry, str) or entry not in fields:
                    raise TypeError("dataclass paths use field names")
                value = getattr(value, entry)
            elif isinstance(value, tuple) and hasattr(type(value), "_fields"):
                fields = tuple(type(value)._fields)
                if isinstance(entry, str):
                    if entry not in fields:
                        raise TypeError("named-tuple path has no such field")
                    value = getattr(value, entry)
                else:
                    value = value[entry]
            elif isinstance(value, (tuple, list)) and isinstance(entry, int):
                value = value[entry]  # type: ignore[index]
            elif isinstance(value, collections.abc.Mapping):
                value = value[entry]
            else:
                raise TypeError("tree paths cannot traverse terminal leaves")
        except (AttributeError, IndexError, KeyError, TypeError) as exception:
            raise KeyError(
                f"tree_get: invalid path prefix {path[: position + 1]!r}"
            ) from exception
    return value


def _replace_named_tuple(
    target: tuple[object, ...], entry: object, value: object
) -> object:
    """Replace one named-tuple field while preserving its concrete type."""
    native_target = typing.cast(typing.Any, target)
    fields = tuple(type(native_target)._fields)
    try:
        if isinstance(entry, bool):
            raise TypeError("named-tuple paths reject Boolean indices")
        field = fields[entry] if isinstance(entry, int) else entry
        if not isinstance(field, str) or field not in fields:
            raise KeyError(entry)
        return native_target._replace(**{field: value})
    except (IndexError, KeyError, TypeError, ValueError) as exception:
        raise KeyError(
            f"tree_replace: invalid path component {entry!r}"
        ) from exception


def tree_replace(tree: object, path: Path, value: object) -> object:
    """Return a tree with one path immutably replaced."""
    if not path:
        return value
    target = tree_get(tree, path[:-1])
    entry = path[-1]
    if isinstance(entry, bool) or not isinstance(entry, (int, str)):
        raise KeyError(f"tree_replace: invalid path component {entry!r}")
    with _REGISTRY_LOCK:
        custom = _REGISTRY.get(type(target))
    if custom is not None:
        if isinstance(entry, bool) or not isinstance(entry, int):
            raise KeyError(f"tree_replace: invalid path component {entry!r}")
        children, metadata = custom.flatten(target)
        items = list(children)
        try:
            items[entry] = value
        except IndexError as exception:
            raise KeyError(
                f"tree_replace: invalid path component {entry!r}"
            ) from exception
        replacement = custom.unflatten(metadata, items)
    elif isinstance(target, collections.abc.Mapping):
        if entry not in target:
            raise KeyError(f"tree_replace: invalid path component {entry!r}")
        items = tuple(
            (key, value if key == entry else item)
            for key, item in target.items()
        )
        factory = (
            target.default_factory
            if isinstance(target, collections.defaultdict)
            else None
        )
        replacement = _rebuild_mapping(type(target), factory, items)
    elif dataclasses.is_dataclass(target) and isinstance(entry, str):
        fields = {field.name: field for field in dataclasses.fields(target)}
        if entry not in fields:
            raise KeyError(f"tree_replace: invalid path component {entry!r}")
        values = {
            name: value if name == entry else getattr(target, name)
            for name in fields
        }
        replacement = _rebuild_dataclass(type(target), values)
    elif isinstance(target, tuple) and hasattr(type(target), "_fields"):
        replacement = _replace_named_tuple(target, entry, value)
    elif (
        isinstance(target, tuple)
        and isinstance(entry, int)
        and not isinstance(entry, bool)
    ):
        items = list(target)
        try:
            items[entry] = value
        except IndexError as exception:
            raise KeyError(
                f"tree_replace: invalid path component {entry!r}"
            ) from exception
        replacement = tuple(items)
    elif (
        isinstance(target, list)
        and isinstance(entry, int)
        and not isinstance(entry, bool)
    ):
        replacement = list(target)
        try:
            replacement[entry] = value
        except IndexError as exception:
            raise KeyError(
                f"tree_replace: invalid path component {entry!r}"
            ) from exception
    else:
        raise KeyError(f"tree_replace: invalid path component {entry!r}")
    return tree_replace(tree, path[:-1], replacement)


def tree_array_namespace(tree: object) -> object:
    """Infer one namespace from all native array leaves."""
    from asc.core.namespace import array_namespace

    arrays = [
        leaf
        for leaf in tree_leaves(tree)
        if _array_api_compat.compat.is_array_api_obj(leaf)
    ]
    if not arrays:
        raise errors.NamespaceError(
            "tree_array_namespace: tree contains no native array leaves"
        )
    return array_namespace(*arrays)


def tree_to_backend(
    tree: object,
    destination: object,
    *,
    dtype: object | None = None,
    device: object | None = "cpu",
    copy: bool | None = True,
) -> object:
    """Explicitly convert every native array leaf to one backend."""
    from asc.conversion import convert_array

    def convert(leaf: object) -> object:
        if not _array_api_compat.compat.is_array_api_obj(leaf):
            return leaf
        return convert_array(
            leaf,
            destination,
            dtype=dtype,
            device=device,
            copy=copy,
        )

    return tree_map(convert, tree)


def tree_to_device(
    tree: object, device: object, *, copy: bool | None = None
) -> object:
    """Explicitly move each native array leaf within its own backend."""
    from asc.conversion import to_device

    return tree_map(
        lambda leaf: (
            to_device(leaf, device, copy=copy)
            if _array_api_compat.compat.is_array_api_obj(leaf)
            else leaf
        ),
        tree,
    )


def tree_to_numpy(
    tree: object,
    *,
    allow_detach: bool = False,
    allow_transfer: bool = False,
    copy: bool = True,
) -> object:
    """Explicitly convert every native array leaf to NumPy."""
    from asc.conversion import to_numpy

    return tree_map(
        lambda leaf: (
            to_numpy(
                leaf,
                allow_detach=allow_detach,
                allow_transfer=allow_transfer,
                copy=copy,
            )
            if _array_api_compat.compat.is_array_api_obj(leaf)
            else leaf
        ),
        tree,
    )


__all__ = [
    "Path",
    "PathEntry",
    "TreeSpec",
    "register_pytree_node",
    "tree_all",
    "tree_any",
    "tree_array_namespace",
    "tree_flatten",
    "tree_get",
    "tree_leaves",
    "tree_map",
    "tree_map_with_path",
    "tree_replace",
    "tree_structure",
    "tree_to_backend",
    "tree_to_device",
    "tree_to_numpy",
    "tree_unflatten",
]
