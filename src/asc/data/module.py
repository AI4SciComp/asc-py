# Copyright 2026 AI4SciComp contributors.
# SPDX-License-Identifier: Apache-2.0
"""Validated stage-oriented dataset and loader orchestration."""

from __future__ import annotations

import collections
import collections.abc
import types
import typing

from asc import errors
from asc.data.combined import CombinedLoader, Mode
from asc.data.dataset import Dataset, is_map_style_dataset
from asc.data.loader import DataLoader
from asc.data.schema import DataSpec, infer_data_spec
from asc.data.split import DatasetSplits, train_validation_test_split
from asc.random import RandomState

Stage = typing.Literal["train", "validation", "test", "predict"]
_STAGES: tuple[Stage, ...] = ("train", "validation", "test", "predict")


class DataModule:
    """Own finite stage registries and idempotent lifecycle hooks."""

    def __init__(self) -> None:
        self._datasets: dict[
            Stage, collections.OrderedDict[str, Dataset[object]]
        ] = {stage: collections.OrderedDict() for stage in _STAGES}
        self._prepared = False
        self._setup_stages: set[Stage] = set()

    @staticmethod
    def _stage(stage: str) -> Stage:
        if stage not in _STAGES:
            raise errors.DatasetError(
                f"DataModule: stage must be one of {_STAGES!r}; received "
                f"{stage!r}"
            )
        return typing.cast(Stage, stage)

    def add_dataset(
        self, stage: Stage, name: str, dataset: Dataset[object]
    ) -> None:
        """Register one uniquely named dataset without performing I/O."""
        normalized = self._stage(stage)
        if not isinstance(dataset, Dataset) or not is_map_style_dataset(
            dataset
        ):
            raise errors.DatasetError(
                "DataModule.add_dataset: dataset must be a finite Dataset"
            )
        if not name or name.strip() != name:
            raise errors.DatasetError(
                "DataModule.add_dataset: name must be non-empty and trimmed"
            )
        if name in self._datasets[normalized]:
            raise errors.DatasetError(
                f"DataModule.add_dataset: duplicate {normalized}/{name!r}"
            )
        self._datasets[normalized][name] = dataset

    def get_dataset(self, stage: Stage, name: str) -> Dataset[object]:
        """Return one registered dataset or a path-aware error."""
        normalized = self._stage(stage)
        try:
            return self._datasets[normalized][name]
        except KeyError as exception:
            raise errors.DatasetError(
                f"DataModule.get_dataset: unknown {normalized}/{name!r}"
            ) from exception

    def remove_dataset(self, stage: Stage, name: str) -> Dataset[object]:
        """Remove and return one registered dataset."""
        normalized = self._stage(stage)
        try:
            return self._datasets[normalized].pop(name)
        except KeyError as exception:
            raise errors.DatasetError(
                f"DataModule.remove_dataset: unknown {normalized}/{name!r}"
            ) from exception

    def datasets(
        self, stage: Stage
    ) -> collections.abc.Mapping[str, Dataset[object]]:
        """Return a read-only view of one stage registry."""
        normalized = self._stage(stage)
        return types.MappingProxyType(self._datasets[normalized])

    def _prepare_data(self) -> None:
        """Override for one-time external preparation."""

    def _setup(self, stage: Stage) -> None:
        """Override for idempotent stage setup."""

    def _teardown(self, stage: Stage) -> None:
        """Override for idempotent stage teardown."""

    def prepare_data(self) -> None:
        """Run the protected preparation hook at most once."""
        if not self._prepared:
            self._prepare_data()
            self._prepared = True

    def setup(self, stage: Stage) -> None:
        """Run a protected setup hook once per stage."""
        normalized = self._stage(stage)
        if normalized not in self._setup_stages:
            self._setup(normalized)
            self._setup_stages.add(normalized)

    def teardown(self, stage: Stage) -> None:
        """Run teardown once for an active stage."""
        normalized = self._stage(stage)
        if normalized in self._setup_stages:
            self._teardown(normalized)
            self._setup_stages.remove(normalized)

    def loader(self, stage: Stage, name: str, **config: object) -> DataLoader:
        """Create a validated loader for one registered dataset."""
        return DataLoader(self.get_dataset(stage, name), **config)

    def combined_loader(
        self,
        stage: Stage,
        *,
        mode: Mode = "min_size",
        configs: (
            collections.abc.Mapping[str, collections.abc.Mapping[str, object]]
            | None
        ) = None,
        limits: int | object | None = None,
    ) -> CombinedLoader:
        """Create a named loader tree without mutable default configuration."""
        normalized = self._stage(stage)
        configuration = {} if configs is None else dict(configs)
        unknown = set(configuration) - set(self._datasets[normalized])
        if unknown:
            raise errors.DataLoaderError(
                "DataModule.combined_loader: unknown loader configs "
                f"{sorted(unknown)!r}"
            )
        loaders = {
            name: DataLoader(dataset, **dict(configuration.get(name, {})))
            for name, dataset in self._datasets[normalized].items()
        }
        return CombinedLoader(loaders, mode=mode, limits=limits)

    def register_splits(
        self,
        dataset: Dataset[object],
        *,
        name: str,
        train: float,
        validation: float,
        test: float,
        state: RandomState | None = None,
    ) -> DatasetSplits:
        """Create and register reproducible train/validation/test subsets."""
        if not name or name.strip() != name:
            raise errors.DatasetError(
                "DataModule.register_splits: name must be non-empty and trimmed"
            )
        duplicate_stages = [
            stage
            for stage in ("train", "validation", "test")
            if name in self._datasets[stage]
        ]
        if duplicate_stages:
            raise errors.DatasetError(
                "DataModule.register_splits: duplicate split name "
                f"{name!r} in stages {duplicate_stages!r}"
            )
        splits = train_validation_test_split(
            dataset,
            train=train,
            validation=validation,
            test=test,
            state=state,
        )
        self.add_dataset("train", name, splits.train)
        self.add_dataset("validation", name, splits.validation)
        self.add_dataset("test", name, splits.test)
        return splits

    def specs(
        self, stage: Stage
    ) -> collections.abc.Mapping[str, DataSpec | None]:
        """Infer first-sample specs without materializing whole datasets."""
        normalized = self._stage(stage)
        return types.MappingProxyType(
            {
                name: infer_data_spec(dataset[0]) if len(dataset) else None
                for name, dataset in self._datasets[normalized].items()
            }
        )

    def diagnostics(self) -> dict[str, object]:
        """Return non-materializing stage counts and lifecycle state."""
        return {
            "prepared": self._prepared,
            "setup_stages": tuple(sorted(self._setup_stages)),
            "datasets": {
                stage: {
                    name: len(dataset) for name, dataset in datasets.items()
                }
                for stage, datasets in self._datasets.items()
            },
        }


__all__ = ["DataModule", "Stage"]
