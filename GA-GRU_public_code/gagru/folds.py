from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .data import DataRepository
from .errors import DataValidationError
from .preprocessing import CurvePreprocessor, compute_class_weights
from .windows import WindowedDataset, build_windows


@dataclass(frozen=True)
class PreparedSplit:
    train: WindowedDataset
    evaluation: WindowedDataset
    preprocessor: CurvePreprocessor
    class_weights: np.ndarray
    train_wells: tuple[str, ...]
    evaluation_wells: tuple[str, ...]
    split_name: str


def prepare_well_split(
    repository: DataRepository,
    train_wells: Iterable[str],
    evaluation_wells: Iterable[str],
    *,
    window_length: int = 9,
    split_name: str,
) -> PreparedSplit:
    train_ids = tuple(train_wells)
    evaluation_ids = tuple(evaluation_wells)
    overlap = set(train_ids).intersection(evaluation_ids)
    if overlap:
        raise DataValidationError(f"Train/evaluation well leakage: {sorted(overlap)}")
    repository.protocol.assert_access_allowed(train_ids)
    repository.protocol.assert_access_allowed(evaluation_ids)

    train_frame = repository.load_wells(train_ids)
    evaluation_frame = repository.load_wells(evaluation_ids)
    features = repository.protocol.feature_names
    train_raw = build_windows(train_frame, features, window_length)
    evaluation_raw = build_windows(evaluation_frame, features, window_length)
    if len(train_raw.y) == 0 or len(evaluation_raw.y) == 0:
        raise DataValidationError(f"{split_name} produced an empty window set")

    preprocessor = CurvePreprocessor.fit(
        train_raw.X,
        features,
        repository.protocol.resistivity_names,
        training_wells=train_ids,
        locked_external_wells=repository.protocol.locked_external_wells,
    )
    train = train_raw.with_features(preprocessor.transform(train_raw.X))
    evaluation = evaluation_raw.with_features(preprocessor.transform(evaluation_raw.X))
    weights = compute_class_weights(train.y, num_classes=len(repository.protocol.class_names))
    return PreparedSplit(
        train=train,
        evaluation=evaluation,
        preprocessor=preprocessor,
        class_weights=weights,
        train_wells=train_ids,
        evaluation_wells=evaluation_ids,
        split_name=split_name,
    )


def prepare_inner_fold(
    repository: DataRepository,
    fold_number: int,
    *,
    window_length: int = 9,
) -> PreparedSplit:
    fold = repository.protocol.fold(fold_number)
    return prepare_well_split(
        repository,
        fold.inner_training_wells,
        fold.inner_validation_wells,
        window_length=window_length,
        split_name=f"outer_fold_{fold_number}_inner",
    )


def prepare_outer_fold(
    repository: DataRepository,
    fold_number: int,
    *,
    window_length: int = 9,
) -> PreparedSplit:
    fold = repository.protocol.fold(fold_number)
    return prepare_well_split(
        repository,
        fold.outer_training_wells,
        [fold.outer_test_well],
        window_length=window_length,
        split_name=f"outer_fold_{fold_number}",
    )
