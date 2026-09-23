from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .errors import DataValidationError


SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class RandomCenterWindows:
    X: np.ndarray
    y: np.ndarray
    wells: np.ndarray
    depths: np.ndarray
    center_ids: np.ndarray
    context_ids: tuple[np.ndarray, ...]
    feature_names: tuple[str, ...]
    window_length: int

    def __post_init__(self) -> None:
        samples = len(self.y)
        if self.X.shape != (samples, self.window_length, len(self.feature_names)):
            raise ValueError("Random-center window tensor has an invalid shape")
        if not (
            len(self.wells)
            == len(self.depths)
            == len(self.center_ids)
            == len(self.context_ids)
            == samples
        ):
            raise ValueError("Random-center window arrays have different lengths")


def add_source_row_ids(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"well_id", "segment_id", "depth", "class_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise DataValidationError(f"Cannot add row IDs; missing columns: {sorted(missing)}")
    result = frame.sort_values(["segment_id", "depth"], kind="stable").reset_index(
        drop=True
    )
    if result["well_id"].astype(str).nunique() != 1:
        raise DataValidationError("Random-center task must contain exactly one well")
    if result.duplicated(["well_id", "depth"]).any():
        raise DataValidationError("Random-center task contains duplicate depths")
    result = result.copy()
    result["source_row_id"] = np.arange(len(result), dtype=np.int64)
    return result


def _continuous_runs(frame: pd.DataFrame) -> Iterable[pd.DataFrame]:
    grouped = frame.groupby(["well_id", "segment_id"], sort=False, observed=True)
    for _, segment in grouped:
        ordered = segment.sort_values("depth", kind="stable")
        depths = ordered["depth"].to_numpy(float)
        start = np.r_[
            True,
            ~np.isclose(np.diff(depths), 0.125, atol=1e-7, rtol=0),
        ]
        run_ids = np.cumsum(start)
        for _, run in ordered.groupby(run_ids, sort=False):
            yield run


def eligible_center_ids(frame: pd.DataFrame, window_length: int) -> np.ndarray:
    if window_length < 1 or window_length % 2 == 0:
        raise ValueError("window_length must be a positive odd integer")
    if "source_row_id" not in frame.columns:
        raise DataValidationError("Random-center frame lacks source_row_id")
    half = window_length // 2
    parts: list[np.ndarray] = []
    for run in _continuous_runs(frame):
        if len(run) >= window_length:
            parts.append(run.iloc[half : len(run) - half]["source_row_id"].to_numpy(np.int64))
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


def stratified_center_assignment(
    frame: pd.DataFrame,
    *,
    window_length: int,
    seed: int,
    fractions: Sequence[float] = (0.70, 0.15, 0.15),
) -> pd.DataFrame:
    values = np.asarray(tuple(fractions), dtype=float)
    if values.shape != (3,) or np.any(values <= 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("fractions must contain three positive values summing to one")
    eligible = set(eligible_center_ids(frame, window_length).tolist())
    centers = frame.loc[frame["source_row_id"].isin(eligible)].copy()
    if centers.empty:
        raise DataValidationError("No eligible random centers were found")
    rng = np.random.default_rng(seed)
    records: list[pd.DataFrame] = []
    for class_id, group in centers.groupby("class_id", sort=True, observed=True):
        ids = group["source_row_id"].to_numpy(np.int64)
        ids = ids[rng.permutation(len(ids))]
        if len(ids) < 3:
            raise DataValidationError(
                f"Class {class_id} has fewer than three eligible centers"
            )
        validation_count = max(1, int(round(len(ids) * values[1])))
        test_count = max(1, int(round(len(ids) * values[2])))
        while validation_count + test_count >= len(ids):
            if validation_count >= test_count and validation_count > 1:
                validation_count -= 1
            elif test_count > 1:
                test_count -= 1
            else:
                raise DataValidationError(f"Class {class_id} cannot cover all splits")
        train_count = len(ids) - validation_count - test_count
        split_values = np.asarray(
            ["train"] * train_count
            + ["validation"] * validation_count
            + ["test"] * test_count,
            dtype=object,
        )
        records.append(
            pd.DataFrame(
                {
                    "source_row_id": ids,
                    "class_id": int(class_id),
                    "split": split_values,
                }
            )
        )
    assignment = pd.concat(records, ignore_index=True)
    assignment = assignment.merge(
        centers.loc[:, ["source_row_id", "well_id", "depth", "segment_id"]],
        on="source_row_id",
        how="left",
        validate="one_to_one",
    )
    return assignment.sort_values("source_row_id", kind="stable").reset_index(drop=True)


def build_random_center_windows(
    frame: pd.DataFrame,
    assignment: pd.DataFrame,
    feature_names: Sequence[str],
    window_length: int,
    *,
    requested_splits: Sequence[str] = SPLITS,
) -> dict[str, RandomCenterWindows]:
    features = tuple(feature_names)
    selected_splits = tuple(str(value) for value in requested_splits)
    if not selected_splits or len(set(selected_splits)) != len(selected_splits):
        raise ValueError("requested_splits must contain unique split names")
    if not set(selected_splits).issubset(SPLITS):
        raise ValueError("requested_splits contains an unknown split")
    required = {"well_id", "depth", "class_id", "source_row_id", *features}
    missing = required.difference(frame.columns)
    if missing:
        raise DataValidationError(f"Random-center frame missing columns: {sorted(missing)}")
    assignment_required = {"source_row_id", "class_id", "split"}
    assignment_missing = assignment_required.difference(assignment.columns)
    if assignment_missing:
        raise DataValidationError(
            f"Random-center assignment missing columns: {sorted(assignment_missing)}"
        )
    if assignment["source_row_id"].duplicated().any():
        raise DataValidationError("A center appears more than once in the assignment")
    if not set(assignment["split"].astype(str)).issubset(selected_splits):
        raise DataValidationError("Random-center assignment contains an unknown split")

    assignment_map = assignment.set_index("source_row_id")["split"].astype(str).to_dict()
    assigned_labels = assignment.set_index("source_row_id")["class_id"].astype(int).to_dict()
    half = window_length // 2
    parts: dict[str, list[list[object]]] = {
        split: [[], [], [], [], [], []] for split in selected_splits
    }
    for run in _continuous_runs(frame):
        if len(run) < window_length:
            continue
        values = run.loc[:, features].to_numpy(np.float32)
        labels = run["class_id"].to_numpy(np.int64)
        row_ids = run["source_row_id"].to_numpy(np.int64)
        depths = run["depth"].to_numpy(float)
        wells = run["well_id"].astype(str).to_numpy()
        for center in range(half, len(run) - half):
            center_id = int(row_ids[center])
            split = assignment_map.get(center_id)
            if split is None:
                continue
            if int(labels[center]) != assigned_labels[center_id]:
                raise DataValidationError("Frozen center label differs from source data")
            context = row_ids[center - half : center + half + 1].copy()
            target = parts[split]
            target[0].append(values[center - half : center + half + 1])
            target[1].append(int(labels[center]))
            target[2].append(str(wells[center]))
            target[3].append(float(depths[center]))
            target[4].append(center_id)
            target[5].append(context)

    result: dict[str, RandomCenterWindows] = {}
    for split, target in parts.items():
        if not target[0]:
            raise DataValidationError(f"Random-center split {split} has no windows")
        result[split] = RandomCenterWindows(
            X=np.stack(target[0]).astype(np.float32),
            y=np.asarray(target[1], dtype=np.int64),
            wells=np.asarray(target[2], dtype=str),
            depths=np.asarray(target[3], dtype=float),
            center_ids=np.asarray(target[4], dtype=np.int64),
            context_ids=tuple(target[5]),
            feature_names=features,
            window_length=window_length,
        )
    return result


def impute_from_training_windows(
    windows: dict[str, RandomCenterWindows],
) -> tuple[dict[str, RandomCenterWindows], dict[str, float]]:
    if "train" not in windows or not set(windows).issubset(SPLITS):
        raise DataValidationError("Windows must include train and use known split names")
    medians = np.nanmedian(windows["train"].X, axis=(0, 1))
    if not np.isfinite(medians).all():
        raise DataValidationError("Training windows do not provide finite feature medians")
    result: dict[str, RandomCenterWindows] = {}
    for split, batch in windows.items():
        values = np.asarray(batch.X, dtype=np.float32).copy()
        missing = ~np.isfinite(values)
        if missing.any():
            feature_indices = np.nonzero(missing)[2]
            values[missing] = medians[feature_indices]
        if not np.isfinite(values).all():
            raise DataValidationError(f"Non-finite values remain in {split} windows")
        result[split] = RandomCenterWindows(
            X=values,
            y=batch.y,
            wells=batch.wells,
            depths=batch.depths,
            center_ids=batch.center_ids,
            context_ids=batch.context_ids,
            feature_names=batch.feature_names,
            window_length=batch.window_length,
        )
    return result, {
        feature: float(medians[index])
        for index, feature in enumerate(windows["train"].feature_names)
    }


def combine_random_center_windows(
    first: RandomCenterWindows,
    second: RandomCenterWindows,
) -> RandomCenterWindows:
    if first.feature_names != second.feature_names:
        raise ValueError("Cannot combine random-center windows with different features")
    if first.window_length != second.window_length:
        raise ValueError("Cannot combine random-center windows with different lengths")
    shared_centers = set(first.center_ids.astype(int)).intersection(
        second.center_ids.astype(int)
    )
    if shared_centers:
        raise DataValidationError(
            f"Cannot combine windows with {len(shared_centers)} shared target centers"
        )
    return RandomCenterWindows(
        X=np.concatenate((first.X, second.X), axis=0),
        y=np.concatenate((first.y, second.y), axis=0),
        wells=np.concatenate((first.wells, second.wells), axis=0),
        depths=np.concatenate((first.depths, second.depths), axis=0),
        center_ids=np.concatenate((first.center_ids, second.center_ids), axis=0),
        context_ids=first.context_ids + second.context_ids,
        feature_names=first.feature_names,
        window_length=first.window_length,
    )


def center_and_context_overlap_audit(
    windows: dict[str, RandomCenterWindows],
) -> dict[str, object]:
    if set(windows) != set(SPLITS):
        raise DataValidationError("Exactly train, validation, and test windows are required")
    center_sets = {
        split: set(batch.center_ids.astype(int).tolist())
        for split, batch in windows.items()
    }
    context_sets = {
        split: set(np.concatenate(batch.context_ids).astype(int).tolist())
        for split, batch in windows.items()
    }
    pair_records: dict[str, dict[str, float | int]] = {}
    for index, first in enumerate(SPLITS):
        for second in SPLITS[index + 1 :]:
            center_overlap = center_sets[first].intersection(center_sets[second])
            if center_overlap:
                raise DataValidationError(
                    f"Random-center splits {first}/{second} share target centers"
                )
            shared_context = context_sets[first].intersection(context_sets[second])
            denominator = max(1, len(context_sets[second]))
            pair_records[f"{first}_to_{second}"] = {
                "center_overlap": 0,
                "shared_context_rows": len(shared_context),
                "second_context_rows": len(context_sets[second]),
                "second_context_overlap_fraction": len(shared_context) / denominator,
            }
    return {
        "center_overlap_across_splits": 0,
        "context_overlap_is_expected_and_disclosed": True,
        "pairs": pair_records,
    }


def remap_labels(
    windows: RandomCenterWindows, classes: Sequence[int]
) -> tuple[np.ndarray, dict[int, int]]:
    ordered = tuple(int(value) for value in classes)
    mapping = {global_id: local_id for local_id, global_id in enumerate(ordered)}
    try:
        local = np.asarray([mapping[int(value)] for value in windows.y], dtype=np.int64)
    except KeyError as exc:
        raise DataValidationError("Window label lies outside the frozen local classes") from exc
    return local, mapping
