from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from .errors import DataValidationError
from .random_center import RandomCenterWindows
from .single_well import (
    SPLITS,
    _apply_purge,
    row_class_counts,
    validate_partition_isolation,
    window_class_counts,
)


@dataclass(frozen=True)
class StrictIntervalCandidate:
    interval_split_codes: np.ndarray
    seed: int
    trial: int
    window_length: int
    objective: float
    missing_evaluation_classes: int
    split_boundaries: int
    purged_rows: int
    retained_rows: int
    row_counts: pd.DataFrame
    window_counts: pd.DataFrame


def add_complete_interval_ids(
    frame: pd.DataFrame, *, sampling_interval_m: float = 0.125
) -> pd.DataFrame:
    """Attach immutable IDs to complete, contiguous labeled intervals."""

    if sampling_interval_m <= 0:
        raise ValueError("sampling_interval_m must be positive")
    required = {
        "well_id",
        "segment_id",
        "depth",
        "class_id",
        "interval_top",
        "interval_bottom",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise DataValidationError(
            f"Strict-interval frame missing columns: {sorted(missing)}"
        )

    result = frame.sort_values(["segment_id", "depth"], kind="stable").reset_index(
        drop=True
    )
    if result["well_id"].astype(str).nunique() != 1:
        raise DataValidationError(
            "A strict-interval task must contain exactly one well"
        )
    if result.duplicated(["well_id", "depth"]).any():
        raise DataValidationError("A strict-interval task contains duplicate depths")
    if "source_row_id" not in result:
        result = result.copy()
        result["source_row_id"] = np.arange(len(result), dtype=np.int64)
    if result["source_row_id"].duplicated().any():
        raise DataValidationError("Strict-interval source_row_id values are not unique")

    result = result.copy()
    result["source_segment_id"] = result["segment_id"].astype(str)
    identity_columns = ["class_id", "interval_top", "interval_bottom"]
    for column in ("source_interval_file", "source_interval_row"):
        if column in result:
            identity_columns.append(column)
    identity_changed = (
        result.loc[:, identity_columns]
        .ne(result.loc[:, identity_columns].shift())
        .any(axis=1)
    )
    depth_gap = ~np.isclose(
        result["depth"].diff().fillna(sampling_interval_m).to_numpy(float),
        sampling_interval_m,
        atol=1e-7,
        rtol=0,
    )
    new_interval = (
        result["source_segment_id"].ne(result["source_segment_id"].shift())
        | identity_changed
        | depth_gap
    )
    new_interval.iloc[0] = True
    sequence = new_interval.cumsum().astype(int)
    well_id = str(result["well_id"].iloc[0])
    result["interval_sequence"] = sequence
    result["lithology_interval_id"] = sequence.map(
        lambda value: f"{well_id}_I{value:05d}"
    )

    class_counts = result.groupby("lithology_interval_id", sort=False)[
        "class_id"
    ].nunique()
    if class_counts.max() != 1:
        raise DataValidationError("A complete lithology interval contains mixed labels")
    return result


def complete_interval_manifest(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "lithology_interval_id",
        "interval_sequence",
        "source_segment_id",
        "class_id",
        "depth",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise DataValidationError(
            f"Cannot create interval manifest; missing columns: {sorted(missing)}"
        )
    manifest = (
        frame.groupby(
            [
                "lithology_interval_id",
                "interval_sequence",
                "source_segment_id",
                "class_id",
            ],
            sort=False,
            as_index=False,
        )
        .agg(
            top_depth_m=("depth", "min"),
            bottom_depth_m=("depth", "max"),
            rows=("depth", "size"),
        )
        .sort_values("interval_sequence", kind="stable")
        .reset_index(drop=True)
    )
    expected = np.arange(1, len(manifest) + 1)
    if not np.array_equal(manifest["interval_sequence"].to_numpy(int), expected):
        raise DataValidationError("Complete interval sequence is not contiguous")
    return manifest


def _allocation_counts(count: int, fractions: np.ndarray) -> np.ndarray:
    if count < len(SPLITS):
        raise DataValidationError(
            "Every modeled class needs at least three complete labeled intervals"
        )
    raw = fractions * count
    allocated = np.maximum(np.floor(raw).astype(int), 1)
    while allocated.sum() > count:
        removable = np.flatnonzero(allocated > 1)
        if not len(removable):
            raise DataValidationError(
                "Cannot allocate complete intervals to three splits"
            )
        index = removable[np.argmax(allocated[removable] - raw[removable])]
        allocated[index] -= 1
    while allocated.sum() < count:
        index = int(np.argmax(raw - allocated))
        allocated[index] += 1
    return allocated


def _purge_mask(
    frame: pd.DataFrame,
    row_split_codes: np.ndarray,
    *,
    purge_each_side_m: float,
    sampling_interval_m: float,
) -> tuple[np.ndarray, int]:
    rows = len(frame)
    if len(row_split_codes) != rows:
        raise ValueError("row_split_codes has the wrong length")
    source_segments = frame["source_segment_id"].astype(str).to_numpy()
    depths = frame["depth"].to_numpy(float)
    same_source = (source_segments[:-1] == source_segments[1:]) & np.isclose(
        np.diff(depths), sampling_interval_m, atol=1e-7, rtol=0
    )
    boundary_positions = np.flatnonzero(
        same_source & (row_split_codes[:-1] != row_split_codes[1:])
    )
    if purge_each_side_m == 0 or not len(boundary_positions):
        return np.zeros(rows, dtype=bool), int(len(boundary_positions))

    side_rows = int(np.ceil(purge_each_side_m / sampling_interval_m - 0.5 - 1e-12))
    if side_rows <= 0:
        return np.zeros(rows, dtype=bool), int(len(boundary_positions))

    segment_start = np.r_[
        True,
        (source_segments[1:] != source_segments[:-1])
        | ~np.isclose(np.diff(depths), sampling_interval_m, atol=1e-7, rtol=0),
    ]
    starts = np.flatnonzero(segment_start)
    ends = np.r_[starts[1:], rows]
    row_start = np.repeat(starts, ends - starts)
    row_end = np.repeat(ends, ends - starts)

    difference = np.zeros(rows + 1, dtype=np.int32)
    purge_starts = np.maximum(
        row_start[boundary_positions], boundary_positions - side_rows + 1
    )
    purge_ends = np.minimum(
        row_end[boundary_positions], boundary_positions + side_rows + 1
    )
    np.add.at(difference, purge_starts, 1)
    np.add.at(difference, purge_ends, -1)
    return np.cumsum(difference[:-1]) > 0, int(len(boundary_positions))


def _candidate_counts(
    frame: pd.DataFrame,
    interval_split_codes: np.ndarray,
    *,
    window_length: int,
    purge_each_side_m: float,
    sampling_interval_m: float,
) -> tuple[np.ndarray, np.ndarray, int, int, int]:
    interval_index = frame["interval_sequence"].to_numpy(int) - 1
    row_splits = interval_split_codes[interval_index]
    labels = frame["class_id"].to_numpy(int)
    if labels.min() < 0 or labels.max() > 9:
        raise DataValidationError("Strict sensitivity expects global class IDs 0-9")
    purge, boundary_count = _purge_mask(
        frame,
        row_splits,
        purge_each_side_m=purge_each_side_m,
        sampling_interval_m=sampling_interval_m,
    )
    retained_indices = np.flatnonzero(~purge)
    row_counts = np.zeros((len(SPLITS), 10), dtype=int)
    np.add.at(row_counts, (row_splits[retained_indices], labels[retained_indices]), 1)

    window_counts = np.zeros((len(SPLITS), 10), dtype=int)
    if len(retained_indices):
        source_segments = frame["source_segment_id"].astype(str).to_numpy()
        depths = frame["depth"].to_numpy(float)
        new_run = np.r_[
            True,
            (np.diff(retained_indices) != 1)
            | (
                source_segments[retained_indices[1:]]
                != source_segments[retained_indices[:-1]]
            )
            | (row_splits[retained_indices[1:]] != row_splits[retained_indices[:-1]])
            | ~np.isclose(
                depths[retained_indices[1:]] - depths[retained_indices[:-1]],
                sampling_interval_m,
                atol=1e-7,
                rtol=0,
            ),
        ]
        run_starts = np.flatnonzero(new_run)
        run_ends = np.r_[run_starts[1:], len(retained_indices)]
        half = window_length // 2
        centers = [
            retained_indices[start + half : end - half]
            for start, end in zip(run_starts, run_ends, strict=True)
            if end - start >= window_length
        ]
        if centers:
            center_indices = np.concatenate(centers)
            np.add.at(
                window_counts,
                (row_splits[center_indices], labels[center_indices]),
                1,
            )
    return (
        row_counts,
        window_counts,
        int(purge.sum()),
        int(len(frame) - purge.sum()),
        boundary_count,
    )


def search_strict_interval_split(
    frame: pd.DataFrame,
    *,
    seed: int,
    required_classes: Sequence[int],
    window_length: int = 9,
    fractions: Sequence[float] = (0.64, 0.16, 0.20),
    purge_each_side_m: float = 1.0,
    sampling_interval_m: float = 0.125,
    trials: int = 4096,
    minimum_train_windows_per_class: int = 1,
) -> StrictIntervalCandidate:
    """Select a model-blind, class-stratified complete-interval assignment."""

    if trials < 1:
        raise ValueError("trials must be positive")
    if window_length < 1 or window_length % 2 == 0:
        raise ValueError("window_length must be a positive odd integer")
    target = np.asarray(tuple(fractions), dtype=float)
    if target.shape != (3,) or np.any(target <= 0) or not np.isclose(target.sum(), 1):
        raise ValueError("fractions must contain three positive values summing to one")
    classes = tuple(sorted(set(int(value) for value in required_classes)))
    if not classes:
        raise DataValidationError("required_classes is empty")

    manifest = complete_interval_manifest(frame)
    interval_classes = manifest["class_id"].to_numpy(int)
    class_intervals = {
        class_id: np.flatnonzero(interval_classes == class_id) for class_id in classes
    }
    absent = [
        class_id for class_id, indices in class_intervals.items() if not len(indices)
    ]
    if absent:
        raise DataValidationError(f"Required classes lack intervals: {absent}")
    allocations = {
        class_id: _allocation_counts(len(indices), target)
        for class_id, indices in class_intervals.items()
    }

    rng = np.random.default_rng(int(seed))
    best: StrictIntervalCandidate | None = None
    required_index = np.asarray(classes, dtype=int)
    for trial in range(trials):
        interval_splits = np.empty(len(manifest), dtype=np.int8)
        for class_id, indices in class_intervals.items():
            shuffled = rng.permutation(indices)
            counts = allocations[class_id]
            train_end = int(counts[0])
            validation_end = train_end + int(counts[1])
            interval_splits[shuffled[:train_end]] = 0
            interval_splits[shuffled[train_end:validation_end]] = 1
            interval_splits[shuffled[validation_end:]] = 2

        row_counts_array, window_counts_array, purged, retained, boundaries = (
            _candidate_counts(
                frame,
                interval_splits,
                window_length=window_length,
                purge_each_side_m=purge_each_side_m,
                sampling_interval_m=sampling_interval_m,
            )
        )
        required_windows = window_counts_array[:, required_index]
        if (
            (required_windows[0] < minimum_train_windows_per_class).any()
            or required_windows[1].sum() == 0
            or required_windows[2].sum() == 0
        ):
            continue

        row_totals = row_counts_array[:, required_index].sum(axis=1).astype(float)
        window_totals = required_windows.sum(axis=1).astype(float)
        row_fractions = row_totals / row_totals.sum()
        window_fractions = window_totals / window_totals.sum()
        class_fractions = required_windows / np.maximum(
            required_windows.sum(axis=0, keepdims=True), 1
        )
        objective = float(
            np.square(window_fractions - target).sum()
            + 0.5 * np.square(row_fractions - target).sum()
            + 1.5 * np.square(class_fractions - target[:, None]).mean()
            + 0.05 * (purged / len(frame))
            + 0.01 * (boundaries / max(1, len(manifest) - 1))
        )
        missing_evaluation = int((required_windows[1:] == 0).sum())
        row_counts = pd.DataFrame(
            row_counts_array, index=SPLITS, columns=range(10), dtype=int
        )
        window_counts = pd.DataFrame(
            window_counts_array, index=SPLITS, columns=range(10), dtype=int
        )
        candidate = StrictIntervalCandidate(
            interval_split_codes=interval_splits.copy(),
            seed=int(seed),
            trial=int(trial),
            window_length=int(window_length),
            objective=objective,
            missing_evaluation_classes=missing_evaluation,
            split_boundaries=boundaries,
            purged_rows=purged,
            retained_rows=retained,
            row_counts=row_counts,
            window_counts=window_counts,
        )
        rank = (
            candidate.missing_evaluation_classes,
            candidate.objective,
            candidate.split_boundaries,
            candidate.purged_rows,
            candidate.trial,
        )
        if best is None:
            best = candidate
        else:
            best_rank = (
                best.missing_evaluation_classes,
                best.objective,
                best.split_boundaries,
                best.purged_rows,
                best.trial,
            )
            if rank < best_rank:
                best = candidate

    if best is None:
        raise DataValidationError(
            "No complete-interval split retained every required training class"
        )
    return best


def materialize_strict_candidate(
    frame: pd.DataFrame,
    candidate: StrictIntervalCandidate,
    *,
    purge_each_side_m: float,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest = complete_interval_manifest(frame)
    if len(candidate.interval_split_codes) != len(manifest):
        raise DataValidationError("Candidate interval assignment has the wrong length")
    assignment = manifest.copy()
    assignment["split"] = [SPLITS[int(code)] for code in candidate.interval_split_codes]
    mapping = assignment.set_index("lithology_interval_id")["split"]
    row_assignment = frame["lithology_interval_id"].map(mapping)
    if row_assignment.isna().any():
        raise DataValidationError("A source row lacks a complete-interval assignment")

    retained, boundaries, purged_count = _apply_purge(
        frame, row_assignment.to_numpy(str), purge_each_side_m
    )
    retained_ids = set(retained["source_row_id"].astype(int))
    purged = frame.loc[~frame["source_row_id"].astype(int).isin(retained_ids)].copy()
    purged["split"] = row_assignment.loc[purged.index].to_numpy(str)
    purged["purged"] = True
    if purged_count != len(purged) or purged_count != candidate.purged_rows:
        raise DataValidationError("Fast and materialized purge counts differ")
    partitions = {
        split: retained.loc[retained["split"].eq(split)].copy().reset_index(drop=True)
        for split in SPLITS
    }
    observed_rows = row_class_counts(retained)
    observed_windows = window_class_counts(retained, candidate.window_length)
    if not observed_rows.equals(candidate.row_counts):
        raise DataValidationError("Fast and materialized row counts differ")
    if not observed_windows.equals(candidate.window_counts):
        raise DataValidationError("Fast and materialized window counts differ")
    return partitions, purged.reset_index(drop=True), boundaries, assignment


def _window_membership(
    frame: pd.DataFrame, window_length: int
) -> tuple[set[int], set[int]]:
    half = window_length // 2
    centers: set[int] = set()
    contexts: set[int] = set()
    for _, segment in frame.groupby(["well_id", "segment_id"], sort=False):
        ordered = segment.sort_values("depth", kind="stable")
        if len(ordered) < window_length:
            continue
        depths = ordered["depth"].to_numpy(float)
        if not np.allclose(np.diff(depths), 0.125, atol=1e-7, rtol=0):
            raise DataValidationError("A strict partition segment is not continuous")
        row_ids = ordered["source_row_id"].to_numpy(int)
        centers.update(row_ids[half : len(row_ids) - half].tolist())
        contexts.update(row_ids.tolist())
    return centers, contexts


def audit_strict_partition_isolation(
    partitions: dict[str, pd.DataFrame],
    *,
    window_length: int,
    purge_each_side_m: float,
) -> dict[str, object]:
    if set(partitions) != set(SPLITS):
        raise DataValidationError("Exactly train, validation, and test are required")
    physical = validate_partition_isolation(
        partitions, purge_each_side_m=purge_each_side_m
    )
    row_sets = {
        split: set(frame["source_row_id"].astype(int))
        for split, frame in partitions.items()
    }
    interval_sets = {
        split: set(frame["lithology_interval_id"].astype(str))
        for split, frame in partitions.items()
    }
    memberships = {
        split: _window_membership(frame, window_length)
        for split, frame in partitions.items()
    }
    pair_records: dict[str, dict[str, int]] = {}
    for index, first in enumerate(SPLITS):
        for second in SPLITS[index + 1 :]:
            center_overlap = len(
                memberships[first][0].intersection(memberships[second][0])
            )
            source_overlap = len(row_sets[first].intersection(row_sets[second]))
            context_overlap = len(
                memberships[first][1].intersection(memberships[second][1])
            )
            interval_overlap = len(
                interval_sets[first].intersection(interval_sets[second])
            )
            if any((center_overlap, source_overlap, context_overlap, interval_overlap)):
                raise DataValidationError(
                    f"Strict partitions {first}/{second} are not fully isolated"
                )
            pair_records[f"{first}_to_{second}"] = {
                "target_center_overlap": center_overlap,
                "source_row_overlap": source_overlap,
                "input_context_overlap": context_overlap,
                "complete_interval_overlap": interval_overlap,
            }
    return {
        "target_center_overlap_across_splits": 0,
        "source_row_overlap_across_splits": 0,
        "input_context_overlap_across_splits": 0,
        "complete_interval_overlap_across_splits": 0,
        "minimum_retained_depth_separation_m": physical[
            "minimum_retained_depth_separation_m"
        ],
        "pairs": pair_records,
    }


def build_strict_partition_windows(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    window_length: int,
) -> RandomCenterWindows:
    """Build traceable windows wholly inside one frozen strict partition."""

    if window_length < 1 or window_length % 2 == 0:
        raise ValueError("window_length must be a positive odd integer")
    features = tuple(str(value) for value in feature_names)
    required = {
        "well_id",
        "segment_id",
        "depth",
        "class_id",
        "source_row_id",
        *features,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise DataValidationError(
            f"Strict partition cannot form windows; missing: {sorted(missing)}"
        )
    if frame["source_row_id"].duplicated().any():
        raise DataValidationError("Strict partition duplicates source-row IDs")

    window_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    well_parts: list[np.ndarray] = []
    depth_parts: list[np.ndarray] = []
    center_parts: list[np.ndarray] = []
    contexts: list[np.ndarray] = []
    half = window_length // 2
    for _, segment in frame.groupby(["well_id", "segment_id"], sort=False):
        ordered = segment.sort_values("depth", kind="stable")
        if len(ordered) < window_length:
            continue
        depths = ordered["depth"].to_numpy(float)
        if not np.allclose(np.diff(depths), 0.125, atol=1e-7, rtol=0):
            raise DataValidationError("A strict window segment is not continuous")
        values = ordered.loc[:, features].to_numpy(np.float32)
        labels = ordered["class_id"].to_numpy(np.int64)
        row_ids = ordered["source_row_id"].to_numpy(np.int64)
        wells = ordered["well_id"].astype(str).to_numpy()
        starts = np.arange(len(ordered) - window_length + 1)[:, None]
        offsets = np.arange(window_length)[None, :]
        index = starts + offsets
        center_positions = np.arange(half, len(ordered) - half)
        window_parts.append(values[index])
        label_parts.append(labels[center_positions])
        well_parts.append(wells[center_positions])
        depth_parts.append(depths[center_positions])
        center_parts.append(row_ids[center_positions])
        contexts.extend(row_ids[row_index].copy() for row_index in index)

    if not window_parts:
        raise DataValidationError("Strict partition produced no windows")
    return RandomCenterWindows(
        X=np.concatenate(window_parts).astype(np.float32, copy=False),
        y=np.concatenate(label_parts),
        wells=np.concatenate(well_parts),
        depths=np.concatenate(depth_parts),
        center_ids=np.concatenate(center_parts),
        context_ids=tuple(contexts),
        feature_names=features,
        window_length=window_length,
    )
