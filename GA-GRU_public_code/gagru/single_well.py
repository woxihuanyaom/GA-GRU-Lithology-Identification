from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .errors import DataValidationError


SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class GroupedSplitCandidate:
    assignment: np.ndarray
    outer_seed: int
    outer_fold: int
    inner_seed: int
    inner_fold: int
    objective: float
    split_boundaries: int
    purged_rows: int
    retained_rows: int
    row_counts: pd.DataFrame
    window_counts: pd.DataFrame


def add_depth_blocks(frame: pd.DataFrame, block_width_m: float) -> pd.DataFrame:
    if block_width_m <= 0:
        raise ValueError("block_width_m must be positive")
    required = {"well_id", "segment_id", "depth", "class_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise DataValidationError(f"Single-well frame missing columns: {sorted(missing)}")

    result = frame.sort_values(["segment_id", "depth"], kind="stable").reset_index(
        drop=True
    )
    if result["well_id"].astype(str).nunique() != 1:
        raise DataValidationError("A single-well split received more than one well")
    if result.duplicated(["well_id", "depth"]).any():
        raise DataValidationError("Single-well frame contains duplicate depths")

    result = result.copy()
    result["source_row_id"] = np.arange(len(result), dtype=np.int64)
    result["source_segment_id"] = result["segment_id"].astype(str)
    segment_top = result.groupby("source_segment_id", sort=False)["depth"].transform(
        "min"
    )
    block_number = np.floor(
        (result["depth"].to_numpy(float) - segment_top.to_numpy(float) + 1e-9)
        / block_width_m
    ).astype(int)
    result["depth_block_number"] = block_number
    result["depth_block_id"] = (
        result["source_segment_id"]
        + "_B"
        + pd.Series(block_number, index=result.index).map(lambda value: f"{value:03d}")
    )
    return result


def resegment_after_class_filter(frame: pd.DataFrame) -> pd.DataFrame:
    """Create continuous source segments after uniformly excluding sparse classes."""

    required = {"well_id", "segment_id", "depth"}
    missing = required.difference(frame.columns)
    if missing:
        raise DataValidationError(f"Cannot resegment; missing columns: {sorted(missing)}")
    result = frame.sort_values(["segment_id", "depth"], kind="stable").reset_index(
        drop=True
    )
    result = result.copy()
    result["pre_filter_segment_id"] = result["segment_id"].astype(str)
    new_run = (
        result["pre_filter_segment_id"].ne(result["pre_filter_segment_id"].shift())
        | result["depth"].sub(result["depth"].shift()).sub(0.125).abs().gt(1e-7)
    )
    run_number = new_run.cumsum().astype(int)
    result["segment_id"] = (
        result["well_id"].astype(str)
        + "_F"
        + run_number.map(lambda value: f"{value:04d}")
    )
    return result


def _apply_purge(
    blocked: pd.DataFrame,
    assignment: np.ndarray,
    purge_each_side_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    if len(assignment) != len(blocked):
        raise ValueError("Split assignment length differs from the source frame")
    if purge_each_side_m < 0:
        raise ValueError("purge_each_side_m cannot be negative")

    work = blocked.copy()
    work["split"] = np.asarray(assignment, dtype=str)
    if not set(work["split"]).issubset(SPLITS):
        raise DataValidationError("Split assignment contains an unknown partition")

    purge = np.zeros(len(work), dtype=bool)
    boundary_records: list[dict[str, object]] = []
    for source_segment, segment in work.groupby(
        "source_segment_id", sort=False, observed=True
    ):
        positions = segment.index.to_numpy(dtype=int)
        depths = segment["depth"].to_numpy(dtype=float)
        partitions = segment["split"].to_numpy(dtype=str)
        if len(depths) > 1 and not np.allclose(
            np.diff(depths), 0.125, atol=1e-7, rtol=0
        ):
            raise DataValidationError(
                f"Source segment {source_segment} is not continuous at 0.125 m"
            )
        changes = np.flatnonzero(partitions[:-1] != partitions[1:])
        for change in changes:
            midpoint = float((depths[change] + depths[change + 1]) / 2.0)
            if purge_each_side_m > 0:
                local = np.abs(depths - midpoint) < purge_each_side_m
                purge[positions[local]] = True
            boundary_records.append(
                {
                    "source_segment_id": str(source_segment),
                    "boundary_depth_m": midpoint,
                    "left_split": str(partitions[change]),
                    "right_split": str(partitions[change + 1]),
                }
            )

    work["purged"] = purge
    retained = work.loc[~purge].copy()
    if retained.empty:
        raise DataValidationError("Purging removed every row")

    new_run = (
        retained["source_segment_id"].ne(retained["source_segment_id"].shift())
        | retained["split"].ne(retained["split"].shift())
        | retained["depth"].sub(retained["depth"].shift()).sub(0.125).abs().gt(1e-7)
    )
    run_number = new_run.cumsum().astype(int)
    retained["partition_segment_id"] = (
        retained["well_id"].astype(str)
        + "_"
        + retained["split"].astype(str)
        + "_R"
        + run_number.map(lambda value: f"{value:04d}")
    )
    retained["segment_id"] = retained["partition_segment_id"]
    boundaries = pd.DataFrame(
        boundary_records,
        columns=[
            "source_segment_id",
            "boundary_depth_m",
            "left_split",
            "right_split",
        ],
    )
    return retained.reset_index(drop=True), boundaries, int(purge.sum())


def window_class_counts(
    retained: pd.DataFrame,
    window_length: int,
    *,
    num_classes: int = 10,
) -> pd.DataFrame:
    if window_length < 1 or window_length % 2 == 0:
        raise ValueError("window_length must be a positive odd integer")
    half = window_length // 2
    records: list[tuple[str, int]] = []
    grouped = retained.groupby(
        ["split", "partition_segment_id"], sort=False, observed=True
    )
    for (split, _), segment in grouped:
        ordered = segment.sort_values("depth", kind="stable")
        if len(ordered) < window_length:
            continue
        centers = ordered.iloc[half : len(ordered) - half]
        records.extend((str(split), int(label)) for label in centers["class_id"])
    if not records:
        return pd.DataFrame(0, index=SPLITS, columns=range(num_classes), dtype=int)
    counts = pd.crosstab(
        pd.Series([item[0] for item in records], name="split"),
        pd.Series([item[1] for item in records], name="class_id"),
    )
    return counts.reindex(index=SPLITS, columns=range(num_classes), fill_value=0).astype(
        int
    )


def row_class_counts(
    retained: pd.DataFrame, *, num_classes: int = 10
) -> pd.DataFrame:
    counts = pd.crosstab(retained["split"], retained["class_id"])
    return counts.reindex(index=SPLITS, columns=range(num_classes), fill_value=0).astype(
        int
    )


def _candidate_objective(
    row_counts: pd.DataFrame,
    window_counts: pd.DataFrame,
    target_fractions: np.ndarray,
    purged_rows: int,
    total_rows: int,
    required_classes: tuple[int, ...],
) -> float:
    row_totals = row_counts.sum(axis=1).to_numpy(dtype=float)
    window_totals = window_counts.sum(axis=1).to_numpy(dtype=float)
    row_fraction = row_totals / row_totals.sum()
    window_fraction = window_totals / window_totals.sum()
    class_fraction = window_counts.loc[:, list(required_classes)].to_numpy(dtype=float)
    class_fraction /= np.maximum(class_fraction.sum(axis=0, keepdims=True), 1.0)
    return float(
        np.square(row_fraction - target_fractions).sum()
        + np.square(window_fraction - target_fractions).sum()
        + 3.0 * np.square(class_fraction - target_fractions[:, None]).mean()
        + 0.05 * (purged_rows / total_rows)
    )


def choose_grouped_split(
    blocked: pd.DataFrame,
    *,
    purge_each_side_m: float,
    window_length: int,
    outer_seeds: Iterable[int],
    inner_seeds: Iterable[int],
    outer_n_splits: int = 5,
    inner_n_splits: int = 4,
    target_fractions: Iterable[float] = (0.6, 0.2, 0.2),
    required_classes: Iterable[int] | None = None,
    minimum_train_windows_per_class: int = 20,
    minimum_evaluation_windows_per_class: int = 5,
) -> GroupedSplitCandidate:
    if outer_n_splits < 2 or inner_n_splits < 2:
        raise ValueError("outer_n_splits and inner_n_splits must be at least two")
    target = np.asarray(tuple(target_fractions), dtype=float)
    if target.shape != (3,) or np.any(target <= 0) or not np.isclose(target.sum(), 1.0):
        raise ValueError("target_fractions must contain three positive values summing to one")
    labels = blocked["class_id"].to_numpy(dtype=int)
    groups = blocked["depth_block_id"].astype(str).to_numpy()
    classes = tuple(
        sorted(
            set(int(value) for value in labels)
            if required_classes is None
            else set(int(value) for value in required_classes)
        )
    )
    if not classes or not set(classes).issubset(set(int(value) for value in labels)):
        raise DataValidationError("Required classes are empty or absent from the source frame")
    best: GroupedSplitCandidate | None = None

    for outer_seed in outer_seeds:
        outer = StratifiedGroupKFold(
            n_splits=outer_n_splits, shuffle=True, random_state=int(outer_seed)
        )
        for outer_fold, (development_index, test_index) in enumerate(
            outer.split(blocked, labels, groups), start=1
        ):
            development = blocked.iloc[development_index]
            development_labels = labels[development_index]
            development_groups = groups[development_index]
            for inner_seed in inner_seeds:
                inner = StratifiedGroupKFold(
                    n_splits=inner_n_splits, shuffle=True, random_state=int(inner_seed)
                )
                for inner_fold, (training_relative, validation_relative) in enumerate(
                    inner.split(
                        development, development_labels, development_groups
                    ),
                    start=1,
                ):
                    assignment = np.full(len(blocked), "test", dtype=object)
                    assignment[development_index[training_relative]] = "train"
                    assignment[development_index[validation_relative]] = "validation"
                    retained, boundaries, purged_rows = _apply_purge(
                        blocked, assignment, purge_each_side_m
                    )
                    row_counts = row_class_counts(retained)
                    window_counts = window_class_counts(retained, window_length)
                    required_counts = window_counts.loc[:, list(classes)]
                    if (
                        (
                            required_counts.loc["train"]
                            < minimum_train_windows_per_class
                        ).any()
                        or (
                            required_counts.loc[["validation", "test"]]
                            < minimum_evaluation_windows_per_class
                        )
                        .any()
                        .any()
                    ):
                        continue
                    objective = _candidate_objective(
                        row_counts,
                        window_counts,
                        target,
                        purged_rows,
                        len(blocked),
                        classes,
                    )
                    candidate = GroupedSplitCandidate(
                        assignment=np.asarray(assignment, dtype=str),
                        outer_seed=int(outer_seed),
                        outer_fold=int(outer_fold),
                        inner_seed=int(inner_seed),
                        inner_fold=int(inner_fold),
                        objective=objective,
                        split_boundaries=int(len(boundaries)),
                        purged_rows=int(purged_rows),
                        retained_rows=int(len(retained)),
                        row_counts=row_counts,
                        window_counts=window_counts,
                    )
                    rank = (
                        candidate.objective,
                        candidate.split_boundaries,
                        candidate.purged_rows,
                        candidate.outer_seed,
                        candidate.outer_fold,
                        candidate.inner_seed,
                        candidate.inner_fold,
                    )
                    if best is None:
                        best = candidate
                    else:
                        best_rank = (
                            best.objective,
                            best.split_boundaries,
                            best.purged_rows,
                            best.outer_seed,
                            best.outer_fold,
                            best.inner_seed,
                            best.inner_fold,
                        )
                        if rank < best_rank:
                            best = candidate

    if best is None:
        raise DataValidationError(
            "No label-covered grouped split satisfied the frozen minimum counts"
        )
    return best


def materialize_candidate(
    blocked: pd.DataFrame,
    candidate: GroupedSplitCandidate,
    *,
    purge_each_side_m: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    retained, boundaries, _ = _apply_purge(
        blocked, candidate.assignment, purge_each_side_m
    )
    purged = blocked.copy()
    purged["split"] = candidate.assignment
    retained_ids = set(retained["source_row_id"].astype(int))
    purged = purged.loc[~purged["source_row_id"].isin(retained_ids)].copy()
    purged["purged"] = True
    return retained, purged.reset_index(drop=True), boundaries


def validate_partition_isolation(
    partitions: dict[str, pd.DataFrame], *, purge_each_side_m: float
) -> dict[str, object]:
    if set(partitions) != set(SPLITS):
        raise DataValidationError("Exactly train, validation, and test partitions are required")
    row_sets = {
        split: set(frame["source_row_id"].astype(int))
        for split, frame in partitions.items()
    }
    for index, first in enumerate(SPLITS):
        for second in SPLITS[index + 1 :]:
            overlap = row_sets[first].intersection(row_sets[second])
            if overlap:
                raise DataValidationError(
                    f"Partitions {first}/{second} share {len(overlap)} source rows"
                )

    minimum_distances: dict[str, float | None] = {}
    for index, first in enumerate(SPLITS):
        for second in SPLITS[index + 1 :]:
            pair_name = f"{first}_to_{second}"
            minimum = np.inf
            for source_segment in set(
                partitions[first]["source_segment_id"].astype(str)
            ).intersection(partitions[second]["source_segment_id"].astype(str)):
                first_depths = partitions[first].loc[
                    partitions[first]["source_segment_id"].astype(str).eq(source_segment),
                    "depth",
                ].to_numpy(float)
                second_depths = partitions[second].loc[
                    partitions[second]["source_segment_id"].astype(str).eq(source_segment),
                    "depth",
                ].to_numpy(float)
                if len(first_depths) and len(second_depths):
                    distance = np.abs(
                        first_depths[:, None] - second_depths[None, :]
                    ).min()
                    minimum = min(minimum, float(distance))
            minimum_distances[pair_name] = None if not np.isfinite(minimum) else minimum
            if np.isfinite(minimum) and minimum < 2.0 * purge_each_side_m:
                raise DataValidationError(
                    f"{pair_name} retained rows are only {minimum:.3f} m apart"
                )

    return {
        "source_row_overlap": 0,
        "minimum_retained_depth_separation_m": minimum_distances,
    }
