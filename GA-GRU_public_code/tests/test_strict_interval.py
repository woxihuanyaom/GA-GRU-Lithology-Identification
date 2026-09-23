from __future__ import annotations

import numpy as np
import pandas as pd

from gagru.strict_interval import (
    add_complete_interval_ids,
    audit_strict_partition_isolation,
    build_strict_partition_windows,
    materialize_strict_candidate,
    search_strict_interval_split,
)
from gagru.within_well import prepare_within_well_frames


def example_frame() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    depth = 1000.0
    for interval_row in range(12):
        class_id = interval_row % 2
        top = depth
        bottom = depth + 31 * 0.125
        for local_row in range(32):
            records.append(
                {
                    "well_id": "W1",
                    "segment_id": "W1_S1",
                    "depth": depth,
                    "class_id": class_id,
                    "class_name": f"C{class_id}",
                    "interval_top": top,
                    "interval_bottom": bottom,
                    "source_interval_file": "labels.xlsx",
                    "source_interval_row": interval_row + 2,
                    "F1": 10.0 + class_id + local_row / 100.0,
                    "F2": 20.0 + class_id + local_row / 100.0,
                }
            )
            depth += 0.125
    return pd.DataFrame(records)


def frozen_candidate():
    frame = add_complete_interval_ids(example_frame())
    candidate = search_strict_interval_split(
        frame,
        seed=17,
        required_classes=(0, 1),
        trials=256,
    )
    return frame, candidate


def test_complete_interval_ids_preserve_original_label_intervals() -> None:
    frame = add_complete_interval_ids(example_frame())
    assert frame["lithology_interval_id"].nunique() == 12
    assert frame.groupby("lithology_interval_id")["class_id"].nunique().max() == 1
    assert not frame["source_row_id"].duplicated().any()


def test_strict_split_assigns_each_interval_once_and_has_zero_overlap() -> None:
    frame, candidate = frozen_candidate()
    partitions, purged, _, assignment = materialize_strict_candidate(
        frame, candidate, purge_each_side_m=1.0
    )
    assert len(assignment) == frame["lithology_interval_id"].nunique()
    assert assignment["lithology_interval_id"].nunique() == len(assignment)
    assert set(assignment["split"]) == {"train", "validation", "test"}
    assert len(purged) == candidate.purged_rows

    audit = audit_strict_partition_isolation(
        partitions, window_length=9, purge_each_side_m=1.0
    )
    assert audit["target_center_overlap_across_splits"] == 0
    assert audit["source_row_overlap_across_splits"] == 0
    assert audit["input_context_overlap_across_splits"] == 0
    assert audit["complete_interval_overlap_across_splits"] == 0


def test_strict_preprocessing_uses_training_values_only() -> None:
    frame, candidate = frozen_candidate()
    partitions, _, _, _ = materialize_strict_candidate(
        frame, candidate, purge_each_side_m=1.0
    )
    train = partitions["train"].copy()
    validation = partitions["validation"].copy()
    train["F1"] = 2.0
    validation["F1"] = 2000.0

    task = prepare_within_well_frames(
        train,
        validation,
        well_id="W1",
        features=("F1", "F2"),
        window_length=9,
        global_classes=(0, 1),
        require_validation_all_classes=False,
    )
    assert task.imputation_medians["F1"] == 2.0
    assert task.train.X.shape[2] == 2
    assert task.validation.X.shape[2] == 2


def test_strict_window_builder_preserves_center_and_context_ids() -> None:
    frame, candidate = frozen_candidate()
    partitions, _, _, _ = materialize_strict_candidate(
        frame, candidate, purge_each_side_m=1.0
    )
    windows = build_strict_partition_windows(partitions["train"], ("F1", "F2"), 9)
    assert len(windows.y) == int(candidate.window_counts.loc["train"].sum())
    assert len(np.unique(windows.center_ids)) == len(windows.center_ids)
    assert all(len(context) == 9 for context in windows.context_ids)
