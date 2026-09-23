from __future__ import annotations

import numpy as np
import pandas as pd

from gagru.random_center import (
    add_source_row_ids,
    build_random_center_windows,
    center_and_context_overlap_audit,
    combine_random_center_windows,
    impute_from_training_windows,
    stratified_center_assignment,
)


def example_frame() -> pd.DataFrame:
    rows = 120
    return pd.DataFrame(
        {
            "well_id": ["W1"] * rows,
            "segment_id": ["S1"] * rows,
            "depth": 1000.0 + np.arange(rows) * 0.125,
            "class_id": np.arange(rows) % 3,
            "F1": np.arange(rows, dtype=float),
            "F2": np.arange(rows, dtype=float) * 2,
        }
    )


def test_stratified_centers_are_disjoint_and_covered() -> None:
    frame = add_source_row_ids(example_frame())
    assignment = stratified_center_assignment(
        frame, window_length=9, seed=17, fractions=(0.7, 0.15, 0.15)
    )
    assert not assignment["source_row_id"].duplicated().any()
    assert set(assignment["split"]) == {"train", "validation", "test"}
    assert assignment.groupby("split")["class_id"].nunique().eq(3).all()


def test_windows_share_context_but_not_target_centers() -> None:
    frame = add_source_row_ids(example_frame())
    assignment = stratified_center_assignment(frame, window_length=9, seed=17)
    windows = build_random_center_windows(frame, assignment, ("F1", "F2"), 9)
    audit = center_and_context_overlap_audit(windows)
    assert audit["center_overlap_across_splits"] == 0
    assert audit["pairs"]["train_to_test"]["shared_context_rows"] > 0


def test_missing_values_use_training_window_medians() -> None:
    frame = add_source_row_ids(example_frame())
    assignment = stratified_center_assignment(frame, window_length=9, seed=17)
    frame.loc[10, "F2"] = np.nan
    windows = build_random_center_windows(frame, assignment, ("F1", "F2"), 9)
    filled, medians = impute_from_training_windows(windows)
    assert np.isfinite(medians["F2"])
    assert all(np.isfinite(batch.X).all() for batch in filled.values())


def test_combine_random_center_windows_preserves_metadata() -> None:
    frame = add_source_row_ids(example_frame())
    assignment = stratified_center_assignment(frame, window_length=9, seed=17)
    windows = build_random_center_windows(frame, assignment, ("F1", "F2"), 9)
    combined = combine_random_center_windows(windows["train"], windows["validation"])
    assert len(combined.y) == len(windows["train"].y) + len(windows["validation"].y)
    assert combined.feature_names == ("F1", "F2")
    assert len(np.unique(combined.center_ids)) == len(combined.center_ids)
