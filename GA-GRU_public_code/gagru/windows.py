from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .errors import DataValidationError


DEFAULT_METADATA_COLUMNS = (
    "well_id",
    "block",
    "segment_id",
    "depth",
    "class_id",
    "class_name",
    "raw_lithology",
    "interval_top",
    "interval_bottom",
    "distance_to_boundary_m",
    "near_boundary_0_25m",
)


@dataclass(frozen=True)
class WindowedDataset:
    X: np.ndarray
    y: np.ndarray
    metadata: pd.DataFrame
    feature_names: tuple[str, ...]
    window_length: int

    def __post_init__(self) -> None:
        if self.X.ndim != 3:
            raise ValueError(f"X must have shape [samples, sequence, features], got {self.X.shape}")
        if self.X.shape[0] != len(self.y) or len(self.y) != len(self.metadata):
            raise ValueError("X, y, and metadata sample counts differ")
        if self.X.shape[1] != self.window_length:
            raise ValueError("X sequence dimension does not match window_length")
        if self.X.shape[2] != len(self.feature_names):
            raise ValueError("X feature dimension does not match feature_names")

    @property
    def wells(self) -> tuple[str, ...]:
        return tuple(self.metadata["well_id"].drop_duplicates().astype(str))

    def subset_wells(self, well_ids: Iterable[str]) -> "WindowedDataset":
        requested = set(well_ids)
        mask = self.metadata["well_id"].isin(requested).to_numpy()
        found = set(self.metadata.loc[mask, "well_id"].astype(str))
        missing = requested.difference(found)
        if missing:
            raise DataValidationError(f"Requested wells have no windows: {sorted(missing)}")
        return WindowedDataset(
            X=self.X[mask],
            y=self.y[mask],
            metadata=self.metadata.loc[mask].reset_index(drop=True),
            feature_names=self.feature_names,
            window_length=self.window_length,
        )

    def with_features(self, transformed_X: np.ndarray) -> "WindowedDataset":
        return replace(self, X=np.asarray(transformed_X, dtype=np.float32))


def build_windows(
    frame: pd.DataFrame,
    feature_names: Sequence[str],
    window_length: int,
    *,
    metadata_columns: Sequence[str] = DEFAULT_METADATA_COLUMNS,
) -> WindowedDataset:
    if window_length < 1 or window_length % 2 == 0:
        raise ValueError("window_length must be a positive odd integer")
    required = {"well_id", "segment_id", "depth", "class_id", *feature_names}
    missing = required.difference(frame.columns)
    if missing:
        raise DataValidationError(f"Cannot build windows; missing columns: {sorted(missing)}")

    features = tuple(feature_names)
    kept_metadata = [column for column in metadata_columns if column in frame.columns]
    windows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    metadata_parts: list[pd.DataFrame] = []
    half = window_length // 2

    grouped = frame.groupby(["well_id", "segment_id"], sort=False, observed=True)
    for (_, _), segment in grouped:
        segment = segment.sort_values("depth", kind="stable")
        sample_count = len(segment)
        if sample_count < window_length:
            continue
        depths = segment["depth"].to_numpy(dtype=float)
        if sample_count > 1 and not np.allclose(np.diff(depths), 0.125, atol=1e-7, rtol=0):
            raise DataValidationError("A requested window segment is not continuous at 0.125 m")

        values = segment.loc[:, features].to_numpy(dtype=np.float32)
        starts = np.arange(sample_count - window_length + 1)[:, None]
        offsets = np.arange(window_length)[None, :]
        windows.append(values[starts + offsets])

        center_positions = np.arange(half, sample_count - half)
        center_rows = segment.iloc[center_positions]
        labels.append(center_rows["class_id"].to_numpy(dtype=np.int64))
        center_metadata = center_rows.loc[:, kept_metadata].copy()
        center_metadata["source_frame_index"] = center_rows.index.to_numpy()
        metadata_parts.append(center_metadata)

    if windows:
        X = np.concatenate(windows, axis=0)
        y = np.concatenate(labels, axis=0)
        metadata = pd.concat(metadata_parts, ignore_index=True)
    else:
        X = np.empty((0, window_length, len(features)), dtype=np.float32)
        y = np.empty((0,), dtype=np.int64)
        metadata = pd.DataFrame(columns=[*kept_metadata, "source_frame_index"])

    if not np.isfinite(X).all():
        raise DataValidationError("Window tensor contains non-finite values")
    return WindowedDataset(
        X=X,
        y=y,
        metadata=metadata,
        feature_names=features,
        window_length=window_length,
    )
