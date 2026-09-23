from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .errors import DataValidationError
from .protocol import FrozenProtocol


TRACE_COLUMNS = (
    "well_id",
    "block",
    "depth",
    "class_id",
    "class_name",
    "legacy_class_id",
    "raw_lithology",
    "interval_top",
    "interval_bottom",
    "distance_to_boundary_m",
    "near_boundary_0_25m",
    "segment_id",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DataRepository:
    """Read and validate only the well files declared by the frozen protocol."""

    def __init__(self, protocol: FrozenProtocol, *, verify_hashes: bool = True) -> None:
        self.protocol = protocol
        self.verify_hashes = verify_hashes
        self._cache: dict[str, pd.DataFrame] = {}

    def source_path(self, well_id: str) -> Path:
        spec = self.protocol.well(well_id)
        root = self.protocol.data_root.resolve()
        path = (root / Path(spec.source_relative_path)).resolve()
        if path != root and root not in path.parents:
            raise DataValidationError(f"Well path escapes the frozen data root: {well_id}")
        return path

    def load_well(
        self,
        well_id: str,
        *,
        allow_locked_external: bool = False,
    ) -> pd.DataFrame:
        self.protocol.assert_access_allowed(
            [well_id], allow_locked_external=allow_locked_external
        )
        spec = self.protocol.well(well_id)
        if spec.role == "four_curve_sensitivity":
            raise DataValidationError(
                f"{well_id} is a four-curve sensitivity well and cannot enter the five-curve loader"
            )

        if well_id not in self._cache:
            path = self.source_path(well_id)
            if not path.is_file():
                raise DataValidationError(f"Well file does not exist: {path}")
            if self.verify_hashes:
                expected_hash = self.protocol.raw["dataset"]["file_sha256"][well_id]
                actual_hash = file_sha256(path)
                if actual_hash != expected_hash:
                    raise DataValidationError(
                        f"{well_id} SHA-256 mismatch; expected {expected_hash}, got {actual_hash}"
                    )
            frame = pd.read_csv(path, encoding="utf-8-sig")
            self._validate_frame(frame, well_id)
            self._cache[well_id] = frame
        return self._cache[well_id].copy(deep=True)

    def load_wells(
        self,
        well_ids: Iterable[str],
        *,
        allow_locked_external: bool = False,
    ) -> pd.DataFrame:
        ordered = tuple(well_ids)
        if not ordered:
            raise DataValidationError("At least one well must be requested")
        if len(set(ordered)) != len(ordered):
            raise DataValidationError("A well was requested more than once")
        self.protocol.assert_access_allowed(
            ordered, allow_locked_external=allow_locked_external
        )
        frames = [
            self.load_well(well_id, allow_locked_external=allow_locked_external)
            for well_id in ordered
        ]
        return pd.concat(frames, ignore_index=True, copy=False)

    def load_role(
        self,
        role: str,
        *,
        allow_locked_external: bool = False,
    ) -> pd.DataFrame:
        return self.load_wells(
            self.protocol.wells_for_role(role),
            allow_locked_external=allow_locked_external,
        )

    def clear_cache(self) -> None:
        self._cache.clear()

    def _validate_frame(self, frame: pd.DataFrame, well_id: str) -> None:
        spec = self.protocol.well(well_id)
        required = set(TRACE_COLUMNS).union(self.protocol.feature_names)
        missing = required.difference(frame.columns)
        if missing:
            raise DataValidationError(f"{well_id} missing columns: {sorted(missing)}")
        if len(frame) != spec.rows_5curve:
            raise DataValidationError(
                f"{well_id} expected {spec.rows_5curve} rows, found {len(frame)}"
            )
        if set(frame["well_id"].astype(str)) != {well_id}:
            raise DataValidationError(f"{well_id} file contains another well ID")
        if frame.duplicated(["well_id", "depth"]).any():
            raise DataValidationError(f"{well_id} contains duplicate sample depths")

        numeric_columns = ["depth", *self.protocol.feature_names, "class_id"]
        numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise DataValidationError(f"{well_id} contains missing or non-finite model values")
        if (numeric[list(self.protocol.resistivity_names)] <= 0).any().any():
            raise DataValidationError(f"{well_id} contains non-positive resistivity")

        class_values = numeric["class_id"].to_numpy(dtype=float)
        if not np.equal(class_values, np.floor(class_values)).all():
            raise DataValidationError(f"{well_id} contains non-integer class_id values")

        actual_classes = tuple(sorted(int(value) for value in frame["class_id"].unique()))
        if actual_classes != spec.class_ids_present:
            raise DataValidationError(
                f"{well_id} class coverage changed: expected {spec.class_ids_present}, "
                f"found {actual_classes}"
            )
        expected_names = frame["class_id"].map(self.protocol.class_names)
        if not expected_names.equals(frame["class_name"]):
            raise DataValidationError(f"{well_id} class_id and class_name do not agree")
        legacy_values = pd.to_numeric(frame["legacy_class_id"], errors="coerce")
        expected_legacy = frame["class_id"].map(self.protocol.legacy_class_ids)
        if legacy_values.isna().any() or not np.array_equal(
            legacy_values.to_numpy(dtype=int), expected_legacy.to_numpy(dtype=int)
        ):
            raise DataValidationError(
                f"{well_id} class_id and legacy_class_id do not agree with the frozen mapping"
            )

        boundary = frame["near_boundary_0_25m"]
        if boundary.dtype != bool:
            normalized = boundary.astype(str).str.strip().str.lower()
            if not normalized.isin(["true", "false"]).all():
                raise DataValidationError(f"{well_id} has invalid boundary flags")
            boundary = normalized.eq("true")
        if int(boundary.sum()) != spec.boundary_rows:
            raise DataValidationError(
                f"{well_id} boundary count changed: expected {spec.boundary_rows}, "
                f"found {int(boundary.sum())}"
            )

        segment_count = frame["segment_id"].nunique()
        if segment_count != spec.continuous_segments:
            raise DataValidationError(
                f"{well_id} expected {spec.continuous_segments} segments, found {segment_count}"
            )
        for segment_id, segment in frame.groupby("segment_id", sort=False):
            depths = segment["depth"].to_numpy(dtype=float)
            if len(depths) > 1 and not np.allclose(np.diff(depths), 0.125, atol=1e-7, rtol=0):
                raise DataValidationError(
                    f"{well_id} segment {segment_id} is not continuous at 0.125 m"
                )
