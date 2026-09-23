from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data import DataRepository
from .errors import DataValidationError, ProtocolError
from .folds import prepare_well_split
from .protocol import FrozenProtocol, load_frozen_protocol
from .windows import build_windows


FROZEN_DEVELOPMENT_WINDOW_COUNTS = {1: 42785, 5: 40991, 9: 39691, 17: 37847}


def _window_count_from_segments(frame: pd.DataFrame, window_length: int) -> int:
    sizes = frame.groupby(["well_id", "segment_id"], observed=True).size()
    return int(np.maximum(sizes.to_numpy() - window_length + 1, 0).sum())


def _aggregate_declared_hash(protocol: FrozenProtocol) -> str:
    hashes = protocol.raw["dataset"]["file_sha256"]
    payload = "".join(
        f"{well_id},{hashes[well_id].lower()}\n" for well_id in sorted(hashes)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _class_counts(labels: Iterable[int], number_of_classes: int) -> dict[str, int]:
    values = np.fromiter(labels, dtype=np.int64)
    counts = np.bincount(values, minlength=number_of_classes)
    return {str(index): int(count) for index, count in enumerate(counts)}


def validate_development_pipeline(
    protocol_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run leakage-safe checks without reading either locked external well."""

    protocol = load_frozen_protocol(protocol_dir)
    repository = DataRepository(protocol, verify_hashes=True)
    development_wells = protocol.wells_for_role("development")
    locked_wells = protocol.locked_external_wells
    if set(development_wells).intersection(locked_wells):
        raise ProtocolError("A development well is also marked as locked external")

    declared_five_curve_rows = sum(
        spec.rows_5curve
        for spec in protocol.wells.values()
        if spec.role != "four_curve_sensitivity"
    )
    expected_five_curve_rows = int(protocol.raw["dataset"]["five_curve_rows"])
    if declared_five_curve_rows != expected_five_curve_rows:
        raise ProtocolError(
            "Five-curve row total differs between the JSON and well manifest"
        )
    aggregate_hash = _aggregate_declared_hash(protocol)
    if aggregate_hash != protocol.raw["dataset"]["aggregate_well_file_sha256"]:
        raise ProtocolError("Declared per-well hashes do not reproduce the aggregate hash")

    frames: list[pd.DataFrame] = []
    well_reports: dict[str, dict[str, Any]] = {}
    for well_id in development_wells:
        frame = repository.load_well(well_id)
        frames.append(frame)
        well_reports[well_id] = {
            "rows": int(len(frame)),
            "segments": int(frame["segment_id"].nunique()),
            "classes_present": sorted(int(value) for value in frame["class_id"].unique()),
            "class_counts": _class_counts(
                frame["class_id"].to_numpy(dtype=np.int64), len(protocol.class_names)
            ),
        }

    development_frame = pd.concat(frames, ignore_index=True, copy=False)
    declared_development_rows = sum(protocol.well(well).rows_5curve for well in development_wells)
    if len(development_frame) != declared_development_rows:
        raise DataValidationError("Loaded development row count differs from the manifest")

    window_reports: dict[str, dict[str, Any]] = {}
    for window_length, expected_count in FROZEN_DEVELOPMENT_WINDOW_COUNTS.items():
        windows = build_windows(
            development_frame,
            protocol.feature_names,
            window_length,
        )
        formula_count = _window_count_from_segments(development_frame, window_length)
        if len(windows.y) != formula_count or len(windows.y) != expected_count:
            raise DataValidationError(
                f"Development L={window_length} window count changed: "
                f"expected {expected_count}, built {len(windows.y)}, formula {formula_count}"
            )
        window_reports[str(window_length)] = {
            "samples": int(len(windows.y)),
            "shape": list(windows.X.shape),
            "class_counts": _class_counts(windows.y, len(protocol.class_names)),
        }

    all_classes = set(protocol.class_names)
    fold_reports: list[dict[str, Any]] = []
    development_set = set(development_wells)
    for fold_number in sorted(protocol.folds):
        fold = protocol.fold(fold_number)
        outer_training = set(fold.outer_training_wells)
        outer_test = {fold.outer_test_well}
        inner_training = set(fold.inner_training_wells)
        inner_validation = set(fold.inner_validation_wells)
        if outer_training.intersection(outer_test):
            raise DataValidationError(f"Outer fold {fold_number} has well leakage")
        if outer_training.union(outer_test) != development_set:
            raise DataValidationError(f"Outer fold {fold_number} does not cover development wells")
        if inner_training.intersection(inner_validation):
            raise DataValidationError(f"Inner fold {fold_number} has well leakage")
        if inner_training.union(inner_validation) != outer_training:
            raise DataValidationError(
                f"Inner fold {fold_number} does not partition outer training wells"
            )

        inner_classes = set().union(
            *(set(protocol.well(well).class_ids_present) for well in inner_training)
        )
        outer_classes = set().union(
            *(set(protocol.well(well).class_ids_present) for well in outer_training)
        )
        if inner_classes != all_classes or outer_classes != all_classes:
            raise DataValidationError(f"Fold {fold_number} training does not cover all classes")
        fold_reports.append(
            {
                "fold": fold_number,
                "outer_test_well": fold.outer_test_well,
                "outer_training_wells": list(fold.outer_training_wells),
                "inner_validation_wells": list(fold.inner_validation_wells),
                "inner_training_wells": list(fold.inner_training_wells),
                "inner_training_classes": sorted(inner_classes),
                "outer_training_classes": sorted(outer_classes),
                "leakage": False,
            }
        )

    final_search = protocol.raw["final_development_search"]
    prepared = prepare_well_split(
        repository,
        final_search["training"],
        final_search["validation"],
        window_length=int(protocol.raw["sampling"]["primary_window_length"]),
        split_name="final_development_search",
    )
    train_means = prepared.train.X.mean(axis=(0, 1), dtype=np.float64)
    train_stds = prepared.train.X.std(axis=(0, 1), dtype=np.float64)
    if not np.allclose(train_means, 0.0, atol=1e-5):
        raise DataValidationError("Training windows are not centered after preprocessing")
    if not np.allclose(train_stds, 1.0, atol=1e-5):
        raise DataValidationError("Training windows do not have unit variance after preprocessing")
    if set(repository._cache).intersection(locked_wells):
        raise DataValidationError("A locked external well entered the development cache")

    return {
        "status": "PASS",
        "checked_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "protocol_id": protocol.raw["protocol"]["id"],
        "protocol_version": protocol.raw["protocol"]["version"],
        "protocol_directory": str(protocol.directory),
        "declared_data_root": str(protocol.declared_data_root),
        "data_root": str(protocol.data_root),
        "locked_external_wells_not_read": sorted(locked_wells),
        "development": {
            "wells": list(development_wells),
            "rows": int(len(development_frame)),
            "well_checks": well_reports,
            "windows": window_reports,
        },
        "folds": fold_reports,
        "final_development_search": {
            "training_wells": list(prepared.train_wells),
            "validation_wells": list(prepared.evaluation_wells),
            "training_windows": int(len(prepared.train.y)),
            "validation_windows": int(len(prepared.evaluation.y)),
            "standardized_training_means": train_means.tolist(),
            "standardized_training_stds": train_stds.tolist(),
            "class_weights": prepared.class_weights.tolist(),
        },
        "declared_aggregate_hash": aggregate_hash,
    }


def write_validation_report(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path
